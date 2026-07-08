#!/usr/bin/env python3
"""
fit_tuned_lens.py  (stable training)

Direct tuned-lens: per-layer affine translator, init identity (= logit lens at
step 0), trained by KL so norm+unembed of the translated hidden matches the
final-layer distribution. The tuned_lens package does not support this model
set, so this implements the same object directly.

Stability fixes over the first version (which oscillated):
  - gradient accumulation over --accum sequences per optimizer step (smooths the
    noisy single-sequence gradient)
  - gradient clipping (kills occasional blow-ups)
  - cosine LR schedule with warmup (late steps settle instead of thrash)
  - FIXED held-out eval reported by layer band (early/mid/late). The all-layer
    average KL floors high because early layers lack the information; watch the
    LATE band, which is where commitment is read and where the lens must work.

Run from repo root, .venv active, GPU free:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  # spike to confirm stable descent (~5 min):
  python scripts/fit_tuned_lens.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen_spike --steps 300 --dtype bfloat16 --attn eager
  # full fits:
  python scripts/fit_tuned_lens.py --model Qwen/Qwen2.5-7B-Instruct     --tag qwen_instruct    --steps 3000 --dtype bfloat16 --attn eager
  python scripts/fit_tuned_lens.py --model google/gemma-2-9b-it         --tag gemma_instruct   --steps 3000 --dtype bfloat16 --attn eager
  python scripts/fit_tuned_lens.py --model mistralai/Mistral-7B-Instruct-v0.3 --tag mistral_instruct --steps 3000 --dtype bfloat16 --attn eager
  python scripts/fit_tuned_lens.py --model meta-llama/Llama-3.1-8B-Instruct   --tag llama_instruct   --steps 3000 --dtype bfloat16 --attn eager

Output: data/tuned_lens_direct_<tag>/lens.pt + meta.json (with per-band eval curve)
"""
import argparse, json, math, glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("data")
CAND = DATA / "cat1_candidates_pull2.jsonl"

def load_text_lines(corpus):
    out = []
    if corpus and Path(corpus).exists():
        for line in open(corpus):
            line = line.strip()
            if line:
                try: out.append(json.loads(line).get("text", line))
                except Exception: out.append(line)
        return out
    for line in open(CAND):
        line = line.strip()
        if line: out.append(json.loads(line).get("question", ""))
    for fp in glob.glob(str(DATA / "responses_*.jsonl")):
        for line in open(fp):
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line); t = r.get("completion") or r.get("text") or r.get("response") or ""
                if isinstance(t, str) and len(t) > 20: out.append(t)
            except Exception: pass
    return [t for t in out if t]

class AffineLens(nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        self.t = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        for lin in self.t:
            nn.init.eye_(lin.weight); nn.init.zeros_(lin.bias)
    def forward(self, h, l):
        return self.t[l](h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=3000)     # optimizer steps
    ap.add_argument("--accum", type=int, default=4)        # sequences per step
    ap.add_argument("--lr", type=float, default=1e-3)      # peak lr
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn).to(args.device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    final_norm, lm_head = model.model.norm, model.get_output_embeddings()
    H, nL = model.config.hidden_size, model.config.num_hidden_layers

    lens = AffineLens(nL, H).to(args.device)
    opt = torch.optim.Adam(lens.parameters(), lr=args.lr)

    lines = load_text_lines(args.corpus)
    eval_lines = lines[:16]; train_lines = lines[16:] or lines
    print(f"{args.tag}: {len(train_lines)} train, {len(eval_lines)} eval, {nL} layers, "
          f"{args.steps} steps x accum {args.accum}")

    def enc_of(text):
        return tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len).to(args.device)

    def lens_logits(h):  # h [T,H] float32
        return lm_head(final_norm(h.to(dtype))).float()

    b1, b2, b3 = nL // 3, 2 * nL // 3, nL
    @torch.no_grad()
    def evaluate():
        band = {"early": [], "mid": [], "late": []}
        for text in eval_lines:
            enc = enc_of(text)
            if enc["input_ids"].shape[1] < 8: continue
            out = model(**enc, output_hidden_states=True)
            hs = [h[0].float() for h in out.hidden_states]
            target = F.log_softmax(lens_logits(hs[-1]), dim=-1)
            for l in range(nL):
                pred = F.log_softmax(lens_logits(lens(hs[l], l)), dim=-1)
                kl = float(F.kl_div(pred, target, log_target=True, reduction="batchmean"))
                (band["early"] if l < b1 else band["mid"] if l < b2 else band["late"]).append(kl)
        return {k: (sum(v) / len(v) if v else None) for k, v in band.items()}

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * min(p, 1.0)))

    ti = 0
    def next_enc():
        nonlocal ti
        while True:
            enc = enc_of(train_lines[ti % len(train_lines)]); ti += 1
            if enc["input_ids"].shape[1] >= 8:
                return enc

    curve = []
    for step in range(1, args.steps + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        opt.zero_grad()
        for _ in range(args.accum):
            enc = next_enc()
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                hs = [h[0].float() for h in out.hidden_states]
                target = F.log_softmax(lens_logits(hs[-1]), dim=-1).detach()
            for l in range(nL):
                pred = F.log_softmax(lens_logits(lens(hs[l], l)), dim=-1)
                (F.kl_div(pred, target, log_target=True, reduction="batchmean")
                 / (args.accum * nL)).backward()
        torch.nn.utils.clip_grad_norm_(lens.parameters(), args.clip)
        opt.step()
        if step == 1 or step % args.eval_every == 0:
            b = evaluate()
            curve.append({"step": step, **b})
            print(f"  step {step:5d}  lr {lr_at(step):.2e}  "
                  f"early {b['early']:.2f}  mid {b['mid']:.2f}  late {b['late']:.3f}")

    out_dir = DATA / f"tuned_lens_direct_{args.tag}"; out_dir.mkdir(exist_ok=True)
    torch.save(lens.state_dict(), out_dir / "lens.pt")
    (out_dir / "meta.json").write_text(json.dumps(
        {"model": args.model, "hidden": H, "n_layers": nL, "steps": args.steps,
         "accum": args.accum, "lr": args.lr, "final_band": curve[-1] if curve else None,
         "curve": curve}, indent=2))
    print(f"{args.tag}: saved -> {out_dir}  final late-band KL "
          f"{curve[-1]['late']:.3f}" if curve else "no steps")

if __name__ == "__main__":
    main()
