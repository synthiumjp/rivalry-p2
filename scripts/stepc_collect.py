#!/usr/bin/env python3
"""
stepc_collect.py

STEP C standardised step-0 collection for ONE instruct model over the 800 Cat1
DEVELOPMENT prompts. Produces, per prompt:
  - per-prompt hallucination rate r_p = 1 - correct/20  (locked judge)
  - prompt-final residual at all layers (captured on the prompt forward pass,
    before any generation)

Protocol (locked): 20 samples, temperature 0.7, top_p 0.9; judge = bidirectional
substring, min 2 chars, GT = value + normalized_value + aliases. This is the
standardised protocol, NOT the mixed-metric spike approach.

Reads DEVELOPMENT ids only. Never reads or touches the hold-out.

Run from ~/jpwork/rivalry-p2/H-Neurons/ with:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python scripts/stepc_collect.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen_instruct
  python scripts/stepc_collect.py --model google/gemma-2-9b-it --tag gemma_instruct --dtype bfloat16 --attn eager

Outputs (checkpoint/resume by prompt_id):
  data/stepc_<tag>.jsonl                 one line per prompt: prompt_id, r_p, n_correct, n
  data/stepc_resid_<tag>/<prompt_id>.npy float16 [n_layers+1, hidden] prompt-final residual
"""
import argparse, json, re, os
from pathlib import Path
import numpy as np
import torch

DATA = Path("data")
CAND = DATA / "cat1_candidates_pull2.jsonl"
DEV  = DATA / "cat1_development_prompt_ids.json"

N_SAMPLES, TEMP, TOP_P, MAX_NEW = 20, 0.7, 0.9, 48

_LEAD = re.compile(r"^(the answer is|answer\s*[:\-]?|it is|it's)\s+", re.I)

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_jsonl(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def gt_set(row):
    g = row.get("ground_truth", row.get("answer"))
    out = set()
    def add(x):
        if isinstance(x, str) and x.strip():
            out.add(norm(x))
    if isinstance(g, dict):
        add(g.get("value")); add(g.get("normalized_value"))
        for a in (g.get("aliases") or []):
            add(a)
    elif isinstance(g, list):
        for a in g:
            add(a)
    else:
        add(g)
    return {x for x in out if len(x) >= 2}

def is_correct(cn, gts):
    if len(cn) < 2:
        return False
    return any(a in cn or cn in a for a in gts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev_ids = json.loads(DEV.read_text()).get("prompt_ids")
    assert dev_ids and len(dev_ids) == 800, f"expected 800 dev ids, got {len(dev_ids) if dev_ids else 0}"
    dev_ids = set(dev_ids)

    cand = {r["prompt_id"]: r for r in load_jsonl(CAND) if r.get("prompt_id") in dev_ids}
    assert len(cand) == len(dev_ids), f"missing candidates: have {len(cand)} of {len(dev_ids)}"

    out_jsonl = DATA / f"stepc_{args.tag}.jsonl"
    resid_dir = DATA / f"stepc_resid_{args.tag}"
    resid_dir.mkdir(exist_ok=True)

    done = set()
    if out_jsonl.exists():
        for r in load_jsonl(out_jsonl):
            done.add(r["prompt_id"])
    todo = [pid for pid in dev_ids if pid not in done]
    print(f"{args.tag}: {len(done)} done, {len(todo)} to collect")
    if not todo:
        print("nothing to do"); return

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn,
    ).to(args.device).eval()

    f_out = open(out_jsonl, "a")
    for i, pid in enumerate(todo):
        row = cand[pid]
        q = row["question"]
        gts = gt_set(row)
        msgs = [{"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(args.device)

        # prompt-final residual at all layers (no generation)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple (n_layers+1) of [1, seq, hidden]
        resid = torch.stack([h[0, -1, :] for h in hs]).to(torch.float16).cpu().numpy()
        np.save(resid_dir / f"{pid}.npy", resid)

        # 20 sampled completions for r_p
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=TEMP, top_p=TOP_P,
                num_return_sequences=N_SAMPLES, max_new_tokens=MAX_NEW,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1]:]
        comps = tok.batch_decode(new, skip_special_tokens=True)
        n_correct = sum(is_correct(norm(c), gts) for c in comps)
        r_p = 1.0 - n_correct / N_SAMPLES

        f_out.write(json.dumps({"prompt_id": pid, "r_p": r_p,
                                "n_correct": n_correct, "n": N_SAMPLES}) + "\n")
        f_out.flush()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)}  last r_p={r_p:.2f}")
    f_out.close()
    print(f"{args.tag}: collection complete -> {out_jsonl}")

if __name__ == "__main__":
    main()
