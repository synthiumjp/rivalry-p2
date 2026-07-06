#!/usr/bin/env python3
"""
recapture_and_rescore.py  (cache-loop capture)

Same purpose as before, but the greedy hidden-state capture uses a manual
KV-cache decode loop instead of generate(output_hidden_states=True), which
stalls on MPS. One prefill pass for prompt_final, then single-token forwards
carrying past_key_values, capturing the per-step hidden state at each answer
token. The 20-sample r_p still uses generate (no hidden states needed).

(A) clean-span r_p: 20 samples 0.7/0.9, truncated at the answer span, locked judge.
(B) corrected layer stacks: prompt_final, answer_tok_0..4, last_answer_tok = true
    last answer token. Generated token ids stored.

Reads development ids only. Checkpoint/resume by prompt_id.

Run from repo root, .venv active:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python scripts/recapture_and_rescore.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen_instruct
  python scripts/recapture_and_rescore.py --model google/gemma-2-9b-it --tag gemma_instruct --dtype bfloat16 --attn eager
  python scripts/recapture_and_rescore.py --model mistralai/Mistral-7B-Instruct-v0.3 --tag mistral_instruct
  python scripts/recapture_and_rescore.py --model meta-llama/Llama-3.1-8B-Instruct --tag llama_instruct

Outputs:
  data/rp_clean_<tag>.jsonl
  data/layer_stacks_v2_<tag>/<pid>_0.npz     (+ gen_token_ids, answer_end_idx)
  data/layer_stack_index_v2_<tag>.jsonl
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import torch

from answer_span import truncate_answer_text, answer_span_end_index

DATA = Path("data")
CAND = DATA / "cat1_candidates_pull2.jsonl"
DEV  = DATA / "cat1_development_prompt_ids.json"
N_SAMPLES, TEMP, TOP_P, MAX_NEW = 20, 0.7, 0.9, 48
N_ANSWER_POS = 5

def norm(s):
    s = s.lower().strip(); s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def load_jsonl(p):
    for line in open(p):
        line = line.strip()
        if line:
            yield json.loads(line)

def gt_set(row):
    g = row.get("ground_truth", row.get("answer")); out = set()
    def add(x):
        if isinstance(x, str) and x.strip(): out.add(norm(x))
    if isinstance(g, dict):
        add(g.get("value")); add(g.get("normalized_value"))
        for a in (g.get("aliases") or []): add(a)
    elif isinstance(g, list):
        for a in g: add(a)
    else:
        add(g)
    return {x for x in out if len(x) >= 2}

def is_correct(cn, gts):
    return len(cn) >= 2 and any(a in cn or cn in a for a in gts)

def stack_hidden(hidden_states):
    """hidden_states: tuple over layers of [1, seq, H]. Return [L+1, H] at last pos."""
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in hidden_states])

@torch.no_grad()
def greedy_capture(model, enc, tok, device):
    """Manual cache loop. Returns (gen_ids, per_token_stacks, prompt_final_stack).
    per_token[k] is [L+1,H] at generated token k."""
    out = model(**enc, output_hidden_states=True, use_cache=True)
    prompt_final = stack_hidden(out.hidden_states)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen_ids, per_token = [], []
    for _ in range(MAX_NEW):
        gen_ids.append(int(next_tok.item()))
        step = model(input_ids=next_tok, past_key_values=past,
                     output_hidden_states=True, use_cache=True)
        per_token.append(stack_hidden(step.hidden_states))
        past = step.past_key_values
        if next_tok.item() == tok.eos_token_id:
            break
        next_tok = step.logits[:, -1, :].argmax(-1, keepdim=True)
    return gen_ids, per_token, prompt_final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev_ids = json.loads(DEV.read_text())["prompt_ids"]
    assert len(dev_ids) == 800
    dev_ids = set(dev_ids)
    cand = {r["prompt_id"]: r for r in load_jsonl(CAND) if r.get("prompt_id") in dev_ids}
    assert len(cand) == len(dev_ids)

    rp_path  = DATA / f"rp_clean_{args.tag}.jsonl"
    idx_path = DATA / f"layer_stack_index_v2_{args.tag}.jsonl"
    stk_dir  = DATA / f"layer_stacks_v2_{args.tag}"; stk_dir.mkdir(exist_ok=True)

    done = {r["prompt_id"] for r in load_jsonl(rp_path)} if rp_path.exists() else set()
    todo = [p for p in dev_ids if p not in done]
    print(f"{args.tag}: {len(done)} done, {len(todo)} to do")
    if not todo:
        return

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn).to(args.device).eval()

    f_rp, f_idx = open(rp_path, "a"), open(idx_path, "a")
    for i, pid in enumerate(todo):
        row = cand[pid]; gts = gt_set(row)
        prompt = tok.apply_chat_template([{"role": "user", "content": row["question"]}],
                                         tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        plen = enc["input_ids"].shape[1]

        gen_ids, per_token, prompt_final = greedy_capture(model, enc, tok, args.device)
        full_text = tok.decode(gen_ids, skip_special_tokens=True)
        answer_text = truncate_answer_text(full_text)
        j = answer_span_end_index(gen_ids, tok, answer_text)
        j = min(j, len(per_token) - 1)

        caps, labels = [prompt_final], ["prompt_final"]
        for k in range(min(N_ANSWER_POS, len(per_token))):
            caps.append(per_token[k]); labels.append(f"answer_tok_{k}")
        caps.append(per_token[j]); labels.append("last_answer_tok")
        hidden_stack = np.stack(caps).astype(np.float16)

        greedy_correct = is_correct(norm(answer_text), gts)
        npz_path = stk_dir / f"{pid}_0.npz"
        np.savez_compressed(npz_path, hidden_stack=hidden_stack,
                            position_labels=np.array(labels),
                            gen_token_ids=np.array(gen_ids, dtype=np.int64),
                            answer_end_idx=np.int64(j))
        f_idx.write(json.dumps({
            "prompt_id": pid, "completion_idx": 0, "question": row["question"],
            "category": row.get("category", "1"), "greedy": True,
            "answer_text": answer_text, "answer_end_idx": int(j),
            "correct": bool(greedy_correct), "npz_path": str(npz_path),
            "n_positions": int(hidden_stack.shape[0]),
            "n_layers": int(hidden_stack.shape[1]),
            "hidden_dim": int(hidden_stack.shape[2])}) + "\n"); f_idx.flush()

        with torch.no_grad():
            s = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=TOP_P,
                               num_return_sequences=N_SAMPLES, max_new_tokens=MAX_NEW,
                               pad_token_id=tok.pad_token_id)
        clean = [truncate_answer_text(t) for t in tok.batch_decode(s[:, plen:], skip_special_tokens=True)]
        n_correct = sum(is_correct(norm(t), gts) for t in clean)
        r_p = 1.0 - n_correct / N_SAMPLES
        f_rp.write(json.dumps({"prompt_id": pid, "r_p": r_p, "n_correct": n_correct,
                               "n": N_SAMPLES, "greedy_answer": answer_text[:80]}) + "\n"); f_rp.flush()

        if args.device == "mps" and (i + 1) % 20 == 0:
            torch.mps.empty_cache()
            print(f"  {i+1}/{len(todo)}  r_p={r_p:.2f}  ans='{answer_text[:40]}'")
    f_rp.close(); f_idx.close()
    print(f"{args.tag}: done")

if __name__ == "__main__":
    main()
