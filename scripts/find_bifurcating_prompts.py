"""
find_bifurcating_prompts.py

H6 step 0 diagnostic. Decides whether the pre-registered bifurcation protocol
(Akarlar) is powered, and in the same run reports whether the corrupted-pair
causal-tracing fallback (Meng/ROME lineage) is viable. No patching.

BIFURCATION (reg 3.6): a prompt bifurcates iff its step-0 next-token distribution
supports >= 2 divergent continuations that diverge by step 1. Operationalised:
  - compute step-0 next-token distribution p0 (last-prompt-token forward pass).
  - take the two most probable first-tokens t_a, t_b with p0 >= --min_branch_prob
    (need >= 2 such tokens, else not bifurcating: single dominant path).
  - condition on each (append t_a / t_b), forward again, get step-1 distributions
    p1_a, p1_b.
  - KL_sym = 0.5*(KL(p1_a||p1_b) + KL(p1_b||p1_a)). Bifurcating iff KL_sym > --kl_thresh.

CORRUPTED-PAIR VIABILITY (fallback): from clean-v2 responses, per prompt count
correct vs hallucinated completions (judges). Pairable iff >= 1 of each. These are
the clean/corrupted pairs for ROME-style patching without needing bifurcation.

Both on clean-v2 dev, per model. Output: per-prompt flags + summary counts.

Usage:
  python scripts/find_bifurcating_prompts.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --responses data/responses_qwen_v2.jsonl \
    --prompt_ids data/cat1_development_prompt_ids.json \
    --source data/cat1_candidates_pull2.jsonl \
    --output data/h6_bifurcation_qwen_v2.json \
    --dtype float16 --attn sdpa
"""
import os, json, argparse
import numpy as np, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--responses", required=True, help="clean-v2 responses jsonl (for pair viability)")
    p.add_argument("--prompt_ids", required=True)
    p.add_argument("--source", required=True, help="cat1_candidates_pull2.jsonl (question text)")
    p.add_argument("--output", required=True)
    p.add_argument("--kl_thresh", type=float, default=1.0)
    p.add_argument("--min_branch_prob", type=float, default=0.05,
                   help="min step-0 prob for a first-token to count as a branch")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    return p.parse_args()


def kl(p, q, eps=1e-10):
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def main():
    a = parse_args()
    dtype = torch.float16 if a.dtype == "float16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=dtype, device_map="auto", attn_implementation=a.attn)
    model.eval()
    dev = set(json.load(open(a.prompt_ids))["prompt_ids"])

    # question text
    q_text = {}
    for line in open(a.source):
        e = json.loads(line)
        if e["prompt_id"] in dev:
            q_text[e["prompt_id"]] = e["question"]

    # corrupted-pair viability from clean-v2 responses
    pair = {}
    for line in open(a.responses):
        d = json.loads(line); qid = next(iter(d))
        if qid not in dev: continue
        j = d[qid]["judges"]
        nc = j.count("true"); nf = len(j) - nc - j.count("uncertain")
        pair[qid] = {"n_correct": nc, "n_halluc": nf,
                     "pairable": (nc >= 1 and nf >= 1)}

    device = model.device
    results = {}
    n_bif = 0
    for qid in tqdm(sorted(dev), desc="bifurcation"):
        if qid not in q_text:
            continue
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q_text[qid]}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

        with torch.no_grad():
            logits0 = model(ids).logits[0, -1, :]
        p0 = torch.softmax(logits0.float(), -1).cpu().numpy()

        top = np.argsort(p0)[::-1]
        branches = [int(t) for t in top[:5] if p0[t] >= a.min_branch_prob]
        rec = {"step0_top_prob": float(p0[top[0]]),
               "n_branches": len(branches),
               "step0_entropy": float(-np.sum(p0 * np.log(p0 + 1e-10)))}

        if len(branches) >= 2:
            ta, tb = branches[0], branches[1]
            ids_a = torch.cat([ids, torch.tensor([[ta]], device=device)], 1)
            ids_b = torch.cat([ids, torch.tensor([[tb]], device=device)], 1)
            with torch.no_grad():
                p1a = torch.softmax(model(ids_a).logits[0, -1, :].float(), -1).cpu().numpy()
                p1b = torch.softmax(model(ids_b).logits[0, -1, :].float(), -1).cpu().numpy()
            kl_sym = 0.5 * (kl(p1a, p1b) + kl(p1b, p1a))
            rec["kl_step1"] = kl_sym
            rec["bifurcating"] = bool(kl_sym > a.kl_thresh)
        else:
            rec["kl_step1"] = 0.0
            rec["bifurcating"] = False

        rec.update(pair.get(qid, {"pairable": False}))
        results[qid] = rec
        n_bif += rec["bifurcating"]

    n = len(results)
    n_pair = sum(1 for r in results.values() if r.get("pairable"))
    n_both = sum(1 for r in results.values() if r["bifurcating"] and r.get("pairable"))
    summary = {
        "n_prompts": n,
        "n_bifurcating": n_bif,
        "bifurcation_rate": n_bif / n if n else 0.0,
        "n_pairable": n_pair,
        "pairable_rate": n_pair / n if n else 0.0,
        "n_bifurcating_and_pairable": n_both,
        "kl_thresh": a.kl_thresh, "min_branch_prob": a.min_branch_prob,
    }
    json.dump({"summary": summary, "per_prompt": results}, open(a.output, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
