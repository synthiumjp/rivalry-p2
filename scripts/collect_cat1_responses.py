"""
collect_cat1_responses.py

Stage 1 for the H-Neuron detector pipeline (reg 6.3).
Sources questions from cat1_candidates_pull2.jsonl, restricted to a dev id list.
T=1.0, 20 samples, instruct chat template. Reuses the judge from
collect_responses_hf.py. Output schema matches answer-token + split stages.

Usage:
  python scripts/collect_cat1_responses.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.3 \
    --source data/cat1_candidates_pull2.jsonl \
    --prompt_ids data/cat1_development_prompt_ids.json \
    --output_path data/responses_mistral_v2.jsonl \
    --sample_num 20 --judge_type rule \
    --dtype float16 --attn sdpa
"""
import os, json, argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(s):
    import re, string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def rule_judge(ans, norm_gts):
    na = normalize_answer(ans)
    for g in norm_gts:
        if g and (g in na or na in g):
            return "true"
    return "false"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--source", required=True, help="cat1_candidates_pull2.jsonl")
    p.add_argument("--prompt_ids", required=True, help="dev ids json {prompt_ids:[...]}")
    p.add_argument("--output_path", required=True)
    p.add_argument("--sample_num", type=int, default=20)
    p.add_argument("--judge_type", choices=["rule"], default="rule")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    return p.parse_args()


def main():
    a = parse_args()
    dtype = torch.float16 if a.dtype == "float16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=dtype, device_map="auto", attn_implementation=a.attn)
    model.eval()
    dev = set(json.load(open(a.prompt_ids))["prompt_ids"])

    # source records for dev ids only
    recs = {}
    for line in open(a.source):
        line = line.strip()
        if not line: continue
        e = json.loads(line)
        if e["prompt_id"] in dev:
            recs[e["prompt_id"]] = e
    print(f"dev ids: {len(dev)}, sourced: {len(recs)}")
    missing = dev - set(recs)
    if missing:
        print(f"WARNING: {len(missing)} dev ids not in source, e.g. {sorted(missing)[:5]}")

    done = set()
    if os.path.exists(a.output_path):
        for line in open(a.output_path):
            line = line.strip()
            if line:
                done.add(next(iter(json.loads(line))))
        print(f"resuming, {len(done)} already done")

    device = model.device
    with open(a.output_path, "a", encoding="utf-8") as out:
        for qid, e in tqdm(recs.items(), desc="collect"):
            if qid in done:
                continue
            question = e["question"]              # suffix already present, do NOT re-add
            raw_aliases = e["ground_truth"]
            norm_gts = [normalize_answer(x) for x in raw_aliases if x]
            messages = [{"role": "user", "content": question}]
            text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text_in, return_tensors="pt").to(device)

            responses, judges = [], []
            for _ in range(a.sample_num):
                with torch.no_grad():
                    oid = model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                                         temperature=1.0, top_p=0.9, top_k=50,
                                         do_sample=True, pad_token_id=tok.pad_token_id)
                new = oid[0][inputs["input_ids"].shape[1]:]
                ans = tok.decode(new, skip_special_tokens=True).strip()
                responses.append(ans)
                low = ans.lower()
                if any(t in low for t in ["don't know", "cannot", "not provided", "no information"]):
                    judges.append("uncertain")
                else:
                    judges.append(rule_judge(ans, norm_gts))

            out.write(json.dumps({qid: {
                "question": question,
                "responses": responses,
                "judges": judges,
                "ground_truth": list(raw_aliases),
            }}, ensure_ascii=False) + "\n")
            out.flush()
    print("done")


if __name__ == "__main__":
    main()
