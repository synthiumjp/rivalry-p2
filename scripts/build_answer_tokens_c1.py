"""
build_answer_tokens_c1.py

Stage 2, C1 rule: one span rule for both classes. The answer span is the
truncated response (answer_span.truncate_answer_text), tokenized. No alias
strmatch, no LLM judge, so no true/false span-source confound. Deviation from
Gao et al. answer-token extraction, documented.

Representative response per prompt: the modal response among the sampled
completions (most common exact string), matching extract_answer_tokens_*'s
rep_response = max(set(responses), key=responses.count).

Output schema matches extract_activations_instruct.py: {qid: {question,
response, answer_tokens}}.

Usage:
  python scripts/build_answer_tokens_c1.py \
    --responses data/responses_mistral_v2.jsonl \
    --tokenizer_path mistralai/Mistral-7B-Instruct-v0.3 \
    --output_path data/answer_tokens_mistral_v2.jsonl
"""
import json, argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_span import truncate_answer_text
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--min_answer_tokens", type=int, default=1,
                   help="Skip items whose truncated answer has fewer tokens.")
    return p.parse_args()


def main():
    a = parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer_path, trust_remote_code=True)

    done = set()
    if os.path.exists(a.output_path):
        for line in open(a.output_path):
            line = line.strip()
            if line:
                done.add(next(iter(json.loads(line))))

    n = written = empty = 0
    with open(a.output_path, "a", encoding="utf-8") as out:
        for line in open(a.responses):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line); qid = next(iter(e)); r = e[qid]
            n += 1
            if qid in done:
                continue
            responses = r["responses"]
            rep = max(set(responses), key=responses.count)  # modal completion
            ans = truncate_answer_text(rep)
            if not ans:
                empty += 1
                continue
            atok_ids = tok.encode(ans, add_special_tokens=False)
            atok = [tok.decode([t]) for t in atok_ids]
            if len(atok) < a.min_answer_tokens:
                empty += 1
                continue
            out.write(json.dumps({qid: {
                "question": r["question"],
                "response": ans,                 # truncated response, C1 span
                "answer_tokens": atok,
            }}, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
    print(f"prompts {n}, written {written}, empty/degenerate {empty}")


if __name__ == "__main__":
    main()
