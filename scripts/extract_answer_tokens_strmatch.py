"""
extract_answer_tokens_strmatch.py

Supplementary script that uses string matching to extract answer tokens
for true (correctly answered) examples. For true examples, we know the
ground_truth aliases and we know the response contains one of them (that
is why the rule judge marked it true). Simple substring matching reliably
finds the answer tokens without needing an LLM.

This script:
1. Reads the raw responses JSONL
2. Finds all unanimous-true examples
3. Skips any already extracted (present in existing answer_tokens file)
4. Matches ground_truth aliases against the response text
5. Maps the character span to the token span using the target tokenizer
6. Appends to the answer_tokens output file

Usage:
    python extract_answer_tokens_strmatch.py \
        --input_path data/responses_mistral_full.jsonl \
        --output_path data/answer_tokens_mistral.jsonl \
        --tokenizer_path mistralai/Mistral-7B-v0.3
"""

import os
import json
import argparse
from typing import List, Optional, Tuple, Set

from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract answer tokens via string matching (true examples only)."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to responses JSONL from collect_responses_hf.py")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to answer_tokens JSONL (appends to existing)")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Target model tokenizer")
    parser.add_argument("--also_false", action="store_true", default=False,
                        help="Also process unanimous-false examples (uses heuristic)")
    return parser.parse_args()


def find_answer_span(
    response: str,
    aliases: List[str],
    tokenizer,
) -> Optional[Tuple[List[str], List[int]]]:
    """Find the token span in response that matches a ground_truth alias.

    Returns (answer_token_strings, answer_token_ids) or None.

    Strategy: try each alias (longest first) as a case-insensitive substring
    match in the response. Map the character span to token indices by
    reconstructing the response from individual token decodes and tracking
    character offsets.
    """
    # Tokenize the response
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    if not token_ids:
        return None

    # Build character offset map from tokens
    token_strings = []
    char_ranges = []
    reconstructed = ""
    for tid in token_ids:
        decoded = tokenizer.decode([tid])
        start = len(reconstructed)
        reconstructed += decoded
        end = len(reconstructed)
        token_strings.append(decoded)
        char_ranges.append((start, end))

    reconstructed_lower = reconstructed.lower()

    # Sort aliases by length (longest first for best match)
    sorted_aliases = sorted(aliases, key=len, reverse=True)

    for alias in sorted_aliases:
        alias_lower = alias.lower().strip()
        if not alias_lower:
            continue

        idx = reconstructed_lower.find(alias_lower)
        if idx == -1:
            continue

        alias_end = idx + len(alias_lower)

        # Find token span covering this character range
        span_start = None
        span_end = None
        for i, (cs, ce) in enumerate(char_ranges):
            if ce > idx and cs < alias_end:
                if span_start is None:
                    span_start = i
                span_end = i + 1

        if span_start is not None and span_end is not None:
            answer_tokens = token_strings[span_start:span_end]
            return answer_tokens, token_ids[span_start:span_end]

    return None


def load_processed_ids(path: str) -> Set[str]:
    """Load already-processed question IDs from existing output file."""
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.update(json.loads(line).keys())
            except Exception:
                continue
    return ids


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    processed_ids = load_processed_ids(args.output_path)
    print(f"Already processed: {len(processed_ids)} IDs")

    total = 0
    skipped_judges = 0
    skipped_processed = 0
    extracted = 0
    failed = 0

    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="String-match extraction"):
            data = json.loads(line)
            qid = list(data.keys())[0]
            content = data[qid]
            total += 1

            if qid in processed_ids:
                skipped_processed += 1
                continue

            # Filter: all judges must agree
            judges = content["judges"]
            if len(set(judges)) != 1:
                skipped_judges += 1
                continue
            if "uncertain" in judges or "error" in judges:
                skipped_judges += 1
                continue

            judge = judges[0]

            # By default, only process true examples (string matching is
            # reliable for these because we know the correct answer)
            if judge != "true" and not args.also_false:
                skipped_judges += 1
                continue

            # Pick the most frequent response
            responses = content["responses"]
            rep_response = max(set(responses), key=responses.count)

            # Get ground truth aliases
            ground_truth = content.get("ground_truth", [])
            if not ground_truth:
                failed += 1
                continue

            # Tokenize the response
            token_ids = tokenizer.encode(rep_response, add_special_tokens=False)
            tokenized_list = [tokenizer.decode([tid]) for tid in token_ids]

            # Find answer span via string matching
            match = find_answer_span(rep_response, ground_truth, tokenizer)

            if match is not None:
                answer_tokens, _ = match
                result = {
                    qid: {
                        "question": content["question"],
                        "response": rep_response,
                        "tokenized_response": tokenized_list,
                        "answer_tokens": answer_tokens,
                        "judge": judge,
                    }
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                extracted += 1
            else:
                failed += 1

    print(f"\nDone.")
    print(f"Total lines: {total}")
    print(f"Already processed (skipped): {skipped_processed}")
    print(f"Skipped (judges/false): {skipped_judges}")
    print(f"Newly extracted: {extracted}")
    print(f"Failed match: {failed}")


if __name__ == "__main__":
    main()
