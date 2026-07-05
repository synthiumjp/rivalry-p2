"""
filter_consistency.py

Preprocesses the raw responses JSONL with a configurable consistency
threshold (default 8/10). Outputs a filtered JSONL where each entry's
judges list is set to unanimous (all true or all false) based on the
supermajority vote. This preserves compatibility with downstream scripts
that check for unanimous judges.

Usage:
    python filter_consistency.py \
        --input_path data/responses_mistral_full.jsonl \
        --output_path data/responses_mistral_filtered.jsonl \
        --threshold 8
"""

import json
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter responses by consistency threshold."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=8,
                        help="Minimum agreement count out of total valid judges (default: 8)")
    parser.add_argument("--sample_num", type=int, default=10,
                        help="Expected number of judges per question (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    true_count = 0
    false_count = 0
    skipped = 0
    total = 0

    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Filtering"):
            data = json.loads(line)
            qid = list(data.keys())[0]
            content = data[qid]
            total += 1

            judges = content["judges"]

            # Count only clean judges (exclude uncertain and error)
            clean_judges = [j for j in judges if j in ("true", "false")]
            if len(clean_judges) < args.sample_num:
                skipped += 1
                continue

            n_true = clean_judges.count("true")
            n_false = clean_judges.count("false")

            label = None
            if n_true >= args.threshold:
                label = "true"
                true_count += 1
            elif n_false >= args.threshold:
                label = "false"
                false_count += 1
            else:
                skipped += 1
                continue

            # Pick the most frequent response matching the majority label
            responses = content["responses"]
            if label == "true":
                # Pick the most common response from the correct ones
                correct_responses = [
                    r for r, j in zip(responses, judges) if j == "true"
                ]
                rep_response = max(
                    set(correct_responses), key=correct_responses.count
                )
            else:
                # Pick the most common response from the incorrect ones
                incorrect_responses = [
                    r for r, j in zip(responses, judges) if j == "false"
                ]
                rep_response = max(
                    set(incorrect_responses), key=incorrect_responses.count
                )

            # Output with unanimous judges (compatibility with downstream)
            output = {
                qid: {
                    "question": content["question"],
                    "responses": [rep_response] * args.sample_num,
                    "judges": [label] * args.sample_num,
                    "ground_truth": content.get("ground_truth", []),
                    "original_judges": judges,
                    "consistency": n_true if label == "true" else n_false,
                }
            }
            f_out.write(json.dumps(output, ensure_ascii=False) + "\n")

    balanced = min(true_count, false_count)
    print(f"\nDone.")
    print(f"Total: {total}")
    print(f"True (>={args.threshold}/{args.sample_num}): {true_count}")
    print(f"False (>={args.threshold}/{args.sample_num}): {false_count}")
    print(f"Skipped: {skipped}")
    print(f"Balanced max: {balanced} per class")


if __name__ == "__main__":
    main()
