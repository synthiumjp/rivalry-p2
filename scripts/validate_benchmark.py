"""
validate_benchmark.py

Validates the prompt benchmark against pre-registration criteria.

Category 1 (unambiguous): correct answer in >= 18/20 greedy generations.
Category 2 (ambiguous): >= 2 semantic clusters, no cluster > 80%, SE > 0.5.
Category 3 (fabricated): model does not consistently produce verifiable factual content.
    (Operationalised as: SE > 0.3 or no single cluster > 60%.)

Usage:
    python validate_benchmark.py \
        --se_path data/se_mistral.jsonl \
        --output_path data/benchmark_validated.jsonl
"""

import json
import argparse
import re
import string
from typing import List

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate prompt benchmark against pre-reg criteria."
    )
    parser.add_argument("--se_path", type=str, required=True,
                        help="SE output from compute_se.py")
    parser.add_argument("--greedy_path", type=str, default=None,
                        help="Greedy generation output (for Cat 1 validation)")
    parser.add_argument("--candidates_path", type=str, default=None,
                        help="Candidate file carrying ground_truth (for Cat 1)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSONL with validation results")
    return parser.parse_args()


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (same as collect_responses_hf.py)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "\u2018\u2019`")
        return "".join(ch if ch not in exclude else " " for ch in text)
    if not s:
        return ""
    return white_space_fix(
        remove_articles(handle_punc(str(s).lower().replace("_", " ")))
    ).strip()


def validate_cat1(
    se_data: dict, greedy_data: dict = None, ground_truth: List[str] = None,
) -> dict:
    """Category 1: correct in >= 18/20 greedy generations."""
    result = {"category": "1", "prompt_id": se_data["prompt_id"]}

    completions = se_data.get("completions", [])
    if ground_truth and completions:
        norm_gts = [normalize_answer(gt) for gt in ground_truth]
        correct = 0
        for comp in completions:
            norm_comp = normalize_answer(comp)
            if any(gt in norm_comp for gt in norm_gts if gt):
                correct += 1
        result["greedy_correct"] = correct
        result["greedy_total"] = len(completions)
        result["greedy_pass"] = correct >= 18
    else:
        result["greedy_pass"] = None
        result["greedy_correct"] = None
        result["greedy_total"] = len(completions)

    result["se"] = se_data["se"]
    result["num_clusters"] = se_data["num_clusters"]
    result["valid"] = bool(result["greedy_pass"]) and se_data["se"] < 1.0

    return result


def validate_cat2(se_data: dict) -> dict:
    """Category 2: >= 2 clusters, no cluster > 80%, SE > 0.5."""
    result = {"category": "2", "prompt_id": se_data["prompt_id"]}

    num_clusters = se_data["num_clusters"]
    cluster_sizes = se_data["cluster_sizes"]
    total = sum(cluster_sizes)
    max_cluster_pct = max(cluster_sizes) / total if total > 0 else 1.0

    result["se"] = se_data["se"]
    result["num_clusters"] = num_clusters
    result["max_cluster_pct"] = round(max_cluster_pct, 4)

    result["has_multiple_clusters"] = num_clusters >= 2
    result["no_dominant_cluster"] = max_cluster_pct <= 0.80
    result["se_above_threshold"] = se_data["se"] > 0.5

    result["valid"] = (
        result["has_multiple_clusters"]
        and result["no_dominant_cluster"]
        and result["se_above_threshold"]
    )

    return result


def validate_cat3(se_data: dict) -> dict:
    """Category 3: model does not produce consistent verifiable content.

    Operationalised as: SE > 0.3 or no single cluster > 60%.
    """
    result = {"category": "3", "prompt_id": se_data["prompt_id"]}

    cluster_sizes = se_data["cluster_sizes"]
    total = sum(cluster_sizes)
    max_cluster_pct = max(cluster_sizes) / total if total > 0 else 1.0

    result["se"] = se_data["se"]
    result["num_clusters"] = se_data["num_clusters"]
    result["max_cluster_pct"] = round(max_cluster_pct, 4)

    result["valid"] = se_data["se"] > 0.3 or max_cluster_pct <= 0.60

    return result


def main():
    args = parse_args()

    # Load SE data
    se_items = []
    with open(args.se_path, "r") as f:
        for line in f:
            se_items.append(json.loads(line))

    # Load greedy data if provided
    greedy_map = {}
    if args.greedy_path:
        with open(args.greedy_path, "r") as f:
            for line in f:
                data = json.loads(line)
                greedy_map[data["prompt_id"]] = data

    gt_map = {}
    if args.candidates_path:
        with open(args.candidates_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if "ground_truth" in data:
                    gt_map[data["prompt_id"]] = data["ground_truth"]

    # Validate
    cat_counts = {"1": {"total": 0, "valid": 0},
                  "2": {"total": 0, "valid": 0},
                  "3": {"total": 0, "valid": 0}}

    with open(args.output_path, "w", encoding="utf-8") as f_out:
        for item in tqdm(se_items, desc="Validating"):
            cat = item.get("category", "")

            if cat == "1":
                result = validate_cat1(
                    item, greedy_map.get(item["prompt_id"]),
                    gt_map.get(item["prompt_id"])
                )
            elif cat == "2":
                result = validate_cat2(item)
            elif cat == "3":
                result = validate_cat3(item)
            else:
                continue

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            if cat in cat_counts:
                cat_counts[cat]["total"] += 1
                if result.get("valid"):
                    cat_counts[cat]["valid"] += 1

    print("\nValidation Summary:")
    for cat, counts in sorted(cat_counts.items()):
        total = counts["total"]
        valid = counts["valid"]
        pct = (valid / total * 100) if total > 0 else 0
        print(f"  Category {cat}: {valid}/{total} valid ({pct:.1f}%)")


if __name__ == "__main__":
    main()
