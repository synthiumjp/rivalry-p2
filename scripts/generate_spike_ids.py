#!/usr/bin/env python3
"""
Generate cat1_spike_ids.json from the existing benchmark.

The spike uses ALL 100 existing Cat1 prompts (the full Cat1 pool before
benchmark expansion). These prompts have existing Mistral data (foreknowledge
managed in the pre-reg). The hold-out (200 Cat1) will come from the
EXPANDED 900 new prompts added during Phase 1, ensuring no overlap.

Run from: ~/jpwork/rivalry-p2/H-Neurons/
Output:   data/cat1_spike_ids.json
"""

import json
from pathlib import Path

BENCHMARK = Path("data/benchmark_final_250.jsonl")
OUTPUT = Path("data/cat1_spike_ids.json")

def main():
    cat1_ids = []
    cat1_questions = []

    with open(BENCHMARK) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("category") == "cat1":
                pid = entry.get("prompt_id", entry.get("id", entry.get("question_id")))
                cat1_ids.append(pid)
                cat1_questions.append(entry.get("question", ""))

    print(f"Found {len(cat1_ids)} Cat1 prompts in {BENCHMARK}")

    # If no prompt_id field, use line indices
    if all(pid is None for pid in cat1_ids):
        print("WARNING: No prompt_id field found. Using line-based indices.")
        cat1_ids = []
        cat1_questions = []
        with open(BENCHMARK) as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                if entry.get("category") == "cat1":
                    cat1_ids.append(f"cat1_{i:04d}")
                    cat1_questions.append(entry.get("question", ""))

    spike_manifest = {
        "description": "Step 0 diagnostic spike prompt IDs. All 100 existing Cat1 prompts.",
        "source": str(BENCHMARK),
        "n": len(cat1_ids),
        "note": "These prompts have existing Mistral data (foreknowledge managed). "
                "Hold-out (200 Cat1) will come from expanded benchmark Phase 1.",
        "prompt_ids": cat1_ids,
    }

    with open(OUTPUT, "w") as f:
        json.dump(spike_manifest, f, indent=2)

    print(f"Wrote {len(cat1_ids)} spike IDs to {OUTPUT}")
    print(f"First 5 questions: {cat1_questions[:5]}")


if __name__ == "__main__":
    main()
