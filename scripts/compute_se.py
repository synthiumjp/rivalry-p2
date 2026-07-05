"""
compute_se.py

Semantic Entropy computation pipeline following Farquhar et al. (2024).
For each prompt:
  1. Generate M completions at T=1.0 (nucleus sampling)
  2. Cluster completions via bidirectional entailment (DeBERTa-large-MNLI)
  3. Compute discrete SE = -sum(p_k * log(p_k))

Supports checkpointing: resumes from existing output file.

Usage:
    python compute_se.py \
        --model_path mistralai/Mistral-7B-v0.3 \
        --prompts_path data/benchmark_prompts.jsonl \
        --output_path data/se_mistral.jsonl \
        --num_completions 30 \
        --temperature 1.0

Input format (prompts_path, JSONL):
    {"prompt_id": "cat1_001", "question": "What is the capital of France?", "category": "1"}

Output format (output_path, JSONL):
    {"prompt_id": "...", "question": "...", "category": "...",
     "completions": [...], "clusters": [[0,1,3],[2,4],...],
     "cluster_sizes": [3,2,...], "se": 0.673}
"""

import os
import json
import argparse
import math
from typing import List, Tuple, Dict
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Semantic Entropy.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Target model for generation")
    parser.add_argument("--prompts_path", type=str, required=True,
                        help="JSONL file with prompts")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSONL with SE values")
    parser.add_argument("--nli_model", type=str,
                        default="microsoft/deberta-large-mnli",
                        help="NLI model for entailment clustering")
    parser.add_argument("--num_completions", type=int, default=30,
                        help="Number of completions per prompt (M)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--instruct", action="store_true",
                        help="Apply chat template (for instruct models)")
    parser.add_argument("--gen_batch_size", type=int, default=1,
                        help="Batch size for generation (1 for base models)")
    parser.add_argument("--nli_batch_size", type=int, default=16,
                        help="Batch size for NLI entailment checks")
    return parser.parse_args()


# -------------------------------------------------------------------
# Generation
# -------------------------------------------------------------------

def generate_completions(
    model, tokenizer, question: str, n: int,
    temperature: float, top_p: float, top_k: int,
    max_new_tokens: int, device: str, instruct: bool = False,
) -> List[str]:
    """Generate n completions for a question using the target model."""
    if instruct:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question.strip()}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = question.strip()
    completions = []

    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature,
                              top_p=top_p, top_k=top_k)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completions.append(text)

    return completions


def generate_completions_greedy(
    model, tokenizer, question: str, n: int,
    max_new_tokens: int, device: str,
) -> List[str]:
    """Generate n greedy completions (for Cat 1 validation)."""
    prompt = question.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    completions = []

    for _ in range(n):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completions.append(text)

    return completions


# -------------------------------------------------------------------
# Entailment Clustering
# -------------------------------------------------------------------

def check_entailment_batch(
    nli_model, nli_tokenizer, pairs: List[Tuple[str, str]],
    device: str, batch_size: int = 16,
) -> List[bool]:
    """Check if each (premise, hypothesis) pair is predicted as entailment.

    Returns a list of booleans, one per pair.
    """
    results = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        premises = [p for p, h in batch]
        hypotheses = [h for p, h in batch]

        inputs = nli_tokenizer(
            premises, hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = nli_model(**inputs).logits

        # DeBERTa-large-MNLI: labels are [contradiction, neutral, entailment]
        preds = logits.argmax(dim=-1).cpu().tolist()
        results.extend([p == 2 for p in preds])  # 2 = entailment

    return results


def cluster_by_entailment(
    completions: List[str],
    question: str,
    nli_model,
    nli_tokenizer,
    device: str,
    batch_size: int = 16,
) -> List[List[int]]:
    """Cluster completions by bidirectional entailment.

    Two completions are in the same cluster if they bidirectionally
    entail each other (directly or transitively).

    Following Farquhar et al.: premise/hypothesis = question + answer.
    """
    n = len(completions)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Build all pairs for bidirectional check
    # For each (i, j) with i < j, check both directions
    forward_pairs = []
    backward_pairs = []
    pair_indices = []

    for i in range(n):
        for j in range(i + 1, n):
            # Premise: question + answer_i, Hypothesis: question + answer_j
            p_fwd = f"{question} {completions[i]}"
            h_fwd = f"{question} {completions[j]}"
            forward_pairs.append((p_fwd, h_fwd))
            backward_pairs.append((h_fwd, p_fwd))
            pair_indices.append((i, j))

    # Check entailment in both directions
    fwd_results = check_entailment_batch(
        nli_model, nli_tokenizer, forward_pairs, device, batch_size
    )
    bwd_results = check_entailment_batch(
        nli_model, nli_tokenizer, backward_pairs, device, batch_size
    )

    # Build adjacency: bidirectional entailment
    adj = defaultdict(set)
    for idx, (i, j) in enumerate(pair_indices):
        if fwd_results[idx] and bwd_results[idx]:
            adj[i].add(j)
            adj[j].add(i)

    # Transitive closure via BFS/DFS to find connected components
    visited = set()
    clusters = []

    for node in range(n):
        if node in visited:
            continue
        cluster = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        clusters.append(sorted(cluster))

    return clusters


# -------------------------------------------------------------------
# Semantic Entropy
# -------------------------------------------------------------------

def compute_discrete_se(cluster_sizes: List[int], total: int) -> float:
    """Compute discrete semantic entropy from cluster sizes.

    SE = -sum(p_k * log(p_k)) where p_k = |cluster_k| / M
    """
    if total == 0:
        return 0.0

    se = 0.0
    for size in cluster_sizes:
        if size == 0:
            continue
        p = size / total
        se -= p * math.log(p)

    return se


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def load_processed_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                ids.add(data["prompt_id"])
            except Exception:
                continue
    return ids


def main():
    args = parse_args()

    # Load target model for generation
    print(f"Loading target model: {args.model_path}")
    gen_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    gen_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map=args.device
    )
    gen_model.eval()
    print("Target model loaded.")

    # Load NLI model for entailment clustering
    print(f"Loading NLI model: {args.nli_model}")
    nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        args.nli_model
    ).to(args.device)
    nli_model.eval()
    print("NLI model loaded.")

    # Load prompts
    prompts = []
    with open(args.prompts_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts.")

    processed_ids = load_processed_ids(args.output_path)
    print(f"Already processed: {len(processed_ids)}")

    with open(args.output_path, "a", encoding="utf-8") as f_out:
        for prompt_data in tqdm(prompts, desc="Computing SE"):
            pid = prompt_data["prompt_id"]
            if pid in processed_ids:
                continue

            question = prompt_data["question"]

            # 1. Generate M completions
            completions = generate_completions(
                gen_model, gen_tokenizer, question,
                n=args.num_completions,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                instruct=args.instruct,
            )

            # 2. Cluster by bidirectional entailment
            clusters = cluster_by_entailment(
                completions, question,
                nli_model, nli_tokenizer,
                device=args.device,
                batch_size=args.nli_batch_size,
            )
            cluster_sizes = [len(c) for c in clusters]

            # 3. Compute discrete SE
            se = compute_discrete_se(cluster_sizes, args.num_completions)

            result = {
                "prompt_id": pid,
                "question": question,
                "category": prompt_data.get("category", ""),
                "completions": completions,
                "clusters": clusters,
                "cluster_sizes": cluster_sizes,
                "num_clusters": len(clusters),
                "se": round(se, 6),
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

    print("Done.")


if __name__ == "__main__":
    main()
