"""
capture_layer_stack.py

Captures the residual stream across ALL layers at answer-token positions.

Rationale: the token-axis analysis showed zero temporal structure
(autocorrelation 0.062) but strong contemporaneous H/Anti-H anti-correlation
(-0.173). This is consistent with rivalry dynamics resolving WITHIN a forward
pass (across layer depth) rather than across generated tokens. This script
captures the across-layer trajectory to test that hypothesis.

For each prompt:
  1. Generate a short greedy answer (to fix the trajectory deterministically)
  2. Judge correctness against ground truth (Cat 1) or record for SE-based labels
  3. Capture hidden_states across all layers at:
     - the last prompt token (position that produces the first answer token)
     - the first N generated tokens
  4. Store the [n_layers x hidden_dim] tensor per position

This is cheap relative to token-by-token capture: one forward pass per prompt
with output_hidden_states=True yields all layers at once.

Output (NPZ per prompt referenced from a JSONL index):
  index JSONL: {prompt_id, question, category, correct, answer_text, npz_path}
  NPZ: hidden_stack [n_positions, n_layers, hidden_dim], positions metadata

Usage:
    python capture_layer_stack.py \
        --model_path mistralai/Mistral-7B-v0.3 \
        --prompts_path data/benchmark_final_250.jsonl \
        --output_dir data/layer_stacks_mistral \
        --index_path data/layer_stack_index_mistral.jsonl \
        --n_answer_positions 5 \
        --n_completions 10
"""

import os
import json
import re
import string
import argparse
from typing import List, Dict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Capture layer stacks.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--n_answer_positions", type=int, default=5,
                        help="Number of generated-token positions to capture")
    parser.add_argument("--n_completions", type=int, default=10,
                        help="Sampled completions per prompt (for variance)")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy_first", action="store_true", default=True,
                        help="First completion is greedy (deterministic anchor)")
    parser.add_argument("--device", type=str, default="mps")
    return parser.parse_args()


def normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower().replace("_", " ")
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def judge_correct(text: str, ground_truth: List[str]) -> bool:
    if not ground_truth:
        return False
    norm_text = normalize(text)
    for gt in ground_truth:
        ng = normalize(gt)
        if ng and ng in norm_text:
            return True
    return False


def capture_one_completion(
    model, tokenizer, prompt: str,
    n_answer_positions: int, max_new_tokens: int,
    temperature: float, greedy: bool, device: str,
):
    """Generate one completion, capturing hidden states across all layers
    at the prompt-final position and the first n_answer_positions tokens.

    Returns: (answer_text, hidden_stack, position_labels)
        hidden_stack: np.ndarray [n_positions, n_layers, hidden_dim]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    generated_ids = input_ids.clone()

    captured = []  # list of [n_layers, hidden_dim] arrays
    position_labels = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors [batch, seq, hidden]
        # Capture at the LAST position (the one producing the next token)
        if step == 0:
            # Prompt-final position: produces the first answer token
            stack = np.stack([
                hs[0, -1, :].cpu().float().numpy()
                for hs in outputs.hidden_states
            ])  # [n_layers+1, hidden_dim]
            captured.append(stack)
            position_labels.append("prompt_final")

        logits = outputs.logits[:, -1, :]
        if not greedy and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Capture the hidden states at this newly generated position
        if step < n_answer_positions:
            # Re-run is wasteful; instead capture on next iteration's forward.
            # Simpler: capture the position we just generated by noting we will
            # see it as the last position in the NEXT forward pass. To avoid an
            # extra pass, we capture here using a targeted forward.
            with torch.no_grad():
                out2 = model(generated_ids, output_hidden_states=True)
            stack = np.stack([
                hs[0, -1, :].cpu().float().numpy()
                for hs in out2.hidden_states
            ])
            captured.append(stack)
            position_labels.append(f"answer_tok_{step}")

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Capture at the LAST generated token position. This is where the
    # PT-CSFT work found the correctness signal is strongest (the model has
    # committed its answer). The prompt_final and early answer positions are
    # before the answer is fully formed, so the signal is weak there.
    with torch.no_grad():
        out_final = model(generated_ids, output_hidden_states=True)
    stack = np.stack([
        hs[0, -1, :].cpu().float().numpy()
        for hs in out_final.hidden_states
    ])
    captured.append(stack)
    position_labels.append("last_answer_tok")

    answer_text = tokenizer.decode(
        generated_ids[0, input_len:], skip_special_tokens=True
    ).strip()

    hidden_stack = np.stack(captured)  # [n_positions, n_layers+1, hidden_dim]
    return answer_text, hidden_stack, position_labels


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map=args.device
    )
    model.eval()

    prompts = []
    with open(args.prompts_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts.")

    processed = set()
    if os.path.exists(args.index_path):
        with open(args.index_path) as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["prompt_id"] + "_" +
                                  str(json.loads(line)["completion_idx"]))
                except Exception:
                    continue

    suffix = " Respond with the answer only, without any explanation."

    with open(args.index_path, "a") as f_idx:
        for prompt_data in tqdm(prompts, desc="Capturing layer stacks"):
            pid = prompt_data["prompt_id"]
            question = prompt_data["question"]
            ground_truth = prompt_data.get("ground_truth", [])
            full_prompt = question.strip() + suffix

            for comp_idx in range(args.n_completions):
                key = f"{pid}_{comp_idx}"
                if key in processed:
                    continue

                greedy = (comp_idx == 0 and args.greedy_first)

                answer, hidden_stack, pos_labels = capture_one_completion(
                    model, tokenizer, full_prompt,
                    n_answer_positions=args.n_answer_positions,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    greedy=greedy,
                    device=args.device,
                )

                correct = judge_correct(answer, ground_truth) if ground_truth else None

                npz_name = f"{pid}_{comp_idx}.npz"
                npz_path = os.path.join(args.output_dir, npz_name)
                np.savez_compressed(
                    npz_path,
                    hidden_stack=hidden_stack.astype(np.float16),
                    position_labels=np.array(pos_labels),
                )

                idx_entry = {
                    "prompt_id": pid,
                    "completion_idx": comp_idx,
                    "question": question,
                    "category": prompt_data.get("category", ""),
                    "greedy": greedy,
                    "answer_text": answer,
                    "correct": correct,
                    "npz_path": npz_path,
                    "n_positions": hidden_stack.shape[0],
                    "n_layers": hidden_stack.shape[1],
                    "hidden_dim": hidden_stack.shape[2],
                }
                f_idx.write(json.dumps(idx_entry) + "\n")
                f_idx.flush()

    print("Layer-stack capture complete.")


if __name__ == "__main__":
    main()
