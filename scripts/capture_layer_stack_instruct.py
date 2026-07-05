"""
capture_layer_stack_instruct.py

Modified capture_layer_stack.py with --instruct flag for chat template
formatting. Used for the diagnostic check: does bimodality emerge with
cleaner labels from an instruct model with greedy generation?

Changes from original:
  - --instruct flag: uses tokenizer.apply_chat_template() for prompt formatting
  - add_special_tokens=False when instruct (template includes BOS)
  - defaults: n_completions=1, greedy_first=True (one deterministic answer)

Usage (Step 2 diagnostic):
    python scripts/capture_layer_stack_instruct.py \
        --model_path mistralai/Mistral-7B-Instruct-v0.3 \
        --prompts_path data/benchmark_final_250.jsonl \
        --output_dir data/layer_stacks_mistral_instruct \
        --index_path data/layer_stack_index_mistral_instruct.jsonl \
        --n_answer_positions 5 --n_completions 1 --instruct
"""

import os
import json
import re
import string
import argparse
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Capture layer stacks (instruct support).")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--n_answer_positions", type=int, default=5)
    parser.add_argument("--n_completions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy_first", action="store_true", default=True)
    parser.add_argument("--instruct", action="store_true", default=False,
                        help="Use chat template for instruct models")
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
    instruct: bool = False,
):
    """Generate one completion, capturing hidden states across all layers."""
    if instruct:
        inputs = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False).to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    generated_ids = input_ids.clone()

    captured = []
    position_labels = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids, output_hidden_states=True)

        if step == 0:
            stack = np.stack([
                hs[0, -1, :].cpu().float().numpy()
                for hs in outputs.hidden_states
            ])
            captured.append(stack)
            position_labels.append("prompt_final")

        logits = outputs.logits[:, -1, :]
        if not greedy and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if step < n_answer_positions:
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

    # Capture at last generated token (where answer is committed)
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

    hidden_stack = np.stack(captured)
    return answer_text, hidden_stack, position_labels


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    print(f"Instruct mode: {args.instruct}")
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
                    e = json.loads(line)
                    processed.add(e["prompt_id"] + "_" + str(e["completion_idx"]))
                except Exception:
                    continue

    suffix = " Respond with the answer only, without any explanation."

    with open(args.index_path, "a") as f_idx:
        for prompt_data in tqdm(prompts, desc="Capturing layer stacks"):
            pid = prompt_data["prompt_id"]
            question = prompt_data["question"]
            ground_truth = prompt_data.get("ground_truth", [])

            # Format prompt
            content = question.strip() + suffix
            if args.instruct:
                messages = [{"role": "user", "content": content}]
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                full_prompt = content

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
                    instruct=args.instruct,
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

                # Print first few for sanity check
                if comp_idx == 0 and prompt_data == prompts[0]:
                    print(f"\n--- Sanity check ---")
                    print(f"Prompt: {full_prompt[:120]}...")
                    print(f"Answer: {answer}")
                    print(f"Correct: {correct}")
                    print(f"Stack shape: {hidden_stack.shape}")
                    print(f"---\n")

    print("Layer-stack capture complete.")


if __name__ == "__main__":
    main()
