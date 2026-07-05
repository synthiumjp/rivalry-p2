"""
extract_activations_base.py

Patched version of extract_activations.py from the H-Neurons pipeline
(Gao et al., 2025). Changes from original:

1. torch.float16 instead of torch.bfloat16 (MPS compatibility)
2. Raw tokenization instead of apply_chat_template (base model compatibility)
3. Simplified get_region_indices for base models (no chat headers)

The CETT computation logic is unchanged from the original.

Usage:
    python extract_activations_base.py \
        --model_path mistralai/Mistral-7B-v0.3 \
        --input_path data/answer_tokens_mistral.jsonl \
        --train_ids_path data/train_qids.json \
        --output_root data/activations_mistral \
        --locations answer_tokens
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CETT activations for base models."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to answer_tokens.jsonl")
    parser.add_argument("--train_ids_path", type=str, required=True,
                        help="Path to train_qids.json")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for saving .npy files")
    parser.add_argument("--locations", nargs="+", default=["answer_tokens"],
                        choices=["input", "output", "answer_tokens",
                                 "all_except_answer_tokens"],
                        help="List of positions to extract activations from")
    parser.add_argument("--method", type=str, choices=["mean", "max"],
                        default="mean")
    parser.add_argument("--use_mag", action="store_true", default=True)
    parser.add_argument("--use_abs", action="store_true", default=True)
    return parser.parse_args()


class CETTManager:
    """Unchanged from original. Computes CETT via forward hooks on down_proj."""

    def __init__(self, model):
        self.model = model
        self.activations = []
        self.output_norms = []
        self.hooks = []
        self._register_hooks()
        self.weight_norms = self._get_weight_norms()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.activations.append(input[0].detach())
            self.output_norms.append(
                torch.norm(output.detach(), dim=-1, keepdim=True)
            )

        for name, module in self.model.named_modules():
            if "down_proj" in name:
                self.hooks.append(module.register_forward_hook(hook_fn))

    def _get_weight_norms(self):
        norms = []
        for name, module in self.model.named_modules():
            if "down_proj" in name:
                norms.append(torch.norm(module.weight.data, dim=0))
        return torch.stack(norms).to(self.model.device)

    def clear(self):
        self.activations.clear()
        self.output_norms.clear()

    def get_cett_tensor(self, use_abs=True, use_mag=True):
        """Returns tensor of shape [layers, tokens, neurons]."""
        self.activations = [
            act.squeeze(0) if act.dim() == 3 and act.size(0) == 1 else act
            for act in self.activations
        ]
        self.output_norms = [
            norm.squeeze(0) if norm.dim() == 3 and norm.size(0) == 1 else norm
            for norm in self.output_norms
        ]

        acts = torch.stack(self.activations).transpose(0, 1).to(
            self.model.device
        )
        norms = torch.stack(self.output_norms).transpose(0, 1).to(
            self.model.device
        )

        if use_abs:
            acts = torch.abs(acts)
        if use_mag:
            acts = acts * self.weight_norms.unsqueeze(0)

        return (acts / (norms + 1e-8)).transpose(0, 1)


def get_region_indices_base(
    full_ids: torch.Tensor,
    tokenizer,
    question: str,
    response: str,
    answer_tokens: List[str],
) -> Dict[str, Optional[Tuple[int, int]]]:
    """Identify token indices for base models (no chat template).

    For base models, the input is just: question + response concatenated.
    The question portion is the "input" region.
    The response portion is the "output" region.
    Answer tokens are found by substring matching within the output region.
    """
    # Tokenize question and response separately to find boundary
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    input_len = len(q_ids)

    # Full sequence includes any special tokens added by the tokenizer
    full_len = full_ids.shape[1]

    # Check if tokenizer added a BOS token
    bos_offset = 0
    if tokenizer.bos_token_id is not None:
        if full_ids[0, 0].item() == tokenizer.bos_token_id:
            bos_offset = 1

    input_start = bos_offset
    input_end = bos_offset + input_len
    output_start = input_end
    output_end = full_len

    # Remove EOS if present at end
    if tokenizer.eos_token_id is not None:
        if full_ids[0, -1].item() == tokenizer.eos_token_id:
            output_end -= 1

    # Find answer tokens within the output region
    ans_start, ans_end = None, None
    if answer_tokens:
        full_tokens = [tokenizer.decode([tid]) for tid in full_ids[0]]
        m = len(answer_tokens)

        # Normalize answer tokens for matching
        normalized_answer = [
            t.replace("\u2581", " ").replace("\u0120", " ")
            for t in answer_tokens
        ]

        for i in range(output_start, full_len - m + 1):
            window = [
                full_tokens[j].replace("\u2581", " ").replace("\u0120", " ")
                for j in range(i, i + m)
            ]
            if window == normalized_answer:
                ans_start, ans_end = i, i + m
                break

    return {
        "input": (input_start, input_end),
        "output": (output_start, output_end),
        "answer_tokens": (ans_start, ans_end) if ans_start is not None else None,
    }


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Changed: float16 instead of bfloat16 for MPS compatibility
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    cett_manager = CETTManager(model)

    with open(args.train_ids_path, "r") as f:
        id_map = json.load(f)
        target_ids = set(id_map["t"] + id_map["f"])
    print(f"Loaded {len(target_ids)} target IDs for extraction.")

    for loc in args.locations:
        os.makedirs(os.path.join(args.output_root, loc), exist_ok=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    extracted = 0
    skipped = 0

    for sample_dict in tqdm(samples, desc="Extracting CETT"):
        qid = list(sample_dict.keys())[0]
        if qid not in target_ids:
            continue
        data = sample_dict[qid]
        cett_manager.clear()

        # Changed: raw tokenization for base models (no chat template)
        # Add a space between question and response to prevent BPE merge
        # at the boundary.
        full_text = data["question"] + " " + data["response"]
        input_ids = tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True
        )["input_ids"].to(model.device)

        with torch.no_grad():
            model(input_ids)

        cett_full = cett_manager.get_cett_tensor(
            use_abs=args.use_abs, use_mag=args.use_mag
        )

        regions = get_region_indices_base(
            input_ids, tokenizer, data["question"], data["response"],
            data["answer_tokens"]
        )

        item_extracted = False
        for loc in args.locations:
            indices = None
            selected_cett = None

            if loc in ["input", "output", "answer_tokens"]:
                indices = regions[loc]
                if indices is None:
                    continue
                selected_cett = cett_full[:, indices[0]:indices[1], :]

            elif loc == "all_except_answer_tokens" and regions["answer_tokens"]:
                ans_s, ans_e = regions["answer_tokens"]
                seg1 = cett_full[:, :ans_s, :]
                seg2 = cett_full[:, ans_e:, :]
                selected_cett = torch.cat([seg1, seg2], dim=1)

            if selected_cett is None or selected_cett.shape[1] == 0:
                continue

            # Aggregate across token positions
            if args.method == "mean":
                final_act = selected_cett.mean(dim=1)
            else:
                final_act, _ = selected_cett.max(dim=1)

            save_path = os.path.join(args.output_root, loc, f"act_{qid}.npy")
            np.save(save_path, final_act.cpu().float().numpy())
            item_extracted = True

        if item_extracted:
            extracted += 1
        else:
            skipped += 1

    print(f"\nDone. Extracted: {extracted}, Skipped (no valid region): {skipped}")


if __name__ == "__main__":
    main()
