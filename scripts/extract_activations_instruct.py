"""
extract_activations_instruct.py

Instruct-model CETT activation extractor. Adapted from extract_activations_base.py.
CETT core (CETTManager, norm math, per-position CETT) is UNCHANGED. The prompt-
boundary reconstruction differs from base:

  base:     full_text = question + " " + response ; output_start = len(encode(question))
  instruct: full_text = apply_chat_template([{user: question}], add_generation_prompt=True)
            + response, tokenized add_special_tokens=False (template emits BOS),
            output_start = token length of the templated prompt alone.

Aggregation (--method):
  mean : mean CETT over the span. LENGTH-CONFOUNDED when span length correlates
         with the label (it does: hallucinated answers are ~60% longer). Do NOT
         use for the detector; kept for reference/parity only.
  max  : max over the span. Same length caveat.
  last : CETT at the single LAST answer-token position. No length signal.
         This is the detector feature (deviation from the reference span-mean,
         justified: reference true/false span lengths are 2.4 vs 11.0 tokens,
         so span-mean confounds answer length with hallucination).

With --locations output, selected = cett_full[:, output_start:output_end, :],
so --method last takes cett_full[:, output_end-1, :], the last answer token
(under answer_span-truncated responses the output region ends at the answer).

Usage:
    python scripts/extract_activations_instruct.py \
        --model_path mistralai/Mistral-7B-Instruct-v0.3 \
        --input_path data/answer_tokens_mistral_v2.jsonl \
        --train_ids_path data/train_qids_mistral_v2.json \
        --output_root data/activations_mistral_v2_last \
        --locations output --method last --dtype float16 --attn sdpa

    # Gemma-2: --dtype bfloat16 --attn eager
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Extract CETT activations, instruct models.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--input_path", type=str, required=True,
                   help="answer_tokens JSONL (one {qid: {...}} per line)")
    p.add_argument("--train_ids_path", type=str, required=True,
                   help="train_qids JSON with 't' and 'f' id lists")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--locations", nargs="+", default=["output"],
                   choices=["input", "output", "answer_tokens", "all_except_answer_tokens"])
    p.add_argument("--method", type=str, choices=["mean", "max", "last"], default="last")
    p.add_argument("--use_mag", action="store_true", default=True)
    p.add_argument("--use_abs", action="store_true", default=True)
    p.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--attn", type=str, choices=["sdpa", "eager"], default="sdpa")
    return p.parse_args()


class CETTManager:
    """UNCHANGED from extract_activations_base.py. CETT via hooks on down_proj.

    cett = (|down_proj_input| * ||W_down column||) / (||down_proj_output|| + eps),
    per neuron, per token. Shape returned: [layers, tokens, neurons].
    """

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
            self.output_norms.append(torch.norm(output.detach(), dim=-1, keepdim=True))
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
        self.activations = [
            act.squeeze(0) if act.dim() == 3 and act.size(0) == 1 else act
            for act in self.activations
        ]
        self.output_norms = [
            norm.squeeze(0) if norm.dim() == 3 and norm.size(0) == 1 else norm
            for norm in self.output_norms
        ]
        acts = torch.stack(self.activations).transpose(0, 1).to(self.model.device)
        norms = torch.stack(self.output_norms).transpose(0, 1).to(self.model.device)
        if use_abs:
            acts = torch.abs(acts)
        if use_mag:
            acts = acts * self.weight_norms.unsqueeze(0)
        return (acts / (norms + 1e-8)).transpose(0, 1)


def get_region_indices_instruct(
    full_ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
    answer_tokens: List[str],
) -> Dict[str, Optional[Tuple[int, int]]]:
    """Instruct boundary: output region begins at prompt_len (templated prompt
    length, tokenized add_special_tokens=False). Answer span (if requested) is
    found by string match within the output region, same normalisation as base.
    """
    full_len = full_ids.shape[1]
    output_start = prompt_len
    output_end = full_len
    if tokenizer.eos_token_id is not None:
        if full_ids[0, -1].item() == tokenizer.eos_token_id:
            output_end -= 1

    ans_start, ans_end = None, None
    if answer_tokens:
        full_tokens = [tokenizer.decode([tid]) for tid in full_ids[0]]
        m = len(answer_tokens)
        normalized_answer = [
            t.replace("\u2581", " ").replace("\u0120", " ") for t in answer_tokens
        ]
        for i in range(output_start, output_end - m + 1):
            window = [
                full_tokens[j].replace("\u2581", " ").replace("\u0120", " ")
                for j in range(i, i + m)
            ]
            if window == normalized_answer:
                ans_start, ans_end = i, i + m
                break

    return {
        "input": (0, output_start),
        "output": (output_start, output_end),
        "answer_tokens": (ans_start, ans_end) if ans_start is not None else None,
    }


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map="auto",
        attn_implementation=args.attn,
    )
    model.eval()
    cett_manager = CETTManager(model)

    with open(args.train_ids_path) as f:
        id_map = json.load(f)
        target_ids = set(id_map["t"] + id_map["f"])
    print(f"Loaded {len(target_ids)} target IDs.")

    for loc in args.locations:
        os.makedirs(os.path.join(args.output_root, loc), exist_ok=True)

    with open(args.input_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    extracted, skipped, no_span = 0, 0, 0
    for sample_dict in tqdm(samples, desc="Extracting CETT (instruct)"):
        qid = list(sample_dict.keys())[0]
        if qid not in target_ids:
            continue
        data = sample_dict[qid]
        cett_manager.clear()

        # stored "question" already includes the answer-only suffix and is the
        # user-turn content. Rebuild the exact templated prompt the model saw.
        user_content = data["question"]
        templated_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        # response concatenates DIRECTLY after the assistant header (no space).
        full_text = templated_prompt + data["response"]

        # add_special_tokens=False: template already emits BOS. Doubling it
        # would shift every index by one.
        full_ids = tokenizer(
            full_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(model.device)

        prompt_ids = tokenizer(
            templated_prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            model(full_ids)

        cett_full = cett_manager.get_cett_tensor(
            use_abs=args.use_abs, use_mag=args.use_mag
        )

        regions = get_region_indices_instruct(
            full_ids, prompt_len, tokenizer, data.get("answer_tokens")
        )

        item_extracted = False
        for loc in args.locations:
            if loc in ["input", "output", "answer_tokens"]:
                indices = regions[loc]
                if indices is None:
                    if loc == "answer_tokens":
                        no_span += 1
                    continue
                selected = cett_full[:, indices[0]:indices[1], :]
            elif loc == "all_except_answer_tokens" and regions["answer_tokens"]:
                a_s, a_e = regions["answer_tokens"]
                selected = torch.cat([cett_full[:, :a_s, :], cett_full[:, a_e:, :]], dim=1)
            else:
                continue

            if selected is None or selected.shape[1] == 0:
                continue

            # Aggregate across token positions.
            #   mean / max : over the span (LENGTH-CONFOUNDED; f answers ~60%
            #                longer, so do not use for the detector)
            #   last       : single last-answer-token position, no length signal
            if args.method == "mean":
                final_act = selected.mean(dim=1)
            elif args.method == "max":
                final_act, _ = selected.max(dim=1)
            elif args.method == "last":
                final_act = selected[:, -1, :]
            else:
                raise ValueError(f"unknown method {args.method}")

            np.save(os.path.join(args.output_root, loc, f"act_{qid}.npy"),
                    final_act.cpu().float().numpy())
            item_extracted = True

        extracted += item_extracted
        skipped += (not item_extracted)

    print(f"\nDone. Extracted: {extracted}, Skipped: {skipped}, no answer-span: {no_span}")


if __name__ == "__main__":
    main()
