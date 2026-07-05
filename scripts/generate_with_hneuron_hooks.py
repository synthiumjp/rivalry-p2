"""
generate_with_hneuron_hooks.py

Generates M completions per prompt while extracting H-Neuron activations
at each generated token. Produces a time series of activation vectors
per completion for downstream clustering and intervention analysis.

The H-Neurons are identified from the trained classifier (detector .pkl).
At each generation step, forward hooks on the relevant down_proj layers
capture the FFN input activations at the H-Neuron positions.

Output format (JSONL, one line per prompt):
{
    "prompt_id": "...",
    "question": "...",
    "category": "...",
    "completions": [
        {
            "text": "...",
            "tokens": ["tok1", "tok2", ...],
            "activations": [[h1, h2, ..., h6], [h1, h2, ..., h6], ...],
            "n_tokens": 42
        },
        ...
    ]
}

Usage:
    python generate_with_hneuron_hooks.py \
        --model_path mistralai/Mistral-7B-v0.3 \
        --classifier_path models/detector_mistral_8of10.pkl \
        --prompts_path data/benchmark_final_250.jsonl \
        --output_path data/hneuron_activations_mistral.jsonl \
        --num_completions 20 \
        --temperature 1.0 \
        --max_new_tokens 100
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
import numpy as np
import joblib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate completions with H-Neuron activation extraction."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--classifier_path", type=str, required=True,
                        help="Path to trained H-Neuron classifier (.pkl)")
    parser.add_argument("--prompts_path", type=str, required=True,
                        help="JSONL with prompts (benchmark_final_250.jsonl)")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_completions", type=int, default=20,
                        help="Completions per prompt (M)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    return parser.parse_args()


def get_hneuron_map(classifier, config) -> Dict[int, List[int]]:
    """Extract H-Neuron locations from the trained classifier.

    Returns {layer_idx: [neuron_indices]} for neurons with positive
    classifier weights (hallucination-predictive).
    """
    weights = classifier.coef_[0]
    inter_size = config.intermediate_size
    selected = np.where(weights > 0)[0]

    neuron_map = {}
    for idx in selected:
        layer = int(idx // inter_size)
        neuron = int(idx % inter_size)
        if layer not in neuron_map:
            neuron_map[layer] = []
        neuron_map[layer].append(neuron)

    return neuron_map


def get_all_neuron_indices(classifier, config) -> List[Tuple[int, int]]:
    """Get flat list of (layer, neuron) for all non-zero classifier features.

    Includes both H-Neurons (positive) and Anti-H (negative) for the
    full activation vector used in clustering.
    """
    weights = classifier.coef_[0]
    inter_size = config.intermediate_size
    selected = np.where(weights != 0)[0]

    indices = []
    for idx in selected:
        layer = int(idx // inter_size)
        neuron = int(idx % inter_size)
        indices.append((layer, neuron))

    return sorted(indices, key=lambda x: (x[0], x[1]))


class HNeuronHookManager:
    """Manages forward hooks to capture H-Neuron activations during generation.

    Registers hooks on down_proj layers at the specific layers where
    H-Neurons were identified. At each forward pass (each generated token),
    captures the input to down_proj (the FFN neuron activations) at the
    positions of interest.
    """

    def __init__(self, model, neuron_indices: List[Tuple[int, int]]):
        """
        Args:
            model: the transformer model
            neuron_indices: list of (layer_idx, neuron_idx) pairs
        """
        self.model = model
        self.neuron_indices = neuron_indices
        self.hooks = []
        self.step_activations = {}  # {layer_idx: tensor}

        # Group neurons by layer for efficient extraction
        self.layer_neurons = {}
        for layer, neuron in neuron_indices:
            if layer not in self.layer_neurons:
                self.layer_neurons[layer] = []
            self.layer_neurons[layer].append(neuron)

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on down_proj modules at relevant layers."""
        for name, module in self.model.named_modules():
            if "down_proj" not in name:
                continue

            # Extract layer index from module name
            parts = name.split(".")
            try:
                layer_idx = next(int(p) for p in parts if p.isdigit())
            except StopIteration:
                continue

            if layer_idx not in self.layer_neurons:
                continue

            neurons = self.layer_neurons[layer_idx]

            def make_hook(layer, neuron_list):
                def hook_fn(module, input, output):
                    # input[0] shape: [batch, seq_len, intermediate_size]
                    # During generation, seq_len = 1 (the new token)
                    act = input[0].detach()
                    if act.dim() == 3:
                        # Take last token (during generation this is the only one)
                        act = act[:, -1, :]
                    elif act.dim() == 2:
                        act = act[-1:, :]
                    # Extract only the neurons we care about
                    selected = act[0, neuron_list].cpu().float().numpy()
                    self.step_activations[layer] = selected
                return hook_fn

            self.hooks.append(
                module.register_forward_hook(make_hook(layer_idx, neurons))
            )

    def get_current_activation_vector(self) -> np.ndarray:
        """Assemble the full activation vector from the current step.

        Returns a 1D array with one value per (layer, neuron) pair,
        in the same order as self.neuron_indices.
        """
        vector = []
        for layer, neuron in self.neuron_indices:
            neurons_in_layer = self.layer_neurons[layer]
            local_idx = neurons_in_layer.index(neuron)
            if layer in self.step_activations:
                vector.append(float(self.step_activations[layer][local_idx]))
            else:
                vector.append(0.0)
        return np.array(vector)

    def clear_step(self):
        """Clear activations from the current step."""
        self.step_activations.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def generate_with_hooks(
    model,
    tokenizer,
    hook_manager: HNeuronHookManager,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
) -> Tuple[str, List[str], List[List[float]]]:
    """Generate a single completion while capturing H-Neuron activations.

    Returns:
        text: the generated text
        tokens: list of token strings
        activations: list of activation vectors (one per token)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    all_activations = []
    all_tokens = []

    # Generate token by token to capture activations at each step
    generated_ids = input_ids.clone()

    for step in range(max_new_tokens):
        hook_manager.clear_step()

        with torch.no_grad():
            outputs = model(generated_ids)

        logits = outputs.logits[:, -1, :]

        # Apply temperature and sampling
        if temperature > 0:
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        # Capture the activation vector for this step
        act_vector = hook_manager.get_current_activation_vector()
        all_activations.append(act_vector.tolist())

        # Decode the token
        token_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
        all_tokens.append(token_str)

        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(
        generated_ids[0, input_len:], skip_special_tokens=True
    ).strip()

    return text, all_tokens, all_activations


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

    # Load model
    print(f"Loading model: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map=args.device
    )
    model.eval()

    # Load classifier and extract neuron locations
    print(f"Loading classifier: {args.classifier_path}")
    classifier = joblib.load(args.classifier_path)

    # Use ALL non-zero features (both H and Anti-H) for clustering
    neuron_indices = get_all_neuron_indices(classifier, config)
    hneuron_map = get_hneuron_map(classifier, config)

    n_hneurons = sum(len(v) for v in hneuron_map.values())
    n_total = len(neuron_indices)
    print(f"H-Neurons: {n_hneurons} (positive coef)")
    print(f"Total tracked neurons: {n_total} (H + Anti-H)")
    print(f"Neuron locations: {neuron_indices}")

    # Register hooks
    hook_manager = HNeuronHookManager(model, neuron_indices)

    # Load prompts
    prompts = []
    with open(args.prompts_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts.")

    processed_ids = load_processed_ids(args.output_path)
    print(f"Already processed: {len(processed_ids)}")

    # Suffix for base model prompting
    suffix = " Respond with the answer only, without any explanation."

    with open(args.output_path, "a", encoding="utf-8") as f_out:
        for prompt_data in tqdm(prompts, desc="Generating with hooks"):
            pid = prompt_data["prompt_id"]
            if pid in processed_ids:
                continue

            question = prompt_data["question"]
            # Add suffix for base models (same as collect_responses_hf.py)
            full_prompt = question.strip() + suffix

            completions = []
            for comp_idx in range(args.num_completions):
                text, tokens, activations = generate_with_hooks(
                    model, tokenizer, hook_manager,
                    full_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    device=args.device,
                )
                completions.append({
                    "text": text,
                    "tokens": tokens,
                    "activations": activations,
                    "n_tokens": len(tokens),
                })

            result = {
                "prompt_id": pid,
                "question": question,
                "category": prompt_data.get("category", ""),
                "neuron_indices": neuron_indices,
                "completions": completions,
            }
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    hook_manager.remove_hooks()
    print("Done.")


if __name__ == "__main__":
    main()
