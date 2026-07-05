"""
run_interventions.py

Implements the H4 (Structure Preservation) and H5 (Asymmetric Intervention)
experiments from the pre-registration.

Pipeline:
1. Load baseline activation data from generate_with_hneuron_hooks.py
2. Cluster activations per prompt (k=2) to identify metastable states
3. Compute steering vectors from cluster centroids
4. Alpha calibration sweep on 10 pilot prompts
5. Run targeted (CAST-style) and indiscriminate (ActAdd) interventions
6. Output intervention completions with activations for DPR computation

Pre-registration specs:
- Targeted (CAST-style): apply steering only when cosine similarity to
  non-dominant centroid exceeds threshold (median from baseline)
- Indiscriminate (ActAdd): apply steering at every token
- Alpha sweep: {1.0, 2.0, 3.0, 5.0, 8.0}
- DPR = median non-dominant dwell time (intervention) /
         median non-dominant dwell time (baseline)
- H4 supported: DPR difference (targeted - indiscriminate) > 0.15, p < 0.05
- H5: 2x2 (amplify/suppress x high-SE/low-SE) on switch rates

Usage:
    # Step 1: Cluster baseline activations
    python run_interventions.py cluster \
        --activations_path data/hneuron_activations_mistral.jsonl \
        --output_path data/clusters_mistral.jsonl

    # Step 2: Alpha calibration
    python run_interventions.py calibrate \
        --model_path mistralai/Mistral-7B-v0.3 \
        --classifier_path models/detector_mistral_8of10.pkl \
        --clusters_path data/clusters_mistral.jsonl \
        --output_path data/alpha_calibration_mistral.jsonl \
        --n_pilot 10

    # Step 3: Run interventions
    python run_interventions.py intervene \
        --model_path mistralai/Mistral-7B-v0.3 \
        --classifier_path models/detector_mistral_8of10.pkl \
        --clusters_path data/clusters_mistral.jsonl \
        --output_path data/interventions_mistral.jsonl \
        --alpha 3.0 \
        --num_completions 20
"""

import os
import json
import argparse
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# -------------------------------------------------------------------
# Clustering and Dwell-Time Analysis (Stage 2)
# -------------------------------------------------------------------

def compute_dwell_times(labels: List[int]) -> Dict[int, List[int]]:
    """Compute dwell times (run lengths) for each cluster label.

    Returns {cluster_id: [dwell_time_1, dwell_time_2, ...]}.
    """
    if not labels:
        return {}

    dwells = defaultdict(list)
    current_label = labels[0]
    current_length = 1

    for i in range(1, len(labels)):
        if labels[i] == current_label:
            current_length += 1
        else:
            dwells[current_label].append(current_length)
            current_label = labels[i]
            current_length = 1
    dwells[current_label].append(current_length)

    return dict(dwells)


def compute_cv(dwells: List[int]) -> float:
    """Coefficient of variation of dwell times."""
    if len(dwells) < 2:
        return 0.0
    mean = np.mean(dwells)
    if mean == 0:
        return 0.0
    return float(np.std(dwells, ddof=1) / mean)


def compute_switches(labels: List[int]) -> int:
    """Count state switches in a label sequence."""
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def cluster_prompt_activations(
    completions: List[Dict],
    n_clusters: int = 2,
    random_state: int = 42,
) -> Dict:
    """Cluster token-level activations across all completions for a prompt.

    Returns clustering results including centroids, labels per completion,
    dwell times, and H1-relevant statistics.
    """
    # Pool all activation vectors across completions
    all_vectors = []
    comp_boundaries = []  # (start_idx, end_idx) per completion
    start = 0

    for comp in completions:
        acts = np.array(comp["activations"])
        n = len(acts)
        all_vectors.append(acts)
        comp_boundaries.append((start, start + n))
        start += n

    if not all_vectors:
        return {"error": "no_activations"}

    pooled = np.vstack(all_vectors)

    if len(pooled) < n_clusters * 2:
        return {"error": "too_few_tokens"}

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    all_labels = kmeans.fit_predict(pooled)
    centroids = kmeans.cluster_centers_

    # Silhouette score (if enough data)
    silhouette = None
    if len(pooled) > n_clusters:
        from sklearn.metrics import silhouette_score
        try:
            silhouette = float(silhouette_score(pooled, all_labels))
        except Exception:
            silhouette = None

    # Per-completion labels and dwell times
    comp_results = []
    all_dwells = defaultdict(list)

    for comp_idx, (s, e) in enumerate(comp_boundaries):
        labels = all_labels[s:e].tolist()
        dwells = compute_dwell_times(labels)
        n_switches = compute_switches(labels)

        for cluster_id, dwell_list in dwells.items():
            all_dwells[cluster_id].extend(dwell_list)

        comp_results.append({
            "labels": labels,
            "dwells": dwells,
            "n_switches": n_switches,
            "n_tokens": e - s,
        })

    # Identify dominant cluster (more total tokens)
    cluster_counts = np.bincount(all_labels, minlength=n_clusters)
    dominant_cluster = int(np.argmax(cluster_counts))
    non_dominant_cluster = 1 - dominant_cluster  # assumes k=2

    # Compute pooled CV for non-dominant dwell times
    nd_dwells = all_dwells.get(non_dominant_cluster, [])
    nd_cv = compute_cv(nd_dwells) if nd_dwells else None

    # Compute median non-dominant dwell time (DPR baseline denominator)
    nd_median_dwell = float(np.median(nd_dwells)) if nd_dwells else None

    # Cosine similarity threshold (median cosine to non-dominant centroid)
    nd_centroid = centroids[non_dominant_cluster]
    cosines = []
    for vec in pooled:
        cos = np.dot(vec, nd_centroid) / (
            np.linalg.norm(vec) * np.linalg.norm(nd_centroid) + 1e-8
        )
        cosines.append(float(cos))
    cosine_threshold = float(np.median(cosines))

    return {
        "centroids": centroids.tolist(),
        "dominant_cluster": dominant_cluster,
        "non_dominant_cluster": non_dominant_cluster,
        "cluster_counts": cluster_counts.tolist(),
        "silhouette": silhouette,
        "nd_cv": nd_cv,
        "nd_median_dwell": nd_median_dwell,
        "nd_dwells": nd_dwells,
        "cosine_threshold": cosine_threshold,
        "completions": comp_results,
    }


def cmd_cluster(args):
    """Cluster baseline activations and compute dwell-time statistics."""
    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    continue

    with open(args.activations_path, "r") as f_in, \
         open(args.output_path, "a") as f_out:

        for line in tqdm(f_in, desc="Clustering"):
            data = json.loads(line)
            pid = data["prompt_id"]
            if pid in processed_ids:
                continue

            result = cluster_prompt_activations(data["completions"])

            output = {
                "prompt_id": pid,
                "question": data["question"],
                "category": data.get("category", ""),
                "neuron_indices": data["neuron_indices"],
            }
            # Copy cluster results (except raw dwells to save space)
            for k, v in result.items():
                if k != "nd_dwells":
                    output[k] = v
            output["nd_dwells"] = result.get("nd_dwells", [])

            f_out.write(json.dumps(output) + "\n")
            f_out.flush()

    print("Clustering complete.")


# -------------------------------------------------------------------
# Intervention (Stages 4-5)
# -------------------------------------------------------------------

class InterventionHookManager:
    """Manages hooks that add steering vectors during generation.

    Two modes:
    - targeted (CAST-style): steer only when current activation is near
      the non-dominant centroid (cosine > threshold)
    - indiscriminate (ActAdd): steer at every token
    """

    def __init__(
        self,
        model,
        neuron_indices: List[Tuple[int, int]],
        steering_vector: np.ndarray,
        alpha: float,
        mode: str,  # "targeted" or "indiscriminate"
        nd_centroid: np.ndarray = None,
        cosine_threshold: float = 0.0,
    ):
        self.model = model
        self.neuron_indices = neuron_indices
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.mode = mode
        self.nd_centroid = nd_centroid
        self.cosine_threshold = cosine_threshold

        # Group neurons by layer
        self.layer_neurons = {}
        for layer, neuron in neuron_indices:
            if layer not in self.layer_neurons:
                self.layer_neurons[layer] = []
            self.layer_neurons[layer].append(neuron)

        # Build per-layer steering slices
        self.layer_steering = {}
        idx = 0
        for layer, neuron in neuron_indices:
            if layer not in self.layer_steering:
                self.layer_steering[layer] = {}
            local_idx = self.layer_neurons[layer].index(neuron)
            self.layer_steering.setdefault(layer, {})[local_idx] = idx
            idx += 1

        self.hooks = []
        self.step_activations = {}
        self.intervention_applied = False

        self._register_hooks()

    def _register_hooks(self):
        """Register hooks that both capture activations AND apply steering."""
        for name, module in self.model.named_modules():
            if "down_proj" not in name:
                continue
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
                    act = input[0]
                    if act.dim() == 3:
                        # Capture activations at last position
                        last_act = act[:, -1, :].detach()
                        selected = last_act[0, neuron_list].cpu().float().numpy()
                        self.step_activations[layer] = selected

                        # Apply steering to the input activations
                        should_steer = self._should_steer()
                        if should_steer:
                            # Build the steering delta for this layer
                            delta = torch.zeros_like(act[:, -1, :])
                            for local_idx, neuron_idx in enumerate(neuron_list):
                                global_key = (layer, neuron_idx)
                                # Find position in steering vector
                                sv_idx = self.neuron_indices.index(global_key)
                                delta[0, neuron_idx] = self.alpha * self.steering_vector[sv_idx]
                            # Modify input in-place (add to last token)
                            act[:, -1, :] += delta.to(act.device)
                            self.intervention_applied = True
                return hook_fn

            self.hooks.append(
                module.register_forward_hook(make_hook(layer_idx, neurons))
            )

    def _should_steer(self) -> bool:
        """Determine whether to apply steering at the current step."""
        if self.mode == "indiscriminate":
            return True

        if self.mode == "targeted":
            # CAST-style: steer only when near non-dominant centroid
            current = self.get_current_vector()
            if current is None or self.nd_centroid is None:
                return False
            cos = np.dot(current, self.nd_centroid) / (
                np.linalg.norm(current) * np.linalg.norm(self.nd_centroid) + 1e-8
            )
            return cos > self.cosine_threshold

        return False

    def get_current_vector(self) -> Optional[np.ndarray]:
        """Assemble current activation vector from captured step data."""
        if not self.step_activations:
            return None
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
        self.step_activations.clear()
        self.intervention_applied = False

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def generate_with_intervention(
    model, tokenizer,
    hook_manager: InterventionHookManager,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
) -> Tuple[str, List[str], List[List[float]], int]:
    """Generate one completion with intervention hooks active.

    Returns: (text, tokens, activations, n_interventions)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    generated_ids = input_ids.clone()

    all_activations = []
    all_tokens = []
    n_interventions = 0

    for step in range(max_new_tokens):
        hook_manager.clear_step()

        with torch.no_grad():
            outputs = model(generated_ids)

        logits = outputs.logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")
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

        vec = hook_manager.get_current_vector()
        if vec is not None:
            all_activations.append(vec.tolist())
        if hook_manager.intervention_applied:
            n_interventions += 1

        token_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
        all_tokens.append(token_str)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(
        generated_ids[0, input_len:], skip_special_tokens=True
    ).strip()

    return text, all_tokens, all_activations, n_interventions


def cmd_calibrate(args):
    """Alpha calibration sweep on pilot prompts."""
    print("Loading model and classifier...")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map=args.device
    )
    model.eval()

    classifier = joblib.load(args.classifier_path)
    inter_size = config.intermediate_size
    weights = classifier.coef_[0]
    neuron_indices = []
    for idx in np.where(weights != 0)[0]:
        neuron_indices.append((int(idx // inter_size), int(idx % inter_size)))
    neuron_indices.sort()

    # Load cluster data
    clusters = {}
    with open(args.clusters_path, "r") as f:
        for line in f:
            d = json.loads(line)
            clusters[d["prompt_id"]] = d

    # Select pilot prompts (first n_pilot Cat 2 prompts with valid clusters)
    pilot_prompts = []
    for pid, data in clusters.items():
        if data.get("category") == "2" and "centroids" in data:
            pilot_prompts.append(data)
            if len(pilot_prompts) >= args.n_pilot:
                break

    print(f"Pilot prompts: {len(pilot_prompts)}")

    alphas = [1.0, 2.0, 3.0, 5.0, 8.0]
    results = []
    suffix = " Respond with the answer only, without any explanation."

    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")
        for mode in ["targeted", "indiscriminate"]:
            mode_results = []
            for prompt_data in tqdm(pilot_prompts, desc=f"{mode} a={alpha}"):
                centroids = np.array(prompt_data["centroids"])
                nd_cluster = prompt_data["non_dominant_cluster"]
                nd_centroid = centroids[nd_cluster]
                d_centroid = centroids[prompt_data["dominant_cluster"]]

                # Steering vector: non-dominant minus dominant, unit normalised
                steering = nd_centroid - d_centroid
                norm = np.linalg.norm(steering)
                if norm > 0:
                    steering = steering / norm

                hook_mgr = InterventionHookManager(
                    model, neuron_indices, steering, alpha, mode,
                    nd_centroid=nd_centroid,
                    cosine_threshold=prompt_data.get("cosine_threshold", 0.0),
                )

                question = prompt_data["question"]
                full_prompt = question.strip() + suffix

                # Generate 5 completions per pilot prompt per condition
                for _ in range(5):
                    text, tokens, acts, n_int = generate_with_intervention(
                        model, tokenizer, hook_mgr,
                        full_prompt,
                        max_new_tokens=100,
                        temperature=1.0, top_p=0.9, top_k=50,
                        device=args.device,
                    )

                    # Check coherence
                    is_degenerate = (
                        len(tokens) < 20
                        or len(set(tokens[-10:])) <= 2  # repetitive
                    )

                    # Cluster the intervention activations
                    if acts:
                        km = KMeans(n_clusters=2, random_state=42, n_init=10)
                        labels = km.fit_predict(np.array(acts))
                        dwells = compute_dwell_times(labels.tolist())
                        nd_dwells = dwells.get(nd_cluster, [])
                        nd_median = float(np.median(nd_dwells)) if nd_dwells else 0
                    else:
                        nd_median = 0
                        is_degenerate = True

                    baseline_median = prompt_data.get("nd_median_dwell", 1)
                    dpr = nd_median / baseline_median if baseline_median > 0 else 0

                    mode_results.append({
                        "prompt_id": prompt_data["prompt_id"],
                        "dpr": dpr,
                        "n_interventions": n_int,
                        "n_tokens": len(tokens),
                        "degenerate": is_degenerate,
                    })

                hook_mgr.remove_hooks()

            # Summarise
            dprs = [r["dpr"] for r in mode_results if not r["degenerate"]]
            n_degen = sum(1 for r in mode_results if r["degenerate"])

            result_summary = {
                "alpha": alpha,
                "mode": mode,
                "mean_dpr": float(np.mean(dprs)) if dprs else None,
                "median_dpr": float(np.median(dprs)) if dprs else None,
                "n_degenerate": n_degen,
                "n_total": len(mode_results),
            }
            results.append(result_summary)
            print(f"  {mode}: DPR={result_summary['mean_dpr']:.3f}, "
                  f"degen={n_degen}/{len(mode_results)}")

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCalibration results saved to {args.output_path}")


def cmd_intervene(args):
    """Run full intervention experiment at selected alpha."""
    print("Loading model and classifier...")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map=args.device
    )
    model.eval()

    classifier = joblib.load(args.classifier_path)
    inter_size = config.intermediate_size
    weights = classifier.coef_[0]
    neuron_indices = []
    for idx in np.where(weights != 0)[0]:
        neuron_indices.append((int(idx // inter_size), int(idx % inter_size)))
    neuron_indices.sort()

    # Load cluster data
    clusters = {}
    with open(args.clusters_path, "r") as f:
        for line in f:
            d = json.loads(line)
            clusters[d["prompt_id"]] = d

    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    continue

    suffix = " Respond with the answer only, without any explanation."
    alpha = args.alpha
    directions = ["suppress", "amplify"] if args.run_h5 else ["suppress"]

    with open(args.output_path, "a") as f_out:
        for pid, prompt_data in tqdm(clusters.items(), desc="Intervening"):
            if pid in processed_ids:
                continue
            if "centroids" not in prompt_data:
                continue

            centroids = np.array(prompt_data["centroids"])
            nd_cluster = prompt_data["non_dominant_cluster"]
            nd_centroid = centroids[nd_cluster]
            d_centroid = centroids[prompt_data["dominant_cluster"]]

            question = prompt_data["question"]
            full_prompt = question.strip() + suffix

            prompt_result = {
                "prompt_id": pid,
                "question": question,
                "category": prompt_data.get("category", ""),
                "alpha": alpha,
                "baseline_nd_median_dwell": prompt_data.get("nd_median_dwell"),
                "conditions": {},
            }

            for direction in directions:
                # Steering vector direction
                if direction == "suppress":
                    # Steer away from dominant (toward non-dominant)
                    steering = nd_centroid - d_centroid
                else:
                    # Steer toward dominant (amplify dominant)
                    steering = d_centroid - nd_centroid

                norm = np.linalg.norm(steering)
                if norm > 0:
                    steering = steering / norm

                for mode in ["targeted", "indiscriminate"]:
                    condition_key = f"{direction}_{mode}"

                    hook_mgr = InterventionHookManager(
                        model, neuron_indices, steering, alpha, mode,
                        nd_centroid=nd_centroid,
                        cosine_threshold=prompt_data.get("cosine_threshold", 0.0),
                    )

                    condition_completions = []
                    for _ in range(args.num_completions):
                        text, tokens, acts, n_int = generate_with_intervention(
                            model, tokenizer, hook_mgr,
                            full_prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=1.0, top_p=0.9, top_k=50,
                            device=args.device,
                        )

                        is_degenerate = (
                            len(tokens) < 20
                            or len(set(tokens[-10:])) <= 2
                        )

                        condition_completions.append({
                            "text": text,
                            "n_tokens": len(tokens),
                            "n_interventions": n_int,
                            "degenerate": is_degenerate,
                            "activations": acts,
                        })

                    hook_mgr.remove_hooks()
                    prompt_result["conditions"][condition_key] = condition_completions

            f_out.write(json.dumps(prompt_result) + "\n")
            f_out.flush()

    print("Intervention experiment complete.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="H4/H5 Intervention Experiments"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Cluster subcommand
    p_cluster = subparsers.add_parser("cluster")
    p_cluster.add_argument("--activations_path", type=str, required=True)
    p_cluster.add_argument("--output_path", type=str, required=True)

    # Calibrate subcommand
    p_cal = subparsers.add_parser("calibrate")
    p_cal.add_argument("--model_path", type=str, required=True)
    p_cal.add_argument("--classifier_path", type=str, required=True)
    p_cal.add_argument("--clusters_path", type=str, required=True)
    p_cal.add_argument("--output_path", type=str, required=True)
    p_cal.add_argument("--n_pilot", type=int, default=10)
    p_cal.add_argument("--device", type=str, default="mps")

    # Intervene subcommand
    p_int = subparsers.add_parser("intervene")
    p_int.add_argument("--model_path", type=str, required=True)
    p_int.add_argument("--classifier_path", type=str, required=True)
    p_int.add_argument("--clusters_path", type=str, required=True)
    p_int.add_argument("--output_path", type=str, required=True)
    p_int.add_argument("--alpha", type=float, required=True)
    p_int.add_argument("--num_completions", type=int, default=20)
    p_int.add_argument("--max_new_tokens", type=int, default=100)
    p_int.add_argument("--run_h5", action="store_true", default=False)
    p_int.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()

    if args.command == "cluster":
        cmd_cluster(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "intervene":
        cmd_intervene(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
