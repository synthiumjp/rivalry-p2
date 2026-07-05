"""
analyse_layer_dynamics.py

Tests the relocated rivalry hypothesis: does the correctness signal show
accumulator/competition dynamics ACROSS LAYER DEPTH (within a forward pass)
rather than across generated tokens?

Pipeline:
  1. Load captured layer stacks + correctness labels
  2. Train a linear probe per layer (correctness from hidden state)
  3. Track probe AUROC across depth (the "buildup" curve)
  4. Project each completion's hidden state onto the probe direction at each
     layer -> across-layer trajectory of the correctness score
  5. Test for LCA signatures:
     a. Monotonic buildup vs non-monotonic (competition produces overshoot)
     b. Separation dynamics: do correct and incorrect trajectories diverge
        with a characteristic depth (the "decision layer")?
     c. Bistability: at the separation layer, is the score distribution
        bimodal (two attractors) rather than unimodal?

Interpretation:
  - Monotonic single buildup, unimodal -> single accumulator, no rivalry
  - Non-monotonic with crossover/overshoot, bimodal at decision layer ->
    competing accumulators (LCA signature), rivalry relocated to depth axis

Usage:
    python analyse_layer_dynamics.py \
        --index_path data/layer_stack_index_mistral.jsonl \
        --output_path data/layer_dynamics_results_mistral.json \
        --position prompt_final
"""

import os
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.signal import find_peaks


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse layer dynamics.")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--position", type=str, default="prompt_final",
                        help="Which captured position to analyse")
    parser.add_argument("--category", type=str, default="1",
                        help="Category to use for correctness labels (1 = has GT)")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_layer_data(
    index_path: str, position: str, category: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load hidden stacks and correctness labels for a given position.

    Returns:
        X: [n_samples, n_layers, hidden_dim]
        y: [n_samples] correctness labels
        prompt_ids: list of prompt ids
    """
    entries = []
    with open(index_path) as f:
        for line in f:
            e = json.loads(line)
            if category and e.get("category") != category:
                continue
            if e.get("correct") is None:
                continue
            entries.append(e)

    X_list = []
    y_list = []
    pids = []

    for e in tqdm(entries, desc="Loading stacks"):
        if not os.path.exists(e["npz_path"]):
            continue
        data = np.load(e["npz_path"], allow_pickle=True)
        hidden_stack = data["hidden_stack"].astype(np.float32)
        pos_labels = list(data["position_labels"])

        if position not in pos_labels:
            continue
        pos_idx = pos_labels.index(position)

        X_list.append(hidden_stack[pos_idx])  # [n_layers, hidden_dim]
        y_list.append(1 if e["correct"] else 0)
        pids.append(e["prompt_id"])

    X = np.stack(X_list)  # [n_samples, n_layers, hidden_dim]
    y = np.array(y_list)
    return X, y, pids


def train_layer_probes(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train a probe per layer; return AUROC curve and OUT-OF-SAMPLE scores.

    CRITICAL: projections must be cross-validated (out-of-sample). Using
    in-sample projections (fit on all data, project same data) makes the
    separation and bimodality tests circular: the probe separates its own
    training labels by construction, producing a fake bimodal distribution
    and inflated Cohen's d. We use cross_val_predict with decision_function
    so every projection is for a held-out sample.

    Returns:
        layer_aurocs: [n_layers] cross-validated AUROC per layer
        projections: [n_samples, n_layers] OUT-OF-SAMPLE probe score
    """
    n_samples, n_layers, hidden_dim = X.shape
    layer_aurocs = []
    projections = np.zeros((n_samples, n_layers))

    for layer in tqdm(range(n_layers), desc="Training layer probes"):
        Xl = X[:, layer, :]

        # Standardize
        mean = Xl.mean(axis=0)
        std = Xl.std(axis=0) + 1e-8
        Xl_norm = (Xl - mean) / std

        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)

        # Cross-validated AUROC
        try:
            cv_probs = cross_val_predict(
                clf, Xl_norm, y, cv=5, method="predict_proba"
            )[:, 1]
            auroc = roc_auc_score(y, cv_probs)
        except Exception:
            auroc = 0.5
        layer_aurocs.append(float(auroc))

        # OUT-OF-SAMPLE projections via cross-validated decision function.
        # This is the fix: each sample's projection comes from a probe that
        # did NOT see it during fitting. No circularity.
        try:
            oos_scores = cross_val_predict(
                clf, Xl_norm, y, cv=5, method="decision_function"
            )
            projections[:, layer] = oos_scores
        except Exception:
            projections[:, layer] = 0.0

    return {
        "layer_aurocs": layer_aurocs,
        "projections": projections,
    }


def test_buildup_shape(layer_aurocs: List[float]) -> Dict:
    """Test whether AUROC buildup across depth is monotonic or non-monotonic.

    Monotonic buildup -> single accumulator.
    Non-monotonic (peak then decline, or overshoot) -> possible competition.
    """
    aurocs = np.array(layer_aurocs)
    n = len(aurocs)

    # Find peak layer
    peak_layer = int(np.argmax(aurocs))
    peak_auroc = float(aurocs[peak_layer])

    # Monotonicity: Spearman correlation with layer index
    rho, p = stats.spearmanr(np.arange(n), aurocs)

    # Does it decline after peak? (overshoot signature)
    if peak_layer < n - 1:
        post_peak_decline = float(aurocs[peak_layer] - aurocs[-1])
    else:
        post_peak_decline = 0.0

    # Detect local peaks (multiple peaks suggest competition oscillation)
    peaks, _ = find_peaks(aurocs, height=0.6)

    return {
        "peak_layer": peak_layer,
        "peak_auroc": peak_auroc,
        "final_auroc": float(aurocs[-1]),
        "monotonicity_rho": float(rho),
        "monotonicity_p": float(p),
        "post_peak_decline": post_peak_decline,
        "n_local_peaks": len(peaks),
        "local_peak_layers": peaks.tolist(),
        "is_monotonic": rho > 0.7 and post_peak_decline < 0.05,
    }


def test_separation_dynamics(
    projections: np.ndarray, y: np.ndarray
) -> Dict:
    """Test how correct vs incorrect trajectories separate across depth.

    LCA prediction: trajectories start together, separate at a characteristic
    "decision layer", and the separation is sharp (competition resolves).
    """
    n_samples, n_layers = projections.shape

    correct_mask = y == 1
    incorrect_mask = y == 0

    separation = []  # mean(correct) - mean(incorrect) per layer
    cohens_d = []

    for layer in range(n_layers):
        proj_c = projections[correct_mask, layer]
        proj_i = projections[incorrect_mask, layer]
        if len(proj_c) < 2 or len(proj_i) < 2:
            separation.append(0.0)
            cohens_d.append(0.0)
            continue
        sep = float(np.mean(proj_c) - np.mean(proj_i))
        pooled_std = np.sqrt((np.var(proj_c) + np.var(proj_i)) / 2) + 1e-8
        d = sep / pooled_std
        separation.append(sep)
        cohens_d.append(float(d))

    cohens_d = np.array(cohens_d)

    # Decision layer: where separation crosses half its max
    max_d = np.max(np.abs(cohens_d))
    decision_layer = None
    if max_d > 0:
        half_max = max_d / 2
        for layer in range(n_layers):
            if abs(cohens_d[layer]) >= half_max:
                decision_layer = layer
                break

    # Sharpness: derivative of separation (how fast it resolves)
    sep_derivative = np.diff(np.abs(cohens_d))
    max_sharpness_layer = int(np.argmax(sep_derivative)) if len(sep_derivative) > 0 else None

    return {
        "separation_curve": separation,
        "cohens_d_curve": cohens_d.tolist(),
        "max_cohens_d": float(max_d),
        "max_d_layer": int(np.argmax(np.abs(cohens_d))),
        "decision_layer": decision_layer,
        "max_sharpness_layer": max_sharpness_layer,
    }


def test_bistability_at_layer(
    projections: np.ndarray, layer: int
) -> Dict:
    """Test whether the projection distribution at the decision layer is
    bimodal (two attractors) rather than unimodal.

    Uses Hartigan dip test approximation via the bimodality coefficient.
    """
    scores = projections[:, layer]

    n = len(scores)
    if n < 10:
        return {"status": "insufficient"}

    # Bimodality coefficient: (skew^2 + 1) / kurtosis
    # BC > 0.555 suggests bimodality
    skew = stats.skew(scores)
    kurt = stats.kurtosis(scores, fisher=False)
    if kurt > 0:
        bc = (skew**2 + 1) / kurt
    else:
        bc = 0.0

    # Also fit 1 vs 2 component GMM and compare BIC
    from sklearn.mixture import GaussianMixture
    X = scores.reshape(-1, 1)
    try:
        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=42).fit(X)
        bic1 = gmm1.bic(X)
        bic2 = gmm2.bic(X)
        prefers_bimodal = bic2 < bic1
        bic_diff = float(bic1 - bic2)
    except Exception:
        prefers_bimodal = False
        bic_diff = 0.0

    return {
        "layer": layer,
        "bimodality_coefficient": float(bc),
        "bc_suggests_bimodal": bc > 0.555,
        "gmm_prefers_bimodal": prefers_bimodal,
        "bic_diff_1_minus_2": bic_diff,
    }


def main():
    args = parse_args()

    print(f"Loading layer data for position '{args.position}'...")
    X, y, pids = load_layer_data(args.index_path, args.position, args.category)
    print(f"Loaded {len(y)} samples. Correct: {y.sum()}, Incorrect: {(1-y).sum()}")
    print(f"Shape: {X.shape} (samples, layers, hidden_dim)")

    if y.sum() < 5 or (1 - y).sum() < 5:
        print("Insufficient class balance for probe training.")
        return

    # Train per-layer probes
    probe_results = train_layer_probes(X, y)
    layer_aurocs = probe_results["layer_aurocs"]
    projections = probe_results["projections"]

    print("\n=== Layer AUROC Buildup ===")
    for i, auroc in enumerate(layer_aurocs):
        bar = "#" * int(auroc * 40)
        print(f"  Layer {i:2d}: {auroc:.3f} {bar}")

    # Test buildup shape
    buildup = test_buildup_shape(layer_aurocs)
    print(f"\n=== Buildup Shape ===")
    print(f"  Peak layer: {buildup['peak_layer']} (AUROC {buildup['peak_auroc']:.3f})")
    print(f"  Final layer AUROC: {buildup['final_auroc']:.3f}")
    print(f"  Post-peak decline: {buildup['post_peak_decline']:.3f}")
    print(f"  Monotonic: {buildup['is_monotonic']}")
    print(f"  Local peaks: {buildup['n_local_peaks']}")

    # Test separation dynamics
    separation = test_separation_dynamics(projections, y)
    print(f"\n=== Separation Dynamics ===")
    print(f"  Max Cohen's d: {separation['max_cohens_d']:.3f} at layer {separation['max_d_layer']}")
    print(f"  Decision layer (half-max): {separation['decision_layer']}")
    print(f"  Sharpest separation at layer: {separation['max_sharpness_layer']}")

    # Test bistability at the AUROC PEAK layer. This is where correctness is
    # most decodable (your PT-CSFT probes peaked at the middle layer). If the
    # peak is mid-network and AUROC declines after it, the decline could be
    # benign dilution (single attractor fading) or competition (bimodal, two
    # attractors). Bimodality at the peak distinguishes them.
    peak_layer = buildup["peak_layer"]
    bistability_peak = test_bistability_at_layer(projections, peak_layer)
    print(f"\n=== Bistability at AUROC Peak Layer {peak_layer} ===")
    if bistability_peak.get("status") != "insufficient":
        print(f"  Bimodality coefficient: {bistability_peak['bimodality_coefficient']:.3f} "
              f"(>0.555 suggests bimodal)")
        print(f"  GMM prefers bimodal: {bistability_peak['gmm_prefers_bimodal']}")
        print(f"  BIC difference (1-2): {bistability_peak['bic_diff_1_minus_2']:.1f}")

    # Test bistability at the decision layer
    decision_layer = separation["max_d_layer"]
    bistability = test_bistability_at_layer(projections, decision_layer)
    print(f"\n=== Bistability at Max-Separation Layer {decision_layer} ===")
    if bistability.get("status") != "insufficient":
        print(f"  Bimodality coefficient: {bistability['bimodality_coefficient']:.3f} "
              f"(>0.555 suggests bimodal)")
        print(f"  GMM prefers bimodal: {bistability['gmm_prefers_bimodal']}")
        print(f"  BIC difference (1-2): {bistability['bic_diff_1_minus_2']:.1f}")

    # Overall interpretation.
    # The decisive question is NOT whether AUROC peaks mid-network and declines.
    # That shape is expected from benign dilution (later layers repurpose the
    # residual stream for next-token prediction). The decisive question is
    # whether, at the peak layer where correctness is most decodable, the
    # out-of-sample projection distribution is BIMODAL (two attractors, the
    # LCA signature) or UNIMODAL (one attractor, ordinary readout).
    print(f"\n=== Interpretation ===")

    peak_auroc = buildup["peak_auroc"]
    if peak_auroc < 0.65:
        print(f"  [!] Peak AUROC {peak_auroc:.3f} is near chance.")
        print("      Signal too weak at this position to test competition.")
        print("      VERDICT: inconclusive. Need cleaner labels (instruct model)")
        print("               or a position where correctness is decodable.")
        competition_signatures = 0
    else:
        competition_signatures = 0

        # The ONLY strong competition signature is bimodality at the peak,
        # computed on cross-validated projections (no circularity).
        if bistability_peak.get("gmm_prefers_bimodal") and \
           bistability_peak.get("bc_suggests_bimodal"):
            competition_signatures += 2
            print(f"  [+] Bimodal at peak layer {peak_layer} (BOTH GMM and BC agree):")
            print("      two attractors at the layer of maximal decodability.")
        elif bistability_peak.get("gmm_prefers_bimodal") or \
             bistability_peak.get("bc_suggests_bimodal"):
            competition_signatures += 1
            print(f"  [~] Weak bimodality at peak layer {peak_layer} "
                  "(only one of GMM/BC agrees).")
        else:
            print(f"  [-] Unimodal at peak layer {peak_layer}: single attractor.")

        # Post-peak decline is supportive ONLY if the peak is also bimodal.
        if buildup["post_peak_decline"] > 0.05 and competition_signatures >= 1:
            print(f"  [+] Post-peak decline ({buildup['post_peak_decline']:.3f}) "
                  "consistent with overshoot-and-settle.")
        elif buildup["post_peak_decline"] > 0.05:
            print(f"  [~] Post-peak decline ({buildup['post_peak_decline']:.3f}) "
                  "present but peak is unimodal: likely benign dilution, "
                  "not competition.")

    print(f"\n  Competition evidence score: {competition_signatures}")
    if competition_signatures >= 2:
        print("  VERDICT: Bimodal structure at the layer of maximal correctness")
        print("           decodability. This is a genuine competition signature.")
        print("           The rivalry framing relocates to the depth axis.")
        print("           NEXT: replicate on Llama; confirm bimodality is not")
        print("           driven by a confound (answer length, token type).")
    else:
        print("  VERDICT: No competition signature at the peak layer.")
        print("           The depth profile is single-accumulator readout.")
        print("           Negative result holds. This is defensible and clean.")

    # JSON-safe conversion (numpy bools/ints/floats are not serializable)
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_native(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    # Save
    results = {
        "position": args.position,
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "layer_aurocs": layer_aurocs,
        "buildup": buildup,
        "separation": {k: v for k, v in separation.items()
                       if k not in ["separation_curve", "cohens_d_curve"]},
        "cohens_d_curve": separation["cohens_d_curve"],
        "bistability_peak": bistability_peak,
        "bistability_maxsep": bistability,
        "competition_signatures": competition_signatures,
    }
    with open(args.output_path, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
