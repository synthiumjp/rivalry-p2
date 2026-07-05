"""
hneuron_layer_correspondence.py  (patched: joblib classifier loader)

Patched: switched classifier loader from raw pickle to joblib-first
(matches the actual save format used by the detector .pkl).

Tests whether the H-Neuron and Anti-H neuron layers (from the CETT classifier)
preferentially fall at high |dAUROC/dlayer| in the depth profile (inflection
layers where the correctness signal is being built).

Statistic: weighted sum of |dAUROC/dlayer| at the 18 H-Neuron/Anti-H layers,
weighted by |classifier coefficient|. Null: permute the 18 features to random
positions in the (n_layers x intermediate_dim) feature space.

Usage:
    python scripts/hneuron_layer_correspondence.py \
        --classifier models/detector_mistral_8of10.pkl \
        --dynamics data/layer_dynamics_lastanswer_mistral_instruct.json \
        --output data/hneuron_layer_correspondence.json \
        --intermediate_dim 14336
"""

import os
import json
import argparse
import pickle

import numpy as np
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", type=str, required=True)
    p.add_argument("--dynamics", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--intermediate_dim", type=int, default=14336)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_classifier_coef(path):
    """Try joblib first, then pickle. Returns the 1D coefficient vector."""
    errors = []
    try:
        import joblib
        obj = joblib.load(path)
        if hasattr(obj, "coef_"):
            return np.asarray(obj.coef_).ravel().astype(np.float64)
        errors.append("joblib: no coef_ attribute on loaded object")
    except Exception as e:
        errors.append(f"joblib: {type(e).__name__}: {e}")
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "coef_"):
            return np.asarray(obj.coef_).ravel().astype(np.float64)
        errors.append("pickle: no coef_ attribute on loaded object")
    except Exception as e:
        errors.append(f"pickle: {type(e).__name__}: {e}")
    raise RuntimeError(
        f"Could not load classifier from {path}. Attempts:\n  "
        + "\n  ".join(errors)
    )


def main():
    args = parse_args()

    with open(args.dynamics) as f:
        dyn = json.load(f)
    aurocs = np.array(dyn["layer_aurocs"], dtype=np.float64)
    n_layers_curve = len(aurocs)

    dauroc = np.zeros_like(aurocs)
    dauroc[0] = aurocs[1] - aurocs[0]
    dauroc[-1] = aurocs[-1] - aurocs[-2]
    for l in range(1, n_layers_curve - 1):
        dauroc[l] = (aurocs[l+1] - aurocs[l-1]) / 2
    abs_d = np.abs(dauroc)

    coef = load_classifier_coef(args.classifier)
    nz_idx = np.where(coef != 0)[0]
    weights = np.abs(coef[nz_idx])

    inferred_n_layers = (len(coef) + args.intermediate_dim - 1) // args.intermediate_dim
    print(f"Classifier feature space: {len(coef)} = ~{inferred_n_layers} layers "
          f"x {args.intermediate_dim} intermediate dim")
    block_offset = 1 if n_layers_curve == inferred_n_layers + 1 else 0

    feat_layers = nz_idx // args.intermediate_dim
    feat_neurons = nz_idx % args.intermediate_dim
    curve_layers = feat_layers + block_offset
    valid = (curve_layers >= 0) & (curve_layers < n_layers_curve)
    if not valid.all():
        print(f"WARN: {(~valid).sum()} features map out of curve range")
    curve_layers = curve_layers[valid]
    weights = weights[valid]

    print(f"\n{len(weights)} nonzero features mapped onto AUROC curve.")
    print(f"  Feature layers (classifier indices): "
          f"{sorted(set(feat_layers[valid].tolist()))}")
    print(f"  Mapped to AUROC-curve layers: "
          f"{sorted(set(curve_layers.tolist()))}  (offset {block_offset})")

    obs_stat = float(np.sum(weights * abs_d[curve_layers]))

    rng = np.random.default_rng(args.seed)
    n_features_total = inferred_n_layers * args.intermediate_dim
    null_stats = np.zeros(args.n_perm)
    for i in range(args.n_perm):
        perm_idx = rng.integers(0, n_features_total, size=len(weights))
        perm_layers = (perm_idx // args.intermediate_dim) + block_offset
        perm_layers = np.clip(perm_layers, 0, n_layers_curve - 1)
        null_stats[i] = float(np.sum(weights * abs_d[perm_layers]))

    p_two = float(np.mean(np.abs(null_stats - null_stats.mean())
                          >= abs(obs_stat - null_stats.mean())))
    p_one = float(np.mean(null_stats >= obs_stat))

    print("\n" + "=" * 60)
    print("H-NEURON LAYER CORRESPONDENCE WITH AUROC INFLECTION")
    print("=" * 60)
    print(f"  Observed statistic (sum |coef| * |dAUROC|): {obs_stat:.4f}")
    print(f"  Null mean: {null_stats.mean():.4f}  std: {null_stats.std():.4f}")
    print(f"  p (one-sided, observed >= null): {p_one:.4f}")
    print(f"  p (two-sided): {p_two:.4f}")
    print(f"  z-score: {(obs_stat - null_stats.mean()) / (null_stats.std()+1e-12):.2f}")

    print(f"\n  Per-layer contribution:")
    print(f"  {'Layer':>5} {'|dAUROC|':>10} {'feat':>5} {'sum|coef|':>10}")
    by_layer = {}
    for l, w in zip(curve_layers.tolist(), weights.tolist()):
        by_layer.setdefault(l, []).append(w)
    for l in sorted(by_layer.keys()):
        ws = by_layer[l]
        print(f"  {l:5d} {abs_d[l]:10.4f} {len(ws):5d} {sum(ws):10.4f}")

    print("\n=== INTERPRETATION ===")
    if p_one < 0.05:
        print("  H-Neuron and Anti-H layers significantly cluster at AUROC")
        print("  inflection layers. The H-Neuron method reads decision-relevant")
        print("  layers; the static probe works because it samples the layers")
        print("  where the answer is being built.")
    else:
        print("  H-Neuron layers do NOT preferentially cluster at AUROC")
        print("  inflection. The H-Neuron method works by some other mechanism")
        print("  than reading decision-relevant layers.")

    def to_native(o):
        if isinstance(o, dict):
            return {str(k) if isinstance(k, (np.integer, int)) else k: to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_native(v) for v in o]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    results = {
        "n_features": int(len(weights)),
        "block_offset": int(block_offset),
        "observed_statistic": obs_stat,
        "null_mean": float(null_stats.mean()),
        "null_std": float(null_stats.std()),
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "z_score": float((obs_stat - null_stats.mean()) / (null_stats.std() + 1e-12)),
        "n_permutations": args.n_perm,
        "per_layer_weight_sum": {int(l): float(sum(by_layer[l])) for l in by_layer},
        "per_layer_abs_dauroc": {int(l): float(abs_d[l]) for l in sorted(set(curve_layers.tolist()))},
        "layer_aurocs": aurocs.tolist(),
    }
    with open(args.output, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
