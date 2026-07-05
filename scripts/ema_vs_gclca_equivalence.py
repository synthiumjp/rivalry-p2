"""
ema_vs_gclca_equivalence.py  (v3: schema-aware)

Patched for the actual hneuron_activations_mistral.jsonl schema:
  - Per prompt: {prompt_id, question, category, neuron_indices, completions[]}
  - Per completion: {text, tokens, activations [n_tokens, 18], n_tokens}
  - Correctness is judged on-the-fly against ground_truth loaded from the
    benchmark file (no 'correct' field stored in the activations file).
  - The 18 activation dimensions are aligned to the 18 nonzero classifier
    coefficients via the (layer, neuron) flat-index mapping:
        flat = layer * intermediate_dim + neuron_idx
        coef_aligned[i] = classifier.coef[flat_i]

The unit of analysis is the completion (not the prompt). Cat1 has ~100 prompts
x ~20 completions = ~2000 samples after filtering.

Tests: at end-of-generation, do EMA(h) and a GC-LCA accumulator on the
per-token h(t) signal beat mean(h) and the instantaneous probe? Pre-spec:
no, because lag-1 autocorrelation ~0.06 means leaky integration reduces to
averaging.

Usage:
    python scripts/ema_vs_gclca_equivalence.py \
        --activations data/hneuron_activations_mistral.jsonl \
        --benchmark   data/benchmark_final_250.jsonl \
        --classifier  models/detector_mistral_8of10.pkl \
        --output      data/ema_vs_gclca_results.json \
        --category 1 \
        --intermediate_dim 14336
"""

import os
import re
import json
import string
import argparse

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activations", type=str, required=True)
    p.add_argument("--benchmark", type=str, required=True,
                   help="benchmark_final_250.jsonl with prompt_id and ground_truth")
    p.add_argument("--classifier", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--category", type=str, default="1")
    p.add_argument("--intermediate_dim", type=int, default=14336)
    p.add_argument("--ema_alphas", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--lca_lambda", type=float, default=0.15)
    p.add_argument("--lca_beta", type=float, default=0.225)
    p.add_argument("--lca_alpha", type=float, default=0.06)
    p.add_argument("--lca_kappa", type=float, default=0.045)
    p.add_argument("--lca_sigma", type=float, default=0.05)
    p.add_argument("--lca_substeps", type=int, default=10)
    p.add_argument("--cross_thr", type=float, default=0.5)
    p.add_argument("--max_completions", type=int, default=20,
                   help="Cap completions per prompt (data has up to 20)")
    return p.parse_args()


def normalize(s):
    if not s:
        return ""
    s = s.lower().replace("_", " ")
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def judge_correct(text, ground_truth):
    if not ground_truth:
        return None
    norm_text = normalize(text)
    for gt in ground_truth:
        ng = normalize(gt)
        if ng and ng in norm_text:
            return True
    return False


def load_classifier(path):
    import joblib
    obj = joblib.load(path)
    coef = np.asarray(obj.coef_).ravel().astype(np.float64)
    bias = float(np.asarray(obj.intercept_).ravel()[0])
    nz = np.where(coef != 0)[0]
    print(f"  Loaded via joblib. coef shape {coef.shape}, "
          f"{len(nz)} nonzero, bias {bias:.4f}")
    return coef, bias, nz


def align_weights(neuron_indices, coef, intermediate_dim):
    """Map [n_features, 2] (layer, neuron) pairs to corresponding classifier
    coefficients. Returns aligned_weights [n_features]."""
    aligned = np.zeros(len(neuron_indices), dtype=np.float64)
    missing = 0
    for i, (l, n) in enumerate(neuron_indices):
        flat = int(l) * intermediate_dim + int(n)
        if flat >= len(coef):
            missing += 1
            continue
        aligned[i] = coef[flat]
    return aligned, missing


def ema_trajectory(h, alpha):
    out = np.zeros_like(h)
    out[0] = h[0]
    for t in range(1, len(h)):
        out[t] = alpha * h[t] + (1 - alpha) * out[t-1]
    return out


def gc_lca_trajectory(h, args, rng):
    T = len(h)
    x_f, x_c = 0.0, 0.0
    a_f, a_c = 0.0, 0.0
    xf_traj = np.zeros(T)
    xc_traj = np.zeros(T)
    dt = 1.0 / args.lca_substeps
    for t in range(T):
        I_f = 1.0 - h[t]
        I_c = h[t]
        for _ in range(args.lca_substeps):
            n_f = rng.normal(0, args.lca_sigma)
            n_c = rng.normal(0, args.lca_sigma)
            dx_f = (-args.lca_lambda * x_f - args.lca_beta * x_c
                    + I_f - args.lca_kappa * a_f + n_f) * dt
            dx_c = (-args.lca_lambda * x_c - args.lca_beta * x_f
                    + I_c - args.lca_kappa * a_c + n_c) * dt
            x_f = max(0.0, x_f + dx_f)
            x_c = max(0.0, x_c + dx_c)
            a_f += args.lca_alpha * (x_f - a_f) * dt
            a_c += args.lca_alpha * (x_c - a_c) * dt
        xf_traj[t] = x_f
        xc_traj[t] = x_c
    return xf_traj, xc_traj


def first_crossing(series, thr):
    above = np.where(series > thr)[0]
    return int(above[0]) if len(above) else len(series)


def load_ground_truth(path):
    """Build prompt_id -> ground_truth_list dict from benchmark file."""
    gt = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            pid = e.get("prompt_id")
            gtl = e.get("ground_truth", [])
            if pid and gtl:
                gt[pid] = gtl
    return gt


def main():
    args = parse_args()
    print(f"Loading classifier: {args.classifier}")
    coef, bias, nz = load_classifier(args.classifier)

    print(f"Loading benchmark: {args.benchmark}")
    gt = load_ground_truth(args.benchmark)
    print(f"  {len(gt)} prompts with ground_truth.")

    prompts = []
    with open(args.activations) as f:
        for line in f:
            d = json.loads(line)
            if args.category and str(d.get("category")) != args.category:
                continue
            prompts.append(d)
    print(f"Loaded {len(prompts)} prompts (category {args.category}).")

    rng = np.random.default_rng(42)
    summary = []
    skipped_no_gt = 0
    skipped_short = 0
    skipped_misaligned = 0

    for prompt in tqdm(prompts, desc="Running detectors"):
        pid = prompt["prompt_id"]
        ground = gt.get(pid)
        if not ground:
            skipped_no_gt += 1
            continue

        ni = prompt.get("neuron_indices")
        if not ni or len(ni) == 0:
            skipped_misaligned += 1
            continue
        aligned, missing = align_weights(ni, coef, args.intermediate_dim)
        if missing > 0:
            # Should not happen if classifier and pipeline are consistent
            skipped_misaligned += 1
            continue

        completions = prompt.get("completions", [])[:args.max_completions]
        for ci, c in enumerate(completions):
            acts = c.get("activations")
            text = c.get("text", "")
            if acts is None:
                continue
            acts = np.asarray(acts, dtype=np.float64)
            if acts.ndim != 2 or acts.shape[1] != len(aligned):
                skipped_misaligned += 1
                continue
            if acts.shape[0] < 3:
                skipped_short += 1
                continue

            correct = judge_correct(text, ground)
            if correct is None:
                continue

            # h(t)
            logit = acts @ aligned + bias
            h = 1.0 / (1.0 + np.exp(-logit))

            mean_h = float(np.mean(h))
            emas = {a: ema_trajectory(h, a) for a in args.ema_alphas}
            ema_end = {a: float(emas[a][-1]) for a in args.ema_alphas}
            xf, xc = gc_lca_trajectory(h, args, rng)
            lca_diff_end = float(xc[-1] - xf[-1])
            lca_xc_end = float(xc[-1])
            cross_raw = first_crossing(h, args.cross_thr)
            cross_emas = {a: first_crossing(emas[a], args.cross_thr)
                          for a in args.ema_alphas}
            cross_lca = first_crossing(xc - xf, 0.0)

            summary.append({
                "prompt_id": pid,
                "completion_idx": ci,
                "correct": bool(correct),
                "n_tokens": int(acts.shape[0]),
                "h_inst_last": float(h[-1]),
                "h_mean": mean_h,
                "ema_end": ema_end,
                "lca_diff_end": lca_diff_end,
                "lca_xc_end": lca_xc_end,
                "cross_raw": cross_raw,
                "cross_emas": cross_emas,
                "cross_lca": cross_lca,
            })

    n = len(summary)
    print(f"\nUsable completions: {n}")
    print(f"  skipped (no ground_truth): {skipped_no_gt}")
    print(f"  skipped (short): {skipped_short}")
    print(f"  skipped (misaligned): {skipped_misaligned}")

    if n < 50:
        print(f"Too few usable samples. Aborting.")
        return

    y = np.array([1 if not s["correct"] else 0 for s in summary])
    n_hall = int(y.sum())
    print(f"  hallucinated: {n_hall}  correct: {n - n_hall}")

    def auroc(scores):
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, scores))

    auroc_inst = auroc(np.array([s["h_inst_last"] for s in summary]))
    auroc_mean = auroc(np.array([s["h_mean"] for s in summary]))
    auroc_emas = {a: auroc(np.array([s["ema_end"][a] for s in summary]))
                  for a in args.ema_alphas}
    auroc_lca_diff = auroc(np.array([s["lca_diff_end"] for s in summary]))
    auroc_lca_xc = auroc(np.array([s["lca_xc_end"] for s in summary]))

    def median_lead(key):
        if key == "lca":
            arr = np.array([s["cross_lca"] - s["cross_raw"] for s in summary])
        else:
            arr = np.array([s["cross_emas"][key] - s["cross_raw"] for s in summary])
        return float(np.median(arr))

    leads = {f"ema_{a}": median_lead(a) for a in args.ema_alphas}
    leads["lca"] = median_lead("lca")

    spread_aurocs = [auroc_mean, auroc_lca_diff] + list(auroc_emas.values())
    auroc_spread = max(spread_aurocs) - min(spread_aurocs)
    max_abs_lead = max(abs(v) for v in leads.values())

    print("\n" + "=" * 60)
    print("DETECTOR COMPARISON (Cat1 end-of-generation)")
    print("=" * 60)
    print(f"  Instantaneous h(T)     AUROC: {auroc_inst:.3f}")
    print(f"  Mean h                 AUROC: {auroc_mean:.3f}")
    for a in args.ema_alphas:
        print(f"  EMA alpha={a:.2f}         AUROC: {auroc_emas[a]:.3f}")
    print(f"  GC-LCA x_c - x_f       AUROC: {auroc_lca_diff:.3f}")
    print(f"  GC-LCA x_c             AUROC: {auroc_lca_xc:.3f}")
    print()
    print(f"  AUROC spread (mean/EMA/LCA): {auroc_spread:.3f}")
    print(f"    Pre-spec: < 0.02 confirms equivalence")
    print(f"    Verdict: {'EQUIVALENT' if auroc_spread < 0.02 else 'DIVERGE'}")
    print()
    print("  Median temporal lead vs instantaneous probe "
          "(negative = earlier crossing):")
    for k, v in leads.items():
        print(f"    {k:>12}: {v:+.2f} tokens")
    print(f"  Max |lead|: {max_abs_lead:.2f}")
    print(f"    Pre-spec: <= 1 token confirms no useful lead")
    print(f"    Verdict: {'NO LEAD' if max_abs_lead <= 1.0 else 'LEAD'}")

    print("\n=== INTERPRETATION ===")
    if auroc_spread < 0.02 and max_abs_lead <= 1.0:
        print("  Pre-spec confirmed. EMA and GC-LCA match mean(h) on AUROC and")
        print("  neither leads the instantaneous probe. Rivalry-style dynamical")
        print("  detection adds no information on this signal. Paper 3 dynamical")
        print("  framing is retired with empirical support.")
    else:
        print("  Pre-spec NOT confirmed. Detectors diverge or one shows lead.")

    def to_native(o):
        if isinstance(o, dict):
            return {(str(k) if isinstance(k, (np.floating, np.integer, float, int)) else k): to_native(v) for k, v in o.items()}
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
        "n_samples": n,
        "n_hallucinated": n_hall,
        "skipped_no_gt": skipped_no_gt,
        "skipped_short": skipped_short,
        "skipped_misaligned": skipped_misaligned,
        "auroc_instantaneous": auroc_inst,
        "auroc_mean": auroc_mean,
        "auroc_emas": auroc_emas,
        "auroc_lca_diff": auroc_lca_diff,
        "auroc_lca_xc": auroc_lca_xc,
        "auroc_spread": auroc_spread,
        "leads_vs_raw_median_tokens": leads,
        "max_abs_lead": max_abs_lead,
        "lca_params": {
            "lambda": args.lca_lambda, "beta": args.lca_beta,
            "alpha": args.lca_alpha, "kappa": args.lca_kappa,
            "sigma": args.lca_sigma, "substeps": args.lca_substeps,
        },
        "ema_alphas": args.ema_alphas,
        "verdict_equivalent": bool(auroc_spread < 0.02 and max_abs_lead <= 1.0),
    }
    with open(args.output, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
