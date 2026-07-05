"""
check_all_layer_bimodality.py

Diagnostic: test bimodality (BC + GMM) at EVERY layer, not just the AUROC peak.

If competition resolves before the peak, bimodality could appear at an earlier
layer and vanish by the peak (where one accumulator has already won). This
script checks that possibility using the existing layer stack captures.

Recomputes cross-validated projections at each layer (no in-sample circularity).
Reports BC and GMM for every layer with AUROC above the floor.

Usage:
    python scripts/check_all_layer_bimodality.py \
        --index_path data/layer_stack_index_mistral.jsonl \
        --position last_answer_tok
"""

import os
import json
import argparse

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--position", type=str, default="last_answer_tok")
    parser.add_argument("--category", type=str, default="1")
    parser.add_argument("--auroc_floor", type=float, default=0.6,
                        help="Only test bimodality at layers above this AUROC")
    return parser.parse_args()


def load_data(index_path, position, category):
    entries = []
    with open(index_path) as f:
        for line in f:
            e = json.loads(line)
            if category and e.get("category") != category:
                continue
            if e.get("correct") is None:
                continue
            entries.append(e)

    X_list, y_list = [], []
    for e in tqdm(entries, desc="Loading stacks"):
        if not os.path.exists(e["npz_path"]):
            continue
        data = np.load(e["npz_path"], allow_pickle=True)
        hs = data["hidden_stack"].astype(np.float32)
        pos = list(data["position_labels"])
        if position not in pos:
            continue
        X_list.append(hs[pos.index(position)])
        y_list.append(1 if e["correct"] else 0)

    return np.stack(X_list), np.array(y_list)


def main():
    args = parse_args()
    X, y = load_data(args.index_path, args.position, args.category)
    n, n_layers, hdim = X.shape
    print(f"\nSamples: {n}  Correct: {y.sum()}  Incorrect: {(1-y).sum()}")
    print(f"Layers: {n_layers}  Hidden dim: {hdim}")
    print(f"Position: {args.position}  Category: {args.category}")
    print(f"AUROC floor for bimodality test: {args.auroc_floor}\n")

    header = f"{'Layer':>5} {'AUROC':>6} {'BC':>6} {'BC>0.555':>8} {'GMM_2':>5} {'BIC_diff':>8} {'BOTH':>5}"
    print(header)
    print("-" * len(header))

    any_both = False
    results = []

    for layer in range(n_layers):
        Xl = X[:, layer, :]
        mu = Xl.mean(axis=0)
        sd = Xl.std(axis=0) + 1e-8
        Xl_n = (Xl - mu) / sd

        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
        try:
            cv_probs = cross_val_predict(
                clf, Xl_n, y, cv=5, method="predict_proba"
            )[:, 1]
            auroc = roc_auc_score(y, cv_probs)
        except Exception:
            auroc = 0.5

        if auroc < args.auroc_floor:
            print(f"{layer:5d} {auroc:6.3f}      -        -     -        -     -")
            results.append({"layer": layer, "auroc": auroc, "tested": False})
            continue

        # Cross-validated decision function (out-of-sample projections)
        try:
            scores = cross_val_predict(
                clf, Xl_n, y, cv=5, method="decision_function"
            )
        except Exception:
            print(f"{layer:5d} {auroc:6.3f}   FAIL")
            results.append({"layer": layer, "auroc": auroc, "tested": False})
            continue

        # Bimodality coefficient
        skew = stats.skew(scores)
        kurt = stats.kurtosis(scores, fisher=False)  # excess=False
        bc = (skew**2 + 1) / kurt if kurt > 0 else 0.0
        bc_yes = bc > 0.555

        # GMM 1 vs 2 components
        Xs = scores.reshape(-1, 1)
        try:
            g1 = GaussianMixture(n_components=1, random_state=42).fit(Xs)
            g2 = GaussianMixture(n_components=2, random_state=42).fit(Xs)
            bic_d = float(g1.bic(Xs) - g2.bic(Xs))
            gmm_yes = bic_d > 0
        except Exception:
            bic_d = 0.0
            gmm_yes = False

        both = bc_yes and gmm_yes
        if both:
            any_both = True

        flag = " <<<" if both else ""
        print(
            f"{layer:5d} {auroc:6.3f} {bc:6.3f} "
            f"{'Y' if bc_yes else 'N':>8} "
            f"{'Y' if gmm_yes else 'N':>5} "
            f"{bic_d:8.1f} "
            f"{'YES' if both else 'no':>5}{flag}"
        )
        results.append({
            "layer": layer, "auroc": auroc, "tested": True,
            "bc": bc, "bc_bimodal": bc_yes,
            "gmm_bimodal": gmm_yes, "bic_diff": bic_d,
            "both": both,
        })

    print()
    if any_both:
        bimodal_layers = [r["layer"] for r in results if r.get("both")]
        print(f"RESULT: Bimodality (BC + GMM agree) at layer(s): {bimodal_layers}")
        print("        Competition may resolve before the AUROC peak.")
        print("        Investigate these layers before closing the negative.")
    else:
        print(f"RESULT: Unimodal at ALL layers with AUROC > {args.auroc_floor:.2f}.")
        print("        No competition signature anywhere in the depth profile.")
        print("        Step 1 clean. Proceed to instruct-model check (Step 2).")


if __name__ == "__main__":
    main()
