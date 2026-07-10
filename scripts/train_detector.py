"""
train_detector.py

Reconstructs the H-Neuron CETT detector trainer (deleted thunlp classifier.py).
Fits an L1 logistic classifier on CETT activation features (mean over answer
tokens, one scalar per FFN neuron, flattened n_layers x intermediate_dim).

y = 1 for hallucinated (f), y = 0 for correct (t). Positive coef = H-Neuron.

Validation mode (--validate_against) sweeps C to match a reference detector's
nonzero sparsity and reports position overlap, before trusting the trainer on
fresh models.

Usage (validate against surviving Mistral detector):
    python scripts/train_detector.py \
        --acts_root data/activations_mistral_8of10/answer_tokens \
        --train_ids data/train_qids_mistral_8of10.json \
        --intermediate_dim 14336 \
        --validate_against models/detector_mistral_8of10.pkl \
        --output models/detector_mistral_REBUILD.pkl

Usage (fresh model, C pinned from validation):
    python scripts/train_detector.py \
        --acts_root data/activations_qwen/answer_tokens \
        --train_ids data/train_qids_qwen.json \
        --intermediate_dim 18944 \
        --C <pinned> \
        --output models/detector_qwen.pkl
"""

import os
import json
import argparse

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts_root", type=str, required=True,
                   help="Dir with act_<id>.npy files, shape (n_layers, inter_dim)")
    p.add_argument("--train_ids", type=str, required=True,
                   help="JSON with 't' and 'f' id lists")
    p.add_argument("--intermediate_dim", type=int, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--C", type=float, default=None,
                   help="L1 inverse reg. If None and --validate_against set, "
                        "sweep to match reference sparsity.")
    p.add_argument("--validate_against", type=str, default=None,
                   help="Reference detector .pkl to match nonzero set against.")
    p.add_argument("--max_iter", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_xy(acts_root, train_ids, inter_dim):
    with open(train_ids) as f:
        ids = json.load(f)
    X, y, kept, missing = [], [], [], 0
    for label, key in [(0, "t"), (1, "f")]:
        for qid in ids[key]:
            path = os.path.join(acts_root, f"act_{qid}.npy")
            if not os.path.exists(path):
                missing += 1
                continue
            a = np.load(path)
            if a.shape[1] != inter_dim:
                raise ValueError(f"{path} has inter_dim {a.shape[1]} != {inter_dim}")
            X.append(a.reshape(-1).astype(np.float64))
            y.append(label)
            kept.append(qid)
    X = np.array(X)
    y = np.array(y)
    n_feat = X.shape[1]
    print(f"Loaded X {X.shape}, y {y.shape}, missing {missing}")
    print(f"  t (correct, y=0): {(y==0).sum()}   f (halluc, y=1): {(y==1).sum()}")
    print(f"  feature space: {n_feat} = {n_feat // inter_dim} layers x {inter_dim}")
    return X, y, n_feat


def positions(coef, inter_dim):
    nz = np.where(coef != 0)[0]
    return set((int(i // inter_dim), int(i % inter_dim)) for i in nz), nz


def fit_one(X, y, C, max_iter, seed):
    clf = LogisticRegression(penalty="l1", solver="saga", C=C,
                             class_weight="balanced", max_iter=max_iter,
                             random_state=seed, tol=1e-4)
    clf.fit(X, y)
    return clf


def main():
    args = parse_args()
    X, y, n_feat = load_xy(args.acts_root, args.train_ids, args.intermediate_dim)

    ref_positions = None
    ref_n = None
    if args.validate_against:
        ref = joblib.load(args.validate_against)
        ref_w = np.asarray(ref.coef_).ravel()
        ref_positions, _ = positions(ref_w, args.intermediate_dim)
        ref_n = len(ref_positions)
        print(f"\nReference detector: {ref_n} nonzero positions")

    if args.C is not None:
        Cs = [args.C]
    else:
        Cs = np.logspace(-3, 1, 25)

    best = None
    for C in Cs:
        clf = fit_one(X, y, C, args.max_iter, args.seed)
        pos, nz = positions(clf.coef_.ravel(), args.intermediate_dim)
        n_nz = len(nz)
        line = f"  C={C:.4g}  nonzero={n_nz}"
        if ref_positions is not None:
            overlap = len(pos & ref_positions)
            pos_layers_ref = set(l for l, _ in ref_positions
                                 if np.sign(ref_w[l*args.intermediate_dim + _]) > 0)
            line += f"  overlap={overlap}/{ref_n}"
            # rank candidate by |nonzero - ref_n| then -overlap
            key = (abs(n_nz - ref_n), -overlap)
            if best is None or key < best[0]:
                best = (key, C, clf, pos, overlap)
        print(line)

    if args.validate_against and args.C is None:
        _, C, clf, pos, overlap = best
        print(f"\nBest match: C={C:.4g}  overlap={overlap}/{ref_n}")
        # sign check on positive (H-Neuron) layers
        w = clf.coef_.ravel()
        cand_pos_layers = sorted(set(l for l, n in pos
                                     if w[l*args.intermediate_dim + n] > 0))
        ref_pos_layers = sorted(set(l for (l, n) in ref_positions
                                    if ref_w[l*args.intermediate_dim + n] > 0))
        print(f"  candidate positive (H) layers: {cand_pos_layers}")
        print(f"  reference positive (H) layers: {ref_pos_layers}")
        gate = overlap >= 15 and set(ref_pos_layers).issubset(cand_pos_layers)
        print(f"  GATE (overlap>=15 AND all ref H-layers present): "
              f"{'PASS' if gate else 'FAIL'}")
    else:
        clf = fit_one(X, y, args.C, args.max_iter, args.seed)
        pos, nz = positions(clf.coef_.ravel(), args.intermediate_dim)
        print(f"\nFitted: {len(nz)} nonzero at C={args.C}")

    # CV AUROC (replication gate 0.65)
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    auroc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"  5-fold CV AUROC: {auroc:.4f}  "
          f"({'PASS' if auroc > 0.65 else 'MARGINAL/FAIL'} vs 0.65)")

    joblib.dump(clf, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
