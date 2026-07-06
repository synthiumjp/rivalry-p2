#!/usr/bin/env python3
"""
stepc_analyse.py

STEP C decisive number. For each collected model, per layer:
  cross_val_predict (5-fold, seed 42) of r_p ~ prompt-final residual, using a
  StandardScaler + RidgeCV pipeline (fit on training folds only), then Pearson r
  between out-of-sample predictions and r_p. Peak layer L* = argmax layer r.

This is the DEVELOPMENT cv-r. It is the strategic read, not the confirmatory
number. The confirmatory H1 selects L* on development and reports on the sealed
hold-out; this does not touch the hold-out.

Interpretation caveat printed at the end: the peak-across-layers cv-r is mildly
optimistic because L* is chosen on the same development set the r is read from.
Treat the peak as an upper-ish bound on the hold-out number; discount slightly
for the go/no-go decision.

Run:
  python scripts/stepc_analyse.py --tags qwen_instruct gemma_instruct
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

DATA = Path("data")
ALPHAS = np.logspace(-1, 4, 12)

def load_model(tag):
    jl = DATA / f"stepc_{tag}.jsonl"
    rd = DATA / f"stepc_resid_{tag}"
    rp = {}
    with open(jl) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rp[r["prompt_id"]] = r["r_p"]
    ids = sorted(pid for pid in rp if (rd / f"{pid}.npy").exists())
    y = np.array([rp[pid] for pid in ids], dtype=np.float64)
    X = np.stack([np.load(rd / f"{pid}.npy") for pid in ids]).astype(np.float32)  # [N, L, H]
    return ids, y, X

def per_layer_cvr(y, X):
    N, L, H = X.shape
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rs = np.zeros(L)
    for l in range(L):
        Xl = X[:, l, :]
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        pred = cross_val_predict(pipe, Xl, y, cv=cv)
        rs[l] = pearsonr(y, pred)[0]
    return rs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    summary = []
    for tag in args.tags:
        ids, y, X = load_model(tag)
        print(f"\n=== {tag} ===")
        print(f"  n={len(ids)}  layers={X.shape[1]}  hidden={X.shape[2]}")
        print(f"  r_p: mean={y.mean():.3f} sd={y.std():.3f} "
              f"min={y.min():.2f} max={y.max():.2f} "
              f"(frac at 0: {(y==0).mean():.2f}, frac at 1: {(y==1).mean():.2f})")
        if y.std() < 1e-6:
            print("  r_p has no variance; H1 undefined on this model. Check collection.")
            continue
        rs = per_layer_cvr(y, X)
        lstar = int(np.argmax(rs))
        print(f"  per-layer cv-r peak: L*={lstar}  cv-r={rs[lstar]:.3f}")
        # neighbourhood, so you can see if the peak is a spike or a plateau
        lo, hi = max(0, lstar - 3), min(len(rs), lstar + 4)
        print("  around L*: " + "  ".join(f"L{l}:{rs[l]:.2f}" for l in range(lo, hi)))
        print(f"  layers with cv-r>0.5: {int((rs>0.5).sum())}/{len(rs)}   "
              f">0.3: {int((rs>0.3).sum())}/{len(rs)}")
        summary.append((tag, lstar, rs[lstar]))

    print("\n=== STEP C decisive read (development cv-r at peak L*) ===")
    for tag, lstar, r in summary:
        verdict = ">0.5" if r > 0.5 else (">0.3" if r > 0.3 else "<0.3")
        print(f"  {tag:16s} L*={lstar:3d}  cv-r={r:.3f}  [{verdict}]")
    print("\n  CAVEAT: peak selected on the same development set the r is read")
    print("  from, so this is mildly optimistic. The hold-out number will be")
    print("  lower. H1 MEI is cv-r>0.5 in >=3/5 models and >0.3 in all 5,")
    print("  reported on the hold-out. This is the development preview.")

if __name__ == "__main__":
    main()
