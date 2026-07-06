#!/usr/bin/env python3
"""
verify_stepc_clean.py

Does the STEP C L* lock survive clean-span scoring? The H1 predictor
(prompt-final residual) was never degenerated, so only the r_p LABEL could have
shifted. This re-runs the per-layer cv-r with the clean r_p and the EXISTING
stepc prompt-final residuals, and compares L* and cv-r to the lock.

Verdict: if L* is unchanged (or within the plateau) and cv-r is within noise,
the lock stands and gets annotated verified-clean. If it moves materially, the
lock is re-cut on clean r_p before the hold-out.

Run:
  python scripts/verify_stepc_clean.py --tags qwen_instruct gemma_instruct mistral_instruct llama_instruct
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

DATA = Path("data")
LOCK = json.loads((DATA / "lstar_lock.json").read_text())
ALPHAS = np.logspace(-1, 4, 12)

def per_layer_cvr(y, X):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rs = []
    for l in range(X.shape[1]):
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        rs.append(float(pearsonr(y, cross_val_predict(pipe, X[:, l, :], y, cv=cv))[0]))
    return np.array(rs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    for tag in args.tags:
        rp = {}
        for line in open(DATA / f"rp_clean_{tag}.jsonl"):
            r = json.loads(line); rp[r["prompt_id"]] = r["r_p"]
        rd = DATA / f"stepc_resid_{tag}"
        ids = sorted(p for p in rp if (rd / f"{p}.npy").exists())
        y = np.array([rp[p] for p in ids])
        X = np.stack([np.load(rd / f"{p}.npy") for p in ids]).astype(np.float32)

        rs = per_layer_cvr(y, X)
        lstar_clean = int(np.argmax(rs))
        cvr_clean = float(rs[lstar_clean])

        locked = LOCK["models"][tag]
        lstar_lock, cvr_lock = locked["lstar"], locked["dev_cv_r_at_lstar"]
        in_plateau = lstar_clean in locked["plateau_layers"]

        print(f"\n=== {tag} ===")
        print(f"  locked:  L*={lstar_lock}  cv-r={cvr_lock:.3f}")
        print(f"  clean :  L*={lstar_clean}  cv-r={cvr_clean:.3f}  "
              f"cv-r at locked L*={rs[lstar_lock]:.3f}")
        print(f"  r_p clean: mean={y.mean():.3f} sd={y.std():.3f}  n={len(ids)}")
        dr = abs(cvr_clean - cvr_lock)
        if in_plateau and dr < 0.03:
            print("  VERDICT: lock STANDS (clean L* in locked plateau, cv-r within noise)")
        elif dr < 0.05:
            print("  VERDICT: lock likely stands (cv-r within 0.05); inspect plateau")
        else:
            print("  VERDICT: RE-CUT the lock on clean r_p before hold-out")

if __name__ == "__main__":
    main()
