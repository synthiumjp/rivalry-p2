#!/usr/bin/env python3
"""
h4_stage_coupling_dev.py
H4 (load-bearing novelty test): stage coupling between the Stage-1 encoding
signal and Stage-2 commitment depth. Reg 3.4: per prompt, Spearman(pi(r_p),
L*(p)); pi(r_p) = out-of-sample step-0 prediction at the locked L*_step0;
L*(p) = commitment depth. Decision (HOLD-OUT): rho>0.3, p<0.005, >=3/5 models.
This is the DEVELOPMENT preview only. Hold-out stays sealed.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

DATA = Path("data")
LOCK = json.loads((DATA / "lstar_lock.json").read_text())
ALPHAS = np.logspace(-1, 4, 12)

def load_rp(tag):
    rp = {}
    for line in open(DATA / f"rp_clean_{tag}.jsonl"):
        line = line.strip()
        if line:
            r = json.loads(line); rp[r["prompt_id"]] = r["r_p"]
    return rp

def load_commit_Lstar(tag):
    fp = DATA / f"commitment_rows_{tag}.jsonl"
    if not fp.exists():
        raise SystemExit(f"MISSING {fp}. Re-run commitment_confirmatory.py first.")
    out = {}
    for line in open(fp):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get("L_star") is not None:
            out[r["prompt_id"]] = float(r["L_star"])
    return out

def step0_pred(tag, ids, rp):
    lstar = LOCK["models"][tag]["lstar"]
    rd = DATA / f"stepc_resid_{tag}"
    have = [p for p in ids if (rd / f"{p}.npy").exists() and p in rp]
    X = np.stack([np.load(rd / f"{p}.npy")[lstar] for p in have]).astype(np.float32)
    y = np.array([rp[p] for p in have])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    pred = cross_val_predict(pipe, X, y, cv=cv)
    return dict(zip(have, pred)), lstar

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    summary = []
    for tag in args.tags:
        rp = load_rp(tag)
        Lstar = load_commit_Lstar(tag)
        common = sorted(set(Lstar) & set(rp))
        pi, l0 = step0_pred(tag, common, rp)
        common = [p for p in common if p in pi]
        x = np.array([pi[p] for p in common])
        L = np.array([Lstar[p] for p in common])
        rho, p = spearmanr(x, L)
        clears = (rho > 0.3) and (p < 0.005)
        summary.append((tag, rho, p, len(common), l0, clears))
        print(f"{tag:16s} n={len(common):4d}  L*_step0={l0:3d}  "
              f"Spearman rho={rho:+.3f}  p={p:.2e}  {'CLEARS' if clears else '-'}")
    n_clear = sum(s[5] for s in summary)
    print(f"\nH4 dev preview: {n_clear}/{len(summary)} clear (rho>0.3, p<0.005). "
          f"MEI needs >=3/5 on the HOLD-OUT. Development only.")
    Path(DATA / "h4_stage_coupling_dev.json").write_text(json.dumps(
        {"models": [{"tag": t, "rho": float(r), "p": float(pp), "n": int(n),
                     "L_step0": int(l0), "clears": bool(c)}
                    for t, r, pp, n, l0, c in summary],
         "n_clear": int(n_clear), "n_models": len(summary)}, indent=2))

if __name__ == "__main__":
    main()
