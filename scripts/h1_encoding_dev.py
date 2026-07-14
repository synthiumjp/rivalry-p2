#!/usr/bin/env python3
"""
h1_encoding_dev.py
H1 (encoding-time prediction): a linear readout of the last-prompt-token
residual, taken before any token is generated, predicts per-prompt
hallucination rate r_p. Reg index 1 / analysis H1: per-layer 5-fold
cross_val_predict RidgeCV Pearson r; peak layer L* SELECTED ON DEVELOPMENT,
reported on the hold-out. Confirmed (HOLD-OUT): cv-r > 0.5 in >=3/5 models
and > 0.3 in all 5.

This is the DEVELOPMENT freeze artifact only. It records the full per-layer
cv-r profile per model and the development-selected peak layer, which is the
layer that will be read on the sealed hold-out. Hold-out stays sealed.

Inputs (all already used by h4_stage_coupling_dev.py, no new capture):
  data/rp_clean_<tag>.jsonl        per-prompt r_p
  data/stepc_resid_<tag>/<pid>.npy (n_layers, hidden) prompt-final residual
Output:
  data/h1_encoding_dev.json
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
ALPHAS = np.logspace(-1, 4, 12)


def load_rp(tag):
    rp = {}
    for line in open(DATA / f"rp_clean_{tag}.jsonl"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rp[r["prompt_id"]] = r["r_p"]
    return rp


def sweep(tag):
    rp = load_rp(tag)
    rd = DATA / f"stepc_resid_{tag}"
    have = [p for p in rp if (rd / f"{p}.npy").exists()]
    have.sort()
    # stack: (n_prompts, n_layers, hidden)
    stack = np.stack([np.load(rd / f"{p}.npy") for p in have]).astype(np.float32)
    y = np.array([rp[p] for p in have])
    n_layers = stack.shape[1]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    profile = []
    for L in range(n_layers):
        X = stack[:, L, :]
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        pred = cross_val_predict(pipe, X, y, cv=cv)
        r, p = pearsonr(pred, y)
        profile.append({"layer": int(L), "cv_r": float(r), "p": float(p)})
    return profile, len(have), n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    models = []
    for tag in args.tags:
        profile, n, n_layers = sweep(tag)
        peak = max(profile, key=lambda d: d["cv_r"])
        # plateau onset: earliest layer within 0.05 cv-r of the peak.
        # reg-silent choice, documented in the deviation doc. reported
        # alongside the reg-literal argmax peak; onset is the encoding-locus
        # evidence, peak is the hold-out read layer.
        onset = next(d for d in profile if d["cv_r"] >= peak["cv_r"] - 0.05)
        clears_dev_05 = peak["cv_r"] > 0.5
        clears_dev_03 = peak["cv_r"] > 0.3
        models.append({
            "tag": tag,
            "n": n,
            "n_layers": n_layers,
            "peak_layer": peak["layer"],
            "peak_cv_r": peak["cv_r"],
            "peak_p": peak["p"],
            "plateau_onset_layer": int(onset["layer"]),
            "plateau_onset_cv_r": float(onset["cv_r"]),
            "plateau_tol": 0.05,
            "dev_gt_0p5": clears_dev_05,
            "dev_gt_0p3": clears_dev_03,
            "profile": profile,
        })
        print(f"{tag:16s} n={n:4d}  peak L={peak['layer']:2d}  "
              f"cv-r={peak['cv_r']:+.3f}  p={peak['p']:.2e}  "
              f"{'>0.5' if clears_dev_05 else ('>0.3' if clears_dev_03 else '<0.3')}")
    n5 = sum(m["dev_gt_0p5"] for m in models)
    n3 = sum(m["dev_gt_0p3"] for m in models)
    print(f"\nH1 dev preview: peak cv-r >0.5 in {n5}/{len(models)}, "
          f">0.3 in {n3}/{len(models)}. "
          f"Confirmatory rule is HOLD-OUT (>0.5 in >=3/5 and >0.3 in all 5). "
          f"Development only; peak layer per model is the locked read layer.")
    (DATA / "h1_encoding_dev.json").write_text(json.dumps(
        {"models": models,
         "dev_peak_gt_0p5": int(n5),
         "dev_peak_gt_0p3": int(n3),
         "n_models": len(models),
         "note": "Development per-layer cv-r sweep. peak_layer selected on "
                 "development is the layer to be read on the sealed hold-out. "
                 "Not a confirmatory result."},
        indent=2))


if __name__ == "__main__":
    main()
