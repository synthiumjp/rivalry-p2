#!/usr/bin/env python3
"""
lock_lstar.py

Pre-commit the per-model development-selected peak layer L* BEFORE the hold-out
is opened. L* selection (peak 5-fold cross-validated Pearson r on the 800
development set) is the registered procedure; the values are already known from
STEP C. This file converts those known development quantities into pre-committed
ones and records the exact procedure and inputs so the selection is verifiable
and tamper-evident.

Deterministic: re-running reproduces identical L* (KFold shuffle seed 42,
RidgeCV, StandardScaler fit on train folds only).

Run:
  python scripts/lock_lstar.py --tags mistral_instruct qwen_instruct gemma_instruct llama_instruct
Then commit data/lstar_lock.json.

Add Phi-3 later (before hold-out) by re-running with its tag appended; the file
is model-keyed and each entry carries its own timestamp.
"""
import argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

DATA = Path("data")
LOCK = DATA / "lstar_lock.json"
ALPHAS = np.logspace(-1, 4, 12)
N_FOLDS, SEED, PLATEAU_TOL = 5, 42, 0.02

RULE = ("L* = argmax over layers of the 5-fold cross-validated Pearson r "
        "between the prompt-final residual and per-prompt hallucination rate "
        "r_p, computed with cross_val_predict on the 800 Cat1 development set. "
        "StandardScaler + RidgeCV(alphas=logspace(-1,4,12)); KFold(5, "
        "shuffle=True, random_state=42). Selection on development only; the "
        "200-prompt hold-out is not opened.")

def load(tag):
    jl = DATA / f"stepc_{tag}.jsonl"
    rd = DATA / f"stepc_resid_{tag}"
    rp = {}
    for line in open(jl):
        line = line.strip()
        if line:
            r = json.loads(line); rp[r["prompt_id"]] = r["r_p"]
    ids = sorted(p for p in rp if (rd / f"{p}.npy").exists())
    y = np.array([rp[p] for p in ids], float)
    X = np.stack([np.load(rd / f"{p}.npy") for p in ids]).astype(np.float32)
    return ids, y, X, jl, rd

def per_layer_cvr(y, X):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rs = []
    for l in range(X.shape[1]):
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        pred = cross_val_predict(pipe, X[:, l, :], y, cv=cv)
        rs.append(float(pearsonr(y, pred)[0]))
    return rs

def sha256_file(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()

def resid_manifest_hash(ids, rd):
    h = hashlib.sha256()
    for pid in ids:
        f = rd / f"{pid}.npy"
        h.update(f"{pid}:{f.stat().st_size}\n".encode())
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    lock = {}
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
    lock.setdefault("rule", RULE)
    lock.setdefault("hold_out_opened", False)
    lock.setdefault("models", {})

    for tag in args.tags:
        ids, y, X, jl, rd = load(tag)
        rs = per_layer_cvr(y, X)
        lstar = int(np.argmax(rs))
        peak = rs[lstar]
        plateau = [l for l, r in enumerate(rs) if r >= peak - PLATEAU_TOL]
        entry = {
            "lstar": lstar,
            "dev_cv_r_at_lstar": round(peak, 4),
            "n_layers": X.shape[1],
            "hidden": X.shape[2],
            "n_prompts": len(ids),
            "plateau_layers": plateau,
            "plateau_span": [min(plateau), max(plateau)],
            "per_layer_cv_r": [round(r, 4) for r in rs],
            "r_p_mean": round(float(y.mean()), 4),
            "r_p_sd": round(float(y.std()), 4),
            "rp_file_sha256": sha256_file(jl),
            "resid_manifest_sha256": resid_manifest_hash(ids, rd),
            "locked_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        lock["models"][tag] = entry
        print(f"{tag:16s} L*={lstar:3d}  dev cv-r={peak:.3f}  "
              f"plateau L{min(plateau)}-L{max(plateau)} ({len(plateau)} layers)")

    lock["procedure"] = {"alphas": "logspace(-1,4,12)", "n_folds": N_FOLDS,
                         "seed": SEED, "scaler": "StandardScaler(train-fold only)",
                         "plateau_tol": PLATEAU_TOL}
    LOCK.write_text(json.dumps(lock, indent=2))
    print(f"\nwrote {LOCK}")
    print("hold_out_opened:", lock["hold_out_opened"], "-- keep False until analysis frozen")

if __name__ == "__main__":
    main()
