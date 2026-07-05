#!/usr/bin/env python3
"""
Spike Gate 2: Does the H-Neuron probe replicate on Llama?

Two tests:
  A) Transfer: use Mistral H-Neuron POSITIONS, train classifier on Llama
     activations. Tests whether the same neuron positions are informative.
  B) Llama-specific: same positions (we only extracted these), but the
     classifier weights are fit fresh on Llama.

PASS iff cross-validated AUROC > 0.65.
"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Ground truth ---
bm = {}
with open("data/benchmark_final_250.jsonl") as f:
    for line in f:
        e = json.loads(line)
        if "ground_truth" in e: bm[e["prompt_id"]] = e["ground_truth"]

def is_correct(text, aliases):
    t = text.lower().strip()
    for a in aliases:
        a = a.lower().strip()
        if len(a) < 2: continue
        if a in t or t in a: return True
    return False

# --- Build feature matrix ---
# Each completion -> feature vector (mean activation per tracked neuron)
X, y = [], []
n_neurons = None

with open("data/hneuron_activations_llama.jsonl") as f:
    for line in f:
        e = json.loads(line)
        pid = e["prompt_id"]
        if pid not in bm: continue
        if e["category"] != "1": continue  # Cat1 only for spike
        gt = bm[pid]
        for c in e["completions"]:
            acts = np.array(c["activations"])  # shape (n_tokens, n_neurons)
            if acts.ndim != 2 or acts.shape[0] == 0:
                continue
            if n_neurons is None:
                n_neurons = acts.shape[1]
            # Feature: mean and max activation per neuron across tokens
            feat = np.concatenate([acts.mean(axis=0), acts.max(axis=0)])
            X.append(feat)
            y.append(0 if is_correct(c["text"], gt) else 1)  # 1 = hallucinated

X = np.array(X)
y = np.array(y)

print(f"Feature matrix: {X.shape}")
print(f"Tracked neurons: {n_neurons} (features = {X.shape[1]}: mean+max)")
print(f"Completions: {len(y)}")
print(f"Hallucination rate: {y.mean():.3f}")
print(f"  correct (y=0): {(y==0).sum()}")
print(f"  hallucinated (y=1): {(y==1).sum()}")

if y.mean() == 0 or y.mean() == 1:
    print("ERROR: only one class. Cannot train.")
    exit(1)

# --- Cross-validated AUROC ---
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
aurocs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

print(f"\n5-fold CV AUROC: {aurocs.mean():.4f} (+/- {aurocs.std():.4f})")
print(f"Per-fold: {[round(a,3) for a in aurocs]}")

passed = aurocs.mean() > 0.65
if passed:
    print(f"\n*** GATE 2: PASS (AUROC {aurocs.mean():.3f} > 0.65) ***")
    print("H-Neuron positions transfer from Mistral to Llama.")
else:
    print(f"\n*** GATE 2: AUROC {aurocs.mean():.3f} <= 0.65 ***")
    print("Mistral positions do not transfer well. Llama may need own H-Neurons.")

json.dump({"gate": 2, "model": "Llama-3.1-8B-base",
           "classifier_source": "Mistral H-Neuron positions, Llama-fit weights",
           "cv_auroc_mean": float(aurocs.mean()),
           "cv_auroc_std": float(aurocs.std()),
           "n_completions": int(len(y)),
           "hallucination_rate": float(y.mean()),
           "passed": bool(passed)},
          open("data/spike_gate2_results.json", "w"), indent=2)
print("Saved data/spike_gate2_results.json")
