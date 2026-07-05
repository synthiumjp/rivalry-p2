#!/usr/bin/env python3
"""
Step 0 Spike - Gate 1: Step-0 prompt encoding predicts hallucination rate.

Loads existing Mistral data only. No new generation needed.
PASS iff cross-validated r > 0.20 at best layer.
"""
import json, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# --- Load ground truth ---
benchmark = {}
with open("data/benchmark_final_250.jsonl") as f:
    for line in f:
        e = json.loads(line)
        if e["category"] == "1":
            benchmark[e["prompt_id"]] = e["ground_truth"]

print(f"Cat1 prompts with ground truth: {len(benchmark)}")

# --- Compute per-prompt hallucination rate ---
def is_correct(text, aliases):
    t = text.lower().strip()
    for a in aliases:
        a = a.lower().strip()
        if len(a) < 2: continue
        if a in t or t in a: return True
    return False

h_rates = {}
with open("data/hneuron_activations_mistral.jsonl") as f:
    for line in f:
        e = json.loads(line)
        pid = e["prompt_id"]
        if pid not in benchmark: continue
        gt = benchmark[pid]
        n_correct = sum(is_correct(c["text"], gt) for c in e["completions"])
        h_rates[pid] = 1.0 - n_correct / len(e["completions"])

print(f"Hallucination rates computed: {len(h_rates)}")
print(f"  Mean h_rate: {np.mean(list(h_rates.values())):.3f}")
print(f"  Prompts with h_rate=0: {sum(1 for v in h_rates.values() if v==0)}")
print(f"  Prompts with h_rate=1: {sum(1 for v in h_rates.values() if v==1)}")

# --- Load prompt_final residuals from layer stacks ---
prompt_ids = sorted(h_rates.keys())
residuals = []  # shape will be (n_prompts, 33, 4096)
rates = []
missing = 0

for pid in prompt_ids:
    npz_path = Path(f"data/layer_stacks_mistral/{pid}_0.npz")
    if not npz_path.exists():
        missing += 1
        continue
    npz = np.load(npz_path)
    # position 0 = prompt_final, shape (33, 4096)
    residuals.append(npz["hidden_stack"][0])
    rates.append(h_rates[pid])

residuals = np.array(residuals)  # (n, 33, 4096)
rates = np.array(rates)
print(f"\nLoaded {len(rates)} prompts (missing {missing} NPZs)")
print(f"Residual shape: {residuals.shape}")

# --- Per-layer cross-validated regression ---
n_prompts, n_layers, hidden_dim = residuals.shape

print(f"\n{'Layer':>5} {'cv-r':>7} {'p':>10}")
print("-" * 25)

results = []
for layer in range(n_layers):
    X = residuals[:, layer, :]  # (n, 4096)
    y = rates

    # Standardise features
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Ridge regression with built-in CV for alpha
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])

    try:
        y_pred = cross_val_predict(ridge, X_s, y, cv=5)
        r, p = pearsonr(y, y_pred)
    except Exception as e:
        r, p = 0.0, 1.0

    results.append({"layer": layer, "cv_r": round(r, 4), "p": p})
    flag = " <-- PEAK" if r == max(res["cv_r"] for res in results) and r > 0.2 else ""
    print(f"{layer:5d} {r:7.4f} {p:10.4g}{flag}")

# --- Summary ---
best = max(results, key=lambda x: x["cv_r"])
print(f"\n{'='*40}")
print(f"BEST LAYER: {best['layer']}")
print(f"BEST cv-r:  {best['cv_r']:.4f}")
print(f"p-value:    {best['p']:.4g}")

if best["cv_r"] > 0.20:
    print(f"\n*** GATE 1: PASS (cv-r {best['cv_r']:.3f} > 0.20) ***")
else:
    print(f"\n*** GATE 1: FAIL (cv-r {best['cv_r']:.3f} <= 0.20) ***")

# Save
output = {"gate": 1, "model": "Mistral-7B-v0.3-base", "best_layer": best["layer"],
          "best_cv_r": best["cv_r"], "best_p": best["p"],
          "pass": best["cv_r"] > 0.20, "per_layer": results}
with open("data/spike_gate1_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved data/spike_gate1_results.json")
