import json, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

benchmark = {}
with open("data/benchmark_final_250.jsonl") as f:
    for line in f:
        e = json.loads(line)
        if e["category"] == "1":
            benchmark[e["prompt_id"]] = e["ground_truth"]

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
        nc = sum(is_correct(c["text"], gt) for c in e["completions"])
        h_rates[pid] = 1.0 - nc / len(e["completions"])

pids = sorted(h_rates.keys())
residuals = np.array([np.load(f"data/layer_stacks_mistral/{p}_0.npz")["hidden_stack"][0]
                       for p in pids])
rates = np.array([h_rates[p] for p in pids])

X = StandardScaler().fit_transform(residuals[:, 1, :])
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])

rs = []
for seed in [42, 123, 456]:
    yp = cross_val_predict(ridge, X, rates, cv=KFold(5, shuffle=True, random_state=seed))
    r, _ = pearsonr(rates, yp)
    rs.append(r)
    print(f"Seed {seed}: cv-r = {r:.4f}")

sd = np.std(rs)
print(f"\nMean: {np.mean(rs):.4f}, SD: {sd:.4f}")
if sd < 0.04:
    print("*** GATE 6: PASS (sd < 0.04, 3-seed floor sufficient) ***")
elif sd < 0.06:
    print("GATE 6: MARGINAL (escalate to 5 seeds)")
else:
    print("GATE 6: HIGH VARIANCE (escalate to 7 seeds)")

json.dump({"gate": 6, "seeds": [42, 123, 456], "cv_rs": rs,
           "mean": float(np.mean(rs)), "sd": float(sd),
           "passed": sd < 0.04},
          open("data/spike_gate6_results.json", "w"), indent=2)
print("Saved data/spike_gate6_results.json")
