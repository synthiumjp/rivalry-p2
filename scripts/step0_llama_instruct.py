import json, numpy as np, os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

rows = []
for line in open("data/layer_stack_index_llama_instruct.jsonl"):
    e = json.loads(line)
    if e.get("category") in ("1", 1) and e.get("greedy", True):
        rows.append((e["prompt_id"], int(e["correct"]), e["npz_path"]))

print(f"Cat1 prompts: {len(rows)}")
y = np.array([1 - c for _, c, _ in rows])
print(f"Hallucination rate: {y.mean():.3f}")
if y.mean() in (0.0, 1.0):
    print("Single class."); exit()

res, keep = [], []
for i, (pid, c, npz_path) in enumerate(rows):
    p = npz_path if os.path.exists(npz_path) else f"data/layer_stacks_llama_instruct/{pid}_0.npz"
    if os.path.exists(p):
        res.append(np.load(p)["hidden_stack"][0]); keep.append(i)
res = np.array(res); y = y[keep]
print(f"Loaded {len(y)} stacks {res.shape}")

clf = LogisticRegressionCV(Cs=10, max_iter=2000, class_weight="balanced", cv=5)
profile, best_a, best_l = [], 0, -1
for L in range(res.shape[1]):
    X = StandardScaler().fit_transform(res[:,L,:])
    yp = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:,1]
    a = roc_auc_score(y, yp)
    profile.append(round(float(a),4))
    if L < 6 or L in (12,15,29,32): print(f"  L{L}: AUROC={a:.3f}")
    if a > best_a: best_a, best_l = a, L
print(f"\nLLAMA INSTRUCT peak: AUROC={best_a:.3f} at L{best_l}")
print("\n=== ALL FOUR CELLS ===")
print("Mistral base:     r=0.448      L1")
print("Mistral instruct: AUROC=0.836  L12")
print("Llama base:       r=0.149      L26")
print(f"Llama instruct:   AUROC={best_a:.3f}  L{best_l}")
json.dump({"model":"Llama-3.1-8B-Instruct","peak_auroc":best_a,"peak_layer":best_l,"profile":profile},
          open("data/step0_llama_instruct.json","w"), indent=2)
