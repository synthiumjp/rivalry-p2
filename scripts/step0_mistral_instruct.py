import json, numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Read instruct index: prompt_id, correct, npz_path, category
rows = []
for line in open("data/layer_stack_index_mistral_instruct.jsonl"):
    e = json.loads(line)
    if e.get("category") in ("1", 1) and e.get("greedy", True):
        rows.append((e["prompt_id"], int(e["correct"]), e["npz_path"]))

print(f"Cat1 instruct prompts: {len(rows)}")
y = np.array([1 - c for _, c, _ in rows])  # 1 = hallucinated (incorrect)
print(f"Hallucination rate: {y.mean():.3f}")

if y.mean() in (0.0, 1.0):
    print("Single class, cannot fit."); exit()

# Load prompt_final residuals
import os
res = []
keep = []
for i, (pid, c, npz_path) in enumerate(rows):
    p = npz_path if os.path.exists(npz_path) else f"data/layer_stacks_mistral_instruct/{pid}_0.npz"
    if os.path.exists(p):
        res.append(np.load(p)["hidden_stack"][0])
        keep.append(i)
res = np.array(res)
y = y[keep]
print(f"Loaded {len(y)} stacks, shape {res.shape}")

clf = LogisticRegressionCV(Cs=10, max_iter=2000, class_weight="balanced", cv=5)
best_a, best_l = 0, -1
for L in range(res.shape[1]):
    X = StandardScaler().fit_transform(res[:,L,:])
    yp = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:,1]
    a = roc_auc_score(y, yp)
    if L < 6 or L in (15,29,32): print(f"  L{L}: AUROC={a:.3f}")
    if a > best_a: best_a, best_l = a, L
print(f"\nMISTRAL INSTRUCT step-0 peak: AUROC={best_a:.3f} at layer {best_l}")
print("Note: AUROC not r (binary greedy correctness, n per prompt =1)")
print("Mistral BASE: r=0.448 L1 | Llama BASE: r=0.149 L26")
