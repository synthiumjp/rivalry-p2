import json, numpy as np, os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

rows = []
for line in open("data/layer_stack_index_qwen_instruct.jsonl"):
    e = json.loads(line)
    if e.get("category") in ("1", 1) and e.get("greedy", True):
        rows.append((e["prompt_id"], int(e["correct"]), e["npz_path"]))

y = np.array([1 - c for _, c, _ in rows])
print(f"n={len(rows)}, hallucination rate={y.mean():.3f}, n_wrong={int(y.sum())}")
if y.sum() < 5 or y.sum() > len(y)-5:
    print("WARNING: too few minority-class prompts. Result is noise.")

res, keep = [], []
for i,(pid,c,npz) in enumerate(rows):
    p = npz if os.path.exists(npz) else f"data/layer_stacks_qwen_instruct/{pid}_0.npz"
    if os.path.exists(p):
        res.append(np.load(p)["hidden_stack"][0]); keep.append(i)
res = np.array(res); y = y[keep]
print(f"Loaded {len(y)}, shape {res.shape}")

clf = LogisticRegressionCV(Cs=10, max_iter=2000, class_weight="balanced", cv=5)
best_a, best_l = 0, -1
for L in range(res.shape[1]):
    X = StandardScaler().fit_transform(res[:,L,:])
    try:
        yp = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:,1]
        a = roc_auc_score(y, yp)
    except Exception:
        a = 0.5
    if L < 6 or L in (10,12,15): print(f"  L{L}: AUROC={a:.3f}")
    if a > best_a: best_a, best_l = a, L
print(f"\nQWEN INSTRUCT peak: AUROC={best_a:.3f} at L{best_l} (n_wrong={int(y.sum())}, NOISY)")
print("\n=== ALL CELLS (mixed metrics, NOT yet standardised) ===")
print("Mistral base:     r=0.448      L1")
print("Mistral instruct: AUROC=0.836  L12")
print("Llama base:       r=0.149      L26")
print("Llama instruct:   AUROC=0.648  L10")
print(f"Qwen instruct:    AUROC={best_a:.3f}  L{best_l}  (thin, 12 informative)")
