import json, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

bm = {}
for line in open("data/benchmark_final_250.jsonl"):
    e = json.loads(line)
    if e["category"]=="1" and "ground_truth" in e:
        bm[e["question"].split("Respond with")[0].strip()] = (e["prompt_id"], e["ground_truth"])

def ok(t, al):
    t=t.lower().strip()
    return any(a.lower() in t or t in a.lower() for a in al if len(a)>1)

# hallucination rate per Cat1 prompt from Llama responses
hr = {}
for line in open("data/responses_llama_full.jsonl"):
    e = json.loads(line)
    for qid, d in e.items():
        core = d["question"].split("Respond with")[0].strip()
        if core not in bm: continue
        pid, gt = bm[core]
        nc = sum(ok(r, gt) for r in d["responses"])
        hr[pid] = 1.0 - nc/len(d["responses"])

pids = sorted(hr)
res, rate = [], []
for p in pids:
    try:
        res.append(np.load(f"data/layer_stacks_llama/{p}_0.npz")["hidden_stack"][0])
        rate.append(hr[p])
    except: pass
res, rate = np.array(res), np.array(rate)
print(f"n={len(rate)}, mean h_rate={rate.mean():.3f}")

ridge = RidgeCV(alphas=[0.1,1,10,100,1000])
best_r, best_l = 0, -1
for L in range(res.shape[1]):
    X = StandardScaler().fit_transform(res[:,L,:])
    yp = cross_val_predict(ridge, X, rate, cv=5)
    r,p = pearsonr(rate, yp)
    if L < 6 or L in (15,29,32): print(f"  L{L}: r={r:.3f} p={p:.2g}")
    if r > best_r: best_r, best_l = r, L
print(f"\nLLAMA BASE peak: r={best_r:.3f} at layer {best_l}")
print(f"MISTRAL BASE was: r=0.448 at layer 1")
json.dump({"model":"Llama-3.1-8B-base","peak_r":float(best_r),"peak_layer":int(best_l)},
          open("data/step0_llama.json","w"), indent=2)
