"""
run_5b_probe.py  (Option 2 + corrected SVCCA)

H5 5b: subspace alignment between the H-Neuron-layer representation and the
decision-layer (last-layer) representation.

Two DOCUMENTED DEVIATIONS from the literal reg wording, both caught on development
rehearsal before the hold-out:

(1) The reg says "CCA between MLP-probe hidden representation at H-Neuron layers
    and at decision layers". Two MLPs both trained on the hallucination label have
    trivially correlated hidden reps (both label-shaped), giving rho_1 ~ 1
    regardless of true alignment (verified: 0.98/0.90). Per the reg's stated
    purpose (lines 466-467, "the mismatch is robust to probe form") and the named
    method (SVCCA, Raghu et al.), alignment is computed between the loci
    REPRESENTATIONS directly, SVD-reduced and mean-centred, WITHOUT label training.
    The MLP-probe AUROCs are retained as a separate "nonlinear probe reads
    hallucination from each locus" check.

(2) The reg names rho_1 (the single largest canonical correlation). rho_1 is the
    most overfit SVCCA statistic: two k-dim subspaces in n items have a first
    principal-angle cosine near 1 for any modest k, so rho_1 cannot discriminate
    aligned from unrelated at n = 200-800 (verified: shuffled rho_1 reaches 0.21
    at k=3, 0.33 at k=10, so the 0.25 threshold is unreachable). Replaced a-priori
    by the MEAN canonical correlation at k=3, read against a permutation null
    (shuffle item correspondence). Threshold rho_1 < 0.25 is dead; the
    interpretable metric is excess = mean_cc - null_mean and p_one.

DEVELOPMENT rehearsal; the confirmatory number is the hold-out (reg 3.5).
Descriptive, no significance test on the metric itself.

Usage:
  python scripts/run_5b_probe.py \
    --acts_root data/hidden_5b_mistral_v2 \
    --output data/h5_5b_mistral_v2.json \
    --svcca_k 3
"""
import os, json, glob, argparse
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--svcca_k", type=int, default=3,
                   help="SVD components per representation before CCA. a-priori 3 "
                        "(only value where the shuffle null is well separated).")
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1.0, help="MLP L2")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load(acts_root):
    H, L, Y = [], [], []
    for f in sorted(glob.glob(os.path.join(acts_root, "act_*.npz"))):
        d = np.load(f)
        H.append(d["hneuron"]); L.append(d["last"]); Y.append(int(d["y"]))
    return np.array(H), np.array(L), np.array(Y)


def svcca(A, B, k, n_perm, seed):
    """SVCCA between representations A, B (items x dim).
    Returns MEAN canonical correlation at k SVD components + a permutation null
    (shuffle item correspondence, recompute). rho_1 is NOT used (degenerate)."""
    def basis(M, k):
        M = M - M.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        k = min(k, int((S > 1e-8).sum()))
        return U[:, :k]                      # items x k, orthonormal columns

    def mean_cc(Qa, Qb):
        kc = min(Qa.shape[1], Qb.shape[1])
        s = np.linalg.svd(Qa[:, :kc].T @ Qb[:, :kc], compute_uv=False)
        return float(np.mean(np.clip(s, 0, 1)))

    Qa = basis(A, k)
    Qb = basis(B, k)
    obs = mean_cc(Qa, Qb)

    rng = np.random.default_rng(seed)
    n = A.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = mean_cc(Qa, Qb[rng.permutation(n)])
    return {
        "mean_cc": obs,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "excess": float(obs - null.mean()),
        "p_one": float((null >= obs).mean()),
        "k": int(Qa.shape[1]),
    }


def main():
    a = parse_args()
    H, L, Y = load(a.acts_root)
    print(f"items {len(Y)}  H-dim {H.shape[1]}  last-dim {L.shape[1]}  "
          f"halluc rate {Y.mean():.3f}")

    # probe-quality half: nonlinear MLP reads hallucination from each locus
    cv = StratifiedKFold(5, shuffle=True, random_state=a.seed)
    auroc = {}
    for name, X in [("hneuron", H), ("last", L)]:
        Xs = StandardScaler().fit_transform(X)
        clf = MLPClassifier(hidden_layer_sizes=(a.hidden,), alpha=a.alpha,
                            max_iter=2000, random_state=a.seed)
        auroc[name] = float(cross_val_score(clf, Xs, Y, cv=cv, scoring="roc_auc").mean())
        print(f"  MLP({name}) CV AUROC {auroc[name]:.4f}")

    # alignment half: SVCCA mean canonical correlation vs permutation null
    sv = svcca(H, L, a.svcca_k, a.n_perm, a.seed)
    print(f"\n  SVCCA mean-CC = {sv['mean_cc']:.4f}  (k={sv['k']})")
    print(f"  null mean {sv['null_mean']:.4f}  null p95 {sv['null_p95']:.4f}")
    print(f"  excess {sv['excess']:+.4f}  p_one {sv['p_one']:.4f}")
    print(f"  low excess / high p_one => loci NOT aligned beyond chance "
          f"=> supports mismatch")

    res = {
        "n_items": int(len(Y)),
        "mlp_auroc_hneuron": auroc["hneuron"],
        "mlp_auroc_last": auroc["last"],
        "svcca_mean_cc": sv["mean_cc"],
        "svcca_null_mean": sv["null_mean"],
        "svcca_null_p95": sv["null_p95"],
        "svcca_excess": sv["excess"],
        "svcca_p_one": sv["p_one"],
        "svcca_k": sv["k"],
        "note": ("DEVELOPMENT rehearsal; confirmatory is hold-out (reg 3.5). "
                 "Deviations: (1) SVCCA between raw loci reps not label-trained "
                 "MLP hiddens; (2) mean canonical correlation at k=3 vs shuffle "
                 "null, NOT rho_1 (degenerate at this n). Descriptive."),
    }
    json.dump(res, open(a.output, "w"), indent=2)
    print(f"saved {a.output}")


if __name__ == "__main__":
    main()
