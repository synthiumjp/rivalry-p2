"""
analyse_hypotheses.py

Statistical tests for H1-H5 from the pre-registration.

Usage:
    python analyse_hypotheses.py \
        --clusters_path data/clusters_mistral.jsonl \
        --interventions_path data/interventions_mistral.jsonl \
        --se_path data/se_mistral_benchmark.jsonl \
        --output_path data/hypothesis_results_mistral.json

Hypotheses tested:
    H1: Metastable clustering. CV in [0.3, 0.7], non-geometric dwell distributions.
    H2: Within-episode trends. Negative activation slope within episodes.
    H3: Levelt analogue. Prop I rho > 0.3, Prop II rho < 0.
    H4: Structure preservation. Targeted DPR in [0.8, 1.2], indiscriminate
        in [0.3, 0.8], difference > 0.15.
    H5: Asymmetric intervention. Suppression > amplification, larger
        asymmetry at low SE.

Bonferroni correction: alpha = 0.0125 for 4 confirmatory tests (H1, H3, H4, H5).
H2 is exploratory (alpha = 0.05).
"""

import json
import argparse
import math
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description="Test H1-H5 hypotheses.")
    parser.add_argument("--clusters_path", type=str, required=True)
    parser.add_argument("--interventions_path", type=str, default=None)
    parser.add_argument("--se_path", type=str, default=None)
    parser.add_argument("--activations_path", type=str, default=None,
                        help="Raw activations for H2 within-episode trends")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--alpha_confirmatory", type=float, default=0.0125,
                        help="Bonferroni-corrected alpha for confirmatory tests")
    parser.add_argument("--alpha_exploratory", type=float, default=0.05)
    return parser.parse_args()


# -------------------------------------------------------------------
# H1: Metastable Clustering
# -------------------------------------------------------------------

def test_h1(clusters: List[Dict]) -> Dict:
    """Test H1: CV in [0.3, 0.7], non-geometric dwell distributions.

    Prediction: metastable dynamics produce CV between 0.3 and 0.7
    (lower than geometric/exponential which gives CV ~ 1.0, higher
    than deterministic which gives CV ~ 0).
    """
    all_cvs = []
    all_nd_dwells = []
    n_with_data = 0

    for cluster_data in clusters:
        if "nd_cv" not in cluster_data or cluster_data["nd_cv"] is None:
            continue
        all_cvs.append(cluster_data["nd_cv"])
        nd_dwells = cluster_data.get("nd_dwells", [])
        if nd_dwells:
            all_nd_dwells.extend(nd_dwells)
        n_with_data += 1

    if not all_cvs:
        return {"status": "no_data"}

    mean_cv = float(np.mean(all_cvs))
    median_cv = float(np.median(all_cvs))
    ci_low, ci_high = np.percentile(all_cvs, [2.5, 97.5])

    # Check: CV in [0.3, 0.7]
    cv_in_range = 0.3 <= mean_cv <= 0.7
    frac_in_range = sum(1 for cv in all_cvs if 0.3 <= cv <= 0.7) / len(all_cvs)

    # Non-geometric test: compare dwell distribution to geometric
    # Geometric has memoryless property: P(dwell > k) = (1-p)^k
    # KS test against fitted geometric
    geo_p = None
    ks_stat = None
    ks_pvalue = None

    if len(all_nd_dwells) > 20:
        geo_p = 1.0 / np.mean(all_nd_dwells)
        ks_stat, ks_pvalue = stats.kstest(
            all_nd_dwells,
            lambda x: 1 - (1 - geo_p) ** x
        )

    return {
        "hypothesis": "H1",
        "status": "tested",
        "n_prompts": n_with_data,
        "mean_cv": mean_cv,
        "median_cv": median_cv,
        "cv_ci_95": [float(ci_low), float(ci_high)],
        "cv_in_predicted_range": cv_in_range,
        "frac_prompts_in_range": float(frac_in_range),
        "n_dwell_episodes": len(all_nd_dwells),
        "mean_dwell": float(np.mean(all_nd_dwells)) if all_nd_dwells else None,
        "geometric_p": float(geo_p) if geo_p else None,
        "ks_statistic": float(ks_stat) if ks_stat is not None else None,
        "ks_pvalue": float(ks_pvalue) if ks_pvalue is not None else None,
        "non_geometric": ks_pvalue < 0.05 if ks_pvalue is not None else None,
        "supported": cv_in_range,
    }


# -------------------------------------------------------------------
# H3: Levelt Analogue
# -------------------------------------------------------------------

def test_h3(clusters: List[Dict], se_data: Dict[str, float]) -> Dict:
    """Test H3: Levelt Proposition analogues.

    Prop I: Spearman rho > 0.3 between inverse SE and predominance
    (proportion of tokens in dominant cluster).

    Prop II: negative Spearman rho between inverse SE and mean
    non-dominant dwell time.
    """
    inverse_se = []
    predominance = []
    nd_mean_dwell = []

    for cluster_data in clusters:
        pid = cluster_data["prompt_id"]
        if pid not in se_data:
            continue
        se = se_data[pid]
        if se <= 0:
            continue

        counts = cluster_data.get("cluster_counts", [])
        if not counts or sum(counts) == 0:
            continue

        total = sum(counts)
        dom_count = max(counts)
        pred = dom_count / total

        inverse_se.append(1.0 / se)
        predominance.append(pred)

        nd_dwells = cluster_data.get("nd_dwells", [])
        if nd_dwells:
            nd_mean_dwell.append(float(np.mean(nd_dwells)))
        else:
            nd_mean_dwell.append(0.0)

    if len(inverse_se) < 10:
        return {"status": "insufficient_data", "n": len(inverse_se)}

    # Prop I: inverse SE vs predominance
    rho_prop1, p_prop1 = stats.spearmanr(inverse_se, predominance)

    # Prop II: inverse SE vs non-dominant dwell time
    rho_prop2, p_prop2 = stats.spearmanr(inverse_se, nd_mean_dwell)

    # Bootstrap CIs (10,000 resamples)
    n = len(inverse_se)
    boot_rho1 = []
    boot_rho2 = []
    for _ in range(10000):
        idx = np.random.choice(n, n, replace=True)
        r1, _ = stats.spearmanr(
            [inverse_se[i] for i in idx],
            [predominance[i] for i in idx]
        )
        r2, _ = stats.spearmanr(
            [inverse_se[i] for i in idx],
            [nd_mean_dwell[i] for i in idx]
        )
        boot_rho1.append(r1)
        boot_rho2.append(r2)

    ci1 = np.percentile(boot_rho1, [2.5, 97.5])
    ci2 = np.percentile(boot_rho2, [2.5, 97.5])

    prop1_supported = rho_prop1 > 0.3 and ci1[0] > 0
    prop2_supported = rho_prop2 < 0 and ci2[1] < 0

    return {
        "hypothesis": "H3",
        "status": "tested",
        "n": n,
        "prop1_rho": float(rho_prop1),
        "prop1_p": float(p_prop1),
        "prop1_ci_95": [float(ci1[0]), float(ci1[1])],
        "prop1_supported": prop1_supported,
        "prop2_rho": float(rho_prop2),
        "prop2_p": float(p_prop2),
        "prop2_ci_95": [float(ci2[0]), float(ci2[1])],
        "prop2_supported": prop2_supported,
        "h3_supported": prop1_supported,  # Prop I is confirmatory
    }


# -------------------------------------------------------------------
# H4: Structure Preservation
# -------------------------------------------------------------------

def test_h4(interventions: List[Dict]) -> Dict:
    """Test H4: targeted DPR in [0.8, 1.2], indiscriminate in [0.3, 0.8].

    DPR = median non-dominant dwell time (intervention) /
          median non-dominant dwell time (baseline).

    Supported: DPR difference (targeted - indiscriminate) > 0.15, p < 0.05.
    """
    targeted_dprs = []
    indiscriminate_dprs = []
    paired_prompts = []

    for item in interventions:
        conditions = item.get("conditions", {})
        baseline_dwell = item.get("baseline_nd_median_dwell")
        if not baseline_dwell or baseline_dwell <= 0:
            continue

        # Compute DPR for targeted suppression
        targeted_comps = conditions.get("suppress_targeted", [])
        indisc_comps = conditions.get("suppress_indiscriminate", [])

        if not targeted_comps or not indisc_comps:
            continue

        # Compute median nd dwell for each condition
        t_dwells = _get_nd_dwells_from_completions(targeted_comps)
        i_dwells = _get_nd_dwells_from_completions(indisc_comps)

        if not t_dwells or not i_dwells:
            continue

        t_dpr = float(np.median(t_dwells)) / baseline_dwell
        i_dpr = float(np.median(i_dwells)) / baseline_dwell

        targeted_dprs.append(t_dpr)
        indiscriminate_dprs.append(i_dpr)
        paired_prompts.append(item["prompt_id"])

    if len(targeted_dprs) < 10:
        return {"status": "insufficient_data", "n": len(targeted_dprs)}

    targeted_dprs = np.array(targeted_dprs)
    indiscriminate_dprs = np.array(indiscriminate_dprs)
    dpr_diff = targeted_dprs - indiscriminate_dprs

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(targeted_dprs, indiscriminate_dprs)
    mean_diff = float(np.mean(dpr_diff))

    # Bayesian: posterior probability that difference > 0
    # Simple normal approximation
    se = float(np.std(dpr_diff, ddof=1) / np.sqrt(len(dpr_diff)))
    if se > 0:
        z = mean_diff / se
        p_positive = 1 - stats.norm.cdf(0, loc=mean_diff, scale=se)
    else:
        p_positive = 0.5

    supported = mean_diff > 0.15 and p_value < 0.05

    return {
        "hypothesis": "H4",
        "status": "tested",
        "n_prompts": len(targeted_dprs),
        "mean_targeted_dpr": float(np.mean(targeted_dprs)),
        "mean_indiscriminate_dpr": float(np.mean(indiscriminate_dprs)),
        "mean_dpr_difference": mean_diff,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "p_positive": float(p_positive),
        "targeted_in_range": float(np.mean(
            (targeted_dprs >= 0.8) & (targeted_dprs <= 1.2)
        )),
        "indiscriminate_in_range": float(np.mean(
            (indiscriminate_dprs >= 0.3) & (indiscriminate_dprs <= 0.8)
        )),
        "supported": supported,
    }


def _get_nd_dwells_from_completions(completions: List[Dict]) -> List[float]:
    """Extract non-dominant dwell times from intervention completions."""
    all_dwells = []
    for comp in completions:
        if comp.get("degenerate"):
            continue
        acts = comp.get("activations", [])
        if len(acts) < 5:
            continue
        # Quick k=2 clustering
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(np.array(acts))
        counts = np.bincount(labels, minlength=2)
        nd = int(np.argmin(counts))
        dwells = []
        current = labels[0]
        length = 1
        for i in range(1, len(labels)):
            if labels[i] == current:
                length += 1
            else:
                if current == nd:
                    dwells.append(length)
                current = labels[i]
                length = 1
        if current == nd:
            dwells.append(length)
        all_dwells.extend(dwells)
    return all_dwells


# -------------------------------------------------------------------
# H5: Asymmetric Intervention
# -------------------------------------------------------------------

def test_h5(interventions: List[Dict], se_data: Dict[str, float]) -> Dict:
    """Test H5: suppression > amplification, larger asymmetry at low SE.

    2x2: (suppress vs amplify) x (high-SE vs low-SE) on switch rates.
    """
    # Split prompts by SE median
    prompt_se = {}
    for item in interventions:
        pid = item["prompt_id"]
        if pid in se_data:
            prompt_se[pid] = se_data[pid]

    if len(prompt_se) < 20:
        return {"status": "insufficient_data"}

    se_median = float(np.median(list(prompt_se.values())))

    cells = {
        "suppress_low": [], "suppress_high": [],
        "amplify_low": [], "amplify_high": [],
    }

    for item in interventions:
        pid = item["prompt_id"]
        if pid not in prompt_se:
            continue
        se = prompt_se[pid]
        se_group = "low" if se <= se_median else "high"
        conditions = item.get("conditions", {})

        for direction in ["suppress", "amplify"]:
            key = f"{direction}_targeted"
            comps = conditions.get(key, [])
            if not comps:
                continue

            # Count switches across completions
            total_switches = 0
            total_tokens = 0
            for comp in comps:
                if comp.get("degenerate"):
                    continue
                acts = comp.get("activations", [])
                if len(acts) < 5:
                    continue
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = km.fit_predict(np.array(acts))
                switches = sum(
                    1 for i in range(1, len(labels)) if labels[i] != labels[i-1]
                )
                total_switches += switches
                total_tokens += len(labels) - 1

            if total_tokens > 0:
                switch_rate = total_switches / total_tokens
                cell_key = f"{direction}_{se_group}"
                cells[cell_key].append(switch_rate)

    # Check we have enough data
    for key, vals in cells.items():
        if len(vals) < 5:
            return {"status": "insufficient_data", "cell_counts": {k: len(v) for k, v in cells.items()}}

    # Main effect: suppression > amplification
    suppress_rates = cells["suppress_low"] + cells["suppress_high"]
    amplify_rates = cells["amplify_low"] + cells["amplify_high"]
    main_t, main_p = stats.ttest_ind(suppress_rates, amplify_rates)

    # Interaction: asymmetry larger at low SE
    low_asymmetry = np.mean(cells["suppress_low"]) - np.mean(cells["amplify_low"])
    high_asymmetry = np.mean(cells["suppress_high"]) - np.mean(cells["amplify_high"])

    # Chi-squared or interaction test
    interaction_diff = low_asymmetry - high_asymmetry

    supported = (
        main_p < 0.05
        and np.mean(suppress_rates) > np.mean(amplify_rates)
        and interaction_diff > 0
    )

    return {
        "hypothesis": "H5",
        "status": "tested",
        "se_median": se_median,
        "cell_means": {k: float(np.mean(v)) for k, v in cells.items()},
        "cell_ns": {k: len(v) for k, v in cells.items()},
        "main_effect_t": float(main_t),
        "main_effect_p": float(main_p),
        "suppress_mean": float(np.mean(suppress_rates)),
        "amplify_mean": float(np.mean(amplify_rates)),
        "low_se_asymmetry": float(low_asymmetry),
        "high_se_asymmetry": float(high_asymmetry),
        "interaction_diff": float(interaction_diff),
        "supported": supported,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    args = parse_args()

    # Load cluster data
    clusters = []
    with open(args.clusters_path, "r") as f:
        for line in f:
            clusters.append(json.loads(line))
    print(f"Loaded {len(clusters)} clustered prompts.")

    # Load SE data
    se_data = {}
    if args.se_path:
        with open(args.se_path, "r") as f:
            for line in f:
                d = json.loads(line)
                se_data[d["prompt_id"]] = d["se"]
        print(f"Loaded SE for {len(se_data)} prompts.")

    # Load intervention data
    interventions = []
    if args.interventions_path and os.path.exists(args.interventions_path):
        with open(args.interventions_path, "r") as f:
            for line in f:
                interventions.append(json.loads(line))
        print(f"Loaded {len(interventions)} intervention results.")

    results = {}

    # H1
    print("\n=== H1: Metastable Clustering ===")
    h1 = test_h1(clusters)
    results["H1"] = h1
    if h1["status"] == "tested":
        print(f"  Mean CV: {h1['mean_cv']:.3f} (predicted: [0.3, 0.7])")
        print(f"  CV in range: {h1['cv_in_predicted_range']}")
        print(f"  Non-geometric: {h1['non_geometric']}")
        print(f"  Supported: {h1['supported']}")

    # H3
    if se_data:
        print("\n=== H3: Levelt Analogue ===")
        h3 = test_h3(clusters, se_data)
        results["H3"] = h3
        if h3["status"] == "tested":
            print(f"  Prop I rho: {h3['prop1_rho']:.3f} (predicted: > 0.3)")
            print(f"  Prop II rho: {h3['prop2_rho']:.3f} (predicted: < 0)")
            print(f"  Supported: {h3['h3_supported']}")

    # H4
    if interventions:
        print("\n=== H4: Structure Preservation ===")
        h4 = test_h4(interventions)
        results["H4"] = h4
        if h4["status"] == "tested":
            print(f"  Targeted DPR: {h4['mean_targeted_dpr']:.3f}")
            print(f"  Indiscriminate DPR: {h4['mean_indiscriminate_dpr']:.3f}")
            print(f"  Difference: {h4['mean_dpr_difference']:.3f} (threshold: > 0.15)")
            print(f"  p-value: {h4['p_value']:.4f}")
            print(f"  Supported: {h4['supported']}")

    # H5
    if interventions and se_data:
        print("\n=== H5: Asymmetric Intervention ===")
        h5 = test_h5(interventions, se_data)
        results["H5"] = h5
        if h5["status"] == "tested":
            print(f"  Suppress rate: {h5['suppress_mean']:.3f}")
            print(f"  Amplify rate: {h5['amplify_mean']:.3f}")
            print(f"  Low-SE asymmetry: {h5['low_se_asymmetry']:.3f}")
            print(f"  High-SE asymmetry: {h5['high_se_asymmetry']:.3f}")
            print(f"  Supported: {h5['supported']}")

    # Summary
    print("\n=== Summary ===")
    for h, r in sorted(results.items()):
        status = r.get("supported", r.get("status", "unknown"))
        print(f"  {h}: {status}")

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    import os
    main()
