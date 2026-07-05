#!/usr/bin/env python3
"""Three preliminary-study figures from existing Mistral data."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

Path("figures").mkdir(exist_ok=True)
plt.rcParams.update({"font.size":11, "font.family":"serif", "figure.dpi":300,
                      "savefig.bbox":"tight"})

HNEURON_LAYERS = [0, 7, 9, 18, 24, 31]
ANTI_H_LAYERS = [9, 12, 13, 14, 18, 20, 24, 25, 28, 30]

# === FIGURE 1: AUROC buildup ===
base = json.load(open("data/layer_dynamics_lastanswer_mistral.json"))
inst = json.load(open("data/layer_dynamics_lastanswer_mistral_instruct.json"))
ba, ia = np.array(base["layer_aurocs"]), np.array(inst["layer_aurocs"])
layers = np.arange(len(ba))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(layers, ba, "o-", color="#2166ac", ms=3, lw=1.5, label="Mistral-7B base")
ax.plot(layers, ia, "s-", color="#b2182b", ms=3, lw=1.5, label="Mistral-7B-Instruct")
for hl in HNEURON_LAYERS:
    ax.axvline(hl, color="#4daf4a", alpha=0.25, lw=6, zorder=1)
for al in ANTI_H_LAYERS:
    ax.axvline(al, color="#ff7f00", alpha=0.15, lw=4, zorder=1)
ax.axhline(0.5, color="grey", ls="--", lw=0.5, alpha=0.5)
ax.legend(handles=[
    plt.Line2D([0],[0], color="#2166ac", marker="o", ms=4, label=f"Base (peak L{base['buildup']['peak_layer']}, {base['buildup']['peak_auroc']:.3f})"),
    plt.Line2D([0],[0], color="#b2182b", marker="s", ms=4, label=f"Instruct (peak L{inst['buildup']['peak_layer']}, {inst['buildup']['peak_auroc']:.3f})"),
    Patch(facecolor="#4daf4a", alpha=0.3, label="H-Neuron layers"),
    Patch(facecolor="#ff7f00", alpha=0.2, label="Anti-H layers"),
], loc="upper left", fontsize=8)
ax.set_xlabel("Layer"); ax.set_ylabel("Cross-validated AUROC")
ax.set_title("Per-layer correctness decodability (Mistral, last-answer-tok)")
ax.set_ylim(0.35, max(max(ba), max(ia)) + 0.05)
fig.savefig("figures/fig_prelim_auroc_buildup.pdf"); plt.close()
print("Saved figures/fig_prelim_auroc_buildup.pdf")

# === FIGURE 2: Commitment depth ===
cd = json.load(open("data/commitment_depth_mistral_instruct.json"))
# Check for per-prompt data
per_prompt_keys = [k for k in cd if "per_prompt" in k or "lstar_values" in k or k == "prompts" or k == "per_prompt_lstar"]
if per_prompt_keys:
    print(f"Found per-prompt key: {per_prompt_keys[0]}")

fig, ax = plt.subplots(figsize=(6, 3.5))
bands = ["Early\n(L 0-10)", "Mid\n(L 11-21)", "Late\n(L 22-32)"]
vals = [cd["band_early"], cd["band_mid"], cd["band_late"]]
colors = ["#92c5de", "#f4a582", "#b2182b"]
bars = ax.bar(bands, vals, color=colors, edgecolor="white", width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v}%", ha="center", fontsize=10)
ax.set_ylabel("Percent of prompts")
ax.set_title("Commitment depth distribution (Mistral-Instruct, Cat1, n=100)")
ax.set_ylim(0, 100)
ax.text(0.98, 0.95,
    f"Mean L* = {cd['L_star_mean']:.1f} (sd {cd['L_star_std']:.1f})\n"
    f"Correct: {cd['L_star_correct_mean']:.1f}\n"
    f"Incorrect: {cd['L_star_incorrect_mean']:.1f}\n"
    f"Cohen's d = {cd['cohens_d_correct_minus_incorrect']:.2f}\n"
    f"Welch p = {cd['welch_p']:.3f}",
    transform=ax.transAxes, fontsize=8, va="top", ha="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
fig.savefig("figures/fig_prelim_commitment_depth.pdf"); plt.close()
print("Saved figures/fig_prelim_commitment_depth.pdf")

# === FIGURE 3: Detector comparison ===
ema = json.load(open("data/ema_vs_gclca_results.json"))
names, aurocs = [], []
names.append("Instantaneous"); aurocs.append(ema["auroc_instantaneous"])
names.append("Mean"); aurocs.append(ema["auroc_mean"])
for k in sorted(ema["auroc_emas"].keys(), key=float):
    names.append(f"EMA a={k}"); aurocs.append(ema["auroc_emas"][k])
names.append("GC-LCA (diff)"); aurocs.append(ema["auroc_lca_diff"])
names.append("GC-LCA (xc)"); aurocs.append(ema["auroc_lca_xc"])

# Get leads
leads = ema.get("leads_vs_raw_median_tokens", {})

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#2166ac"] + ["#92c5de"] * (len(aurocs) - 1)
bars = ax.bar(range(len(aurocs)), aurocs, color=colors, edgecolor="white", width=0.7)
ax.set_xticks(range(len(aurocs)))
ax.set_xticklabels(names, fontsize=7, rotation=25, ha="right")
ax.set_ylabel("AUROC (token-level)")
ax.set_title("Detector family comparison (Mistral, token-level)")

ymin = min(aurocs) - 0.02; ymax = max(aurocs) + 0.02
ax.set_ylim(ymin, ymax)
ax.axhline(0.5, color="grey", ls="--", lw=0.5, alpha=0.3)

spread = max(aurocs) - min(aurocs)
integrating = aurocs[1:]  # everything except instantaneous
int_spread = max(integrating) - min(integrating)
ax.text(0.02, 0.95,
    f"Full spread: {spread:.3f}\n"
    f"Integrating-detector spread: {int_spread:.3f}\n"
    f"Max lead vs raw: {max(leads.values()) if leads else 'n/a'} tokens\n"
    f"Dynamical detection = smoothing",
    transform=ax.transAxes, fontsize=8, va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig.savefig("figures/fig_prelim_detector_comparison.pdf"); plt.close()
print("Saved figures/fig_prelim_detector_comparison.pdf")
print("\nDone. All three figures in figures/")
