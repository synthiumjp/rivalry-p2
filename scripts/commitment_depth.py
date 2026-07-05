"""
commitment_depth.py

Exploratory: at what layer does the final answer token first become and remain
top-1 under the logit lens? Operationalises the trajectory-level / committed-
early finding (token-axis autocorrelation ~0, classifier works by averaging)
into the depth axis.

For each Cat1 prompt at prompt_final:
  - Project each layer's residual through final_norm + lm_head (logit lens)
  - top1(l) = argmax at layer l
  - L* = smallest layer such that top1(l) == top1(L_final) for all l >= L*
  - That is the commitment depth.

Distribution across prompts, split by correctness.

Pre-specified: concentrated in mid-to-late band. No a priori prediction on
correct vs incorrect difference; report what shows.

Usage:
    python scripts/commitment_depth.py \
        --model_path mistralai/Mistral-7B-Instruct-v0.3 \
        --index_path data/layer_stack_index_mistral_instruct.jsonl \
        --output data/commitment_depth_mistral_instruct.json \
        --position prompt_final --category 1 --greedy_only
"""

import os
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--index_path", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--position", type=str, default="prompt_final")
    p.add_argument("--category", type=str, default="1")
    p.add_argument("--greedy_only", action="store_true", default=True)
    p.add_argument("--device", type=str, default="mps")
    return p.parse_args()


def load_lens(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map=device
    )
    model.eval()
    return tok, model, model.model.norm, model.lm_head


@torch.no_grad()
def per_layer_top1(stack, final_norm, lm_head, device):
    """stack: [n_layers, hidden_dim]. Returns top-1 token id per layer."""
    n_layers = stack.shape[0]
    top1 = np.zeros(n_layers, dtype=np.int64)
    for l in range(n_layers):
        h = torch.tensor(stack[l], dtype=torch.float16, device=device).unsqueeze(0)
        logits = lm_head(final_norm(h)).float().squeeze(0)
        top1[l] = int(logits.argmax().item())
    return top1


def find_commitment(top1):
    """Earliest layer L* such that top1[l] == top1[-1] for all l >= L*."""
    final = top1[-1]
    n = len(top1)
    L_star = n - 1  # at least the last layer
    for L in range(n - 1, -1, -1):
        if top1[L] == final:
            L_star = L
        else:
            break
    return L_star


def main():
    args = parse_args()
    print(f"Loading lens from {args.model_path} ...")
    tok, model, final_norm, lm_head = load_lens(args.model_path, args.device)

    entries = []
    with open(args.index_path) as f:
        for line in f:
            e = json.loads(line)
            if args.category and e.get("category") != args.category:
                continue
            if e.get("correct") is None:
                continue
            if args.greedy_only and not e.get("greedy", False):
                continue
            entries.append(e)
    print(f"{len(entries)} entries.")

    per_prompt = []
    for e in tqdm(entries, desc="Commitment depth"):
        if not os.path.exists(e["npz_path"]):
            continue
        data = np.load(e["npz_path"], allow_pickle=True)
        stack = data["hidden_stack"].astype(np.float32)
        pos_labels = list(data["position_labels"])
        if args.position not in pos_labels:
            continue
        hs = stack[pos_labels.index(args.position)]  # [layers, hdim]

        top1 = per_layer_top1(hs, final_norm, lm_head, args.device)
        L_star = find_commitment(top1)

        per_prompt.append({
            "prompt_id": e["prompt_id"],
            "correct": bool(e["correct"]),
            "L_star": int(L_star),
            "n_layers": int(len(top1)),
            "final_token": tok.decode([int(top1[-1])]),
        })

    n = len(per_prompt)
    if n < 10:
        print(f"Only {n} prompts. Aborting.")
        return

    L = np.array([p["L_star"] for p in per_prompt])
    n_layers = per_prompt[0]["n_layers"]
    correct_mask = np.array([p["correct"] for p in per_prompt])
    Lc = L[correct_mask]
    Li = L[~correct_mask]

    print("\n" + "=" * 60)
    print("COMMITMENT DEPTH (instruct prompt_final, Cat1)")
    print("=" * 60)
    print(f"n = {n}  layers per stack = {n_layers}")
    print(f"  L* overall   : mean {L.mean():.2f}  median {np.median(L):.1f}  "
          f"std {L.std():.2f}")
    print(f"  L* correct   : n={len(Lc)} mean {Lc.mean():.2f} median {np.median(Lc):.1f}")
    print(f"  L* incorrect : n={len(Li)} mean {Li.mean():.2f} median {np.median(Li):.1f}")

    # Histogram by layer band
    early = int((L < n_layers / 3).sum())
    mid = int(((L >= n_layers / 3) & (L < 2 * n_layers / 3)).sum())
    late = int((L >= 2 * n_layers / 3).sum())
    print(f"\n  Commitment band counts (of {n}):")
    print(f"    early (< {n_layers//3}):  {early} ({100*early/n:.1f}%)")
    print(f"    mid:                       {mid} ({100*mid/n:.1f}%)")
    print(f"    late (>= {2*n_layers//3}): {late} ({100*late/n:.1f}%)")

    # Correct vs incorrect test
    if len(Lc) >= 5 and len(Li) >= 5:
        pooled_std = np.sqrt((Lc.var(ddof=1) + Li.var(ddof=1)) / 2) + 1e-8
        d = (Lc.mean() - Li.mean()) / pooled_std
        t, p_two = stats.ttest_ind(Lc, Li, equal_var=False)
        u, p_mw = stats.mannwhitneyu(Lc, Li, alternative="two-sided")
        print(f"\n  Correct vs incorrect:")
        print(f"    Cohen's d (correct - incorrect): {d:+.3f}")
        print(f"    Welch t: t={t:+.3f}  p={p_two:.4f}")
        print(f"    Mann-Whitney U: p={p_mw:.4f}")
    else:
        d, p_two, p_mw = None, None, None

    print("\n=== INTERPRETATION ===")
    mid_late_frac = (mid + late) / n
    if mid_late_frac > 0.7:
        print(f"  Commitment concentrated mid-to-late "
              f"({100*mid_late_frac:.0f}% of prompts). Decision happens in")
        print(f"  a depth band, consistent with trajectory-level reading: the")
        print(f"  answer is set in a small set of layers and the probe reads")
        print(f"  the post-decision state.")
    else:
        print(f"  Commitment NOT concentrated mid-to-late. Distribution is")
        print(f"  spread. Re-examine the pre-spec.")

    def to_native(o):
        if isinstance(o, dict):
            return {k: to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_native(v) for v in o]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    results = {
        "n_prompts": n,
        "n_layers": n_layers,
        "L_star_mean": float(L.mean()),
        "L_star_median": float(np.median(L)),
        "L_star_std": float(L.std()),
        "L_star_correct_mean": float(Lc.mean()) if len(Lc) else None,
        "L_star_incorrect_mean": float(Li.mean()) if len(Li) else None,
        "band_early": early, "band_mid": mid, "band_late": late,
        "cohens_d_correct_minus_incorrect": d,
        "welch_p": p_two,
        "mannwhitney_p": p_mw,
        "per_prompt": per_prompt,
    }
    with open(args.output, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
