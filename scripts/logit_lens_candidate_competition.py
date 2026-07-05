"""
logit_lens_candidate_competition.py

Exploratory depth-axis test (pre-registered in
paper2_depth_competition_prereg_stub.md). Tests whether the two leading answer
candidates compete across layer depth at the prompt_final position, using the
logit lens (nostalgebraist 2020; cf. tuned lens, Belrose et al. 2023).

Subspace rationale: the confirmatory bimodality test projected onto a
correctness axis, which is the resolved OUTCOME of competition, not the
competition. Here we track two genuine competing representations (answer
candidates) across the same depth axis via the unembedding. This maps directly
onto two LCA channels evolving across the forward pass.

Measures per prompt (prompt_final position):
  A = final-layer top-1 token (committed first answer token)
  B = final-layer top-2 token (leading alternative)
  crossover         : B led A at some earlier layer, A won by the end (WTA)
  coactivation_depth: layers where BOTH softmax probs exceed a floor
  suppression_r     : Pearson r of (logit_A, logit_B) across layers
                      (LOGITS, not probs: avoids softmax sum-to-1 coupling)

Decision rule (pre-specified, all three required):
  1. crossover present in > 40% of prompts
  2. mean suppression_r < -0.30
  3. coactivation_depth higher for incorrect than correct, d > 0.30, p < 0.0125

Logit lens detail: apply the model's final norm to each layer's residual, then
the unembedding. This is the standard logit-lens approximation (the per-layer
affine correction is what tuned lens adds; that is the confirmation step if
this pass is positive).

Usage:
    python scripts/logit_lens_candidate_competition.py \
        --model_path mistralai/Mistral-7B-v0.3 \
        --index_path data/layer_stack_index_mistral.jsonl \
        --output_path data/logit_lens_competition_mistral.json \
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
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--position", type=str, default="prompt_final")
    p.add_argument("--category", type=str, default="1")
    p.add_argument("--greedy_only", action="store_true", default=False,
                   help="Use only greedy completions (deterministic candidates)")
    p.add_argument("--prob_floor", type=float, default=0.05,
                   help="Co-activation probability floor (pre-reg placeholder)")
    p.add_argument("--device", type=str, default="mps")
    # Decision-rule thresholds (pre-registered)
    p.add_argument("--thr_crossover_frac", type=float, default=0.40)
    p.add_argument("--thr_suppression_r", type=float, default=-0.30)
    p.add_argument("--thr_cohens_d", type=float, default=0.30)
    p.add_argument("--thr_alpha", type=float, default=0.0125)
    return p.parse_args()


def load_lens(model_path, device):
    """Load final norm + unembedding only (full model load on the Studio)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map=device
    )
    model.eval()
    # Mistral / Llama HF layout
    final_norm = model.model.norm
    lm_head = model.lm_head
    return tok, model, final_norm, lm_head


@torch.no_grad()
def layer_logits(hidden_layer, final_norm, lm_head, device):
    """Project one layer's residual [hidden_dim] to vocab logits via logit lens."""
    h = torch.tensor(hidden_layer, dtype=torch.float16, device=device).unsqueeze(0)
    normed = final_norm(h)
    logits = lm_head(normed).float().squeeze(0)  # [vocab]
    return logits


def is_variant(tok, a_id, b_id):
    """True if B is a casing/whitespace/subword variant of A (not a competitor)."""
    a = tok.decode([a_id]).strip().lower()
    b = tok.decode([b_id]).strip().lower()
    if not a or not b:
        return True
    return a == b or a.startswith(b) or b.startswith(a)


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
    print(f"{len(entries)} entries (category {args.category}, "
          f"greedy_only={args.greedy_only}).")

    per_prompt = []
    sanity_hits = 0
    sanity_total = 0
    excluded_variant = 0

    for e in tqdm(entries, desc="Logit-lens competition"):
        if not os.path.exists(e["npz_path"]):
            continue
        data = np.load(e["npz_path"], allow_pickle=True)
        stack = data["hidden_stack"].astype(np.float32)  # [pos, layers, hdim]
        pos_labels = list(data["position_labels"])
        if args.position not in pos_labels:
            continue
        hs = stack[pos_labels.index(args.position)]  # [layers, hdim]
        n_layers = hs.shape[0]

        # Final-layer logits define the candidates
        final_logits = layer_logits(hs[-1], final_norm, lm_head, args.device)
        top2 = torch.topk(final_logits, k=2)
        a_id = int(top2.indices[0].item())
        b_id = int(top2.indices[1].item())

        # Sanity: greedy final top-1 should match first answer token
        if e.get("greedy", False) and e.get("answer_text"):
            sanity_total += 1
            ans_ids = tok(e["answer_text"], add_special_tokens=False)["input_ids"]
            if ans_ids and ans_ids[0] == a_id:
                sanity_hits += 1

        # Confound filter: B a variant of A
        if is_variant(tok, a_id, b_id):
            excluded_variant += 1
            continue

        # Trace A, B across depth
        logit_A = np.zeros(n_layers)
        logit_B = np.zeros(n_layers)
        prob_A = np.zeros(n_layers)
        prob_B = np.zeros(n_layers)
        for l in range(n_layers):
            ll = layer_logits(hs[l], final_norm, lm_head, args.device)
            logit_A[l] = float(ll[a_id].item())
            logit_B[l] = float(ll[b_id].item())
            probs = torch.softmax(ll, dim=-1)
            prob_A[l] = float(probs[a_id].item())
            prob_B[l] = float(probs[b_id].item())

        # crossover: B logit exceeds A logit at any layer before the final
        crossover = bool(np.any(logit_B[:-1] > logit_A[:-1]))

        # coactivation depth (probabilities, floor)
        coact = int(np.sum((prob_A > args.prob_floor) & (prob_B > args.prob_floor)))

        # suppression r on LOGITS (no softmax coupling)
        if np.std(logit_A) > 1e-6 and np.std(logit_B) > 1e-6:
            supp_r = float(np.corrcoef(logit_A, logit_B)[0, 1])
        else:
            supp_r = 0.0

        # answer token length covariate
        ans_ids = tok(e.get("answer_text", ""), add_special_tokens=False)["input_ids"]

        per_prompt.append({
            "prompt_id": e["prompt_id"],
            "correct": bool(e["correct"]),
            "a_token": tok.decode([a_id]),
            "b_token": tok.decode([b_id]),
            "crossover": crossover,
            "coactivation_depth": coact,
            "suppression_r": supp_r,
            "answer_n_tokens": len(ans_ids),
        })

    n = len(per_prompt)
    if n < 10:
        print(f"Only {n} usable prompts. Insufficient.")
        return

    crossover_frac = np.mean([p["crossover"] for p in per_prompt])
    mean_supp_r = np.mean([p["suppression_r"] for p in per_prompt])

    coact_correct = np.array([p["coactivation_depth"] for p in per_prompt if p["correct"]])
    coact_incorrect = np.array([p["coactivation_depth"] for p in per_prompt if not p["correct"]])

    if len(coact_correct) >= 2 and len(coact_incorrect) >= 2:
        diff = coact_incorrect.mean() - coact_correct.mean()
        pooled = np.sqrt((coact_incorrect.var(ddof=1) + coact_correct.var(ddof=1)) / 2) + 1e-8
        cohens_d = diff / pooled
        # one-sided: incorrect > correct
        t_stat, p_two = stats.ttest_ind(coact_incorrect, coact_correct, equal_var=False)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:
        cohens_d = 0.0
        p_one = 1.0

    # answer-length confound check
    lengths = np.array([p["answer_n_tokens"] for p in per_prompt])
    coacts = np.array([p["coactivation_depth"] for p in per_prompt])
    len_r, len_p = stats.spearmanr(lengths, coacts)

    print("\n" + "=" * 60)
    print("DEPTH-AXIS CANDIDATE COMPETITION  (logit lens)")
    print("=" * 60)
    print(f"Usable prompts: {n}  (excluded as B-variant-of-A: {excluded_variant})")
    if sanity_total:
        print(f"Sanity (greedy final top-1 == first answer token): "
              f"{sanity_hits}/{sanity_total} = {sanity_hits/sanity_total:.2f}")
    print(f"Correct: {len(coact_correct)}  Incorrect: {len(coact_incorrect)}")
    print()
    print(f"Criterion 1  crossover fraction : {crossover_frac:.3f}  "
          f"(thr > {args.thr_crossover_frac})  "
          f"{'PASS' if crossover_frac > args.thr_crossover_frac else 'fail'}")
    print(f"Criterion 2  mean suppression_r : {mean_supp_r:.3f}  "
          f"(thr < {args.thr_suppression_r})  "
          f"{'PASS' if mean_supp_r < args.thr_suppression_r else 'fail'}")
    print(f"Criterion 3  coact d (inc>cor)  : {cohens_d:.3f}  p1={p_one:.4f}  "
          f"(thr d>{args.thr_cohens_d}, p<{args.thr_alpha})  "
          f"{'PASS' if (cohens_d > args.thr_cohens_d and p_one < args.thr_alpha) else 'fail'}")
    print()
    print(f"  coact_depth correct  : mean {coact_correct.mean():.2f} "
          f"(n={len(coact_correct)})")
    print(f"  coact_depth incorrect: mean {coact_incorrect.mean():.2f} "
          f"(n={len(coact_incorrect)})")
    print(f"  answer-length confound: Spearman r(len, coact)={len_r:.3f} p={len_p:.4f}")

    c1 = crossover_frac > args.thr_crossover_frac
    c2 = mean_supp_r < args.thr_suppression_r
    c3 = cohens_d > args.thr_cohens_d and p_one < args.thr_alpha

    print("\n=== VERDICT ===")
    if c1 and c2 and c3:
        print("  All three criteria PASS. Depth-axis candidate competition")
        print("  supported AND tied to hallucination. Rivalry relocates to depth.")
        print("  NEXT: float32 recapture + tuned-lens confirmation + Llama replication.")
    elif c1 and c2 and not c3:
        print("  Criteria 1-2 pass, 3 fails. Competition exists across depth but")
        print("  does NOT track correctness. Weaker, non-confirmatory. Report as such.")
    else:
        print("  Decision rule not met. Depth-axis dynamical reading REJECTED.")
        print("  Negative result holds. Finalise the negative paper.")

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
        "excluded_variant": excluded_variant,
        "sanity_top1_match": (sanity_hits / sanity_total) if sanity_total else None,
        "crossover_frac": crossover_frac,
        "mean_suppression_r": mean_supp_r,
        "coact_correct_mean": float(coact_correct.mean()) if len(coact_correct) else None,
        "coact_incorrect_mean": float(coact_incorrect.mean()) if len(coact_incorrect) else None,
        "cohens_d": cohens_d,
        "p_one_sided": p_one,
        "answer_length_spearman_r": float(len_r),
        "answer_length_spearman_p": float(len_p),
        "thresholds": {
            "crossover_frac": args.thr_crossover_frac,
            "suppression_r": args.thr_suppression_r,
            "cohens_d": args.thr_cohens_d,
            "alpha": args.thr_alpha,
        },
        "criteria": {"c1_crossover": c1, "c2_suppression": c2, "c3_correctness": c3},
        "verdict_supported": bool(c1 and c2 and c3),
        "per_prompt": per_prompt,
    }
    with open(args.output_path, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
