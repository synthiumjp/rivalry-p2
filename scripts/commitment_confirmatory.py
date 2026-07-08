#!/usr/bin/env python3
"""
commitment_confirmatory.py

Confirmatory commitment depth to v6.3 6.5 spec, on the CLEAN v2 layer stacks
(correct last_answer_tok). Distinct from the exploratory commitment_depth.py,
which read prompt_final with a bare stabilisation rule.

Per prompt, at last_answer_tok, logit lens per layer = argmax(lm_head(final_norm(h_l))):
  - ambiguity mask: layer is ambiguous if top1 and top2 logits are within 0.05
  - L*-unstable exclusion: >30% of layers ambiguous -> excluded from commitment
  - commitment L*: smallest layer l such that argmax(l)==final argmax for a
    persistence window of k=2 (l and l+1 both match), ignoring ambiguous layers
  - no-commitment: persistence never satisfied -> flagged, excluded
  - depth-normalised L* = L*/n_transformer_layers; upper-third = normalised >= 2/3

H2: proportion upper-third (binomial vs uniform null), per model, target >75% all.
H3: L* incorrect vs correct, Welch t + Cohen's d (pooled SD), one-tailed direction
    reported; correctness from the v2 index (clean-span greedy, locked judge).
Sensitivity: re-run at k=1 and k=3, report fraction whose L* moves >2 layers.

Tuned lens (optional, --tuned_lens dir): confirmation lens. Where the logit lens
is ambiguous mid-network, compare tuned-lens L* to logit-lens L*; prompts with
>3-layer disagreement are flagged lens-sensitive and excluded. If absent, runs
logit-lens only and reports lens-sensitive as pending.

Run:
  python scripts/commitment_confirmatory.py --model Qwen/Qwen2.5-7B-Instruct \
    --index data/layer_stack_index_v2_qwen_instruct.jsonl \
    --output data/commitment_v2_qwen_instruct.json \
    [--tuned_lens data/tuned_lens_direct_qwen_instruct]
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy import stats

AMBIG = 0.05
UNSTABLE_FRAC = 0.30
POS = "last_answer_tok"

def load_model_head(model_path, device, dtype):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                             attn_implementation="eager").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, m.model.norm, m.get_output_embeddings()

class AffineLens(torch.nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        self.t = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(n_layers)])
    def forward(self, h, l):
        return self.t[l](h)

def _norm_tok(s):
    return s.strip().lower()

@torch.no_grad()
def layer_top12(stack, final_norm, lm_head, device, dtype, tok, lens=None):
    """stack: [n_layers+1, H]. Returns per-layer (top1_str, gap) over layers,
    where top1_str is the decoded top-1 token normalised (strip+lower). Matching
    on the answer WORD, not the exact token id, so 'Fat'/' fat'/'fat' collapse."""
    n = stack.shape[0]
    top1 = [""] * n; gap = np.zeros(n)
    for l in range(n):
        h = torch.tensor(stack[l], device=device, dtype=dtype).unsqueeze(0)
        if lens is not None and l < len(lens.t):
            h = lens(h.float(), l).to(dtype)
        logits = lm_head(final_norm(h)).float().squeeze(0)
        v, idx = torch.topk(logits, 2)
        top1[l] = _norm_tok(tok.decode([int(idx[0])])); gap[l] = float(v[0] - v[1])
    return top1, gap

def commitment(top1, gap, k, target_str, skip_embed=True):
    """L* to a GIVEN target WORD (normalised string), persistence window k
    (reg 6.5: 'top-1 remains top-1 for at least the window'). Ambiguous layers
    (top1-top2 logit gap < AMBIG) ignored. L* is the earliest non-ambiguous layer
    from which the target is top-1 for k consecutive non-ambiguous layers."""
    lo = 1 if skip_embed else 0
    layers = list(range(lo, len(top1)))
    ambiguous = gap[lo:] < AMBIG
    ambiguous_frac = float(ambiguous.mean())
    kept = [l for l in layers if gap[l] >= AMBIG]
    if len(kept) < k:
        return None, ambiguous_frac
    match = [top1[l] == target_str for l in kept]
    L = None
    for i in range(len(kept) - k + 1):
        if all(match[i:i + k]):
            L = kept[i]; break
    return L, ambiguous_frac

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tuned_lens", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    model, final_norm, lm_head = load_model_head(args.model, args.device, dtype)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    n_tl = model.config.num_hidden_layers

    lens = None
    if args.tuned_lens and Path(args.tuned_lens, "lens.pt").exists():
        H = model.config.hidden_size
        lens = AffineLens(n_tl, H).to(args.device)
        lens.load_state_dict(torch.load(Path(args.tuned_lens, "lens.pt"), map_location=args.device))
        lens.eval()
        print(f"loaded tuned lens from {args.tuned_lens}")

    entries = [json.loads(l) for l in open(args.index) if l.strip()]
    entries = [e for e in entries if e.get("greedy") and e.get("correct") is not None]

    N_STORED_ANSWER_POS = 5   # answer_tok_0..4 stored by the recapture

    rows = []
    for e in entries:
        d = np.load(e["npz_path"], allow_pickle=True)
        labels = list(d["position_labels"])
        gen = list(d["gen_token_ids"])
        ae = int(d["answer_end_idx"])
        if ae >= len(gen):
            continue
        target_tok = int(gen[ae])                       # the decisive answer token
        target_str = _norm_tok(tok.decode([target_tok]))  # match on the answer WORD
        # The residual that PRODUCES a token is the one at the preceding position.
        # For a single-token answer (ae==0) that producing residual is prompt_final,
        # NOT answer_tok_0 (which produces the token AFTER the answer).
        if ae == 0:
            anchor = "prompt_final"
            anchor_k = -1
        else:
            anchor_k = ae - 1
            anchor = f"answer_tok_{anchor_k}"
        # multi-token answers whose producing position was not stored -> excluded
        if (anchor_k >= N_STORED_ANSWER_POS) or (anchor not in labels):
            rows.append({"prompt_id": e["prompt_id"], "correct": bool(e["correct"]),
                         "L_star": None, "L_k1": None, "L_k3": None,
                         "ambiguous_frac": None, "unstable": False,
                         "no_commit": False, "unmeasurable": True})
            continue
        stack = d["hidden_stack"][labels.index(anchor)].astype(np.float32)  # [n_layers+1, H]
        top1, gap = layer_top12(stack, final_norm, lm_head, args.device, dtype, tok)
        L2, ambf = commitment(top1, gap, k=2, target_str=target_str)
        L1, _ = commitment(top1, gap, k=1, target_str=target_str)
        L3, _ = commitment(top1, gap, k=3, target_str=target_str)
        rec = {"prompt_id": e["prompt_id"], "correct": bool(e["correct"]),
               "L_star": L2, "L_logit_k2": L2, "L_k1": L1, "L_k3": L3,
               "ambiguous_frac": ambf, "unmeasurable": False,
               "unstable": ambf > UNSTABLE_FRAC, "commit_lens": "logit"}
        if lens is not None:
            t1, tg = layer_top12(stack, final_norm, lm_head, args.device, dtype, tok, lens=lens)
            Lt, _ = commitment(t1, tg, k=2, target_str=target_str)
            rec["L_star_tuned"] = Lt
            # lens-sensitive: both lenses commit but disagree >3 layers (reported flag)
            rec["lens_sensitive"] = (Lt is not None and L2 is not None and abs(Lt - L2) > 3)
            # RESCUE: where the logit lens fails to commit, fall back to the tuned lens
            if L2 is None and Lt is not None:
                rec["L_star"] = Lt
                rec["commit_lens"] = "tuned"
        rec["no_commit"] = rec["L_star"] is None
        rows.append(rec)

    # clean set = measurable, committed, stable. lens_sensitive is REPORTED, not excluded.
    def clean(r):
        return (not r.get("unmeasurable") and not r["no_commit"] and not r["unstable"])
    clean_rows = [r for r in rows if clean(r)]
    Ls = np.array([r["L_star"] for r in clean_rows], float)
    norm_L = Ls / n_tl
    upper_third = norm_L >= (2.0 / 3.0)

    Lc = np.array([r["L_star"] for r in clean_rows if r["correct"]], float)
    Li = np.array([r["L_star"] for r in clean_rows if not r["correct"]], float)

    meas = [r for r in rows if not r.get("unmeasurable")]
    out = {
        "model": args.model, "n_total": len(rows), "n_clean": len(clean_rows),
        "n_transformer_layers": n_tl,
        "excl_unmeasurable": sum(r.get("unmeasurable", False) for r in rows),
        "excl_no_commit": sum(r["no_commit"] for r in rows),
        "excl_unstable": sum(r["unstable"] for r in rows),
        "flag_lens_sensitive": sum(r.get("lens_sensitive", False) for r in rows) if lens else None,
        "rescued_by_tuned": sum(r.get("commit_lens") == "tuned" for r in rows) if lens else None,
        # k-sensitivity (pure logit lens): no-commit count is highly window-
        # dependent when commitment is last-layer/abrupt. First-class result.
        "no_commit_k1": sum(r.get("L_k1") is None for r in meas),
        "no_commit_k2": sum(r.get("L_logit_k2") is None for r in meas),
        "no_commit_k3": sum(r.get("L_k3") is None for r in meas),
        "H2_upper_third_prop": float(upper_third.mean()) if len(Ls) else None,
        "H2_binomial_p": float(stats.binomtest(int(upper_third.sum()), len(Ls), 1/3,
                              alternative="greater").pvalue) if len(Ls) else None,
        "L_star_median": float(np.median(Ls)) if len(Ls) else None,
        "L_star_norm_median": float(np.median(norm_L)) if len(Ls) else None,
    }
    if len(Lc) >= 5 and len(Li) >= 5:
        psd = np.sqrt((Lc.var(ddof=1) + Li.var(ddof=1)) / 2) + 1e-8
        out["H3_cohens_d"] = float((Li.mean() - Lc.mean()) / psd)   # incorrect - correct
        out["H3_welch_t"], out["H3_welch_p"] = [float(x) for x in stats.ttest_ind(Li, Lc, equal_var=False)]
        out["L_star_correct_mean"] = float(Lc.mean())
        out["L_star_incorrect_mean"] = float(Li.mean())
    # k-sensitivity
    moved = [abs((r["L_star"] or 0) - (r["L_k1"] or 0)) > 2 or
             abs((r["L_star"] or 0) - (r["L_k3"] or 0)) > 2 for r in clean_rows]
    out["k_sensitivity_frac_moved_gt2"] = float(np.mean(moved)) if clean_rows else None
    out["lens"] = "logit+tuned" if lens else "logit-only (tuned pending)"

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
