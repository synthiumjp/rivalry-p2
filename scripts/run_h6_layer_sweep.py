"""
run_h6_layer_sweep.py

H6 first causal read (Alternative A, bifurcation, layer sweep).

For each bifurcating prompt: resample the first generated token to find a branch
that yields a CORRECT answer (donor/clean) and one that yields a HALLUCINATED
answer (recipient/corrupted). Denoising patch: overwrite the corrupted branch's
residual at the first generated position, at layer L, with the clean branch's,
then continue greedily and judge. Correction rate per layer.

Tests the depth confound: L* (Mistral 31) sits on the late cv-r plateau; the probe
median is 19. If correction rate rises monotonically with layer, the "regime"
advantage is a depth artifact. If it spikes at L* above depth-matched neighbours,
it is regime-specific.

Patch is via forward hook on model.model.layers[L] output residual, at the single
first-generated position, magnitude 1.0x (full replacement). Judge = rule judge
(substring, normalised).

Usage:
  python scripts/run_h6_layer_sweep.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.3 \
    --bifurcation data/h6_bifurcation_mistral_v2.json \
    --source data/cat1_candidates_pull2.jsonl \
    --sweep_layers 10 15 19 25 29 31 \
    --n_prompts 30 \
    --output data/h6_sweep_mistral_v2.json \
    --dtype float16 --attn sdpa
"""
import os, json, argparse, re
import numpy as np, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--bifurcation", required=True, help="h6_bifurcation_<model>.json")
    p.add_argument("--source", required=True, help="cat1_candidates_pull2.jsonl (question + ground_truth)")
    p.add_argument("--sweep_layers", type=int, nargs="+", required=True)
    p.add_argument("--n_prompts", type=int, default=30)
    p.add_argument("--output", required=True)
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--branch_samples", type=int, default=24,
                   help="first-token samples to find a correct and a halluc branch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    return p.parse_args()


def normalize(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def judge_correct(text, ground_truth):
    t = normalize(text)
    for gt in ground_truth if isinstance(ground_truth, list) else [ground_truth]:
        g = normalize(str(gt))
        if g and g in t:
            return True
    return False


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    dtype = torch.float16 if a.dtype == "float16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=dtype, device_map="auto", attn_implementation=a.attn)
    model.eval()
    device = model.device

    # ground truth + question text
    meta = {}
    for line in open(a.source):
        e = json.loads(line)
        meta[e["prompt_id"]] = {"question": e["question"],
                                "ground_truth": e.get("ground_truth")}

    bif = json.load(open(a.bifurcation))["per_prompt"]
    # bifurcating + pairable prompts, capped
    cand = [q for q, r in bif.items()
            if r.get("bifurcating") and r.get("pairable") and q in meta]
    cand = cand[:a.n_prompts]
    print(f"candidate bifurcating+pairable prompts: {len(cand)}")

    layers_module = model.model.layers  # Mistral/Llama/Qwen: .model.layers[L]

    # patching hook state
    patch_state = {"active": False, "layer": None, "pos": None, "vec": None}

    def make_hook(layer_idx):
        def hook(mod, inp, out):
            if not patch_state["active"] or patch_state["layer"] != layer_idx:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            hidden[:, patch_state["pos"], :] = patch_state["vec"].to(hidden.dtype)
            if isinstance(out, tuple):
                return (hidden,) + tuple(out[1:])
            return hidden
        return hook

    handles = [layers_module[L].register_forward_hook(make_hook(L))
               for L in a.sweep_layers]

    def gen_from_first_token(ids, first_tok, cache_layer=None, cache_pos=None):
        """Greedy generate after forcing first_tok; optionally cache residual at
        (cache_layer, cache_pos). Returns text and (if requested) cached vec."""
        cached = {}
        if cache_layer is not None:
            def cache_hook(mod, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                cached["vec"] = hidden[:, cache_pos, :].detach().clone()
                return out
            ch = layers_module[cache_layer].register_forward_hook(cache_hook)
        seq = torch.cat([ids, torch.tensor([[first_tok]], device=device)], 1)
        with torch.no_grad():
            for _ in range(a.max_new_tokens):
                logits = model(seq).logits[0, -1, :]
                nxt = int(torch.argmax(logits))
                seq = torch.cat([seq, torch.tensor([[nxt]], device=device)], 1)
                if nxt == tok.eos_token_id:
                    break
        if cache_layer is not None:
            ch.remove()
        text = tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True)
        return text, cached.get("vec")

    results = {}
    layer_correct = {L: 0 for L in a.sweep_layers}
    layer_total = {L: 0 for L in a.sweep_layers}
    n_usable = 0

    for qid in tqdm(cand, desc="h6 sweep"):
        gt = meta[qid]["ground_truth"]
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": meta[qid]["question"]}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        plen = ids.shape[1]
        pos = plen  # first generated position (0-indexed into full seq)

        # sample first tokens, find a correct-branch and a halluc-branch
        with torch.no_grad():
            logits0 = model(ids).logits[0, -1, :]
        probs = torch.softmax(logits0.float(), -1)
        sampled = torch.multinomial(probs, a.branch_samples, replacement=True).tolist()
        first_toks = list(dict.fromkeys(sampled))  # unique, keep order

        clean_tok = corrupt_tok = None
        clean_text = corrupt_text = None
        for ft in first_toks:
            txt, _ = gen_from_first_token(ids, ft)
            ok = judge_correct(txt, gt)
            if ok and clean_tok is None:
                clean_tok, clean_text = ft, txt
            if (not ok) and corrupt_tok is None:
                corrupt_tok, corrupt_text = ft, txt
            if clean_tok is not None and corrupt_tok is not None:
                break
        if clean_tok is None or corrupt_tok is None:
            results[qid] = {"usable": False,
                            "reason": "no clean+corrupt branch pair found"}
            continue
        n_usable += 1

        rec = {"usable": True, "clean_first": clean_tok, "corrupt_first": corrupt_tok,
               "clean_text": clean_text[:120], "corrupt_text": corrupt_text[:120],
               "patched": {}}

        for L in a.sweep_layers:
            # cache clean-branch residual at (L, first-gen pos)
            _, clean_vec = gen_from_first_token(ids, clean_tok, cache_layer=L, cache_pos=pos)
            # patch it into the corrupt branch at (L, first-gen pos)
            patch_state.update({"active": True, "layer": L, "pos": pos, "vec": clean_vec})
            ptxt, _ = gen_from_first_token(ids, corrupt_tok)
            patch_state["active"] = False
            corrected = judge_correct(ptxt, gt)
            rec["patched"][str(L)] = {"corrected": corrected, "text": ptxt[:120]}
            layer_total[L] += 1
            layer_correct[L] += int(corrected)

        results[qid] = rec

    for h in handles:
        h.remove()

    curve = {str(L): (layer_correct[L] / layer_total[L] if layer_total[L] else None)
             for L in a.sweep_layers}
    summary = {
        "n_candidate": len(cand),
        "n_usable": n_usable,
        "sweep_layers": a.sweep_layers,
        "correction_rate_by_layer": curve,
        "note": ("H6 Alt-A bifurcation layer sweep, dev rehearsal. Monotonic rise "
                 "with depth = depth artifact; spike at L* above neighbours = "
                 "regime-specific. Confirmatory is hold-out."),
    }
    json.dump({"summary": summary, "per_prompt": results}, open(a.output, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
