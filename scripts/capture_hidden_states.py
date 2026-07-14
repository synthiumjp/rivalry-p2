"""
capture_hidden_states.py

H5 5b capture. Raw residual-stream hidden states at the single last-answer-token
position, at (a) a set of "source" layers and (b) the last layer.

Source layers come from EITHER the detector's H-Neuron layers (--detector) OR an
explicit list (--layers), the latter for the random-layer control that tests
whether H-Neuron/last-layer alignment is H-Neuron-specific or a generic
residual-stream sharing artifact.

Boundary logic identical to extract_activations_instruct.py. Single forward pass
with output_hidden_states=True (no generation, no MPS stall).

Output per item: act_<qid>.npz with:
  hneuron : (n_src_layers * hidden,)  concat residual at the source layers
  last    : (hidden,)                 residual at final layer
  y       : 0 (correct) / 1 (halluc)

Usage (H-Neuron layers, from detector):
  python scripts/capture_hidden_states.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.3 \
    --input_path data/answer_tokens_mistral_v2.jsonl \
    --train_ids_path data/train_qids_mistral_v2.json \
    --detector models/detector_mistral_v2_last.pkl \
    --intermediate_dim 14336 \
    --output_root data/hidden_5b_mistral_v2 \
    --dtype float16 --attn sdpa

Usage (random-layer control, explicit layers):
  python scripts/capture_hidden_states.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.3 \
    --input_path data/answer_tokens_mistral_v2.jsonl \
    --train_ids_path data/train_qids_mistral_v2.json \
    --layers 2 5 7 10 14 16 21 24 26 \
    --output_root data/hidden_5b_mistral_v2_randctrl \
    --dtype float16 --attn sdpa
"""
import os, json, argparse
import numpy as np, torch, joblib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--input_path", required=True)
    p.add_argument("--train_ids_path", required=True)
    p.add_argument("--detector", default=None,
                   help="detector .pkl; nonzero-coef layers define the source layers")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="explicit source block indices (overrides --detector); "
                        "for the random-layer control")
    p.add_argument("--intermediate_dim", type=int, default=None,
                   help="required with --detector, to decode layer from flat coef index")
    p.add_argument("--output_root", required=True)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    return p.parse_args()


def detector_layers(detector_path, inter_dim):
    c = joblib.load(detector_path)
    w = np.asarray(c.coef_).ravel()
    nz = np.where(w != 0)[0]
    return sorted(set(int(i // inter_dim) for i in nz))


def main():
    a = parse_args()

    if a.layers is not None:
        src_layers = sorted(set(a.layers))
        src_desc = "explicit"
    elif a.detector is not None:
        if a.intermediate_dim is None:
            raise ValueError("--intermediate_dim required with --detector")
        src_layers = detector_layers(a.detector, a.intermediate_dim)
        src_desc = "detector H-Neuron"
    else:
        raise ValueError("give either --detector (+ --intermediate_dim) or --layers")
    print(f"source layers ({src_desc}, block indices): {src_layers}")

    dtype = torch.float16 if a.dtype == "float16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=dtype, device_map="auto",
        attn_implementation=a.attn, output_hidden_states=True)
    model.eval()

    # hidden_states is a tuple length n_layers+1: index 0 = embeddings,
    # index L+1 = output of block L. Source block L -> hidden_states[L+1].
    hs_idx = [L + 1 for L in src_layers]

    with open(a.train_ids_path) as f:
        idm = json.load(f)
    target = set(idm["t"] + idm["f"])
    label = {q: 0 for q in idm["t"]}
    label.update({q: 1 for q in idm["f"]})

    os.makedirs(a.output_root, exist_ok=True)

    with open(a.input_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    n = 0
    for s in tqdm(samples, desc="capture 5b hidden"):
        qid = next(iter(s))
        if qid not in target:
            continue
        r = s[qid]
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["question"]}],
            tokenize=False, add_generation_prompt=True)
        full = prompt + r["response"]
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        plen = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
        pos = ids.shape[1] - 1
        if pos < plen:
            continue

        with torch.no_grad():
            out = model(ids)
        hs = out.hidden_states

        src = np.concatenate([hs[j][0, pos, :].float().cpu().numpy() for j in hs_idx])
        last = hs[-1][0, pos, :].float().cpu().numpy()

        np.savez(os.path.join(a.output_root, f"act_{qid}.npz"),
                 hneuron=src, last=last, y=label[qid])
        n += 1

    with open(os.path.join(a.output_root, "_meta.json"), "w") as f:
        json.dump({"source_layers": src_layers, "source_desc": src_desc,
                   "hs_index": hs_idx, "n_items": n}, f, indent=2)
    print(f"captured {n} items; source layers {src_layers} ({src_desc})")


if __name__ == "__main__":
    main()
