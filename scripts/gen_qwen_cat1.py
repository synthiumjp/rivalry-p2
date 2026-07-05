import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N = 20
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
model.eval()

# Load Cat1 prompts
prompts = []
for line in open("data/benchmark_final_250.jsonl"):
    e = json.loads(line)
    if e["category"] == "1" and "ground_truth" in e:
        prompts.append(e)
print(f"{len(prompts)} Cat1 prompts")

def ok(t, al):
    t = t.lower().strip()
    return any(a.lower() in t or t in a.lower() for a in al if len(a) > 1)

out = open("data/responses_qwen_instruct.jsonl", "w")
for e in tqdm(prompts):
    msgs = [{"role": "user", "content": e["question"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(device)
    resps = []
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=100, do_sample=True,
                             temperature=0.7, top_p=0.9, num_return_sequences=N,
                             pad_token_id=tok.eos_token_id)
    for g in gen:
        r = tok.decode(g[inp["input_ids"].shape[1]:], skip_special_tokens=True)
        resps.append(r)
    nc = sum(ok(r, e["ground_truth"]) for r in resps)
    out.write(json.dumps({
        "prompt_id": e["prompt_id"],
        "question": e["question"],
        "ground_truth": e["ground_truth"],
        "responses": resps,
        "n_correct": nc,
        "h_rate": 1.0 - nc / N
    }) + "\n")
    out.flush()
out.close()
print("Done. Saved data/responses_qwen_instruct.jsonl")
