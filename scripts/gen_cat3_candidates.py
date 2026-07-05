#!/usr/bin/env python3
"""
gen_cat3_candidates.py  (v2 - pronounceable phonotactics)
Deterministic construction of fabricated-entity (Cat3) candidates.

Method: phonotactically-plausible invented names built from simple CV(C)
syllables plus real-sounding place/name suffixes, crossed with domain-specific
question frames. Names are designed to sound like they COULD be real
(Vestronia, Lake Thandor) rather than alien gibberish. Fully deterministic
under --seed. Dedupes against existing cat3 and Cat1/Cat2 strings on disk.
The instruct SE screen (refusal-aware gate) is the primary real-referent guard.

Output schema matches existing cat3_candidates.jsonl:
  prompt_id, question, category="3", domain, source="fabricated", fabricated_name
"""
import json, random, argparse, os, re
from collections import Counter

# --- pronounceable inventories ---
# single-consonant onsets + a few common clusters that ARE pronounceable
ONSETS = ["b","d","f","g","k","l","m","n","p","r","s","t","v","z","th","br",
          "dr","tr","gr","fl","gl","pl","kr","st","sk","vr","str","cl","sh"]
VOWELS = ["a","e","i","o","u","a","e","i","o"]   # weight toward common vowels
# codas: sonorants and simple single consonants only (keeps names sayable)
CODAS  = ["n","r","l","s","m","th","","","",""]   # bias toward open syllables

# real-sounding terminal morphemes that make a name read as a place/entity
PLACE_SUFFIX  = ["ia","ovia","landia","stan","heim","gard","wick","mor","dor",
                 "ess","aria","onia","este","vik","holm","burg","thal"]
PERSON_SUFFIX = ["ov","enko","sson","ai","eux","ard","ison","escu","opal","ner",
                 "as","ic","ek","oja"]
THING_SUFFIX  = ["ium","ite","ol","ase","on","yl","ine","ate","ox","yr"]

DOMAINS = {
    "geography": [
        "What is the capital of {N}?",
        "What is the population of {N}?",
        "Which river flows through {N}?",
        "What is the highest mountain in {N}?",
        "On which continent is {N} located?",
        "What currency is used in {N}?",
        "What is the official language of {N}?",
    ],
    "people": [
        "In what year was {N} born?",
        "What did {N} win the Nobel Prize for?",
        "Which novel did {N} write?",
        "What instrument did {N} play?",
        "What theory is {N} best known for?",
        "In which city did {N} die?",
        "What was {N}'s most famous work?",
    ],
    "science": [
        "What is the atomic number of {N}?",
        "What is the chemical formula of {N}?",
        "Who discovered {N}?",
        "What is the boiling point of {N}?",
        "What is the half-life of {N}?",
        "In which taxonomic order is the {N} classified?",
        "What is the primary industrial use of {N}?",
    ],
    "culture": [
        "Who composed the opera {N}?",
        "In what year was the film {N} released?",
        "Who directed {N}?",
        "Which author wrote the novel {N}?",
        "Who painted {N}?",
        "In which museum is {N} displayed?",
        "What award did the film {N} receive?",
    ],
}

SUFFIX = " Respond with the answer only, without any explanation."

def syllable(rng, allow_coda=True):
    o = rng.choice(ONSETS)
    v = rng.choice(VOWELS)
    c = rng.choice(CODAS) if allow_coda else ""
    return o + v + c

def make_name(rng, kind):
    # 1-2 core syllables + a real-sounding suffix -> pronounceable, plausible
    n_core = rng.choice([1, 2, 2])
    core = ""
    for i in range(n_core):
        core += syllable(rng, allow_coda=(i < n_core - 1))  # last core syl open before suffix
    if kind == "geography":
        suf = rng.choice(PLACE_SUFFIX)
    elif kind == "people":
        suf = rng.choice(PERSON_SUFFIX)
    elif kind == "science":
        suf = rng.choice(THING_SUFFIX)
    else:  # culture: mix of place/person-ish titles
        suf = rng.choice(PLACE_SUFFIX + PERSON_SUFFIX)
    name = core + suf
    # avoid awkward triple letters
    name = re.sub(r"(.)\1\1+", r"\1\1", name)
    name = name[0].upper() + name[1:]
    # people frequently get a given name
    if kind == "people" and rng.random() < 0.75:
        given = rng.choice(["Alden","Verena","Marek","Korin","Elyse","Dristan",
                            "Thalia","Renard","Silva","Boren","Ivo","Nela","Cael","Rosa"])
        name = given + " " + name
    if kind == "culture" and rng.random() < 0.4:
        name = "The " + name
    return name

# small stoplist of real short words the syllable generator can accidentally hit
STOPWORDS = {
    "koon","gaol","paard","goon","moon","noon","soon","boon","loon","toon",
    "roar","soar","boar","gaon","naan","koan","loan","moan","roan","bean",
    "lean","mean","dean","peon","neon","aeon","onion","union","ration","nation",
    "salon","talon","melon","felon","baron","apron","moron","canon","demon",
    "lemon","colon","vison","bison","raven","haven","maven","siren","liven",
    "oval","opal","vital","tidal","modal","nodal","penal","renal","venal",
}

def is_realish(name):
    bare = name.replace("The ", "").split()[-1].lower()
    if bare in STOPWORDS:
        return True
    return False

def load_guard(paths):
    seen = set()
    for p in paths:
        if not os.path.exists(p): continue
        for line in open(p):
            try: d = json.loads(line)
            except Exception: continue
            for key in ("question","fabricated_name"):
                if isinstance(d.get(key), str): seen.add(d[key].lower())
            if isinstance(d.get("ground_truth"), list):
                for g in d["ground_truth"]: seen.add(str(g).lower())
    return seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_per_domain", type=int, default=90)
    ap.add_argument("--start_idx", type=int, default=1000)
    ap.add_argument("--output_path", type=str, default="data/cat3_candidates_pull2.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    guard = load_guard(["data/cat3_candidates.jsonl","data/cat1_candidates.jsonl",
                        "data/cat1_candidates_pull2.jsonl","data/cat2_candidates.jsonl"])
    used = set(); rows = []; idx = args.start_idx
    for domain, frames in DOMAINS.items():
        made = attempts = 0
        while made < args.n_per_domain and attempts < args.n_per_domain*80:
            attempts += 1
            name = make_name(rng, domain)
            key = name.lower()
            bare = name.replace("The ","").split()[-1]
            if key in used or key in guard: continue
            if is_realish(name): continue
            min_len = 6 if domain in ("science","people") else 5
            if len(bare) < min_len or len(bare) > 13: continue   # length sanity
            used.add(key)
            frame = rng.choice(frames)
            rows.append({"prompt_id": f"cat3_{idx:04d}",
                         "question": frame.replace("{N}", name)+SUFFIX,
                         "category":"3","domain":domain,"source":"fabricated",
                         "fabricated_name":name})
            idx += 1; made += 1

    with open(args.output_path,"w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    dc = Counter(r["domain"] for r in rows)
    print(f"wrote {len(rows)} cat3 candidates to {args.output_path}")
    print(f"domain balance: {dict(dc)}")
    print(f"prompt_id range: {rows[0]['prompt_id']} .. {rows[-1]['prompt_id']}")
    print("samples across domains:")
    for dom in DOMAINS:
        exs = [r for r in rows if r["domain"]==dom][:4]
        for r in exs:
            print(f"  {dom[:4]} | {r['fabricated_name']!r:28} | {r['question'].split('Respond')[0].strip()[:50]}")

if __name__ == "__main__":
    main()
