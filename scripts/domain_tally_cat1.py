#!/usr/bin/env python3
"""
domain_tally_cat1.py

APPROXIMATE domain tally over the Cat1 development set, to check whether the
TriviaQA-only build under-fills the registered science/biomedical domain (the
registration names TriviaQA + NQ + BioASQ; the build is TriviaQA-only).

This is a heuristic keyword classifier over question text, NOT ground-truth
domain labels. TriviaQA carries no clean domain field. Use the output only to
answer "is biomedical near zero" and to size the source-deviation note. Do not
report these as precise domain proportions.

Run from repo root:
  python scripts/domain_tally_cat1.py
"""
import json, re
from collections import Counter
from pathlib import Path

DATA = Path("data")
CAND = DATA / "cat1_candidates_pull2.jsonl"
DEV  = DATA / "cat1_development_prompt_ids.json"

# ordered: first match wins, so put specific domains before broad ones
DOMAINS = [
    ("biomedical", r"\b(disease|syndrome|virus|bacteri|cancer|tumou?r|vaccine|"
                   r"anatomy|hormone|enzyme|protein|gene|dna|rna|blood|organ|"
                   r"muscle|bone|nerve|brain|heart|drug|medicine|medical|"
                   r"symptom|diagnos|surgery|antibiotic|vitamin)\b"),
    ("science",    r"\b(element|atom|molecul|chemical|physic|force|energy|"
                   r"planet|star|galaxy|orbit|gravity|electron|proton|"
                   r"equation|theorem|speed of light|periodic table|isotope|"
                   r"compound|reaction|acid|metal|temperature|boiling|melting)\b"),
    ("geography",  r"\b(capital|country|countries|river|mountain|ocean|sea|"
                   r"continent|city|island|border|lake|desert|population|"
                   r"located|flag|currency)\b"),
    ("history",    r"\b(war|battle|king|queen|emperor|president|century|"
                   r"ancient|empire|revolution|treaty|dynasty|invasion|"
                   r"founded|independence|assassinat|reign)\b"),
    ("sport",      r"\b(olympic|world cup|championship|football|cricket|tennis|"
                   r"golf|boxing|athlete|team|league|medal|tournament|player|"
                   r"scored|goal|race)\b"),
    ("arts_media", r"\b(novel|author|book|poem|play|painting|artist|composer|"
                   r"symphony|opera|album|song|band|singer|film|movie|actor|"
                   r"director|character|wrote|starred|directed)\b"),
]
COMPILED = [(name, re.compile(pat, re.I)) for name, pat in DOMAINS]

def load_jsonl(p):
    for line in open(p):
        line = line.strip()
        if line:
            yield json.loads(line)

def classify(q):
    for name, rx in COMPILED:
        if rx.search(q):
            return name
    return "other"

dev_ids = set(json.loads(DEV.read_text())["prompt_ids"])
cand = {r["prompt_id"]: r for r in load_jsonl(CAND) if r.get("prompt_id") in dev_ids}
assert len(cand) == len(dev_ids), f"have {len(cand)} of {len(dev_ids)}"

counts = Counter()
examples = {}
for pid, r in cand.items():
    d = classify(r.get("question", ""))
    counts[d] += 1
    examples.setdefault(d, r.get("question", "")[:80])

n = len(cand)
print(f"APPROXIMATE domain tally, Cat1 development (n={n}). Heuristic, not labels.\n")
for d, c in counts.most_common():
    print(f"  {d:12s} {c:4d}  {100*c/n:5.1f}%   e.g. {examples[d]}")

bio = counts.get("biomedical", 0)
sci = counts.get("science", 0)
print(f"\nbiomedical={bio} ({100*bio/n:.1f}%)  science={sci} ({100*sci/n:.1f}%)  "
      f"science+bio={bio+sci} ({100*(bio+sci)/n:.1f}%)")
print("read: if biomedical is a few percent or less, the BioASQ drop left the "
      "registered biomedical domain effectively unrepresented -> limitation "
      "sentence; if you want it represented, backfill from NQ/BioASQ.")
