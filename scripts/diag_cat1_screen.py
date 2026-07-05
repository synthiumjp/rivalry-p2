#!/usr/bin/env python3
"""
diag_cat1_screen.py

Quantify the damage from the 18/20 correctness screen on Cat1, and characterise
an SE-free answer-distribution consistency measure, to ground the B-vs-C re-draw
decision. Reads only, no GPU, no writes.

Inputs (run from ~/jpwork/rivalry-p2/H-Neurons/):
  data/cat1_candidates_pull2.jsonl     ground truth per prompt_id
  data/se_cat1_pull2_instruct.jsonl    20 completions per prompt_id
  data/cat1_benchmark_1000_ids.json    the drawn 1000 (kept set)

Prints:
  - r_p distribution on the KEPT 1000 (expect floor-bunched near 0: the range
    restriction on the Mistral-instruct confirmatory cell)
  - gold-item loss among DISCARDED, split into:
      consistent-wrong  (r_p high, modal_freq high)  systematic-error gold
      variable-wrong    (r_p high, modal_freq low)   fabrication gold  <- C also drops these
  - modal_freq distribution (the SE-free consistency measure)
  - cross-tab: 18/20 correctness gate vs consistency gate (they select different sets)

NOTE ON SCORING: this reproduces the reg's judge (bidirectional substring, min 2
chars, GT = value + normalized_value + aliases). Confirm it matches your
reference inline Cat1 scorer; if that used word-boundary matching, mirror it here
so the KEPT set reproduces exactly.
"""
import json, re
from collections import Counter
from pathlib import Path

DATA = Path("data")
CAND = DATA / "cat1_candidates_pull2.jsonl"
SE   = DATA / "se_cat1_pull2_instruct.jsonl"
KEPT = DATA / "cat1_benchmark_1000_ids.json"

# high-r_p cut for "gold" and consistency cut separating the two gold modes
R_GOLD   = 0.75
MODE_CUT = 0.50

_LEAD = re.compile(r"^(the answer is|answer\s*[:\-]?|it is|it's)\s+", re.I)

def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_jsonl(p: Path):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def comp_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return c.get("text") or c.get("completion") or c.get("output") or ""
    return str(c)

def gt_set(row):
    g = row.get("ground_truth", row.get("answer"))
    out = set()
    def add(x):
        if isinstance(x, str) and x.strip():
            out.add(norm(x))
    if isinstance(g, dict):
        add(g.get("value")); add(g.get("normalized_value"))
        for a in (g.get("aliases") or []):
            add(a)
    elif isinstance(g, list):
        for a in g:
            add(a)
    else:
        add(g)
    return {x for x in out if len(x) >= 2}

def is_correct(comp_norm, gts):
    if len(comp_norm) < 2:
        return False
    for a in gts:
        if a in comp_norm or comp_norm in a:
            return True
    return False

def answer_key(comp):
    line = next((l for l in comp.splitlines() if l.strip()), comp)
    line = _LEAD.sub("", line.strip())
    return norm(line)

# --- load GT and kept ids ---
gt = {}
for r in load_jsonl(CAND):
    pid = r.get("prompt_id")
    if pid:
        gt[pid] = gt_set(r)

kj = json.loads(KEPT.read_text())
if isinstance(kj, list):
    kept_ids = set(kj)
else:
    kept_ids = set(kj.get("prompt_ids") or kj.get("ids") or [])

# --- score every SE row ---
rows = []  # (pid, n, r_p, modal_freq, kept)
skipped = 0
for r in load_jsonl(SE):
    pid = r.get("prompt_id")
    comps = [comp_text(c) for c in (r.get("completions") or [])]
    comps = [c for c in comps if c.strip()]
    if pid not in gt or not comps:
        skipped += 1
        continue
    gts = gt[pid]
    n = len(comps)
    n_correct = sum(is_correct(norm(c), gts) for c in comps)
    r_p = 1.0 - n_correct / n
    keys = [answer_key(c) for c in comps]
    modal_freq = Counter(keys).most_common(1)[0][1] / n
    rows.append((pid, n, r_p, modal_freq, pid in kept_ids))

def hist(vals, edges):
    c = Counter()
    for v in vals:
        for e in edges:
            if v <= e + 1e-9:
                c[e] += 1
                break
    return c

kept = [x for x in rows if x[4]]
disc = [x for x in rows if not x[4]]

print(f"scored={len(rows)}  skipped(no gt/comps)={skipped}  "
      f"kept(drawn)={len(kept)}  discarded={len(disc)}")

print("\n--- r_p distribution, KEPT 1000  (range restriction on the confirmatory cell) ---")
edges = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
h = hist([x[2] for x in kept], edges)
for e in edges:
    print(f"  r_p<={e:.2f}: {h.get(e,0):4d}")

print("\n--- DISCARDED: gold-item loss ---")
cw  = [x for x in disc if x[2] >= R_GOLD and x[3] >= MODE_CUT]
vw  = [x for x in disc if x[2] >= R_GOLD and x[3] <  MODE_CUT]
mid = [x for x in disc if x[2] <  R_GOLD]
print(f"  discarded total:                              {len(disc):4d}")
print(f"  consistent-wrong (r_p>={R_GOLD}, mode>={MODE_CUT}):  {len(cw):4d}   gold, systematic (18/20 dropped, C keeps)")
print(f"  variable-wrong   (r_p>={R_GOLD}, mode<{MODE_CUT}):   {len(vw):4d}   gold, fabrication (18/20 AND C drop) <-- the B-vs-C number")
print(f"  mid r_p (<{R_GOLD}):                              {len(mid):4d}")

print("\n--- modal_freq distribution, ALL scored (the SE-free consistency measure) ---")
edges2 = [0.25, 0.50, 0.75, 0.90, 1.0]
h2 = hist([x[3] for x in rows], edges2)
for e in edges2:
    print(f"  modal_freq<={e:.2f}: {h2.get(e,0):4d}")

print("\n--- cross-tab: 18/20 correctness gate vs consistency gate (mode>=k) ---")
for k in (0.60, 0.70, 0.80):
    pass18 = sum(1 for x in rows if (1 - x[2]) * x[1] >= 18)
    passC  = sum(1 for x in rows if x[3] >= k)
    both   = sum(1 for x in rows if x[3] >= k and (1 - x[2]) * x[1] >= 18)
    consC_not18 = sum(1 for x in rows if x[3] >= k and (1 - x[2]) * x[1] < 18)
    print(f"  k={k:.2f}: 18/20-pass={pass18:4d}  cons-pass={passC:4d}  both={both:4d}  "
          f"cons-admits-but-18/20-drops={consC_not18:4d}")
