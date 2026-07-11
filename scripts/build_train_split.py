"""
build_train_split.py

train_qids_<model>_v2.json for the CETT detector, per reg 6.3.
Reads the T=1.0 collection (NOT rp_clean, which is the 0.7/0.9 H1 run).
  t iff n_correct/n >= 0.8   (8/10 consistency)
  f iff n_correct/n <= 0.2
  middle dropped
intersect with successful answer-token extraction, then balance (seed 42).
"""
import json, argparse, random

def answer_token_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(next(iter(json.loads(line).keys())))
    return ids

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True)
    p.add_argument("--answer_tokens", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hi", type=float, default=0.8)
    p.add_argument("--lo", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    at = answer_token_ids(args.answer_tokens)
    t, f, mid, incomplete = [], [], 0, 0
    with open(args.responses) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid = next(iter(d))
            rec = d[qid]
            judges = rec.get("judges", [])
            n = len(judges)
            if n == 0:
                incomplete += 1
                continue
            frac = judges.count("true") / n
            if qid not in at:
                continue
            if frac >= args.hi:
                t.append(qid)
            elif frac <= args.lo:
                f.append(qid)
            else:
                mid += 1

    random.seed(args.seed)
    k = min(len(t), len(f))
    random.shuffle(t); random.shuffle(f)
    t, f = sorted(t[:k]), sorted(f[:k])
    json.dump({"t": t, "f": f}, open(args.output, "w"), indent=2)
    print(f"answer-token ids: {len(at)}  incomplete: {incomplete}")
    print(f"t: {len(t)}  f: {len(f)}  dropped middle: {mid}  balanced to {k}/class")
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
