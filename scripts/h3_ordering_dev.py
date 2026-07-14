#!/usr/bin/env python3
"""
h3_ordering_dev.py
H3 (commitment ordering): incorrect answers commit LATER than correct.
Reg analysis H3: Welch t of L* (incorrect vs correct) per model; confirmed if
pooled Cohen's d > 0.3 OR per-model d > 0.3 in >=3/5. Directional, one-tailed.

SIGN CONVENTION (critical): the registered direction is "incorrect commits
later", so we compute d = (mean L*_incorrect - mean L*_correct) / pooled_sd.
POSITIVE d = registered direction (incorrect later) = supports H3.
NEGATIVE d = reversal (incorrect commits EARLIER). The Step-0 spike and the
development record both show reversals on instruct; a negative d here is the
expected, pre-documented disconfirmation, not a bug.

Reg exclusions (commitment analyses): drop L*-unstable prompts
(unstable / no_commit / unmeasurable flags), report exclusion rate per model.

Development freeze artifact only. Hold-out sealed.

Input:  data/commitment_rows_<tag>.jsonl  (tracked; has correct, L_star, flags)
Output: data/h3_ordering_dev.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path("data")


def load_rows(tag):
    rows = []
    for line in open(DATA / f"commitment_rows_{tag}.jsonl"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def excluded(r):
    # reg L*-stability exclusions for commitment analyses
    return bool(r.get("unstable")) or bool(r.get("no_commit")) or bool(r.get("unmeasurable"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    models = []
    ds = []
    for tag in args.tags:
        rows = load_rows(tag)
        n_total = len(rows)
        kept = [r for r in rows if not excluded(r) and r.get("correct") is not None
                and r.get("L_star") is not None]
        n_excl = n_total - len(kept)
        L = np.array([float(r["L_star"]) for r in kept])
        ok = np.array([bool(r["correct"]) for r in kept])
        Lc = L[ok]        # correct
        Li = L[~ok]       # incorrect
        if len(Lc) < 2 or len(Li) < 2:
            print(f"{tag:16s} insufficient split (correct={len(Lc)}, incorrect={len(Li)})")
            models.append({"tag": tag, "n_kept": len(kept), "n_excluded": n_excl,
                           "n_correct": int(len(Lc)), "n_incorrect": int(len(Li)),
                           "cohens_d_incorrect_minus_correct": None,
                           "t": None, "p_one_tailed": None, "clears": False})
            continue
        pooled_sd = np.sqrt((Lc.var(ddof=1) + Li.var(ddof=1)) / 2) + 1e-8
        d = (Li.mean() - Lc.mean()) / pooled_sd   # REGISTERED direction
        t, p_two = stats.ttest_ind(Li, Lc, equal_var=False)
        # one-tailed in the registered direction (incorrect > correct)
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
        clears = d > 0.3
        ds.append(d)
        models.append({
            "tag": tag,
            "n_kept": int(len(kept)),
            "n_excluded": int(n_excl),
            "n_correct": int(len(Lc)),
            "n_incorrect": int(len(Li)),
            "L_correct_mean": float(Lc.mean()),
            "L_incorrect_mean": float(Li.mean()),
            "cohens_d_incorrect_minus_correct": float(d),
            "t": float(t),
            "p_one_tailed": float(p_one),
            "clears": bool(clears),
        })
        arrow = "incorrect LATER" if d > 0 else "incorrect EARLIER (reversal)"
        print(f"{tag:16s} n={len(kept):4d} (excl {n_excl:3d})  "
              f"Lc={Lc.mean():5.2f} Li={Li.mean():5.2f}  "
              f"d={d:+.3f}  p1={p_one:.2e}  {'CLEARS' if clears else '-'}  [{arrow}]")

    n_clear = sum(m["clears"] for m in models)
    pooled_d = float(np.mean(ds)) if ds else None
    print(f"\nH3 dev preview: {n_clear}/{len(models)} clear (d>0.3, incorrect later). "
          f"unweighted mean d={pooled_d:+.3f} if computed. "
          f"Reg confirms if pooled d>0.3 OR per-model d>0.3 in >=3/5, on the HOLD-OUT. "
          f"Development only. Negative d = pre-documented reversal.")
    (DATA / "h3_ordering_dev.json").write_text(json.dumps(
        {"models": models,
         "n_clear": int(n_clear),
         "n_models": len(models),
         "unweighted_mean_d": pooled_d,
         "sign_convention": "d = (L*_incorrect - L*_correct)/pooled_sd; "
                            "positive = registered direction (incorrect later)",
         "note": "Development commitment-ordering result. Reversals (negative d) "
                 "are the pre-documented expected outcome on instruct. "
                 "Not a confirmatory result; hold-out sealed."},
        indent=2))


if __name__ == "__main__":
    main()
