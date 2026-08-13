#!/usr/bin/env python
"""How much of a candidate pool is GENUINELY NEW versus a restatement of what we already train on?

§41bz built a 335,990-problem decontaminated pool and called it "22x the current pool". That number is
only meaningful if the problems are actually different ones. Two of its three sources are synthetic
expansions of exactly the distributions this line already uses -- `orca_math` is GSM8K-style and
`synthetic_math` is MATH-style -- so a large fraction could be paraphrases of the same 15,212 problems.
§41bs says the binding constraint is DIVERSITY across distinct problems, so "22x the volume, 1.1x the
diversity" would make the whole coverage branch a much weaker bet than it looks.

Same exact prefix filter as decontam_pool.py (J(A,B)>=t implies |A∩B| >= t|A|), run against the TRAINING
pool instead of the eval pools.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
WORD = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(WORD.findall(s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/project/rcc/youzhi/data/numina_pool/pool_clean.jsonl")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--out", default="", help="write the NOVEL-only subset here")
    ap.add_argument("--report", default="report/numina_novelty.json")
    a = ap.parse_args()

    from effort_probe import load_pool
    old = []
    for p in ("gsm8k_train", "math_train_easy", "math_train_hard"):
        old += [q for q, _ in load_pool(p, 0)]
    rows = [json.loads(l) for l in open(a.pool)]
    print(f"existing training pool {len(old)} | candidate pool {len(rows)}", flush=True)

    ntok = [toks(r["question"]) for r in rows]
    df = Counter()
    for t in ntok:
        df.update(t)
    T = a.threshold
    idx = defaultdict(list)
    for i, t in enumerate(ntok):
        if not t:
            continue
        for w in sorted(t, key=lambda w: df[w])[:int((1 - T) * len(t)) + 1]:
            idx[w].append(i)
    print(f"index built ({len(idx)} prefix tokens)", flush=True)

    hit, bysrc = set(), Counter()
    for k, q in enumerate(old):
        et = toks(q)
        if not et:
            continue
        cand = set()
        for w in sorted(et, key=lambda w: df.get(w, 0))[:int((1 - T) * len(et)) + 1]:
            cand.update(idx.get(w, ()))
        for i in cand:
            if len(et & ntok[i]) / len(et | ntok[i]) >= T:
                if i not in hit:
                    bysrc[rows[i]["source"]] += 1
                hit.add(i)
        if (k + 1) % 3000 == 0:
            print(f"  ... {k+1}/{len(old)} probed, {len(hit)} overlaps", flush=True)

    tot = Counter(r["source"] for r in rows)
    novel = len(rows) - len(hit)
    print(f"\noverlapping with an existing training problem (J>={T}): {len(hit)} "
          f"({100*len(hit)/len(rows):.2f}%)")
    for s, c in tot.most_common():
        print(f"    {s:16s} {c:7d} total, {bysrc.get(s,0):6d} overlap ({100*bysrc.get(s,0)/c:5.2f}%)")
    print(f"\n=> GENUINELY NEW: {novel} ({100*novel/len(rows):.2f}%), {novel/len(old):.1f}x the "
          f"existing pool")

    if a.out:
        with open(a.out, "w") as f:
            for i, r in enumerate(rows):
                if i not in hit:
                    f.write(json.dumps(r) + "\n")
        print(f"wrote {novel} novel problems -> {a.out}")
    json.dump({"pool": a.pool, "n_pool": len(rows), "n_existing": len(old),
               "n_overlap": len(hit), "n_novel": novel,
               "overlap_by_source": dict(bysrc), "total_by_source": dict(tot)},
              open(a.report, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
