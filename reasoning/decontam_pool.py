#!/usr/bin/env python
"""Exact near-duplicate sweep of a candidate TRAINING pool against every judged EVAL item.

WHY A NEW SCRIPT. `pool_decontam.py flag` answers the reverse question (which eval items are
near-dups of a mix) and takes an HF dataset dir. Here the pool is a 341k-row JSONL and the job is to
DROP contaminated training problems before they are ever generated on. §41bh found the coverage
hole was 57% competition MATH near-dupping MATH-500, and §41bt had to re-measure decontamination
after the pool grew 27%; a 22x expansion pulled from a corpus that literally contains `math` and
`gsm8k` sources is exactly where this bites.

WHY IT IS FAST AND STILL EXACT. Brute force is |pool| x |eval| = ~964M Jaccard computations. Prefix
filtering makes it exact and cheap: if J(A,B) >= t then |A ∩ B| >= t|A|, so B must share at least
one of A's ceil((1-t)|A|) + 1 RAREST tokens. Indexing the pool on those prefixes and probing with
each eval item's prefix cannot miss a pair at or above the threshold -- this is a filter, not a
sample, and every surviving candidate gets a real Jaccard computed.
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
    ap.add_argument("--pool", required=True, help="JSONL with a `question` field")
    ap.add_argument("--out", default="", help="write the CLEANED pool here")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--pools", nargs="+",
                    default=["asdiv", "svamp", "gsmplus", "mawps", "math500"])
    ap.add_argument("--n", nargs="+", type=int, default=[1000, 1000, 500, 500, 319],
                    help="MUST match the gate's per-pool n so the judged items are the same ones")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="report/numina_decontam.json")
    a = ap.parse_args()

    from pool_decontam import load_judged

    rows = [json.loads(l) for l in open(a.pool)]
    print(f"pool: {len(rows)} problems", flush=True)
    ptoks = [toks(r["question"]) for r in rows]

    df = Counter()
    for t in ptoks:
        df.update(t)

    # index each pool row on its rarest ceil((1-t)*|A|)+1 tokens
    idx = defaultdict(list)
    for i, t in enumerate(ptoks):
        if not t:
            continue
        order = sorted(t, key=lambda w: df[w])
        plen = int((1 - a.threshold) * len(t)) + 1
        for w in order[:plen]:
            idx[w].append(i)
    print(f"index: {len(idx)} prefix tokens", flush=True)

    hits = {}          # pool row -> (best J, pool name, eval question)
    per_pool = {}
    for pool, n in zip(a.pools, a.n):
        judged = load_judged(pool, n, a.seed)
        worst = 0.0
        flagged = 0
        for q, _ in judged:
            et = toks(q)
            if not et:
                continue
            order = sorted(et, key=lambda w: df.get(w, 0))
            plen = int((1 - a.threshold) * len(et)) + 1
            cand = set()
            for w in order[:plen]:
                cand.update(idx.get(w, ()))
            for i in cand:
                pt = ptoks[i]
                j = len(et & pt) / len(et | pt)
                if j >= a.threshold:
                    flagged += 1
                    if j > hits.get(i, (0,))[0]:
                        hits[i] = (j, pool, q[:120])
                if j > worst:
                    worst = j
        per_pool[pool] = {"judged": len(judged), "max_jaccard": round(worst, 4),
                          "pairs_at_threshold": flagged}
        print(f"  {pool:9s} judged={len(judged):4d}  max J={worst:.3f}  pairs>=t {flagged}", flush=True)

    print(f"\nCONTAMINATED pool rows to drop: {len(hits)} of {len(rows)} "
          f"({100*len(hits)/len(rows):.3f}%)")
    for i, (j, pool, q) in sorted(hits.items(), key=lambda kv: -kv[1][0])[:5]:
        print(f"  J={j:.3f} vs {pool}: {q}")
        print(f"          pool row: {rows[i]['question'][:120]}")

    if a.out:
        kept = 0
        with open(a.out, "w") as f:
            for i, r in enumerate(rows):
                if i in hits:
                    continue
                f.write(json.dumps(r) + "\n")
                kept += 1
        print(f"\nwrote {kept} clean problems -> {a.out}")

    json.dump({"pool": a.pool, "threshold": a.threshold, "n_pool": len(rows),
               "n_dropped": len(hits), "per_pool": per_pool},
              open(a.report, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
