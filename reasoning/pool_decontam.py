#!/usr/bin/env python
"""Quantify indirect leakage of an eval pool into a training mix, and re-score a gate without it.

WHY THIS EXISTS. `clean_eval.load_clean` labels its pools by hand: svamp/asdiv/mawps "clean",
gsmplus "semi-clean", gsm8k "CONTAMINATED", and math500 carries the free-text warning "never
directly trained but OpenMathReasoning/Mixture-of-Thoughts carry indirect-leak risk". That warning
had never been measured. Measured on 2026-08-04 against `cot_sft_mix_v6_gen`, math500 has 4 of its
319 judged items at Jaccard >= 0.85 and 17 at >= 0.70 -- e.g. the eval item

    "What is the remainder when $1 + 2 + ... + 10$ is divided by 9?"

against a `med_math` training row identical except for "divided by 8". Zero EXACT leaks in any pool,
and svamp/asdiv/mawps/gsmplus have nothing above 0.70, so this is a math500-specific problem.

WHAT IT DOES NOT CLAIM. A near-duplicate is not proof a score is inflated: the numbers differ, so
the answer differs, and the model still has to execute the method. But at J=0.93 the *method* is
given, which is a far stronger transfer than the perturbation logic that makes GSM-Plus defensible.
The honest move is to report both numbers -- full pool and clean subset -- and let the gap speak.

USAGE
  # 1. which judged items are near-duplicates of a training row?
  python reasoning/pool_decontam.py flag --pool math500 --n 319 --mix <hf dataset dir> \
      --threshold 0.70 --out report/decontam_math500.json

  # 2. re-score gate JSONs on the clean subset only
  python reasoning/pool_decontam.py rescore --flags report/decontam_math500.json \
      --gate-json <a.json> [<b.json> ...] --pool math500

Item ordering matches the gate exactly: load_clean() does `Random(seed).shuffle(probs)` then
`probs[:n]`, and this reproduces that build order, filter, shuffle and truncation. Change --seed/--n
here if the gate is ever run with different ones, or the indices will not correspond.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def norm_q(s):
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", s)


def toks(s):
    return set(norm_q(s).split())


def load_judged(pool, n, seed):
    """The exact (question, gold) items the gate judges, in the gate's order."""
    from clean_eval import load_clean
    return load_clean(pool, n, seed=seed)


def load_mix(path):
    """(questions, tiers) from an HF chat-format SFT mix."""
    from datasets import load_from_disk
    d = load_from_disk(path)
    d = d["train"] if hasattr(d, "keys") and "train" in d else d
    qs, tiers = [], []
    for r in d:
        q = ""
        if "messages" in r:
            for m in r["messages"]:
                if m.get("role") == "user":
                    q = m.get("content", "")
                    break
        else:
            for c in ("question", "prompt", "problem", "instruction", "input"):
                if c in r:
                    q = r[c]
                    break
        qs.append(q)
        tiers.append(r.get("tier", "?"))
    return qs, tiers


def cmd_flag(a):
    judged = load_judged(a.pool, a.n, a.seed)
    mix_q, mix_tier = load_mix(a.mix)
    mix_tok = [toks(q) for q in mix_q]
    exact_ix = {}
    for i, q in enumerate(mix_q):
        exact_ix.setdefault(norm_q(q), i)

    # Inverted index on non-ubiquitous tokens. Any pair with high Jaccard necessarily shares a rare
    # token, so candidate generation is lossless at the thresholds we care about -- not sampling.
    df = Counter()
    for t in mix_tok:
        df.update(t)
    ubiq = {w for w, c in df.items() if c > len(mix_tok) * 0.10}
    index = defaultdict(list)
    for i, t in enumerate(mix_tok):
        for w in t:
            if w not in ubiq:
                index[w].append(i)

    flagged, rows = [], []
    for k, (q, gold) in enumerate(judged):
        jt = toks(q)
        cand = Counter()
        for w in jt:
            if w in index:
                cand.update(index[w])
        best, bi = 0.0, -1
        for i in cand:
            j = len(jt & mix_tok[i]) / len(jt | mix_tok[i])
            if j > best:
                best, bi = j, i
        is_exact = norm_q(q) in exact_ix
        if best >= a.threshold or is_exact:
            flagged.append(k)
            rows.append({"index": k, "jaccard": round(best, 4), "exact": is_exact,
                         "tier": mix_tier[bi] if bi >= 0 else None,
                         "eval_q": q[:300], "mix_q": (mix_q[bi][:300] if bi >= 0 else "")})

    out = {"pool": a.pool, "n": len(judged), "seed": a.seed, "mix": a.mix,
           "threshold": a.threshold, "n_flagged": len(flagged),
           "frac_flagged": round(len(flagged) / max(1, len(judged)), 4),
           "flagged_indices": flagged, "detail": rows}
    print(f"{a.pool}: {len(flagged)}/{len(judged)} items flagged at J>={a.threshold} "
          f"({100*len(flagged)/max(1,len(judged)):.1f}%)")
    for r in rows[:8]:
        print(f"  [{r['index']:>4}] J={r['jaccard']:.3f} tier={r['tier']}")
        print(f"        EVAL: {r['eval_q'][:120]}")
        print(f"        MIX : {r['mix_q'][:120]}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return out


def cmd_rescore(a):
    flags = json.load(open(a.flags))
    bad = set(flags["flagged_indices"])
    pool = a.pool or flags["pool"]
    print(f"pool={pool}   excluding {len(bad)} flagged of {flags['n']} "
          f"(threshold J>={flags['threshold']})\n")
    hdr = f"{'model':<14} {'cfg':<10} {'n_all':>6} {'acc_all':>8} {'n_clean':>8} {'acc_clean':>10} {'delta':>7}"
    print(hdr)
    print("-" * len(hdr))
    for jf in a.gate_json:
        d = json.load(open(jf))
        for model, pools in d.get("res", {}).items():
            if pool not in pools:
                continue
            for cfg, r in pools[pool].items():
                if a.cfg and cfg != a.cfg:
                    continue
                ok = r.get("ok") or []
                if not ok:
                    continue
                keep = [v for i, v in enumerate(ok) if i not in bad]
                aa = 100.0 * sum(ok) / len(ok)
                ac = 100.0 * sum(keep) / max(1, len(keep))
                print(f"{model:<14} {cfg:<10} {len(ok):>6} {aa:>8.2f} {len(keep):>8} "
                      f"{ac:>10.2f} {ac-aa:>+7.2f}")


def cmd_clean_mix(a):
    """Write a copy of the mix with every row that near-duplicates ANY eval item removed.

    Decontaminates against the FULL pools, not the judged slice, deliberately: the gate judges
    `Random(0).shuffle(pool)[:n]`, so a mix cleaned only against that slice would silently re-leak
    the moment anyone changes --n. Stricter is the right default for a training-side fix.
    """
    from datasets import load_from_disk

    mix_q, mix_tier = load_mix(a.mix)
    mix_tok = [toks(q) for q in mix_q]

    df = Counter()
    for t in mix_tok:
        df.update(t)
    ubiq = {w for w, c in df.items() if c > len(mix_tok) * 0.10}
    index = defaultdict(list)
    for i, t in enumerate(mix_tok):
        for w in t:
            if w not in ubiq:
                index[w].append(i)

    from clean_eval import load_clean
    dirty = {}                                   # mix row -> (jaccard, pool)
    for pool in a.pools:
        items = load_clean(pool, 0, seed=0)      # n=0 -> the WHOLE pool
        hits = 0
        for q, _ in items:
            jt = toks(q)
            cand = Counter()
            for w in jt:
                if w in index:
                    cand.update(index[w])
            for i in cand:
                j = len(jt & mix_tok[i]) / len(jt | mix_tok[i])
                if j >= a.threshold and j > dirty.get(i, (0, ""))[0]:
                    dirty[i] = (j, pool)
                    hits += 1
        print(f"  {pool:<10} {len(items):>5} eval items -> {hits} mix-row hits at J>={a.threshold}")

    per_tier = Counter(mix_tier[i] for i in dirty)
    print(f"\nremoving {len(dirty)} of {len(mix_q)} rows ({100*len(dirty)/len(mix_q):.2f}%)")
    for t, c in per_tier.most_common():
        tot = sum(1 for x in mix_tier if x == t)
        print(f"  {t:<22} {c:>4} of {tot:>5} ({100*c/tot:5.2f}%)")

    d = load_from_disk(a.mix)
    d = d["train"] if hasattr(d, "keys") and "train" in d else d
    keep = [i for i in range(len(mix_q)) if i not in dirty]
    out = d.select(keep)
    out.save_to_disk(a.out)
    print(f"\nwrote {a.out}  ({len(out)} rows)")
    if a.report:
        json.dump({"mix": a.mix, "out": a.out, "threshold": a.threshold, "pools": a.pools,
                   "n_removed": len(dirty), "n_kept": len(keep),
                   "per_tier_removed": dict(per_tier),
                   "removed": [{"row": i, "jaccard": round(j, 4), "pool": p,
                                "tier": mix_tier[i], "q": mix_q[i][:200]}
                               for i, (j, p) in sorted(dirty.items())]},
                  open(a.report, "w"), indent=1)
        print(f"wrote {a.report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("flag", help="find judged items that near-duplicate a training row")
    f.add_argument("--pool", required=True)
    f.add_argument("--n", type=int, required=True, help="MUST match the gate's per-pool n")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--mix", required=True)
    f.add_argument("--threshold", type=float, default=0.70)
    f.add_argument("--out", default="")
    f.set_defaults(fn=cmd_flag)

    r = sub.add_parser("rescore", help="re-score gate JSONs with flagged items removed")
    r.add_argument("--flags", required=True)
    r.add_argument("--gate-json", nargs="+", required=True)
    r.add_argument("--pool", default="")
    r.add_argument("--cfg", default="", help="restrict to one config, e.g. greedy")
    r.set_defaults(fn=cmd_rescore)

    c = sub.add_parser("clean-mix", help="write a mix copy with near-duplicate rows removed")
    c.add_argument("--mix", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--pools", nargs="+",
                   default=["svamp", "asdiv", "mawps", "gsmplus", "math500"])
    c.add_argument("--threshold", type=float, default=0.70)
    c.add_argument("--report", default="")
    c.set_defaults(fn=cmd_clean_mix)

    a = ap.parse_args()
    a.fn(a)
