#!/usr/bin/env python
"""Build a DIVERSE, numeric-verifiable math problem pool from NuminaMath-CoT.

WHY. §41bs diagnosed the coverage null as a GENERALISATION failure: the coverage arm raised the
per-sample solve rate 44% relative on the 1,392 problems it trained on and transferred exactly
nothing (held-out pass@8 −0.03, p=1.00). Generalisation is what scales with problem DIVERSITY, and
this line's entire training pool is 15,212 problems (GSM8K-train + MATH-train, §41bt) -- every round,
every DPO set and every coverage arm has recycled the same ones. NuminaMath-CoT is ~860k rows drawn
from genuinely different sources.

WHAT IS KEPT, and why not everything:
  * gold must survive `norm()` -- the whole verification stack downstream is numeric-only
    (star_generate.norm, and every grader through it). Measured yield on a 30k stream: 52.7%.
  * SOURCE FILTER. Default keeps cn_k12 / orca_math / synthetic_math and drops
    olympiads / aops_forum / amc_aime / synthetic_amc, which are far out of range for a model at
    ~59% on grade-school word problems: neither the student nor a Qwen3-14B completer solves them,
    so they are dead weight in a coverage corpus. Pass --sources to override.
  * `math` and `gsm8k` sources are DROPPED by default: they are the pools we already train on, and
    NuminaMath's `math` rows overlap MATH-test, of which math500 is a subset.

⚠️THIS SCRIPT DOES NOT DECONTAMINATE. It writes a candidate pool; `--decontam` runs the §41bt Jaccard
sweep against the judged eval items and drops anything at or above the threshold. Run it. The four
clean pools are not MATH-derived so the risk is mostly gsmplus (GSM8K-derived), but "mostly" is not a
measurement and §41bh already caught a 57%-competition-MATH coverage hole this way.

⚠️Streams rather than downloading: the full corpus is GBs and we need two short fields. Do NOT import
torch in this process -- `datasets` streaming plus torch segfaults at interpreter teardown (observed
2026-08-12), so `norm` is reimplemented here rather than imported from star_generate.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

KEEP_DEFAULT = ["cn_k12", "orca_math", "synthetic_math"]


def norm(s):
    """Byte-identical to star_generate.norm -- kept in sync by hand to avoid importing torch."""
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "").replace(" ", "")
    s = s.rstrip(".")
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    v = m.group(0)
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return None


def boxed_gold(sol):
    m = re.findall(r"\\boxed\{([^{}]*)\}", sol)
    return norm(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/rcc/youzhi/data/numina_pool/pool.jsonl")
    ap.add_argument("--sources", nargs="*", default=KEEP_DEFAULT)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = stream the whole corpus")
    ap.add_argument("--max-keep", type=int, default=0, help="stop once this many problems are kept")
    ap.add_argument("--min-chars", type=int, default=40, help="drop degenerate stubs")
    ap.add_argument("--max-chars", type=int, default=1200, help="drop essay-length prompts")
    # ⚠️Streaming threw `Bad file descriptor` mid-pull on a 20k smoke and retried; over 860k rows a
    # dropped connection yields a SILENTLY TRUNCATED pool, which is the worst possible failure for a
    # corpus whose whole point is size. The parquet is 1.15 GB once -- take the download.
    ap.add_argument("--stream", type=int, default=0, help="1 = stream (fragile), 0 = download parquet")
    ap.add_argument("--cache", default="/project/rcc/youzhi/data/hf_datasets")
    a = ap.parse_args()

    from datasets import load_dataset
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train",
                      streaming=bool(a.stream), cache_dir=a.cache)

    keep = set(a.sources)
    seen_q = set()
    n = kept = 0
    by_src, drop = Counter(), Counter()
    with open(a.out, "w") as f:
        for r in ds:
            n += 1
            if a.max_rows and n > a.max_rows:
                break
            src = r.get("source", "?")
            if src not in keep:
                drop["source"] += 1
                continue
            q = (r.get("problem") or "").strip()
            if not (a.min_chars <= len(q) <= a.max_chars):
                drop["length"] += 1
                continue
            if q in seen_q:
                drop["dup_question"] += 1
                continue
            g = boxed_gold(r.get("solution", "") or "")
            if g is None:
                drop["no_numeric_gold"] += 1
                continue
            seen_q.add(q)
            f.write(json.dumps({"question": q, "gold": g, "source": src}) + "\n")
            kept += 1
            by_src[src] += 1
            if a.max_keep and kept >= a.max_keep:
                break
            if kept % 25000 == 0:
                print(f"  ... {n} streamed, {kept} kept", flush=True)

    print(f"streamed {n} rows -> KEPT {kept} problems -> {a.out}")
    print("  by source:", dict(by_src))
    print("  drops    :", dict(drop))
    json.dump({"streamed": n, "kept": kept, "by_source": dict(by_src), "drops": dict(drop),
               "sources": a.sources},
              open("report/numina_pool_build.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
