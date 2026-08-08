#!/usr/bin/env python
"""Per-tier THINK-token length, and what a given --max_think_tokens would have removed.

WHY THIS EXISTS. §34 found that `cot-sft.py --max_think_tokens 128` silently severed reasoning traces
mid-derivation, and §36's fix was worth +7.06pt. But "how much did it cut, per tier" was only ever
answered in aggregate, and the aggregate hides the thing that matters: truncation is not uniform, and a
tier whose whole purpose lives at the END of the think block loses that purpose entirely.

Run on `cot_mix_robust` (the §33 verify-tier mix) on 2026-08-04 this produced the finding that
retrospectively explains §33's failure:

    tier                    p50    p90    max   % >128
    pert_verify_fix         246    287    557    100%
    pert_verify_rederive    249    286    291    100%
    verify_rederive         202    288    613     98%
    verify_fix              208    292    605     96%
    verify_confirm          150    216    530     71%
    hard_strict             286    441    637     89%
    gsm8k_train_short       193    301    376     80%
    med_math                110    266    617     42%
    synth_arith              15     32     33      0%

In a self-verification trace the verification step comes AFTER the initial solve, inside the think
block. Cutting at 128 think-tokens therefore removed the verification from ~100% of the rows meant to
teach it — the model learned "solve, start re-checking, stop", which is exactly the double-application
failure §33v documented and the −23.8pt one-step-arithmetic regression that blocked the §33 ship.
Meanwhile `synth_arith` shows 0%: the arithmetic drill was never truncated, it was DROPPED by the other
flag. Two separate mechanisms, and only a per-tier view separates them.

USE IT BEFORE trusting any negative result from a CoT-SFT arm: if the tier that carries your
hypothesis is >50% truncated, the experiment did not test the hypothesis.

  python reasoning/think_len_audit.py --data <hf dataset dir> [--max-think 128] [--tokenizer <dir>]
"""
import argparse
import re
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="/project/rcc/youzhi/models/a35_reason/dpo")
    ap.add_argument("--max-think", type=int, default=128,
                    help="the --max_think_tokens value to evaluate exposure against")
    ap.add_argument("--sample", type=int, default=400, help="rows per tier (0 = all)")
    a = ap.parse_args()

    import numpy as np
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    d = load_from_disk(a.data)
    d = d["train"] if hasattr(d, "keys") and "train" in d else d

    by = defaultdict(list)
    no_think = defaultdict(int)
    for r in d:
        t = r.get("tier", "?")
        if a.sample and len(by[t]) >= a.sample:
            continue
        ass = next((m["content"] for m in r["messages"] if m.get("role") == "assistant"), "")
        m = re.search(r"<think>(.*?)</think>", ass, re.S)
        if not m:
            no_think[t] += 1
            continue
        by[t].append(len(tok(m.group(1), add_special_tokens=False)["input_ids"]))

    print(f"{a.data}\nthink-token length per tier; exposure to --max_think_tokens {a.max_think}\n")
    hdr = f"{'tier':<24}{'rows':>7}{'p50':>7}{'p90':>7}{'max':>7}{'% cut':>8}{'tok lost':>10}"
    print(hdr)
    print("-" * len(hdr))
    tot_lost = tot_tok = 0
    for t in sorted(by, key=lambda x: -float(np.mean(by[x])) if by[x] else 0):
        v = np.array(by[t])
        if not len(v):
            continue
        lost = np.clip(v - a.max_think, 0, None).sum()
        tot_lost += lost
        tot_tok += v.sum()
        print(f"{t:<24}{len(v):>7}{np.percentile(v,50):>7.0f}{np.percentile(v,90):>7.0f}"
              f"{v.max():>7}{100*(v>a.max_think).mean():>7.0f}%{100*lost/max(1,v.sum()):>9.0f}%")
    if tot_tok:
        print("-" * len(hdr))
        print(f"{'ALL SAMPLED TIERS':<24}{'':>7}{'':>7}{'':>7}{'':>7}{'':>8}{100*tot_lost/tot_tok:>9.0f}%")
    if no_think:
        print(f"\nrows with no <think> block (unaffected by --max_think_tokens): "
              f"{dict(sorted(no_think.items(), key=lambda kv: -kv[1]))}")
    print("\nRead: a tier whose purpose lives at the END of the think block (self-verification, "
          "re-derivation, final-answer restatement) loses that purpose entirely at high % cut, even "
          "though the row still trains and reports no error.")


if __name__ == "__main__":
    main()
