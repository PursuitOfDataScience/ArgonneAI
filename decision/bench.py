"""Jev-comparison benchmarks for a trained decision model, on one GPU.

    python decision/bench.py --model DIR --built /path/to/built --out bench.json

(1) LATENCY of system_one, the way Jev is quoted (median ms per request): a short support ticket with three
    questions, a Banking77 query with all 77 intents, and a ~6k-token document with two questions. 5 warmup calls
    are discarded; median, p10 and p90 over --reps calls are reported with n.
(2) OUT-OF-CATEGORY: Jev's documented failure is that inputs matching no option still get >=0.99 confidence
    (30 of 30 in the TDS test). Here: Banking77 test queries whose gold intent is REMOVED from the options,
    (a) without an escape option: how often is max confidence >= 0.99, and the median max confidence;
    (b) with "None of the above" offered: how often the model picks it.
    Compared against the same queries WITH their gold intent present (accuracy, confidence).
"""
import argparse
import json
import os
import random
import statistics as st
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision.api import ArgonneDecision, Choice, Noul, Score  # noqa: E402

TICKET = ("Hi, I ordered a laptop two weeks ago (order #88213) and I was charged twice on my credit card. "
          "I have called three times and nobody has fixed it. This is ridiculous. I want the duplicate charge "
          "refunded today or I will cancel my account.")


def timeit(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return {"median_ms": round(st.median(ts), 1), "p10_ms": round(ts[len(ts) // 10], 1),
            "p90_ms": round(ts[(9 * len(ts)) // 10], 1), "n": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--built", required=True, help="built/ dir with ood.jsonl and test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--ooc_n", type=int, default=500)
    args = ap.parse_args()
    dm = ArgonneDecision(args.model)
    res = {"model": args.model, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    bank = [json.loads(l) for l in open(os.path.join(args.built, "ood.jsonl")) if '"banking77"' in l]
    intents = sorted(set(bank[0]["options"]))
    long_doc = next(json.loads(l) for l in open(os.path.join(args.built, "test.jsonl")) if '"quality"' in l)

    ticket_qs = {
        "dept": Choice("Which team should handle this ticket?", ["billing", "technical support", "shipping", "other"]),
        "refund": Noul("The customer is asking for a refund."),
        "anger": Score("How angry is the customer?", {0: "Calm", 1: "Frustrated but civil", 2: "Very angry"}),
    }
    res["latency"] = {
        "ticket_3_questions": timeit(lambda: dm.system_one(TICKET, ticket_qs), args.reps),
        "banking77_77_options": timeit(lambda: dm.system_one(bank[0]["state"], {"intent": Choice(bank[0]["instructions"], intents)}), args.reps),
        "document_2_questions": timeit(lambda: dm.system_one(long_doc["state"], {
            "q": Choice(long_doc["instructions"], long_doc["options"]),
            "fiction": Noul("The text is a work of fiction.")}), max(10, args.reps // 5)),
    }
    res["latency"]["document_2_questions"]["input_tokens"] = dm.system_one(long_doc["state"], {
        "q": Choice(long_doc["instructions"], long_doc["options"])})["usage"]["input_tokens"]
    res["example_ticket"] = dm.system_one(TICKET, ticket_qs)
    print(json.dumps(res["latency"], indent=1), flush=True)

    rng = random.Random(0)
    sample = rng.sample(bank, min(args.ooc_n, len(bank)))
    with_gold, no_gold, escape = [], [], []
    for r in sample:
        gold = r["options"][r["label"]]
        out = dm.system_one(r["state"], {"a": Choice(r["instructions"], r["options"])})["answers"]["a"]
        with_gold.append((out["choice"] == gold, max(out["probabilities"].values())))
        opts = [o for o in r["options"] if o != gold]
        out = dm.system_one(r["state"], {"a": Choice(r["instructions"], opts)})["answers"]["a"]
        no_gold.append(max(out["probabilities"].values()))
        out = dm.system_one(r["state"], {"a": Choice(r["instructions"], opts + ["None of the above"])})["answers"]["a"]
        escape.append(out["choice"] == "None of the above")
    res["out_of_category"] = {
        "n": len(sample),
        "gold_present": {"acc": round(sum(c for c, _ in with_gold) / len(sample), 4),
                         "median_max_conf": round(st.median(p for _, p in with_gold), 4)},
        "gold_removed_no_escape": {"share_conf_ge_0.99": round(sum(p >= 0.99 for p in no_gold) / len(sample), 4),
                                   "median_max_conf": round(st.median(no_gold), 4)},
        "gold_removed_with_none_option": {"picked_none": round(sum(escape) / len(sample), 4)},
        "jev_reference": "TDS: 30 of 30 out-of-category inputs got confidence >= 0.99",
    }
    print(json.dumps(res["out_of_category"], indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
