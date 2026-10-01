"""Zero-shot classifier baselines on the decision sets argonne-4.5-decision never trained on, scored like the model.

    python decision/baseline_suite.py --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --data ood.jsonl --out x.npz

The standard recipe (the Hugging Face zero-shot pipeline's). Choice: one premise / hypothesis pair per option ("This
example is {option}." for label lists, "The answer is: {option}" for questions), softmax over the options' entailment
logits. Noul: the statement itself is the hypothesis, P(entailment). Writes evaluate.py's npz layout (ood_logits,
ood_label, ood_source, ood_ids, ood_source_names, ood_scored), so decision/bench_report.py scores every model the same.
"""
import argparse
import json

import numpy as np

LABEL_LISTS = ("banking77", "newsgroups", "fin_topic", "fin_sentiment")   # options are label names, not answers


def hyp_choice(r, opt):
    if r["source"] in LABEL_LISTS:
        return f"This example is {opt}."
    return f"The answer is: {opt}"


def premise(r):
    q = r["instructions"] if r["type"] == "choice" else ""
    return (r["state"] + ("\n\n" + q if q else "")).strip() if r["state"] else q


def run_nli(args, recs):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    mod = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    lab = {k.lower(): v for k, v in mod.config.label2id.items()}
    ent = lab.get("entailment")
    other = lab.get("contradiction", lab.get("not_entailment"))
    pairs, owner = [], []
    for i, r in enumerate(recs):
        if r["type"] == "choice":
            for o in r["options"]:
                pairs.append((premise(r), hyp_choice(r, o)))
                owner.append(i)
        else:
            pairs.append((premise(r), r["instructions"]))
            owner.append(i)
    scores = np.zeros((len(pairs), 2), dtype=np.float32)       # (entailment logit, the other logit)
    order = np.argsort([len(p[0]) + len(p[1]) for p in pairs])  # length-sorted batches
    with torch.no_grad():
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            enc = tok([pairs[j][0] for j in idx], [pairs[j][1] for j in idx], truncation="only_first",
                      max_length=512, padding=True, return_tensors="pt").to("cuda")
            lg = mod(**enc).logits.float().cpu().numpy()
            scores[idx, 0], scores[idx, 1] = lg[:, ent], lg[:, other]
            if s // args.batch % 200 == 0:
                print(f"[nli] {s:,}/{len(pairs):,} pairs", flush=True)
    kmax = max(len(r["options"]) for r in recs)
    L = np.full((len(recs), kmax), -np.inf, dtype=np.float32)
    pos = 0
    for i, r in enumerate(recs):
        if r["type"] == "choice":
            k = len(r["options"])
            L[i, :k] = scores[pos:pos + k, 0]                     # softmax over entailment logits
            pos += k
        else:
            L[i, 0], L[i, 1] = scores[pos, 0], scores[pos, 1]     # entailment vs the other class
            pos += 1
    return L, np.ones(len(recs), dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("nli",), default="nli")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.data)]
    L, scored = run_nli(args, recs)
    names = sorted({r["source"] for r in recs})
    labels = np.array([r["label"] for r in recs])
    np.savez_compressed(args.out, ood_logits=L, ood_label=labels, ood_type=np.array([r["type"] == "noul" for r in recs]) * 2,
                        ood_source=np.array([names.index(r["source"]) for r in recs]),
                        ood_ids=np.array([r["id"] for r in recs]), ood_source_names=np.array(names), ood_scored=scored)
    print(f"[baseline] {args.kind} {args.model}: {len(recs)} items, accuracy {(L.argmax(1) == labels).mean():.4f}; "
          f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
