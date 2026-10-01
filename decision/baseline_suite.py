"""Open-model baselines on the decision sets argonne-4.5-decision never trained on, scored like the model itself.

    python decision/baseline_suite.py --kind nli --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --data ood.jsonl --out x.npz
    python decision/baseline_suite.py --kind llm --model Qwen/Qwen2.5-7B-Instruct --data ood.jsonl --out y.npz

nli: a zero-shot NLI classifier, the standard recipe (the Hugging Face zero-shot pipeline's). Choice: one premise /
hypothesis pair per option ("This example is {option}." for label lists, "The answer is: {option}" for questions),
softmax over the options' entailment logits. Noul: the statement itself is the hypothesis, P(entailment).
llm: an instruction model through vLLM, thinking off. Choice with up to 26 options: lettered options, the first answer
token's log-probabilities over the letters. More than 26 options: the model writes the option (greedy), matched to the
list; accuracy only, since no single token carries a probability. Noul: p("true") / (p("true") + p("false")).
Writes evaluate.py's npz layout (ood_logits, ood_label, ood_source, ood_ids, ood_source_names, ood_scored), so the
same accuracy and calibration code reads every model. ood_scored marks the items whose probabilities are real.
"""
import argparse
import difflib
import json
import math

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_LISTS = ("banking77", "newsgroups", "fin_topic", "fin_sentiment")   # options are label names, not answers


def norm(t):
    return " ".join("".join(c if c.isalnum() else " " for c in t.lower()).split())


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


def _shim_tokenizer_for_vllm():
    """transformers 5.x removed `all_special_tokens_extended`, which vLLM 0.11's tokenizer cache still reads.
    Rebuild it from added_tokens_decoder (the same shim reasoning/vllm_argonne.py applies)."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    def _aste(self):
        dec = getattr(self, "added_tokens_decoder", {}) or {}
        out = [dec[i] for i in self.all_special_ids if i in dec]
        return out if out else list(self.all_special_tokens)
    PreTrainedTokenizerBase.all_special_tokens_extended = property(_aste)


def run_llm(args, recs):
    _shim_tokenizer_for_vllm()
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, max_logprobs=26, seed=0)
    kmax = max(len(r["options"]) for r in recs)
    L = np.full((len(recs), kmax), -np.inf, dtype=np.float32)
    scored = np.ones(len(recs), dtype=bool)
    lettered = [i for i, r in enumerate(recs) if r["type"] == "choice" and len(r["options"]) <= 26]
    written = [i for i, r in enumerate(recs) if r["type"] == "choice" and len(r["options"]) > 26]
    noul = [i for i, r in enumerate(recs) if r["type"] == "noul"]
    ctk = {"enable_thinking": False}

    def body(r):
        return (r["state"] + "\n\n") if r["state"] else ""
    if lettered:
        convs = [[{"role": "user", "content": body(recs[i]) + f"Question: {recs[i]['instructions']}\n"
                   + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(recs[i]["options"]))
                   + "\n\nAnswer with the letter of the best option and nothing else."}] for i in lettered]
        outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=1, logprobs=26), use_tqdm=True,
                        chat_template_kwargs=ctk)
        for i, o in zip(lettered, outs):
            k = len(recs[i]["options"])
            mass = [0.0] * k
            for v in o.outputs[0].logprobs[0].values():
                t = (v.decoded_token or "").strip().rstrip(".").upper()
                if len(t) == 1 and t in LETTERS[:k]:
                    mass[LETTERS.index(t)] += math.exp(v.logprob)
            tot = sum(mass)
            L[i, :k] = [math.log(m / tot) if tot > 0 and m > 0 else -30.0 for m in mass]
    if written:
        convs = [[{"role": "user", "content": body(recs[i]) + f"Question: {recs[i]['instructions']}\nOptions:\n"
                   + "\n".join(f"- {o}" for o in recs[i]["options"])
                   + "\n\nAnswer with the exact text of one option and nothing else."}] for i in written]
        outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=40), use_tqdm=True, chat_template_kwargs=ctk)
        for i, o in zip(written, outs):
            ans, opts = norm(o.outputs[0].text), [norm(x) for x in recs[i]["options"]]
            j = opts.index(ans) if ans in opts else max(range(len(opts)),
                                                         key=lambda k: difflib.SequenceMatcher(None, ans, opts[k]).ratio())
            L[i, :len(opts)] = -20.0
            L[i, j] = 0.0
            scored[i] = False
    if noul:
        convs = [[{"role": "user", "content": body(recs[i]) + f"Statement: {recs[i]['instructions']}\n\n"
                   "Is the statement true for the text above? Answer with exactly one word: true or false."}] for i in noul]
        outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), use_tqdm=True,
                        chat_template_kwargs=ctk)
        for i, o in zip(noul, outs):
            pt = pf = 0.0
            for v in o.outputs[0].logprobs[0].values():
                t = (v.decoded_token or "").strip().lower()
                pt += math.exp(v.logprob) if t == "true" else 0.0
                pf += math.exp(v.logprob) if t == "false" else 0.0
            if pt + pf == 0:
                pt = pf = 1.0
            L[i, 0], L[i, 1] = math.log(max(pt, 1e-12)), math.log(max(pf, 1e-12))
    return L, scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("nli", "llm"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpu_mem", type=float, default=0.55)
    ap.add_argument("--max_model_len", type=int, default=16384)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.data)]
    L, scored = (run_nli if args.kind == "nli" else run_llm)(args, recs)
    names = sorted({r["source"] for r in recs})
    labels = np.array([r["label"] for r in recs])
    np.savez_compressed(args.out, ood_logits=L, ood_label=labels, ood_type=np.array([r["type"] == "noul" for r in recs]) * 2,
                        ood_source=np.array([names.index(r["source"]) for r in recs]),
                        ood_ids=np.array([r["id"] for r in recs]), ood_source_names=np.array(names), ood_scored=scored)
    print(f"[baseline] {args.kind} {args.model}: {len(recs)} items, accuracy {(L.argmax(1) == labels).mean():.4f}; "
          f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
