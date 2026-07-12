#!/usr/bin/env python3
"""Best-of-N reranking with the trained generative verifier (§22 lever #1 — the step-change).

pass@256 ≈ 82% but single-sample ≈ 4%: the correct answer is in the sample set, unpicked.
This harness: sample K candidates from the POLICY, score each with the VERIFIER (P('Yes')),
and keep the highest-scored one. Unlike majority-vote (saturates ~K64), best-of-N keeps
improving with K toward the pass@K ceiling — so large K is genuinely useful here (and uses the
GPU well). Reports best-of-N vs majority-vote vs single-sample vs pass@K on the same samples.

Two models loaded (policy + verifier). Generation is K identical copies of one prompt (KV-cached,
no padding). Verifier scoring is one forward per closed+boxed candidate (prefill only; fast).
"""
import argparse
import datetime as _dt
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401
except ModuleNotFoundError:
    pass

from star_generate import extract_boxed, norm, load_problems, batched_sample, autofit_k
from build_verifier_data import PROMPT_TMPL


@torch.inference_mode()
def verifier_p_yes(verifier, tok, question, solution, yes_id, no_id, dtype, max_ctx):
    msgs = [{"role": "user", "content": PROMPT_TMPL.format(q=question, sol=solution)}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    ids = ids[:, -max_ctx:].to(verifier.embed_tokens.weight.device)
    with torch.autocast("cuda", dtype=dtype):
        logits = verifier(ids).logits[0, -1, :].float()
    two = torch.tensor([logits[yes_id], logits[no_id]])
    return float(F.softmax(two, dim=-1)[0])  # P(Yes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="/project/rcc/youzhi/models/instruct/soup_blend_a085")
    ap.add_argument("--verifier", default="/project/rcc/youzhi/models/instruct/verifier_soup_r1")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--target-hbm", type=float, default=0.0, help="autofit gen K to fill HBM (0=off)")
    ap.add_argument("--max-k", type=int, default=256)
    ap.add_argument("--source", choices=["gsm8k", "math", "both"], default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=150)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: needs GPU"); sys.exit(1)
    fh = open(args.log, "a") if args.log else None

    def out(*p):
        line = " ".join(str(x) for x in p)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.policy, trust_remote_code=True)
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]

    policy = AutoModelForCausalLM.from_pretrained(
        args.policy, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True).to("cuda").eval()
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True).to("cuda").eval()
    max_ctx = policy.config.max_position_embeddings

    problems = load_problems(args.source, args.n_problems, seed=args.seed)
    eff_k = args.k
    if args.target_hbm > 0:
        longest = max(problems, key=lambda p: len(p[0]))[0]
        penc = tok.apply_chat_template([{"role": "user", "content": longest}], tokenize=True,
                                       add_generation_prompt=True, enable_thinking=True,
                                       return_tensors="pt")
        pid = (penc["input_ids"] if hasattr(penc, "keys") else penc).to("cuda")
        eff_k = autofit_k(policy, pid, eos_id=eos_id, target_frac=args.target_hbm,
                          max_k=args.max_k, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p)

    out("=" * 64)
    out(f"best-of-N rerank  {_dt.datetime.now().isoformat(timespec='seconds')}")
    out(f"policy={args.policy}\nverifier={args.verifier}")
    out(f"K={eff_k} src={args.source} n={len(problems)} yes_id={yes_id} no_id={no_id}")
    out("=" * 64)

    n = len(problems)
    n_bon = n_maj = n_pass = n_single = 0
    for i, (q, gold, tier) in enumerate(problems):
        enc = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=True,
                                      return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        with torch.autocast("cuda", dtype=dtype):
            gens = batched_sample(policy, ids.repeat(eff_k, 1), max_new_tokens=args.max_new_tokens,
                                  eos_id=eos_id, temperature=args.temperature,
                                  top_k=args.top_k, top_p=args.top_p)
        cands, votes = [], Counter()
        for j, g in enumerate(gens):
            text = tok.decode(g, skip_special_tokens=True)
            if "</think>" not in text:
                continue
            pred = extract_boxed(text)
            if pred is None:
                continue
            cands.append((text, pred))
            votes[pred] += 1
            if j == 0 and pred == gold:
                n_single += 1
        if any(p == gold for _, p in cands):
            n_pass += 1
        if votes and votes.most_common(1)[0][0] == gold:
            n_maj += 1
        # best-of-N: verifier-score each candidate, pick max P(Yes)
        if cands:
            best_pred, best_score = None, -1.0
            for text, pred in cands:
                s = verifier_p_yes(verifier, tok, q, text, yes_id, no_id, dtype, max_ctx)
                if s > best_score:
                    best_score, best_pred = s, pred
            if best_pred == gold:
                n_bon += 1
        if (i + 1) % 20 == 0:
            out(f"  [{i+1}/{n}] best-of-N={100*n_bon/(i+1):.1f}%  maj={100*n_maj/(i+1):.1f}%  "
                f"single={100*n_single/(i+1):.1f}%  pass@{eff_k}={100*n_pass/(i+1):.1f}%")

    out(f"\n  BEST-OF-N (verifier)   : {100*n_bon/n:.2f}%  ({n_bon}/{n})")
    out(f"  majority vote          : {100*n_maj/n:.2f}%  ({n_maj}/{n})")
    out(f"  single-sample          : {100*n_single/n:.2f}%  ({n_single}/{n})")
    out(f"  pass@{eff_k} (ceiling)  : {100*n_pass/n:.2f}%  ({n_pass}/{n})")
    out(f"  verifier lift over vote: {100*(n_bon-n_maj)/n:+.2f} pts")
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
