#!/usr/bin/env python3
"""§24 Tier-1 — EXTERNAL-verifier best-of-N (the pivotal capturability experiment).

§22i/§23b established that a *same-base* picker (learned verifier, majority vote, confidence /
budget-forced vote) is saturated: it is as weak as the base, so it cannot cash the pass@K latent
capability (SVAMP/ASDiv pass@32 ~73% vs self-consistency ~40-51%) into pass@1. §24 Tier 1 asks the
one untried question that decides the whole go-forward tree:

    Can a STRONGER, EXTERNAL model rerank the shipped model's own candidates and capture the gap?

This harness is the external analogue of `vllm_bon.py` (which reranks with a same-base verifier).
It runs in two processes (separate vLLM engines, don't co-reside):

  --mode generate : the shipped Argonne policy (default x_v6v2_040 = HF v3) samples K candidates per
                    CLEAN problem (SVAMP/ASDiv via clean_eval.load_clean -- contamination-free), and
                    dumps {source, question, gold, candidates}. Prints self-consistency + pass@K free.

  --mode rerank   : an EXTERNAL verifier (default Qwen3-4B: tokenizer-aligned Qwen BPE, a strong
                    drop-in judge) reranks. Three scoring lenses (fair, complementary; run all):
                      yesno    - non-thinking 1-token judge; rank distinct answers by P("Yes").
                      reasoned - thinking judge; reasons then emits "Verdict: Yes/No"; rank by verdict.
                      solver   - the verifier SOLVES each problem itself; pick the candidate answer
                                 that matches its solution (the strongest external picker + an upper
                                 reference; INTERPRET via solver_solo/solver_cov -- a high solver-bon
                                 can just mean "Qwen solved it", not "gap captured").
                    Reports, per source: single / self-consistency (closed-only AND full-pool) /
                    best-of-N (each lens) / pass@K, with Wilson 95% CIs and a paired McNemar p-value
                    for each best-of-N vs the closed-only vote.

DESIGN NOTES (all raised by the pre-launch adversarial review, §24):
- Score DISTINCT candidate answers (not all K): a verifier that reranks answers is the right frame,
  and identical answers get identical scores.
- best-of-N candidate pool = every candidate carrying a boxed answer (closed OR not), so best-of-N
  sees the SAME pool as pass@K -> its honest ceiling IS pass@K.
- The FAIR same-base baseline is the CLOSED-ONLY self-consistency (require '</think>', exactly as
  clean_eval / §23a-g) -> it reproduces the banked 40.3 (SVAMP) / 48.0 (ASDiv) SC that §24 cites.
  The full-pool vote is reported too, but the go/no-go lift is quoted against the closed-only vote.
- Ties in _pick break by the vote's own most_common order, so an UNINFORMATIVE verifier reproduces
  the vote EXACTLY (the "no-op => tie" guarantee actually holds).
- generate stops on '<|im_end|>' so candidates match the DEPLOYED bf16 model (the fp32 config's
  eos=151643 alone would let it ramble and a second \boxed could flip extract_boxed's last-match).

Decision (per §24): external best-of-N approaches pass@K -> the gap is CAPTURABLE, build a 2-model
serving recipe. External best-of-N ~= closed-vote -> the gap is base-LOCKED (verification is
base-limited even for a stronger model on these traces) -> only Tier 3 (a better base) remains.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from star_generate import extract_boxed, norm  # noqa: E402  verified primitives
from clean_eval import load_clean, build_ids   # noqa: E402  clean loaders + chat-template ids


# ----------------------------- verifier prompts ------------------------------
JUDGE_YESNO = (
    "Problem:\n{q}\n\n"
    "A student's proposed final answer is: {ans}\n\n"
    "Is this the correct final answer to the problem? Reply with only \"Yes\" or \"No\"."
)
JUDGE_REASONED = (
    "Problem:\n{q}\n\n"
    "A student's proposed final answer is: {ans}\n\n"
    "Solve the problem yourself, then decide whether the student's answer is correct. "
    "Finish your reply with exactly one line: 'Verdict: Yes' or 'Verdict: No'."
)
SOLVE_PROMPT = (
    "Solve the following problem. Reason step by step, then give ONLY the final "
    "numeric answer as \\boxed{{...}}.\n\n{q}"
)


# ------------------------------- statistics ----------------------------------
def wilson(k, n, z=1.96):
    """95% Wilson score interval (percent) for k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (center - half), 100 * (center + half))


def mcnemar_p(b, c):
    """Two-sided McNemar p (continuity-corrected normal approx). b,c = discordant counts."""
    nn = b + c
    if nn == 0:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / nn
    return math.erfc(math.sqrt(chi) / math.sqrt(2))


# ------------------------------ generate mode --------------------------------
def run_generate(args):
    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    import datetime as _dt

    tok = AutoTokenizer.from_pretrained(args.policy, trust_remote_code=True)
    llm = LLM(model=args.policy, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              trust_remote_code=True)
    # Stop on the trained turn terminator (<|im_end|>=151645) so candidates match the DEPLOYED bf16
    # model. The fp32 config eos=151643 alone would let it ramble past the answer (a 2nd \boxed
    # would flip extract_boxed's last-match). Must use stop_token_ids, NOT a stop string: vLLM
    # skips special tokens in the output text, so a "<|im_end|>" stop string would never match.
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_new_tokens, stop_token_ids=[im_end])

    rows = []
    for src in args.sources:
        probs = load_clean(src, args.n_problems, seed=args.seed)
        if not probs:
            print(f"[generate] {src}: NO PROBLEMS -- skipped", flush=True)
            continue
        prompts = [TokensPrompt(prompt_token_ids=build_ids(tok, q, think=True)) for q, _ in probs]
        t0 = _dt.datetime.now()
        outs = llm.generate(prompts, sp)
        dt = (_dt.datetime.now() - t0).total_seconds()
        for (q, gold), o in zip(probs, outs):
            rows.append({"source": src, "question": q, "gold": gold,
                         "candidates": [co.text for co in o.outputs]})
        # free metrics (sanity gate: pass@K should reproduce §23g ~73-77%)
        srows = [r for r in rows if r["source"] == src]
        n = len(srows)
        n_pass = maj_c = maj_f = tot = corr = unclosed = 0
        for r in srows:
            boxed_all = [(extract_boxed(c), "</think>" in c) for c in r["candidates"]]
            boxed = [p for p, _ in boxed_all if p is not None]
            closed = [p for p, cl in boxed_all if p is not None and cl]
            unclosed += sum(1 for c in r["candidates"] if "</think>" not in c)
            tot += len(r["candidates"])
            corr += sum(1 for p in boxed if p == r["gold"])
            if any(p == r["gold"] for p in boxed):
                n_pass += 1
            vc = Counter(closed)
            vf = Counter(boxed)
            if vc and vc.most_common(1)[0][0] == r["gold"]:
                maj_c += 1
            if vf and vf.most_common(1)[0][0] == r["gold"]:
                maj_f += 1
        print(f"[generate] {src}: {n} probs x K={args.k} in {dt:.1f}s "
              f"({args.k*n/max(dt,1):.0f} samp/s) | single {100*corr/max(tot,1):.1f}% | "
              f"SC-closed {100*maj_c/n:.1f}% | SC-full {100*maj_f/n:.1f}% | "
              f"pass@{args.k} {100*n_pass/n:.1f}% | unclosed {100*unclosed/max(tot,1):.1f}%",
              flush=True)

    json.dump(rows, open(args.gen_out, "w"))
    print(f"[generate] wrote {len(rows)} problems -> {args.gen_out}", flush=True)


# ------------------------------- rerank mode ---------------------------------
def _p_yes_from_logprobs(lp, ids):
    """lp: {token_id: Logprob}. ids = ([yes_ids],[no_ids]). Return P(Yes) or None."""
    yes_ids, no_ids = ids
    ly = max((lp[i].logprob for i in yes_ids if i in lp), default=None)
    ln = max((lp[i].logprob for i in no_ids if i in lp), default=None)
    if ly is None and ln is None:
        return None
    if ly is None:
        return 0.0
    if ln is None:
        return 1.0
    return math.exp(ly) / (math.exp(ly) + math.exp(ln))


def _parse_verdict(text):
    """Reasoned reply -> 1.0 (Yes) / 0.0 (No) / None (unparsed).

    Requires a CLOSED think block: if '</think>' is absent the reply was truncated at the token
    budget mid-thought, so scanning the raw CoT for a bare 'yes/no' would fabricate a confident
    (arbitrary) verdict -> return None (counts as unparsed -> neutral 0.5). Only the post-'</think>'
    verdict text is inspected.
    """
    import re
    if "</think>" not in text:
        return None
    t = text.split("</think>", 1)[-1]
    m = re.findall(r"verdict\s*[:\-]?\s*(yes|no)", t, re.I)
    if m:
        return 1.0 if m[-1].lower() == "yes" else 0.0
    m = re.findall(r"\b(yes|no)\b", t, re.I)  # fallback: last bare yes/no in the verdict text
    if m:
        return 1.0 if m[-1].lower() == "yes" else 0.0
    return None


def _pick(scores, votes):
    """scores: {answer: float}. votes: Counter. Pick argmax score; break ties by the VOTE's own
    most_common order so a constant/uninformative verifier reproduces the majority vote exactly."""
    if not scores:
        return None
    order = [a for a, _ in votes.most_common()]
    order += [a for a in scores if a not in votes]  # answers with no votes (shouldn't happen)
    best, bs = None, None
    for a in order:
        s = scores.get(a, 0.5)
        if bs is None or s > bs:   # strict '>' keeps the first (= highest-vote) among ties
            bs, best = s, a
    return best


def compute_metrics(rows, vscore, solo, kinds):
    """Pure scoring/aggregation (no vLLM) -> {source: {...}}. Unit-testable.

    rows carry precomputed _boxed / _votes (full pool) and _votes_closed (closed-only) + _answers.
    vscore = {kind: {(ri, ans): score}}; solo = {ri: qwen_pred}.
    """
    by_src = defaultdict(list)
    for ri, r in enumerate(rows):
        by_src[r["source"]].append(ri)
    res = {}
    for src, idxs in by_src.items():
        n = len(idxs)
        tot = corr = vote_c = vote_f = n_pass = 0
        bon = {k: 0 for k in kinds}
        disc = {k: [0, 0] for k in kinds}   # [b: bon right & vote_closed wrong, c: bon wrong & vote_closed right]
        solver_solo = solver_cov = solver_noans = 0
        for ri in idxs:
            r = rows[ri]
            gold = r["gold"]
            votes, votesc = r["_votes"], r["_votes_closed"]
            tot += len(r["candidates"])
            corr += sum(1 for p in r["_boxed"] if p == gold)
            if any(p == gold for p in r["_boxed"]):
                n_pass += 1
            vc_ok = bool(votesc and votesc.most_common(1)[0][0] == gold)
            vf_ok = bool(votes and votes.most_common(1)[0][0] == gold)
            vote_c += vc_ok
            vote_f += vf_ok
            for k in kinds:
                if k == "solver":
                    qp = solo.get(ri)
                    if qp is None:
                        solver_noans += 1
                    if qp is not None and qp == gold:
                        solver_solo += 1
                    if qp is not None and qp in votes:
                        solver_cov += 1
                        pick = qp                          # external oracle present -> take it
                    else:
                        pick = votes.most_common(1)[0][0] if votes else None  # fallback: vote
                else:
                    sc = {a: vscore[k].get((ri, a), 0.5) for a in r["_answers"]}
                    pick = _pick(sc, votes)
                ok = (pick == gold)
                bon[k] += ok
                if ok and not vc_ok:
                    disc[k][0] += 1
                if (not ok) and vc_ok:
                    disc[k][1] += 1
        res[src] = {"n": n, "single": 100 * corr / max(tot, 1),
                    "vote_closed": 100 * vote_c / n, "vote_full": 100 * vote_f / n,
                    "passk": 100 * n_pass / n,
                    "bon": {k: 100 * bon[k] / n for k in kinds},
                    "cnt": {"vote_closed": vote_c, "vote_full": vote_f, "passk": n_pass,
                            "bon": dict(bon)},
                    "disc": disc,
                    "solver_solo": 100 * solver_solo / n, "solver_cov": 100 * solver_cov / n,
                    "solver_noans": 100 * solver_noans / n}
    return res


def run_rerank(args):
    import vllm_argonne
    vllm_argonne.register()   # applies the transformers-5.x tokenizer shim (needed for ANY vLLM
    #                           tokenizer load in this env, incl. native Qwen3-4B); arch reg is inert here
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    rows = json.load(open(args.gen_out))
    kinds = ["yesno", "reasoned", "solver"] if args.kind == "all" else [args.kind]

    tok = AutoTokenizer.from_pretrained(args.verifier, trust_remote_code=True)
    yes_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes", " yes")]
    no_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no", " no")]

    llm = LLM(model=args.verifier, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.verifier_max_model_len,
              trust_remote_code=True)

    # per-problem pool: distinct boxed answers + vote counts (full and closed-only)
    for r in rows:
        boxed_all = [(extract_boxed(c), "</think>" in c) for c in r["candidates"]]
        r["_boxed"] = [p for p, _ in boxed_all if p is not None]
        r["_boxed_closed"] = [p for p, cl in boxed_all if p is not None and cl]
        r["_votes"] = Counter(r["_boxed"])
        r["_votes_closed"] = Counter(r["_boxed_closed"])
        # rerank only the top-M distinct answers by vote (the right answer is almost always among
        # them when pass@K is high); caps verifier cost at high K + focuses on plausible candidates.
        if args.rerank_topm and args.rerank_topm > 0:
            r["_answers"] = [a for a, _ in r["_votes"].most_common(args.rerank_topm)]
        else:
            r["_answers"] = sorted(r["_votes"])

    vscore = {k: {} for k in kinds}
    solo = {}

    if "yesno" in kinds:
        vp, meta = [], []
        for ri, r in enumerate(rows):
            for a in r["_answers"]:
                vp.append(TokensPrompt(prompt_token_ids=build_ids(
                    tok, JUDGE_YESNO.format(q=r["question"], ans=a), think=False)))
                meta.append((ri, a))
        outs = llm.generate(vp, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
        n_neutral = 0
        for (ri, a), o in zip(meta, outs):
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            py = _p_yes_from_logprobs(lp, (yes_ids, no_ids))
            if py is None:
                n_neutral += 1
                py = 0.5
            vscore["yesno"][(ri, a)] = py
        print(f"[rerank] yesno: scored {len(vp)} (problem,answer) pairs "
              f"({n_neutral} no-yes/no-token -> 0.5)", flush=True)

    if "reasoned" in kinds:
        vp, meta = [], []
        for ri, r in enumerate(rows):
            for a in r["_answers"]:
                vp.append(TokensPrompt(prompt_token_ids=build_ids(
                    tok, JUDGE_REASONED.format(q=r["question"], ans=a), think=True)))
                meta.append((ri, a))
        outs = llm.generate(vp, SamplingParams(max_tokens=args.verifier_max_tokens, temperature=0.0))
        n_unparsed = 0
        for (ri, a), o in zip(meta, outs):
            v = _parse_verdict(o.outputs[0].text)
            if v is None:
                n_unparsed += 1
                v = 0.5
            vscore["reasoned"][(ri, a)] = v
        frac = 100 * n_unparsed / max(len(vp), 1)
        print(f"[rerank] reasoned: scored {len(vp)} pairs ({n_unparsed} unparsed -> 0.5 = "
              f"{frac:.1f}%{'  <-- HIGH: budget truncation may null this lens' if frac > 15 else ''})",
              flush=True)

    if "solver" in kinds:
        vp = [TokensPrompt(prompt_token_ids=build_ids(tok, SOLVE_PROMPT.format(q=r["question"]),
                                                      think=True)) for r in rows]
        outs = llm.generate(vp, SamplingParams(max_tokens=args.solver_max_tokens, temperature=0.0))
        n_noans = 0
        for ri, o in enumerate(outs):
            qp = extract_boxed(o.outputs[0].text)
            if qp is None:
                n_noans += 1
            solo[ri] = qp
        print(f"[rerank] solver: solved {len(vp)} problems ({n_noans} no boxed answer)", flush=True)

    # --------------------------- score & report ---------------------------
    log = open(args.log, "a") if args.log else None

    def out(*p):
        line = " ".join(str(x) for x in p)
        print(line, flush=True)
        if log:
            log.write(line + "\n"); log.flush()

    res = compute_metrics(rows, vscore, solo, kinds)
    out("=" * 92)
    out(f"EXTERNAL-VERIFIER BEST-OF-N  policy={Path(args.gen_out).stem}  "
        f"verifier={Path(args.verifier).name}  K={args.k}  kinds={kinds}")
    out("  (self-consistency 'closed' = closed-only vote, the fair same-base baseline reproducing "
        "the banked §23g SC;")
    out("   best-of-N & pass@K pool = all boxed candidates; CIs are Wilson 95%; McNemar vs closed vote)")
    out("=" * 92)
    for src, m in res.items():
        n = m["n"]

        def ci(k):
            lo, hi = wilson(k, n)
            return f"[{lo:.1f}-{hi:.1f}]"
        out(f"\n  ---- [{src}]  n={n} ----")
        out(f"    single-sample acc     : {m['single']:.1f}%")
        out(f"    self-cons (closed)    : {m['vote_closed']:.1f}%  {ci(m['cnt']['vote_closed'])}   "
            f"<- fair same-base baseline")
        out(f"    self-cons (full pool) : {m['vote_full']:.1f}%  {ci(m['cnt']['vote_full'])}")
        for k in kinds:
            b, c = m["disc"][k]
            p = mcnemar_p(b, c)
            sig = "SIG" if p < 0.05 else "ns"
            out(f"    BEST-OF-N [{k:<8}]   : {m['bon'][k]:.1f}%  {ci(m['cnt']['bon'][k])}   "
                f"lift vs closed-vote {m['bon'][k]-m['vote_closed']:+.1f}pts  "
                f"(McNemar b={b} c={c} p={p:.3f} {sig})")
        out(f"    pass@{args.k} (ceiling)    : {m['passk']:.1f}%  {ci(m['cnt']['passk'])}")
        if "solver" in kinds:
            out(f"    [solver diag] Qwen-solo acc {m['solver_solo']:.1f}%  |  its answer in "
                f"candidates (coverage) {m['solver_cov']:.1f}%  |  no-boxed {m['solver_noans']:.1f}%")
    out("=" * 92)
    if log:
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["generate", "rerank"])
    ap.add_argument("--policy", default="/project/rcc/youzhi/models/instruct/x_v6v2_040")
    ap.add_argument("--verifier", default="Qwen/Qwen3-4B")
    ap.add_argument("--kind", default="all", choices=["yesno", "reasoned", "solver", "all"])
    ap.add_argument("--gen-out", default="/project/rcc/youzhi/data/ext_verify_candidates.json")
    ap.add_argument("--sources", nargs="+", default=["svamp", "asdiv"])
    ap.add_argument("--n-problems", type=int, default=500)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)        # policy gen
    ap.add_argument("--max-model-len", type=int, default=1536)        # policy ctx
    ap.add_argument("--verifier-max-tokens", type=int, default=3072)  # reasoned judge (avoid trunc null)
    ap.add_argument("--solver-max-tokens", type=int, default=3072)    # solver
    ap.add_argument("--verifier-max-model-len", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rerank-topm", type=int, default=0,
                    help="rerank only the top-M distinct answers by vote (0=all); caps cost at high K")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    if args.mode == "generate":
        run_generate(args)
    else:
        run_rerank(args)


if __name__ == "__main__":
    main()
