#!/usr/bin/env python3
"""PAIRED ship-gate over DEPLOYABLE reasoning-effort configurations (§33).

Every number this campaign produced up to now was an unpaired accuracy at n=300, which cannot
separate arms: the measured noise floor on this model is large (the eager-vs-compiled kernel path
alone rewrites ~30% of greedy traces and flips correctness on 4-6% of problems, for a net accuracy
change of only ~1pt). So the decision has to rest on PAIRED comparisons at n=1000 with McNemar,
which is what this does.

It evaluates, per model, four configurations that a server could actually run:
  greedy            plain greedy, max_new_tokens 512                      (the current product)
  budget            s1 force-close of an unclosed `</think>`              (clean_eval's "+budget")
  extend<N>         suppress `</think>`, inject the cue, regenerate, xN, then force-close
  selfcons(K)       sample K at T/top_p/top_k, majority vote over closed+boxed
and keeps the per-problem correctness vector for each, so any two cells -- across models or across
configurations -- can be compared with McNemar's exact test on the discordant pairs.

Grading is `clean_eval.grade`'s rule, unchanged, so numbers stay comparable to §32's published
table: a sample counts as correct only if it closed `</think>` AND its `\\boxed{}` matches gold.
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path

RDIR = str(Path(__file__).resolve().parent)
REPO = str(Path(__file__).resolve().parent.parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLOSE_STR = "\n</think>\n\nThe answer is $\\boxed{"
DEFAULT_CUE = "\nWait, let me double-check that.\n"


def mcnemar(a, b):
    """Exact two-sided McNemar on paired boolean vectors. Returns (n01, n10, p)."""
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if not x and y)      # b fixes what a missed
    n10 = sum(1 for x, y in zip(a, b) if x and not y)      # b breaks what a had
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return n01, n10, p


def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return "[--]"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return f"[{100*(c-h):.1f}-{100*(c+h):.1f}]"


def evaluate_model(model, pools, args):
    """Return {pool: {config: {'ok': [bool], 'decoded': float, 'fm': dict}}}."""
    from effort_probe import load_pool, make_llm, n_think_tokens, prompt_ids
    from star_generate import extract_boxed
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm, tok = make_llm(model, args.max_model_len, args.gpu_util, seed=args.seed,
                        enforce_eager=args.enforce_eager)
    cue_ids = tok.encode(args.cue, add_special_tokens=False)
    close_ids = tok.encode(CLOSE_STR, add_special_tokens=False)
    res = {}

    for pool in pools:
        probs = load_pool(pool, args.n, seed=args.seed)
        ids = [prompt_ids(tok, q) for q, _ in probs]
        golds = [g for _, g in probs]
        n = len(golds)
        out = {}

        def grade(texts):
            ok, fm = [], Counter()
            for t, g in zip(texts, golds):
                pred = extract_boxed(t)
                if "</think>" not in t:
                    fm["unclosed"] += 1
                elif pred is None:
                    fm["no_answer"] += 1
                elif pred == g:
                    fm["correct"] += 1
                else:
                    fm["wrong"] += 1
                ok.append(bool(pred is not None and pred == g and "</think>" in t))
            return ok, dict(fm)

        def force_close(prefix_ids, texts, temp=0.0):
            """Append the deployed close to anything unclosed / unanswered, then read the answer."""
            need, meta, finals = [], [], list(texts)
            for i, t in enumerate(texts):
                if "</think>" in t and extract_boxed(t) is not None:
                    continue
                pre = prefix_ids[i] + ([] if "</think>" in t else close_ids)
                need.append(TokensPrompt(prompt_token_ids=ids[i] + pre))
                meta.append((i, "</think>" in t))
            if need:
                o2 = llm.generate(need, SamplingParams(n=1, temperature=temp,
                                                       max_tokens=args.tail))
                for (i, closed), o in zip(meta, o2):
                    finals[i] = texts[i] + ("" if closed else CLOSE_STR) + o.outputs[0].text
            return finals

        # ---- greedy ----------------------------------------------------------
        g_out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids],
                             SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens))
        g_ids = [list(o.outputs[0].token_ids) for o in g_out]
        g_txt = [o.outputs[0].text for o in g_out]
        ok, fm = grade(g_txt)
        out["greedy"] = {"ok": ok, "fm": fm, "decoded": sum(len(x) for x in g_ids) / n,
                         "think_len": sum(n_think_tokens(tok, t) for t in g_txt) / n}

        # ---- greedy + budget-forcing ----------------------------------------
        b_out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids],
                             SamplingParams(n=1, temperature=0.0, max_tokens=args.think_budget))
        b_ids = [list(o.outputs[0].token_ids) for o in b_out]
        b_txt = force_close(b_ids, [o.outputs[0].text for o in b_out])
        ok, fm = grade(b_txt)
        out["budget"] = {"ok": ok, "fm": fm, "decoded": sum(len(x) for x in b_ids) / n + args.tail,
                         "think_len": sum(n_think_tokens(tok, t) for t in b_txt) / n}

        # ---- greedy + N forced continuations --------------------------------
        cur_ids = [list(x) for x in b_ids]
        cur_txt = [o.outputs[0].text for o in b_out]
        dec = [len(x) for x in cur_ids]
        for r in range(1, args.extensions + 1):
            reqs, nxt = [], []
            for i, t in enumerate(cur_txt):
                j = t.find("</think>")
                pre = cur_ids[i] if j < 0 else tok.encode(t[:j], add_special_tokens=False)
                pre = pre + cue_ids
                reqs.append(TokensPrompt(prompt_token_ids=ids[i] + pre))
                nxt.append(pre)
            ox = llm.generate(reqs, SamplingParams(n=1, temperature=0.0,
                                                   max_tokens=args.extend_tokens))
            new_txt = []
            for i, o in enumerate(ox):
                add = list(o.outputs[0].token_ids)
                nxt[i] = nxt[i] + add
                dec[i] += len(cue_ids) + len(add)
                new_txt.append(tok.decode(nxt[i], skip_special_tokens=True))
            cur_ids, cur_txt = nxt, new_txt
            fin = force_close(cur_ids, cur_txt)
            ok, fm = grade(fin)
            out[f"extend{r}"] = {"ok": ok, "fm": fm,
                                 "decoded": sum(dec) / n + args.tail,
                                 "think_len": sum(n_think_tokens(tok, t) for t in fin) / n}

        # ---- sampled K -> self-consistency + pass@K -------------------------
        s_out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids],
                             SamplingParams(n=args.k, temperature=args.temperature,
                                            top_p=args.top_p, top_k=args.top_k,
                                            max_tokens=args.max_new_tokens, seed=args.seed))
        sc_ok, pk_ok, dec_s, fm = [], [], 0, Counter()
        for o, g in zip(s_out, golds):
            votes, any_c = Counter(), False
            for c in o.outputs:
                t = c.text
                dec_s += len(c.token_ids)
                pred = extract_boxed(t)
                if "</think>" not in t:
                    fm["unclosed"] += 1
                elif pred is None:
                    fm["no_answer"] += 1
                elif pred == g:
                    fm["correct"] += 1
                else:
                    fm["wrong"] += 1
                if pred is not None and "</think>" in t:
                    votes[pred] += 1
                    if pred == g:
                        any_c = True
            sc_ok.append(bool(votes and votes.most_common(1)[0][0] == g))
            pk_ok.append(any_c)
        out[f"selfcons{args.k}"] = {"ok": sc_ok, "fm": dict(fm), "decoded": dec_s / n,
                                    "think_len": None}
        out[f"pass{args.k}"] = {"ok": pk_ok, "fm": {}, "decoded": dec_s / n, "think_len": None}

        # ---- PARALLEL x SEQUENTIAL: extend every sampled candidate, then re-vote ----
        # The two scaling axes are usually reported separately. If the effort knob is real, each
        # of the K votes should get better and the majority should improve on plain self-cons --
        # and if it does not, the knob is only repairing the *greedy path*, which is a narrower
        # claim worth stating as such.
        if args.sc_extensions > 0:
            flat_pid, flat_ids, flat_txt = [], [], []
            for pi, o in enumerate(s_out):
                for c in o.outputs:
                    flat_pid.append(pi)
                    flat_ids.append(list(c.token_ids))
                    flat_txt.append(c.text)
            dec_f = [len(x) for x in flat_ids]
            for r in range(1, args.sc_extensions + 1):
                reqs, nxt = [], []
                for j, t in enumerate(flat_txt):
                    kk = t.find("</think>")
                    pre = flat_ids[j] if kk < 0 else tok.encode(t[:kk], add_special_tokens=False)
                    pre = pre + cue_ids
                    reqs.append(TokensPrompt(prompt_token_ids=ids[flat_pid[j]] + pre))
                    nxt.append(pre)
                ox = llm.generate(reqs, SamplingParams(n=1, temperature=0.0,
                                                       max_tokens=args.extend_tokens))
                nt = []
                for j, o in enumerate(ox):
                    add = list(o.outputs[0].token_ids)
                    nxt[j] = nxt[j] + add
                    dec_f[j] += len(cue_ids) + len(add)
                    nt.append(tok.decode(nxt[j], skip_special_tokens=True))
                flat_ids, flat_txt = nxt, nt
            # force-close the flattened set (indices are into `flat_pid`, not `ids`)
            need, meta, fin = [], [], list(flat_txt)
            for j, t in enumerate(flat_txt):
                if "</think>" in t and extract_boxed(t) is not None:
                    continue
                pre = flat_ids[j] + ([] if "</think>" in t else close_ids)
                need.append(TokensPrompt(prompt_token_ids=ids[flat_pid[j]] + pre))
                meta.append((j, "</think>" in t))
            if need:
                o2 = llm.generate(need, SamplingParams(n=1, temperature=0.0,
                                                       max_tokens=args.tail))
                for (j, closed), o in zip(meta, o2):
                    fin[j] = flat_txt[j] + ("" if closed else CLOSE_STR) + o.outputs[0].text
            pv = [Counter() for _ in range(n)]
            pany = [False] * n
            for j, t in enumerate(fin):
                pred = extract_boxed(t)
                if pred is not None and "</think>" in t:
                    pv[flat_pid[j]][pred] += 1
                    if pred == golds[flat_pid[j]]:
                        pany[flat_pid[j]] = True
            out[f"sc{args.k}+ext{args.sc_extensions}"] = {
                "ok": [bool(v and v.most_common(1)[0][0] == g) for v, g in zip(pv, golds)],
                "fm": {}, "decoded": sum(dec_f) / n, "think_len": None}
            out[f"pass{args.k}+ext{args.sc_extensions}"] = {
                "ok": pany, "fm": {}, "decoded": sum(dec_f) / n, "think_len": None}
        res[pool] = out
        print(f"  [{model} / {pool}] " +
              "  ".join(f"{k}={100*sum(v['ok'])/n:.2f}%" for k, v in out.items()), flush=True)

    # A second LLM() in the same process fails with "Free memory on device (18.86/139.73 GiB) is
    # less than desired GPU memory utilization" -- vLLM does not release the KV cache when the
    # handle goes out of scope. Tear it down explicitly; --report-from also lets each model run in
    # its own process, which is what the launcher actually does.
    import gc
    try:
        from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                                     destroy_model_parallel)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="NAME=PATH")
    ap.add_argument("--report-from", nargs="*", default=[],
                    help="merge these gate JSONs and print the report (no GPU)")
    ap.add_argument("--pools", nargs="+", default=["svamp", "asdiv"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--extensions", type=int, default=3)
    ap.add_argument("--sc-extensions", type=int, default=0,
                    help="also extend every sampled candidate N times and re-vote")
    ap.add_argument("--extend-tokens", type=int, default=160)
    ap.add_argument("--cue", default=DEFAULT_CUE)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--think-budget", type=int, default=256)
    ap.add_argument("--tail", type=int, default=48)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--enforce-eager", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.report_from:
        all_res, a0 = {}, None
        for f in args.report_from:
            o = json.load(open(f))
            a0 = a0 or o["args"]
            all_res.update(o["res"])
        for k in ("pools", "n", "k", "extensions", "cue"):
            setattr(args, k, a0[k])
        report(all_res, args)
        return

    # MULTIPLE MODELS MUST NOT SHARE A PROCESS. vLLM does not release its KV cache when an LLM object
    # goes out of scope, so a second engine in the same process dies with
    #   "Free memory on device (25.85/139.73 GiB) on startup is less than desired ... (0.85, 118.77)"
    # and everything after the first model is silently lost -- the run still exits 0 and still prints
    # the FIRST model's numbers, which is exactly how it gets mistaken for a result. This cost two runs
    # on 2026-08-04 (the alpha sweep, then the long-trace screen) even though the loop-per-model
    # workaround was already known and used in the gate scripts. So the isolation lives HERE now, and
    # callers can pass as many --models as they like.
    if len(args.models) > 1:
        import subprocess
        import tempfile
        all_res, parts = {}, []
        tmpdir = tempfile.mkdtemp(prefix="effort_gate_")
        try:
            for i, spec in enumerate(args.models):
                name = spec.split("=", 1)[0]
                out = os.path.join(tmpdir, f"{i}.json")
                argv = [sys.executable, os.path.abspath(__file__), "--models", spec,
                        "--pools", *args.pools, "--n", str(args.n), "--k", str(args.k),
                        "--extensions", str(args.extensions),
                        "--extend-tokens", str(args.extend_tokens),
                        "--max-new-tokens", str(args.max_new_tokens),
                        "--max-model-len", str(args.max_model_len),
                        "--gpu-util", str(args.gpu_util), "--seed", str(args.seed),
                        "--json-out", out]
                print(f"\n=== [{i+1}/{len(args.models)}] {spec}  (isolated subprocess)", flush=True)
                r = subprocess.run(argv)
                if r.returncode != 0 or not os.path.exists(out):
                    print(f"!!! {name} FAILED (rc={r.returncode}) -- dropped from the merge, "
                          f"NOT silently treated as absent", flush=True)
                    continue
                o = json.load(open(out))
                all_res.update(o["res"])
                parts.append(out)
            if not all_res:
                raise SystemExit("every model failed; nothing to report")
            missing = [s.split("=", 1)[0] for s in args.models if s.split("=", 1)[0] not in all_res]
            if missing:
                print(f"\n⚠️MISSING FROM THIS REPORT: {missing}", flush=True)
        finally:
            pass
        if args.json_out:
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            json.dump({"args": vars(args), "res": all_res}, open(args.json_out, "w"))
        report(all_res, args)
        return

    all_res = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        print(f"\n=== {name} = {path}", flush=True)
        all_res[name] = evaluate_model(path, args.pools, args)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump({"args": vars(args), "res": all_res}, open(args.json_out, "w"))
    report(all_res, args)


def report(all_res, args):
    names = list(all_res)
    cfgs = list(all_res[names[0]][args.pools[0]])
    print("\n" + "=" * 100)
    print(f"EFFORT GATE   n={args.n}/pool   K={args.k}   extensions<={args.extensions}   "
          f"cue={args.cue!r}")
    print("=" * 100)
    for pool in args.pools:
        print(f"\n---- {pool} ----")
        print(f"{'model':<14}" + "".join(f"{c:>13}" for c in cfgs))
        for nm in names:
            r = all_res[nm][pool]
            n = len(r[cfgs[0]]["ok"])
            print(f"{nm:<14}" + "".join(f"{100*sum(r[c]['ok'])/n:>12.2f}%" for c in cfgs))
        print(f"{'decoded tok':<14}" + "".join(
            f"{all_res[names[0]][pool][c]['decoded']:>13.0f}" for c in cfgs)
            + f"   ({names[0]})")

    # ---- paired tests -------------------------------------------------------
    print("\n" + "=" * 100)
    print("PAIRED McNEMAR (same problems; n01 = B fixes what A missed, n10 = B breaks what A had)")
    print("=" * 100)
    ref = names[0]
    best_cfg = {}
    for pool in args.pools:
        # the deployable single-pass family only -- self-cons/pass@K are a different compute class
        fam = [c for c in cfgs if c == "greedy" or c == "budget" or c.startswith("extend")]
        for nm in names:
            r = all_res[nm][pool]
            best_cfg[(nm, pool)] = max(fam, key=lambda c: sum(r[c]["ok"]))
        print(f"\n---- {pool} ----")
        for nm in names:
            if nm == ref:
                continue
            for cfg_a, cfg_b in [("greedy", "greedy"),
                                 (best_cfg[(ref, pool)], best_cfg[(nm, pool)])]:
                a = all_res[ref][pool][cfg_a]["ok"]
                b = all_res[nm][pool][cfg_b]["ok"]
                n01, n10, p = mcnemar(a, b)
                d = 100 * (sum(b) - sum(a)) / len(a)
                print(f"  {ref}/{cfg_a:<9} -> {nm}/{cfg_b:<9}  delta {d:+6.2f}pt  "
                      f"n01={n01:<4} n10={n10:<4} p={p:.4g}")
        # within-model: does the effort knob help THIS model?
        for nm in names:
            r = all_res[nm][pool]
            bc = best_cfg[(nm, pool)]
            if bc == "greedy":
                print(f"  [{nm}] best single-pass config IS plain greedy (knob does not help)")
                continue
            n01, n10, p = mcnemar(r["greedy"]["ok"], r[bc]["ok"])
            d = 100 * (sum(r[bc]["ok"]) - sum(r["greedy"]["ok"])) / len(r["greedy"]["ok"])
            print(f"  [{nm}] greedy -> {bc:<9} delta {d:+6.2f}pt  n01={n01:<4} n10={n10:<4} "
                  f"p={p:.4g}   (decoded {r['greedy']['decoded']:.0f} -> {r[bc]['decoded']:.0f} tok)")

    print("\nAGGREGATE over pools (sum of per-pool accuracy, deployable single-pass best):")
    for nm in names:
        tot = sum(100 * sum(all_res[nm][p][best_cfg[(nm, p)]]["ok"]) /
                  len(all_res[nm][p]["greedy"]["ok"]) for p in args.pools)
        totg = sum(100 * sum(all_res[nm][p]["greedy"]["ok"]) /
                   len(all_res[nm][p]["greedy"]["ok"]) for p in args.pools)
        sc = sum(100 * sum(all_res[nm][p][f"selfcons{args.k}"]["ok"]) /
                 len(all_res[nm][p]["greedy"]["ok"]) for p in args.pools)
        print(f"  {nm:<14} greedy {totg:7.2f}   best-single-pass {tot:7.2f}   "
              f"self-cons@{args.k} {sc:7.2f}")


if __name__ == "__main__":
    main()
