"""
Two-axis BASE-QUALITY GATE probe for an argonne4.0 pretrain checkpoint.

Follows reasoning/thinking_training.md: for a BASE (which is all a4 is right now — it has
had no SFT/DPO/CoT), the doc's judge is Step 0's two-axis few-shot probe, and the gate to
justify running the reasoning recipe is the §15 real-base bar:

    >= 14/20 MATH  AND  >= 14/15 GENERAL, simultaneously, on the STANDARD set.

This is the same instrument as reasoning/probe_pretrain_ckpt.py (used for 3.5), but that
script hard-imports 3.5's 2.88B arch constants from continue_pretrain, which are wrong for
a4 (1.04B: hidden 1536 / 32L / 6 heads / 2 KV / inter 4096). Here we load a4's already
EXTRACTED HF dir instead, so the arch comes from its own config.json and no constants are
assumed. Everything else — probe items, few-shot prefixes, greedy decoding, answer
extraction, keyword grading — is imported VERBATIM so the numbers are directly comparable
to every §11-§16 reading and to 3.5's gate history.

Two item sets, reported separately and pooled (the doc built the extension set precisely to
stop anyone over-reading a single 20/15 set):
  STANDARD  20 math / 15 general   <- the gate is defined on this one
  EXTENSION 20 math / 15 general   <- fresh items, same format/grading

Env:
  MODEL_DIR  extracted a4 HF dir (required)
  TOK        tokenizer dir (default: the Qwen3-0.6B-Base tokenizer a4 pretrains with)
  LABEL      name to print for this arm (default: basename of MODEL_DIR)
"""

import os
import sys

import torch

REPO_ROOT = "/home/youzhi/ArgonneAI"
REASONING_DIR = os.path.join(REPO_ROOT, "reasoning")
for _p in (REASONING_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer

import base_probe_general as bpg
# Fresh extension items live in eval_intermix_base; importing it is safe (module-level
# code is only constants + defs -- its 3.5-specific paths are used inside main()).
from eval_intermix_base import GEN_EXT, MATH_EXT

MODEL_DIR = os.environ["MODEL_DIR"].strip()
TOK = os.environ.get("TOK", "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base").strip()
LABEL = os.environ.get("LABEL", os.path.basename(MODEL_DIR.rstrip("/"))).strip()
# Generation budget. 60 == base_probe_general.gen_line's hard-coded value, i.e. what every
# §11-§16 reading (3.0-base 3/20, 3.5 @140k 14/20) was measured with -> the GATE number must
# always be read at 60. Larger values are a TRUNCATION CONTROL only: a4 restates the setup
# verbosely before computing, so at 60 several answers are cut off mid-arithmetic ("6 factorial
# is 6 times 5 times ... ####" with no number). Raising the budget separates "cannot do it"
# from "was not allowed to finish". Report both; never quote the raised number as the gate.
MAXNEW = int(os.environ.get("MAXNEW", "60"))

MATH_GATE, GEN_GATE = 14, 14  # /20 and /15, on the STANDARD set


@torch.no_grad()
def gen_line(model, tok, prompt, max_new):
    """Byte-identical to base_probe_general.gen_line except the token budget is a parameter."""
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_length=ids.shape[1] + max_new, do_sample=False)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return txt.split("Question:")[0].strip(), out.shape[1] - ids.shape[1]


@torch.no_grad()
def probe(model, tok, items, fewshot, kind, label):
    """Greedy few-shot probe. kind='math' -> exact-match on extracted number;
    kind='gen' -> keyword match. Identical grading to base_probe_general/eval_intermix_base."""
    correct, rows, hit_cap = 0, [], 0
    print(f"\n  -- {label} ({len(items)}) --", flush=True)
    for q, gold in items:
        line, n_new = gen_line(model, tok, fewshot + f"Question: {q}\nAnswer:", MAXNEW)
        capped = n_new >= MAXNEW
        if kind == "math":
            pred = bpg.extract_answer(line)
            ok = pred == gold
            shown = f"gold={str(gold):<5} pred={str(pred):<6}"
        else:
            ok = any(k in line.lower() for k in gold)
            shown = f"want={gold[0]:<12}"
        correct += ok
        hit_cap += (capped and not ok)
        rows.append((ok, q, gold, line, capped))
        flag = "CAP" if capped else "   "
        print(f"    [{'Y' if ok else 'n'}]{flag} {q[:44]:44s} {shown} | {line.replace(chr(10), ' ')[:58]}", flush=True)
    if hit_cap:
        print(f"    ^ {hit_cap} of the {len(items) - correct} misses ran to the {MAXNEW}-token cap "
              f"(possible truncation, not capability)", flush=True)
    return correct, rows


def main():
    print("=" * 84)
    print(f"ARGONNE4.0 BASE-QUALITY GATE PROBE  --  {LABEL}")
    print(f"  model: {MODEL_DIR}")
    print(f"  tok  : {TOK}")
    print(f"  gate : MATH >= {MATH_GATE}/20 AND GENERAL >= {GEN_GATE}/15 (standard set, §15 bar)")
    print(f"  budget: {MAXNEW} new tokens" + ("  <- the comparable/gate setting" if MAXNEW == 60
          else "  <- TRUNCATION CONTROL, not comparable to §11-§16 or the gate"))
    print("=" * 84, flush=True)

    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    model = bpg.load_any(MODEL_DIR, None, tok)  # HF-dir branch: arch+theta from config.json
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded: {n_params:,} params  rope_theta={getattr(model.config, 'rope_theta', '?')} "
          f"vocab={model.config.vocab_size}", flush=True)

    ms, _ = probe(model, tok, bpg.MATH_PROBES, bpg.MATH_FEWSHOT, "math", "MATH standard")
    gs, _ = probe(model, tok, bpg.GEN_PROBES, bpg.GEN_FEWSHOT, "gen", "GENERAL standard")
    me, _ = probe(model, tok, MATH_EXT, bpg.MATH_FEWSHOT, "math", "MATH extension")
    ge, _ = probe(model, tok, GEN_EXT, bpg.GEN_FEWSHOT, "gen", "GENERAL extension")

    print("\n" + "=" * 84)
    print(f"SUMMARY  {LABEL}")
    print(f"  MATH     standard {ms:2d}/{len(bpg.MATH_PROBES)}   extension {me:2d}/{len(MATH_EXT)}"
          f"   pooled {ms + me:2d}/{len(bpg.MATH_PROBES) + len(MATH_EXT)}")
    print(f"  GENERAL  standard {gs:2d}/{len(bpg.GEN_PROBES)}   extension {ge:2d}/{len(GEN_EXT)}"
          f"   pooled {gs + ge:2d}/{len(bpg.GEN_PROBES) + len(GEN_EXT)}")
    math_ok, gen_ok = ms >= MATH_GATE, gs >= GEN_GATE
    print(f"  GATE     math {'PASS' if math_ok else 'FAIL'} ({ms}/{len(bpg.MATH_PROBES)} vs >={MATH_GATE})"
          f"   general {'PASS' if gen_ok else 'FAIL'} ({gs}/{len(bpg.GEN_PROBES)} vs >={GEN_GATE})"
          f"   -> {'CLEARED' if (math_ok and gen_ok) else 'NOT cleared'}")
    print("=" * 84, flush=True)

    out = os.environ.get("OUT_JSON")
    if out:
        import json
        json.dump({LABEL: {"maxnew": MAXNEW,
                           "math_standard": ms, "math_ext": me, "gen_standard": gs, "gen_ext": ge,
                           "math_total": len(bpg.MATH_PROBES), "gen_total": len(bpg.GEN_PROBES),
                           "math_ext_total": len(MATH_EXT), "gen_ext_total": len(GEN_EXT),
                           "gate_math": bool(math_ok), "gate_gen": bool(gen_ok),
                           "gate_cleared": bool(math_ok and gen_ok), "model_dir": MODEL_DIR}},
                  open(out, "w"), indent=1)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
