#!/usr/bin/env python3
"""Trivial-input regression probe on the DEPLOYED (transformers) path.

WHY: the release staging's own smoke test asked "What is 17 - 5?" and the candidate answered **7** --
it computed 17-5=12 correctly, then subtracted 5 again, and its trained self-verification pass
re-derived the SAME wrong way and reported "Both ways give 7". Aggregate benchmarks (+2.4pt over five
held-out sets) cannot see this: a self-consistency-style check that repeats the same error confirms
it, which is worse than no check because it adds false confidence. Trivial one-step queries are also
exactly what a user types first. So both models are run head-to-head on simple prompts through
`from_pretrained` + `.generate()`, the path the HF card actually exposes.
"""
import argparse, importlib.util, re, sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = "/home/youzhi/ArgonneAI"


def register_argonne():
    """Register `argonne2` with the Auto* classes directly.

    run_arm.sh's config patch REMOVES `auto_map` (the vLLM path supplies the classes itself), so
    `from_pretrained(trust_remote_code=True)` on a raw arm dir raises "Transformers does not
    recognize this architecture". Earlier probe runs only worked by accident, because they loaded the
    staged HF bundle (which has auto_map) FIRST and that registered the classes for the whole
    process; running a candidate alone failed. Registering explicitly removes the ordering
    dependency -- the same reason `vllm_argonne.register()` exists for the vLLM side.
    """
    spec = importlib.util.spec_from_file_location("_argonne_arch", f"{REPO}/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        AutoConfig.register("argonne2", mod.ArgonneConfig)
        AutoModelForCausalLM.register(mod.ArgonneConfig, mod.ArgonneModel)
    except ValueError:
        pass          # already registered (a dir with auto_map got there first)
    return mod

def _gen_probes(n=64, seed=11):
    """Programmatic ONE-STEP arithmetic. The five held-out benchmarks are all multi-step word
    problems, so none of them can see a regression on single-step queries -- which is what a user
    types first. Built with a fixed seed so both models see identical items."""
    import random
    r = random.Random(seed); out = []
    for _ in range(n):
        op = r.choice("+-*/")
        if op == "+":   a, b = r.randint(2, 500), r.randint(2, 500);      g = a + b
        elif op == "-": a, b = r.randint(20, 900), r.randint(2, 19);      g = a - b
        elif op == "*": a, b = r.randint(2, 25), r.randint(2, 12);        g = a * b
        else:           b, q = r.randint(2, 12), r.randint(2, 30); a = b * q; g = q
        sym = {"+": "+", "-": "-", "*": "*", "/": "/"}[op]
        out.append((f"What is {a} {sym} {b}?", str(g)))
    return out


PROBES = [
    ("What is 17 - 5?", "12"), ("What is 7 times 6?", "42"),
    ("What is 100 divided by 4?", "25"), ("What is half of 80?", "40"),
    ("What is 15% of 80?", "12"), ("What is 23 + 19?", "42"),
    ("What is 9 * 9?", "81"), ("What is 50 - 17?", "33"),
    ("What is 144 / 12?", "12"), ("What is 2 + 2?", "4"),
    ("What is 1000 - 1?", "999"), ("What is 6 * 7 + 8?", "50"),
    ("Tom has 5 apples and buys 3 more. How many does he have?", "8"),
    ("A pen costs $2. How much do 4 pens cost?", "8"),
    ("There are 12 eggs in a carton. How many in 3 cartons?", "36"),
    ("Sara had 20 stickers and gave away 6. How many are left?", "14"),
] + _gen_probes()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--probe-seed", type=int, default=11,
                    help="seed 11 = the items build_arith_tier.py EXCLUDES. Any other seed is a "
                         "fresh draw, i.e. a generalisation check beyond the excluded set.")
    ap.add_argument("--n-gen", type=int, default=64)
    a = ap.parse_args()
    register_argonne()
    global PROBES
    PROBES = PROBES[:16] + _gen_probes(a.n_gen, a.probe_seed)
    print(f"probe: 16 hand-written + {a.n_gen} generated (seed {a.probe_seed}) = {len(PROBES)} items")
    for spec in a.models:
        nm, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                 dtype=torch.bfloat16).to("cuda").eval()
        ok = 0; bad = []
        print(f"\n{'='*80}\n### {nm}  ({path})\n{'='*80}")
        for q, gold in PROBES:
            enc = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                          add_generation_prompt=True)
            if hasattr(enc, "keys"):
                enc = enc["input_ids"]
            if len(enc) and isinstance(enc[0], (list, tuple)):
                enc = enc[0]
            ids = torch.tensor([[int(x) for x in enc]], device="cuda")
            with torch.no_grad():
                # NOTE: ArgonneModel.generate() is the model's OWN implementation -- it takes
                # `max_length` (total), not HF's `max_new_tokens`. A user calling
                # .generate(max_new_tokens=...) gets a TypeError on this card.
                out = m.generate(ids, max_length=ids.shape[-1] + a.max_new, do_sample=False,
                                 eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
            txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
            mb = re.findall(r"\\boxed\{([^}]*)\}", txt)
            pred = mb[-1].strip() if mb else None
            hit = pred is not None and re.sub(r"[^\d.-]", "", pred) == gold
            ok += hit
            ntok = out.shape[-1] - ids.shape[-1]
            print(f"  [{'OK ' if hit else 'BAD'}] {q:<58} gold={gold:<5} got={pred}  ({ntok} tok)")
            if not hit:
                bad.append((q, gold, pred, txt))
        print(f"\n  {nm}: {ok}/{len(PROBES)} correct")
        for q, g, p, t in bad[:4]:
            print(f"\n  --- MISS: {q}  (gold {g}, got {p})\n  {t[:700]}")
        del m; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
