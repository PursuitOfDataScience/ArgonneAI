"""Correctness gate for the vLLM port (§22 / §7 discipline): the vLLM engine's GREEDY output
must match the reference model.py's greedy output token-for-token. Any arch bug (wrong norm,
missing v_norm, sliding window, softcap misplacement) diverges within a few tokens.

Run in 3 separate processes (no GPU co-residence): --mode vllm | ref | compare.
  vllm   -> greedy-decode prompts with the vLLM engine, save token ids
  ref    -> greedy-decode the SAME prompts with HF ArgonneModel (model.py), save token ids
  compare-> load both, report per-prompt match length + first divergence
"""
import argparse
import json
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
RDIR = str(Path(__file__).resolve().parent)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL = "/project/rcc/youzhi/models/instruct/soup_blend_a085"
MAX_NEW = 64
# math + general, think and no-think — cover both code paths.
PROMPTS = [
    ("What is 17 - 5?", True), ("What is 7 times 6?", False),
    ("If 2x + 5 = 17, what is x?", True),
    ("A shop sells pencils 3 for $2. How much do 12 pencils cost?", True),
    ("What is the capital of France?", False),
    ("Which planet is known as the Red Planet?", True),
    ("Who wrote Romeo and Juliet?", False),
    ("What is the sum 1+2+3+4+5+6+7+8+9+10?", True),
]


def build_prompt_ids(tok):
    ids = []
    for q, think in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=think,
                                      return_tensors=None)
        # transformers 5.x may return a BatchEncoding (dict) or nested list — extract flat ids.
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        if len(enc) > 0 and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        ids.append([int(x) for x in enc])
    return ids


def run_vllm(out_path):
    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    pids = build_prompt_ids(tok)
    llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=0.55, max_model_len=2048, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)
    gen = [list(o.outputs[0].token_ids) for o in outs]
    json.dump({"prompt_ids": pids, "gen": gen}, open(out_path, "w"))
    print(f"[vllm] saved {len(gen)} generations -> {out_path}")


def run_ref(out_path):
    import torch
    import model  # noqa: F401 (registers argonne2 for AutoModel)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from eval_math import sample_batch
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    pids = build_prompt_ids(tok)
    m = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                             dtype=torch.bfloat16, low_cpu_mem_usage=True)
    m.to("cuda").eval()
    gen = []
    for p in pids:
        ids = torch.tensor([p], dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = sample_batch(m, tok, ids, max_new_tokens=MAX_NEW, eos_id=eos_id,
                             do_sample=False, temperature=1.0, top_k=0, top_p=0.0)
        gen.append(g[0])
    json.dump({"prompt_ids": pids, "gen": gen}, open(out_path, "w"))
    print(f"[ref] saved {len(gen)} generations -> {out_path}")


def compare(vllm_path, ref_path, tok_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    V = json.load(open(vllm_path))
    R = json.load(open(ref_path))
    assert V["prompt_ids"] == R["prompt_ids"], "prompt ids differ — tokenization mismatch!"
    n_exact = 0
    total_match_frac = 0.0
    print("=" * 70)
    print("vLLM-vs-reference GREEDY token match")
    print("=" * 70)
    for i, (vg, rg) in enumerate(zip(V["gen"], R["gen"])):
        L = min(len(vg), len(rg))
        div = next((j for j in range(L) if vg[j] != rg[j]), L)
        exact = (vg == rg)
        n_exact += exact
        frac = div / max(len(rg), 1)
        total_match_frac += frac
        q = PROMPTS[i][0][:42]
        status = "EXACT" if exact else f"diverge@{div}/{len(rg)}"
        print(f"  [{i}] {status:20} | {q}")
        if not exact and div < L:
            print(f"       ref:  ...{tok.decode(rg[max(0,div-3):div+3])!r}")
            print(f"       vllm: ...{tok.decode(vg[max(0,div-3):div+3])!r}")
    n = len(V["gen"])
    print("-" * 70)
    print(f"  EXACT-match prompts : {n_exact}/{n}")
    print(f"  mean matched-prefix : {100*total_match_frac/n:.1f}% of tokens")
    print("  VERDICT:", "PASS (port is numerically faithful)" if n_exact == n
          else ("CLOSE (accumulation drift — inspect)" if total_match_frac/n > 0.9
                else "FAIL (arch bug — do NOT use the port)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["vllm", "ref", "compare"])
    ap.add_argument("--vllm-out", default="report/vllm_val_vllm.json")
    ap.add_argument("--ref-out", default="report/vllm_val_ref.json")
    args = ap.parse_args()
    if args.mode == "vllm":
        run_vllm(args.vllm_out)
    elif args.mode == "ref":
        run_ref(args.ref_out)
    else:
        compare(args.vllm_out, args.ref_out, MODEL)


if __name__ == "__main__":
    main()
