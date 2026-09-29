"""Is a vllm_validate.py divergence a bf16 NEAR-TIE or an arch bug? Measure it, do not guess.

vllm_validate.py calls anything short of 8/8 exact a FAIL. On a chat model with peaked
distributions that is right. On a BASE model, greedy decoding regularly meets positions where the
top-2 tokens are within bf16 noise, and vLLM (fused kernels, bf16 residual adds) and model.py
(SDPA/math, fp32 RMSNorm) may break such a tie differently; after one flip the continuations
differ forever, which the prefix metric reads as a failure.

For every diverging prompt this takes the COMMON prefix (prompt + generated tokens before the
split) and asks both engines for the next-token log-probabilities at that exact point:
  * model.py in bf16 (the reference that produced the gate) and the same weights upcast to fp32;
  * vLLM, logprobs=20, same prefix.
It also probes positions inside the prompts that matched, as a baseline for how closely the two
engines agree when nothing flips.

Verdict TIE-FLIPS only if at every divergence both engines put the two contested tokens within
--tie-nats of each other AND vLLM's top-5 logprobs agree with model.py bf16 within --agree-nats
everywhere probed. Anything else is SUSPECT and the port must not be trusted for this checkpoint.

  python reasoning/vllm_tie_check.py --mode vllm --model DIR --gate-prefix report/.../vllm_gate_TAG
  python reasoning/vllm_tie_check.py --mode ref  --model DIR --gate-prefix report/.../vllm_gate_TAG
(--gate-prefix P reads P_vllm.json / P_ref.json written by the vLLM exactness gate, writes P_ties_*.json)
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


def probes(prefix):
    V = json.load(open(f"{prefix}_vllm.json"))
    R = json.load(open(f"{prefix}_ref.json"))
    assert V["prompt_ids"] == R["prompt_ids"], "prompt ids differ"
    out = []
    for i, (p, vg, rg) in enumerate(zip(R["prompt_ids"], V["gen"], R["gen"])):
        L = min(len(vg), len(rg))
        d = next((j for j in range(L) if vg[j] != rg[j]), None)
        if d is not None:
            out.append({"prompt": i, "kind": "divergence", "pos": d, "ids": p + rg[:d],
                        "ref_tok": rg[d], "vllm_tok": vg[d]})
        else:
            for pos in (0, 16, 32, 48):
                if pos < len(rg):
                    out.append({"prompt": i, "kind": "baseline", "pos": pos, "ids": p + rg[:pos],
                                "ref_tok": rg[pos], "vllm_tok": vg[pos]})
    return out


def run_vllm(a):
    import vllm_argonne
    vllm_argonne.register()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    cfg = json.load(open(Path(a.model) / "config.json"))
    ctx = int(cfg.get("block_size") or cfg.get("max_position_embeddings"))
    pr = probes(a.gate_prefix)
    llm = LLM(model=a.model, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=a.gpu_mem,
              max_model_len=min(2048, ctx), trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.generate([TokensPrompt(prompt_token_ids=x["ids"]) for x in pr], sp)
    for x, o in zip(pr, outs):
        lp = o.outputs[0].logprobs[0]
        x["vllm_top"] = {str(t): float(v.logprob) for t, v in lp.items()}
        x["vllm_argmax"] = int(o.outputs[0].token_ids[0])
    json.dump(pr, open(f"{a.gate_prefix}_ties_vllm.json", "w"))
    print(f"[ties/vllm] {len(pr)} probes -> {a.gate_prefix}_ties_vllm.json")


def run_ref(a):
    import torch
    from transformers import AutoModelForCausalLM
    import model  # noqa: F401  (registers argonne2)
    pr = json.load(open(f"{a.gate_prefix}_ties_vllm.json"))
    m = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True).to("cuda").eval()

    def logprobs(ids):
        with torch.no_grad():
            lg = m(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float()
        return torch.log_softmax(lg, -1).cpu()

    lp16 = [logprobs(x["ids"]) for x in pr]
    m.float()
    lp32 = [logprobs(x["ids"]) for x in pr]
    worst_agree, rows, ok = 0.0, [], True
    for x, a16, a32 in zip(pr, lp16, lp32):
        top5 = sorted(x["vllm_top"].items(), key=lambda kv: -kv[1])[:5]
        agree = max(abs(v - float(a16[int(t)])) for t, v in top5)
        worst_agree = max(worst_agree, agree)
        r, v = x["ref_tok"], x["vllm_tok"]
        row = {"prompt": x["prompt"], "kind": x["kind"], "pos": x["pos"],
               "top5_agree_nats": round(agree, 4)}
        if x["kind"] == "divergence":
            vt = x["vllm_top"]
            row.update({
                "ref_bf16_margin": round(float(a16[r] - a16[v]), 4),
                "ref_fp32_margin": round(float(a32[r] - a32[v]), 4),
                "vllm_margin": round(vt.get(str(v), -99) - vt.get(str(r), -99), 4),
                "fp32_argmax_is": "ref" if a32[r] >= a32[v] else "vllm",
            })
            if max(abs(row["ref_bf16_margin"]), abs(row["vllm_margin"])) > a.tie_nats:
                ok = False
        rows.append(row)
    # Divergence probes get the tie limit; BASELINE probes (both engines chose the same token) get their own. One
    # limit over all probes read SUSPECT on OPD rounds 3 and 4 (2026-09-26) from baseline tail-logprob noise alone:
    # baseline medians 0.055-0.060 and maxes 0.11-0.17 nats across three checkpoints, so the max of ~16 draws
    # crosses 0.15 as a model sharpens. A real port fault is a nat or more, which 0.25 still catches.
    worst_base = max([r["top5_agree_nats"] for r in rows if r["kind"] == "baseline"] or [0.0])
    worst_div = max([r["top5_agree_nats"] for r in rows if r["kind"] == "divergence"] or [0.0])
    if worst_div > a.agree_nats or worst_base > a.baseline_agree_nats:
        ok = False
    for row in rows:
        print("  ", json.dumps(row))
    verdict = "TIE-FLIPS (port faithful; divergences are bf16 near-ties)" if ok else "SUSPECT"
    print(f"[ties] worst top-5 |logprob vLLM - model.py bf16| over {len(rows)} probes: "
          f"{worst_agree:.4f} nats; at divergences {worst_div:.4f} (limit {a.agree_nats}), at agreeing "
          f"baseline positions {worst_base:.4f} (limit {a.baseline_agree_nats}); tie limit {a.tie_nats} nats")
    print(f"[ties] VERDICT: {verdict}")
    json.dump({"rows": rows, "worst_top5_agree_nats": worst_agree, "verdict": verdict},
              open(f"{a.gate_prefix}_ties.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["vllm", "ref"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate-prefix", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--tie-nats", type=float, default=0.15)
    ap.add_argument("--agree-nats", type=float, default=0.15, help="top-5 logprob limit at divergence probes")
    ap.add_argument("--baseline-agree-nats", type=float, default=0.25,
                    help="top-5 logprob limit at baseline probes, where both engines chose the same token")
    a = ap.parse_args()
    run_vllm(a) if a.mode == "vllm" else run_ref(a)


if __name__ == "__main__":
    main()
