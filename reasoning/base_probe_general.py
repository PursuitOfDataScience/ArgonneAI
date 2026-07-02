"""
Few-shot BASE probe on TWO axes: numeracy (math) AND world/general knowledge.

Motivation (§12/§13): FineMath midtraining lifted numeracy to 10/10 downstream
but the latest checkpoint's downstream model is catastrophically broken on general
chat, and full SFT/DPO could NOT recover it. That points at the *base* having
forgotten world knowledge -- but so far that's inferred from downstream, never
measured on the base. This probe measures it directly, and answers the operational
question for the "retrain-with-intermix from the longmino base" plan:
  * Is `midtrain/` (longmino) a general-HEALTHY canvas, or did it already forget?
  * How bad is the forgetting at the latest FineMath checkpoint (base level)?

Loads HF dirs via ArgonneModel.from_pretrained (self-healing: rebuilds RoPE, ties
lm_head). Loads a raw midtraining `.pt` by building ArgonneModel from the
continue_pretrain constants (rope_theta=1e4 for the FineMath base). Greedy decode.
Reuses the math tier from quick_base_probe.py so numeracy numbers stay comparable.
"""

import os
import re
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/youzhi/ArgonneAI")
from model import ArgonneConfig, ArgonneModel
from transformers import AutoTokenizer

# Reuse the exact math tier (keeps numeracy numbers comparable to §11's probe).
# base_probe_general.py lives in reasoning/, so its own dir is on sys.path[0].
from quick_base_probe import FEWSHOT as MATH_FEWSHOT, PROBES as MATH_PROBES, extract_answer

from continue_pretrain import (
    ENABLE_INTERLEAVED_LOCAL_ATTENTION,
    ENABLE_QK_NORM,
    ENABLE_SANDWICH_NORM,
    ENABLE_V_NORM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LOCAL_ATTENTION_WINDOW,
    LOGIT_SOFTCAP,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Z_LOSS_WEIGHT,
)

# (label, path, rope_theta-for-.pt). HF dirs read their own theta from config.json.
BASES = [
    ("argonne-3.0-base (pre-midtrain, PROVEN general)", "/project/rcc/youzhi/models/pretrain/argonne-3.0-base", None),
    ("midtrain longmino (Phase1, the proposed CANVAS)", "/project/rcc/youzhi/models/midtrain/final_model_complete_longmino", None),
    ("midtrain_finemath 864124 (Phase2 latest, .pt)",   "/project/rcc/youzhi/models/midtrain_finemath/checkpoint_step_864124.pt", 10000.0),
]

# Optional extra checkpoint to probe (e.g. the intermix smoke-test result), via env.
_extra = os.environ.get("EXTRA_CKPT", "").strip()
if _extra:
    _label = os.environ.get("EXTRA_LABEL", "EXTRA (intermix smoke-test)")
    _theta = float(os.environ.get("EXTRA_THETA", "1000000"))
    BASES.append((_label, _extra, _theta))

# 4-shot general-knowledge exemplars (same "#### <answer>" convention as math).
GEN_FEWSHOT = """Question: What is the capital of Germany?
Answer: The capital of Germany is Berlin. #### Berlin

Question: Who painted the Mona Lisa?
Answer: The Mona Lisa was painted by Leonardo da Vinci. #### Leonardo da Vinci

Question: What is the largest mammal on Earth?
Answer: The largest mammal is the blue whale. #### blue whale

Question: What planet do humans live on?
Answer: Humans live on Earth. #### Earth

"""

# (question, [acceptable lowercase keywords]) -- unambiguous single-fact world knowledge.
GEN_PROBES = [
    ("What is the capital of France?", ["paris"]),
    ("Which planet is known as the Red Planet?", ["mars"]),
    ("What gas do humans breathe in to stay alive?", ["oxygen"]),
    ("Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    ("What is the largest planet in the solar system?", ["jupiter"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is the capital of Italy?", ["rome"]),
    ("Who was the first president of the United States?", ["washington"]),
    ("How many continents are there on Earth?", ["seven", "7"]),
    ("What is the opposite of hot?", ["cold"]),
    ("What is the chemical symbol for water?", ["h2o", "h₂o", "h2 o"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("What language is mainly spoken in Brazil?", ["portuguese"]),
    ("How many days are there in a week?", ["seven", "7"]),
    ("Is the sun a star or a planet?", ["star"]),
]


def load_any(path, rope_theta, tok):
    if path.endswith(".pt"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        vocab = len(tok)
        cfg = ArgonneConfig(
            vocab_size=vocab, hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS,
            num_attention_heads=NUM_HEADS, num_key_value_heads=NUM_KV_HEADS,
            intermediate_size=INTERMEDIATE_SIZE, max_position_embeddings=13568,
            rope_theta=rope_theta, use_flash_attention=True, qk_norm=ENABLE_QK_NORM,
            v_norm=ENABLE_V_NORM, sandwich_norm=ENABLE_SANDWICH_NORM,
            z_loss_weight=Z_LOSS_WEIGHT,
            interleaved_local_attention=ENABLE_INTERLEAVED_LOCAL_ATTENTION,
            local_attention_window=LOCAL_ATTENTION_WINDOW if ENABLE_INTERLEAVED_LOCAL_ATTENTION else None,
            logit_softcap=LOGIT_SOFTCAP, tie_word_embeddings=True,
        )
        cfg.block_size = 13568
        cfg.rope_theta = rope_theta
        cfg._keep_in_fp32_modules = []
        model = ArgonneModel(cfg)
        model.load_state_dict(state, strict=False)
        embed = model.get_input_embeddings()
        if embed.weight.shape[0] > vocab:
            embed.weight = nn.Parameter(embed.weight[:vocab])
            lm = model.get_output_embeddings()
            if lm is not None:
                lm.weight = nn.Parameter(lm.weight[:vocab])
            model.config.vocab_size = vocab
        return model.to(torch.bfloat16).to("cuda").eval()
    return ArgonneModel.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()


@torch.no_grad()
def gen_line(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_length=ids.shape[1] + 60, do_sample=False)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return txt.split("Question:")[0].strip()


@torch.no_grad()
def run_math(model, tok):
    correct, rows = 0, []
    for q, gold in MATH_PROBES:
        line = gen_line(model, tok, MATH_FEWSHOT + f"Question: {q}\nAnswer:")
        pred = extract_answer(line)
        ok = (pred == gold)
        correct += ok
        rows.append((ok, q, gold, pred, line.replace("\n", " ")[:70]))
    return correct, rows


@torch.no_grad()
def run_general(model, tok):
    correct, rows = 0, []
    for q, keys in GEN_PROBES:
        line = gen_line(model, tok, GEN_FEWSHOT + f"Question: {q}\nAnswer:")
        low = line.lower()
        ok = any(k in low for k in keys)
        correct += ok
        rows.append((ok, q, keys[0], line.replace("\n", " ")[:70]))
    return correct, rows


def main():
    tok = AutoTokenizer.from_pretrained(BASES[0][1], trust_remote_code=True)
    summary = []
    for name, path, theta in BASES:
        print(f"\n{'='*74}\n{name}\n  {path}\n{'='*74}", flush=True)
        try:
            model = load_any(path, theta, tok)
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            summary.append((name, -1, -1))
            continue
        mc, mrows = run_math(model, tok)
        print(f"\n  -- MATH ({len(MATH_PROBES)}) --", flush=True)
        for ok, q, gold, pred, snip in mrows:
            print(f"    [{'Y' if ok else 'n'}] {q[:44]:44s} gold={str(gold):<5} pred={str(pred):<6} | {snip}", flush=True)
        gc, grows = run_general(model, tok)
        print(f"\n  -- GENERAL ({len(GEN_PROBES)}) --", flush=True)
        for ok, q, gold, snip in grows:
            print(f"    [{'Y' if ok else 'n'}] {q[:44]:44s} want={gold:<12} | {snip}", flush=True)
        print(f"\n  ==> {name}\n      MATH {mc}/{len(MATH_PROBES)}   GENERAL {gc}/{len(GEN_PROBES)}", flush=True)
        summary.append((name, mc, gc))
        del model
        torch.cuda.empty_cache()

    print(f"\n{'#'*74}\nSUMMARY (few-shot base probe)   MATH /{len(MATH_PROBES)}   GENERAL /{len(GEN_PROBES)}\n{'#'*74}")
    for name, mc, gc in summary:
        print(f"  MATH {str(mc):>3}   GENERAL {str(gc):>3}   {name}")


if __name__ == "__main__":
    main()
