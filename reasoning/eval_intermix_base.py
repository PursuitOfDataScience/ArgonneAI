"""
Deep evaluation of the INTERMIX midtraining checkpoint vs its seed (§16 follow-up).

Compares the ACTUAL intermix seed (pretrain/checkpoint_step_329148.pt), the
argonne-3.0-base HF dir (probe-history continuity), the FineMath Phase-2 base
(the trade-off frame), and the latest intermix checkpoint, on three views:

1. Few-shot probe on BOTH axes: the standard 20-math/15-general set (comparable
   with every §13-§16 number) PLUS a fresh EXTENSION set (20 math + 15 general)
   no probe has used before -- doubles the sample and guards against reading
   too much into any single item set.
2. Held-out perplexity at ctx 1024: NLL on FineWeb windows beyond the intermix
   carve and on FineMath tail-of-last-shard docs. A continuous measure of
   general-forgetting vs math-gain, independent of generation/grading.
   (FineWeb "held-out" means held out from MIDTRAINING; pretraining covered the
   corpus, equally for every model here, so relative deltas are what count.)
3. Long-context NLL curve at 13568 (position-bucketed): the intermix phase
   replaced longmino as the context-extension stage; the seed trained at ctx
   1024 and only extrapolates, so intermix should be flat where the seed climbs.

Env: INTERMIX_CKPT (default: latest checkpoint_step_*.pt in models/midtrain),
     INTERMIX_THETA (default 1000000).
"""

import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/youzhi/ArgonneAI")
from transformers import AutoTokenizer

from base_probe_general import (
    GEN_FEWSHOT,
    GEN_PROBES,
    MATH_FEWSHOT,
    MATH_PROBES,
    gen_line,
    load_any,
)
from quick_base_probe import extract_answer

FINEWEB_BIN = "/project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21-binary/train.bin"
FINEWEB_HEADER_BYTES = 256 * 4
# The 60:40 carve uses tokens [0, 780,099*13570). Start well past it.
FINEWEB_HOLDOUT_START = 11_000_000_000
FINEMATH_LAST_SHARD = "/project/rcc/youzhi/data/finemath/finemath-4plus_qwen3_docbin/train-00063-of-00064"

PPL_WINDOWS = 50
PPL_CTX = 1024
LONGCTX_WINDOWS = 4
LONGCTX_LEN = 13568
LONGCTX_BUCKETS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 13568)]


def latest_intermix_ckpt():
    cands = glob.glob("/project/rcc/youzhi/models/midtrain/checkpoint_step_*.pt")
    return max(cands, key=lambda p: int(re.search(r"_step_(\d+)\.pt$", p).group(1)))


INTERMIX_CKPT = os.environ.get("INTERMIX_CKPT", "").strip() or latest_intermix_ckpt()
INTERMIX_THETA = float(os.environ.get("INTERMIX_THETA", "1000000"))

MODELS = [
    ("argonne-3.0-base (HF dir, probe-history ref)", "/project/rcc/youzhi/models/pretrain/argonne-3.0-base", None),
    ("SEED pretrain 329148 (what intermix started from)", "/project/rcc/youzhi/models/pretrain/checkpoint_step_329148.pt", 1000000.0),
    ("finemath 864124 (pure-math Phase-2)", "/project/rcc/youzhi/models/midtrain_finemath/checkpoint_step_864124.pt", 10000.0),
    (f"INTERMIX {os.path.basename(INTERMIX_CKPT)}", INTERMIX_CKPT, INTERMIX_THETA),
]

# ---- EXTENSION probe sets (fresh items, same format/grading as the standard sets) ----
MATH_EXT = [
    ("What is 7 times 8?", 56),
    ("What is 45 plus 28?", 73),
    ("What is 91 minus 47?", 44),
    ("What is 96 divided by 8?", 12),
    ("What is half of 68?", 34),
    ("What is 15 percent of 200?", 30),
    ("If 3x + 7 = 22, what is x?", 5),
    ("What is the sum of all integers from 1 to 20?", 210),
    ("How many positive divisors does 18 have?", 6),
    ("A rectangle has length 12 and width 5. What is its perimeter?", 34),
    ("A rectangle has length 9 and width 4. What is its area?", 36),
    ("What is 6 factorial (6!)?", 720),
    ("What is the next prime number after 13?", 17),
    ("Tom has 4 bags with 7 apples each. How many apples in total?", 28),
    ("A car travels 50 miles per hour for 4 hours. How far does it go?", 200),
    ("What is 1000 minus 387?", 613),
    ("If a book costs $30 and is 20% off, what is the sale price?", 24),
    ("What is 25 times 4?", 100),
    ("Anna is three times as old as Ben. Ben is 6. How old is Anna?", 18),
    ("What is 2 to the power of 8?", 256),
]

GEN_EXT = [
    ("What is the capital of Spain?", ["madrid"]),
    ("What is the capital of England?", ["london"]),
    ("What is the capital of Russia?", ["moscow"]),
    ("Who wrote the novel Nineteen Eighty-Four?", ["orwell"]),
    ("Who developed the theory of relativity?", ["einstein"]),
    ("What is the longest river in the world?", ["nile", "amazon"]),
    ("What is the tallest mountain in the world?", ["everest"]),
    ("How many legs does a spider have?", ["eight", "8"]),
    ("What animal is known as the king of the jungle?", ["lion"]),
    ("In which year did World War II end?", ["1945"]),
    ("What is the currency of the United States?", ["dollar"]),
    ("What do bees make?", ["honey"]),
    ("How many hours are there in a day?", ["twenty-four", "twenty four", "24"]),
    ("What color is the sky on a clear day?", ["blue"]),
    ("Which country is home to the Eiffel Tower?", ["france", "paris"]),
]


@torch.no_grad()
def probe_math(model, tok, probes, label):
    correct = 0
    print(f"\n  -- {label} ({len(probes)}) --", flush=True)
    for q, gold in probes:
        line = gen_line(model, tok, MATH_FEWSHOT + f"Question: {q}\nAnswer:")
        pred = extract_answer(line)
        ok = pred == gold
        correct += ok
        print(f"    [{'Y' if ok else 'n'}] {q[:46]:46s} gold={str(gold):<5} pred={str(pred):<6} | {line.replace(chr(10), ' ')[:60]}", flush=True)
    return correct


@torch.no_grad()
def probe_gen(model, tok, probes, label):
    correct = 0
    print(f"\n  -- {label} ({len(probes)}) --", flush=True)
    for q, keys in probes:
        line = gen_line(model, tok, GEN_FEWSHOT + f"Question: {q}\nAnswer:")
        ok = any(k in line.lower() for k in keys)
        correct += ok
        print(f"    [{'Y' if ok else 'n'}] {q[:46]:46s} want={keys[0]:<12} | {line.replace(chr(10), ' ')[:60]}", flush=True)
    return correct


@torch.no_grad()
def window_nll(model, ids_1d):
    """Mean next-token NLL (nats) over one window; CE computed chunked in fp32."""
    x = torch.from_numpy(np.asarray(ids_1d[:-1], dtype=np.int64))[None].to("cuda")
    y = torch.from_numpy(np.asarray(ids_1d[1:], dtype=np.int64))[None].to("cuda")
    logits = model(x).logits
    total, n = 0.0, x.shape[1]
    for s in range(0, n, 1024):
        e = min(s + 1024, n)
        total += F.cross_entropy(
            logits[0, s:e].float(), y[0, s:e], reduction="sum"
        ).item()
    del logits
    return total / n


@torch.no_grad()
def longctx_bucket_nll(model, ids_1d):
    """Per-position-bucket NLL over one LONGCTX_LEN window."""
    x = torch.from_numpy(np.asarray(ids_1d[:-1], dtype=np.int64))[None].to("cuda")
    y = torch.from_numpy(np.asarray(ids_1d[1:], dtype=np.int64))[None].to("cuda")
    logits = model(x).logits
    sums = {}
    for lo, hi in LONGCTX_BUCKETS:
        hi = min(hi, x.shape[1])
        total = 0.0
        for s in range(lo, hi, 1024):
            e = min(s + 1024, hi)
            total += F.cross_entropy(logits[0, s:e].float(), y[0, s:e], reduction="sum").item()
        sums[(lo, hi)] = total / max(1, hi - lo)
    del logits
    return sums


def load_holdout_windows():
    fw = np.memmap(FINEWEB_BIN, dtype=np.uint32, mode="r", offset=FINEWEB_HEADER_BYTES)
    fw_windows = [
        np.asarray(fw[FINEWEB_HOLDOUT_START + i * (PPL_CTX + 1):FINEWEB_HOLDOUT_START + (i + 1) * (PPL_CTX + 1)])
        for i in range(PPL_WINDOWS)
    ]
    fw_long = [
        np.asarray(fw[FINEWEB_HOLDOUT_START + 10**8 + i * (LONGCTX_LEN + 1):FINEWEB_HOLDOUT_START + 10**8 + (i + 1) * (LONGCTX_LEN + 1)])
        for i in range(LONGCTX_WINDOWS)
    ]
    lengths = np.load(FINEMATH_LAST_SHARD + ".lengths.npy")
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    fm = np.memmap(FINEMATH_LAST_SHARD + ".bin", dtype=np.uint32, mode="r")
    tail = range(len(lengths) - PPL_WINDOWS, len(lengths))
    fm_windows = [np.asarray(fm[offsets[i]:offsets[i] + PPL_CTX + 1]) for i in tail]
    return fw_windows, fm_windows, fw_long


def main():
    tok = AutoTokenizer.from_pretrained(MODELS[0][1], trust_remote_code=True)
    fw_windows, fm_windows, fw_long = load_holdout_windows()
    summary = []
    for name, path, theta in MODELS:
        print(f"\n{'=' * 78}\n{name}\n  {path}\n{'=' * 78}", flush=True)
        try:
            model = load_any(path, theta, tok)
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            continue

        m_std = probe_math(model, tok, MATH_PROBES, "MATH standard")
        m_ext = probe_math(model, tok, MATH_EXT, "MATH extension")
        g_std = probe_gen(model, tok, GEN_PROBES, "GENERAL standard")
        g_ext = probe_gen(model, tok, GEN_EXT, "GENERAL extension")

        fw_nll = float(np.mean([window_nll(model, w) for w in fw_windows]))
        fm_nll = float(np.mean([window_nll(model, w) for w in fm_windows]))
        print(f"\n  held-out NLL@{PPL_CTX}: FineWeb {fw_nll:.4f} (ppl {np.exp(fw_nll):.2f})   "
              f"FineMath {fm_nll:.4f} (ppl {np.exp(fm_nll):.2f})", flush=True)

        try:
            buckets = [longctx_bucket_nll(model, w) for w in fw_long]
            avg = {k: float(np.mean([b[k] for b in buckets])) for k in buckets[0]}
            print(f"  long-ctx NLL by position (FineWeb, {LONGCTX_WINDOWS} x {LONGCTX_LEN}):", flush=True)
            for (lo, hi), v in avg.items():
                print(f"    pos {lo:>5}-{hi:<5}: {v:.4f} (ppl {np.exp(v):.2f})", flush=True)
            longctx = avg
        except Exception as e:
            print(f"  long-ctx eval failed: {e}", flush=True)
            longctx = None

        summary.append((name, m_std, m_ext, g_std, g_ext, fw_nll, fm_nll, longctx))
        del model
        torch.cuda.empty_cache()

    print(f"\n{'#' * 78}\nSUMMARY\n{'#' * 78}")
    print(f"  {'model':<52} MATH std/ext   GEN std/ext   NLL fw / fm")
    for name, ms, me, gs, ge, fw, fm, _ in summary:
        print(f"  {name:<52} {ms:>2}/20 {me:>2}/20    {gs:>2}/15 {ge:>2}/15   {fw:.3f} / {fm:.3f}")


if __name__ == "__main__":
    main()
