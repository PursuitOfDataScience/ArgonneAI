# experiments.md — Argonne-Next architecture search (1× H200, ~30 min/run [35 hard cap], 50 valid experiments)

> **Read this top to bottom before touching anything.** This document is the operating manual for an
> autonomous agent that will run a 50-experiment pretraining-architecture search to find the *next*
> Argonne LM, starting from the current production design. It is a plan — **do not run anything yet
> beyond what each numbered step authorizes.**
>
> **Prime directive on scope:** Everything you create or modify lives under `experiments/`. **Never edit
> any file in the existing codebase** (`model.py`, `pretrain.py`, `continue_pretrain.py`, `*.sh`, etc.).
> You will *copy* the bits you need into `experiments/` and modify the copies.

---

## 0. Table of contents

1. Mission and the hard constraints (non-negotiable)
2. Verified environment facts (paths, SLURM, data, tokenizer)
3. The proxy model — what "the model" is during the search, and why
4. Anti-cheating invariants (locked for all 50 runs)
5. The benchmark metric and the exact evaluation protocol
6. The comparison protocol: iso-data test loss + a measured noise floor
7. Directory layout under `experiments/`
8. Artifacts you must create (with skeletons): `model.py` copy, `train_probe.py`, `run_experiment.sh`, config schema
9. Experiment 0 — calibration (locks the global token budget)
10. The 50-experiment curriculum (phased prior over a greedy adaptive search)
11. The adaptive decision procedure ("after each job, plan and justify the next")
12. The submit-one-job-and-wait-patiently loop (exact commands, robustness, resume)
13. Guardrails: OOM, NaN, divergence, timeout, compile failure
14. Bookkeeping: `results.jsonl`, `leaderboard.md`, `decisions.md`
15. Cleanup rules
16. Final deliverable
17. Timing/throughput math (worked example)
18. Risks and honest caveats

---

## 1. Mission and the hard constraints

**Mission.** Starting from the current production architecture (`model.py` / `pretrain.py`, ~2.88B
"Argonne 3.0"), discover a set of architecture and optimization changes that **lower held-out test
loss** at a fixed training budget, and end with a concrete, justified recipe for "Argonne-Next" that
can be ported to the full 2.88B pretrain.

**Hard constraints (every one of these is a gate — violating any invalidates the run):**

| # | Constraint | How it is enforced |
|---|---|---|
| C1 | **1× H200 GPU per experiment.** | `--gres=gpu:1 --constraint=H200`; single-process (no DDP). |
| C2 | **Target 30 min/run; soft margin +5 min (hard cap 35 min).** A run may run past 30 to *finish its full budget*, but never past ~35. | SLURM `--time=00:35:00` hard kill; in-script abort guard at 33 min (1980 s) writes diagnostics before the kill. The token budget is calibrated (Exp 0) so runs *complete* well inside this window — see C13. |
| C3 | **Benchmark = held-out test loss** (next-token cross-entropy / perplexity). | Section 5. |
| C4 | **Context length == pretrain base context = 1024 tokens, for train AND eval.** Never change it — a longer context lowers per-token loss artificially and is **cheating**. | `block_size` is hard-pinned to 1024 in `train_probe.py`; configs may not set it. |
| C5 | **Model size stays in the 1B–3B ballpark.** Default proxy ≈ 1.03B (Section 3). | Param count is computed and logged every run; reject configs outside ~0.8B–3B. |
| C6 | **One job in flight at a time.** Submit, then **wait patiently** for completion; only then parse, decide, and submit the next. | Section 12 loop; never `sbatch` a new run while a prior one is queued/running. |
| C7 | **Save no checkpoints, no model exports, nothing heavy.** | `train_probe.py` never writes `.pt`/`safetensors`/HF dirs. Only a small JSON result + the SLURM log. |
| C8 | **Same test data every run** (same file, same tail offset, same #eval tokens, same block size, deterministic). | Section 5; offsets are constants. |
| C9 | **Test time stays small/reasonable** (~1 minute). | Fixed `eval_tokens ≈ 3.07M` (3000 windows × 1024); eval once at end (+ optional 1 mid-point). |
| C10 | **Run exactly 50 *valid* experiments** (Exp 0 calibration is setup, not one of the 50; Exp 1..50 are the search). The campaign ends only when 50 valid results exist. | `#valid/50` counter in `leaderboard.md`; see C13. |
| C11 | **Touch nothing in the codebase.** All work under `experiments/`. | Section 7; you copy `model.py`, you never edit the original. |
| C12 | **No build/cache artifacts left behind** (`__pycache__`, `.pyc`, …) per repo `CLAUDE.md`. | Section 15 cleanup after every job. |
| C13 | **Every one of the 50 must be valid; a failed run is retried as the SAME experiment, never skipped.** A run is *valid* only if it completed the full locked budget, produced a finite test loss, and kept all invariants. Operational failures (timeout, OOM, crash, transient NaN) are diagnosed and re-run under the same id; the `#valid/50` counter does **not** advance until the slot holds a valid result. | Section 11 (validity gate) + Section 13 (failure taxonomy). |

---

## 2. Verified environment facts

These were verified on the machine; use them verbatim.

**SLURM (absolute paths — `$PATH` may not have them):**
- `sbatch`  = `/software/slurm-current-el8-x86_64/bin/sbatch`
- `squeue`  = `/software/slurm-current-el8-x86_64/bin/squeue`
- `scancel` = `/software/slurm-current-el8-x86_64/bin/scancel`
- `sacct`   = `/software/slurm-current-el8-x86_64/bin/sacct`
- Account: `rcc-staff`. Partition: `test` (H200 nodes live here; `gpu` is a fallback). Partition time limit is `infinite`, so **you** set `--time=00:35:00` (30-min target + 5-min soft margin, C2).
- H200 nodes exist and are sometimes idle (e.g. an `epyc-9335,768g,H200` node and a `gold-6542Y,1t,H200` node). Request with `--constraint=H200 --gres=gpu:1`.
- **Nodes to avoid:** carry the maintained exclude list from `continue.sh` **verbatim** — `--exclude=midway3-0423,midway3-[0298,0377-0378,0603-0606]`. If a run dies on a node for hardware reasons, add that node here, treat it as an operational failure (C13), and re-run the same experiment elsewhere.

**Python environment (mirror `run_full_training.sh` exactly):**
```bash
module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
This env has torch + flash-attn (the production runs use it). `flash_attn` being present matters: the
interleaved **local/sliding-window attention only takes effect on the flash-attn path** (see the
warning printed by `ArgonneModel.__init__`). Confirm the startup log says
`flash-attn-2; sliding window 256 active on odd layers` in Exp 0; if it says "SDPA/math … IGNORED",
your local-attention experiments are meaningless — fix the env first.

**Data (the pretrain corpus — this is the only data we use):**
- `train.bin` = `/project/rcc/youzhi/fineweb-binary-qwen3/train.bin`
- Format: 256×`int32` header (1024 bytes, magic `20240801`) then `uint32` token stream.
- Size = 83,356,086,840 bytes → **20,839,021,454 tokens (≈ 20.84B)**.
- There is **no separate val/test `.bin`.** We carve a held-out tail (Section 5).

**Tokenizer:**
- `/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base`, `vocab_size = 151936`, eos `<|endoftext|>`.
- The model's default `vocab_size` already matches (151936). Use this tokenizer/vocab for everything.

**Production reference (what we are trying to beat / improve on), from `pretrain.py` + `run_full_training.sh`:**
- 2.88B: hidden 3072, 24 layers, 12 query / 4 KV heads, head_dim 256, SwiGLU intermediate 8192.
- qk_norm ✓, v_norm ✓, sandwich_norm ✓, interleaved local attention (window 256) ✓, logit softcap 15.0, rope_theta 1e6, tied embeddings, z_loss 0, MTP off.
- Production optimizer: AdamW, **LR 3e-4**, betas (0.9, 0.95), wd 0.1, grad_clip 1.0, WSD schedule, warmup 1000, min_lr_ratio 0.1, bf16, torch.compile on, grad-checkpointing on, effective batch ≈ 233K tokens, block 1024.

---

## 3. The proxy model — what "the model" is during the search, and why

We cannot run the full 2.88B for a meaningful number of steps in 30 minutes on one GPU (it would do
only ~60–90 optimizer steps — pure noise). So the search runs on a **faithfully down-scaled Argonne
3.0 proxy** that keeps *every* production structural feature but is small enough to take hundreds of
steps in the budget. This is the standard small-proxy methodology (nanoGPT-speedrun / Chinchilla-style
small-scale ablations).

**Baseline proxy `exp001` (the reference all experiments are measured against):**

| Field (`ArgonneConfig`) | Value | Note |
|---|---|---|
| `hidden_size` | 2048 | down from 3072 |
| `num_hidden_layers` | 16 | down from 24 |
| `num_attention_heads` | 16 | head_dim = 2048/16 = **128** (standard; production's 256 is itself an axis — see Phase 3) |
| `num_key_value_heads` | 4 | GQA ratio 4 |
| `intermediate_size` | 5632 | ≈ 2.75× hidden (SwiGLU) |
| `qk_norm` / `v_norm` / `sandwich_norm` | True / True / True | production defaults ON |
| `interleaved_local_attention` | True | local window on odd layers |
| `local_attention_window` | 256 | |
| `logit_softcap` | 15.0 | |
| `rope_theta` | 1e6 | |
| `z_loss_weight` | 0.0 | |
| `mtp_horizon` / `mtp_loss_weight` | 1 / 0.0 | MTP off |
| `tie_word_embeddings` | True | |
| `max_position_embeddings` (block) | **1024** | **pinned (C4)** |
| `vocab_size` | 151936 | Qwen3 |

**Parameter count ≈ 1.03B** (computed):
- tied embed/lm_head: 151,936 × 2048 = 311,164,928
- per layer: attn 10,485,760 + SwiGLU MLP 34,603,008 + norms ≈ 8,576 = 45,097,344 → ×16 = 721,557,504
- final norm ≈ 2,048
- **total ≈ 1,032,724,480 (1.03B)**; non-embedding ≈ 721.6M.

This sits at the low end of the allowed 1B–3B band **on purpose**: smaller → more tokens/steps in 30
min → lower-variance test-loss comparisons. It is still a faithful proxy (same head structure, same
norm stack, same attention pattern, same vocab). If you later want a heavier proxy, you may raise it,
but re-run calibration (Section 9) and re-establish the noise floor (Section 6) — don't mix proxies
within one comparison.

**Why this is a valid proxy for a 2.88B decision:** changes that help *optimization and stability*
(LR/schedule/warmup, qk/v/sandwich norms, softcap, z-loss, init, betas, clip, GQA, rope) transfer
well from 1B→3B because they act on early-training dynamics. Changes that buy *capacity* (MLP ratio,
depth, untying embeddings) transfer less reliably at this short horizon — those are Phase 4 and are
explicitly flagged lower-confidence and validated at the largest feasible token budget. See Section 18.

---

## 4. Anti-cheating invariants (locked for all 50 runs)

Print these from `train_probe.py` at startup and assert them. If any differs across runs, the
comparison is void.

1. **`block_size == 1024`** for training and evaluation. Hard-coded; not a config field. (C4)
2. **Test set is identical every run:** same file, same `test_offset_tokens = 20_600_000_000`, same
   `eval_tokens = 3_072_000` (3000 windows of 1024), same block size, deterministic order, `model.eval()`,
   no dropout, no sampling.
3. **Train and test never overlap.** Training always starts at token offset 0 and reads forward; the
   maximum a run consumes is `steps × effective_batch ≈ 67M` tokens — three orders of magnitude below
   the test offset at 20.6B. (The tail region has 239M tokens; we use 3.07M of it.)
4. **The reported test loss is *pure next-token cross-entropy*** computed from the model's returned
   logits with **no auxiliary terms** (z-loss / MTP must not leak into the metric — call the model
   *without* `labels` to get logits, then compute CE yourself). The model's own `logit_softcap` stays
   (it is part of the architecture under test). See Section 5.
5. **Identical tokenizer/vocab (151936)** every run.
6. **Same fixed training-token budget** (`steps × effective_batch`, both fixed) for the headline
   comparison — this makes it iso-data. Effective batch in tokens is held constant **except** in the
   dedicated batch-size experiments, which co-scale LR and are labeled.
7. **Deterministic data stream.** The loader is sequential from offset 0 (no shuffling), so every run
   sees the identical token stream in the identical order; the only randomness is weight init (seed).
   This is why the noise floor (Section 6) is small and measurable.
8. **No checkpoints / no exports** (C7). **No codebase edits** (C11).

---

## 5. Benchmark metric and exact evaluation protocol

**Metric:** mean next-token cross-entropy (nats/token) on the fixed held-out tail, reported as
`test_loss` and `test_ppl = exp(test_loss)`. Lower is better.

**Protocol (deterministic, ~1 minute):**
- Open `train.bin`, seek to `test_offset_tokens = 20_600_000_000`.
- Take `N_eval = 3000` contiguous, **non-overlapping** windows of length 1025 (1024 inputs + 1 shift),
  i.e. ~3.07M eval tokens predicting ~3.069M positions. Fixed for all runs.
- `model.eval()`, `torch.no_grad()`, bf16 autocast (same as training forward).
- For each window: `logits = model(x).logits` **(no `labels`)**; `ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), y[:, ...].reshape(-1))` over the shifted targets. Accumulate token-weighted mean.
- **Do not** use `outputs.loss`: when z-loss or MTP is enabled, `outputs.loss` includes those terms and
  would contaminate the metric (invariant #4). The logit softcap *is* applied inside `forward` before
  logits are returned, so it correctly stays part of this architecture's measured loss.
- Optionally take a second identical eval at the schedule's mid-point to record a learning-curve point
  (cheap, deterministic). The end-of-run value is the headline.

---

## 6. Comparison protocol: iso-data test loss + a measured noise floor

**Headline comparison = iso-data:** every run trains on exactly the same number of tokens
(`steps × effective_batch`, both fixed by Exp 0), so a lower `test_loss` means the change learned more
from the *same data*. Always also log `params`, `tokens_per_sec`, `wall_seconds`, `peak_mem_gb` so
capacity/speed trade-offs are visible.

**You must measure the noise floor before trusting any "win."** The only randomness is weight init
(seed). In Exp 1–3 run the baseline at 3 seeds (444, 445, 446) and compute the test-loss standard
deviation **σ**. Then:

- **Accept** a change into the running-best config only if it improves test loss by
  **> max(2σ, 0.003 nats)** versus the current running-best.
- When you *adopt* a new best (especially when stacking several changes), **confirm it with one extra
  seed**; if the second seed regresses past the noise band, do not adopt.
- Treat anything within ±2σ as **neutral** — record it, prefer the simpler/faster option, move on.
- Re-measure σ once mid-campaign (the noise floor can drift as the config changes).

**Search strategy = greedy coordinate descent with re-validation.** Maintain a single "running-best"
config (starts = baseline). Test one change at a time against it; fold in accepted changes; periodically
(every ~10 experiments) re-validate the stacked best by re-running the immediately-prior winner to
catch interactions. Reallocate budget away from dead axes toward promising ones (Section 11).

---

## 7. Directory layout under `experiments/`

Create this tree (and nothing outside it):

```
experiments/
├── experiments.md            # a synced copy of THIS plan (optional but recommended)
├── README.md                 # 10-line quickstart you write for future-you
├── model.py                  # VERBATIM copy of ../model.py (edit only this copy if an axis needs code)
├── train_probe.py            # the stripped trainer (Section 8.2) — single GPU, no ckpt, evals test loss
├── run_experiment.sh         # SLURM wrapper (Section 8.3): 1×H200, 30-min cap, env, calls train_probe.py
├── configs/
│   ├── exp000_calibration.json
│   ├── exp001_baseline.json
│   └── expNNN_<slug>.json     # one per experiment
├── results/
│   ├── results.jsonl          # append-only ledger, one JSON object per run (Section 14)
│   ├── leaderboard.md         # human-readable, sorted; current running-best; status counter
│   ├── decisions.md           # the narrative: per-experiment hypothesis + verdict + next-step justification
│   └── logs/expNNN.out|.err   # SLURM stdout/stderr
└── .gitignore                 # ignore results/logs/*, __pycache__/, *.pt just in case
```

---

## 8. Artifacts you must create

### 8.1 `experiments/model.py` — verbatim copy

```bash
mkdir -p experiments/configs experiments/results/logs
cp model.py experiments/model.py        # copy, never symlink-then-edit the original
cp experiments.md experiments/experiments.md   # optional synced copy of this plan
```
`train_probe.py` imports `ArgonneConfig, ArgonneModel` from **this copy** (it inserts the
`experiments/` dir on `sys.path`). Most experiments are pure `ArgonneConfig` toggles and need **no**
code edit. Only a few axes need code (e.g. alternative weight-init scaling, embedding scaling) — for
those, edit **`experiments/model.py`** and note in the config which code variant is active. The
original `model.py` is never touched.

### 8.2 `experiments/train_probe.py` — contract + skeleton

**Contract:** single-process, single-GPU; build `ArgonneConfig` from a JSON config; train *exactly*
`steps` optimizer steps from token offset 0 at the fixed effective batch; **no checkpointing, no
generation, no export**; evaluate pure next-token CE on the fixed tail; write one JSON result and
append to `results.jsonl`; honor a 33-minute abort guard (Section 1, C2/C13); print and assert the anti-cheating invariants.

Key correctness points (these are the easy-to-get-wrong parts — implement them exactly):
- `BLOCK_SIZE = 1024` is a module constant, **not** read from config (C4).
- Two data loaders over the same memmap: **train** from `start_token_offset=0`; **test** from
  `start_token_offset=20_600_000_000`. Both yield `(x, y)` with `T=1024`.
- Eval computes CE from logits with `labels=None` (invariant #4).
- Abort guard: if `time()-t0 > 1980` (33 min) the run has blown past the soft window — stop, set
  `status="timeout_invalid"`, write diagnostics. This is a FAILED run to be re-run (C13), **not** an
  acceptable result; correct calibration (Exp 0) makes it essentially never fire.
- OOM fallback: on `torch.cuda.OutOfMemoryError`, halve micro-batch and double grad-accum (keeps
  effective batch → iso-data preserved), record `oom_fallback=True`; if still OOM, enable
  gradient checkpointing for that run and record it.
- Determinism: set seeds; `cudnn.benchmark=True` and `allow_tf32=True` (matches production); accept the
  small nondeterminism — that is exactly what the seed-noise floor (Section 6) quantifies.

```python
#!/usr/bin/env python3
"""experiments/train_probe.py — short-horizon proxy trainer for Argonne-Next arch search.
No checkpoints. No export. Fixed block_size=1024. Reports pure next-token test CE."""
import os, sys, json, time, math, argparse, random
import numpy as np, torch, torch.nn.functional as F

BLOCK_SIZE   = 1024                       # PINNED — context length == pretrain base (C4)
TRAIN_OFFSET = 0
TEST_OFFSET  = 20_600_000_000             # tail held-out region (>>> any training reach)
EVAL_WINDOWS = 3000                       # 3000*1024 ≈ 3.07M eval tokens (~1 min)
WALL_GUARD_S = 1980                       # 33 min abort guard (SLURM --time=00:35:00 = 30 target + 5 soft)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # use experiments/model.py
from model import ArgonneConfig, ArgonneModel
from transformers import AutoTokenizer

def load_tokens(path):
    with open(path, "rb") as f:
        magic = np.frombuffer(f.read(256*4), dtype=np.int32)[0]
        assert magic == 20240801, f"bad magic {magic}"
    return np.memmap(path, dtype=np.uint32, mode="r", offset=256*4)

class Loader:
    def __init__(self, toks, B, T, offset):
        self.toks, self.B, self.T, self.start = toks, B, T, int(offset)
        self.pos = self.start
    def batch(self):
        need = self.B*self.T + 1
        if self.pos + need > len(self.toks):      # wrap within the *train* region only
            self.pos = self.start
        buf = torch.from_numpy(self.toks[self.pos:self.pos+need].astype(np.int64))
        x = buf[:-1].view(self.B, self.T); y = buf[1:].view(self.B, self.T)
        self.pos += self.B*self.T
        return x, y

def build_config(cfg, vocab):
    # Whitelist of ArgonneConfig fields the search may set. block_size is NOT here (pinned).
    allowed = {"hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads",
               "intermediate_size","qk_norm","v_norm","sandwich_norm","z_loss_weight",
               "mtp_horizon","mtp_loss_weight","interleaved_local_attention","local_attention_window",
               "logit_softcap","rope_theta","tie_word_embeddings","rms_norm_eps","attention_dropout"}
    arch = {k: cfg[k] for k in cfg if k in allowed}
    return ArgonneConfig(vocab_size=vocab, max_position_embeddings=BLOCK_SIZE,
                         use_flash_attention=True, **arch)

def lr_lambda_factory(warmup, steps, cooldown, min_ratio, sched):
    def f(step):
        if step < warmup: return step / max(1, warmup)
        if sched == "cosine":
            p = (step - warmup) / max(1, steps - warmup)
            return max(min_ratio, 0.5*(1+math.cos(math.pi*p)))
        cd_start = max(warmup, steps - cooldown)            # WSD
        if cooldown <= 0 or step < cd_start: return 1.0
        return 1.0 - min(1.0,(step-cd_start)/max(1,cooldown))*(1.0-min_ratio)
    return f

@torch.no_grad()
def evaluate(model, toks, B, device, dtype):
    model.eval()
    ld = Loader(toks, B, BLOCK_SIZE, TEST_OFFSET)
    tot_loss, tot_tok, done = 0.0, 0, 0
    while done < EVAL_WINDOWS:
        b = min(B, EVAL_WINDOWS - done)
        ld.B = b; x, y = ld.batch(); ld.B = B
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model(x).logits                          # pure logits; NO labels (invariant #4)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        n = y.numel(); tot_loss += loss.item()*n; tot_tok += n; done += b
    model.train()
    return tot_loss/tot_tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/project/rcc/youzhi/fineweb-binary-qwen3/train.bin")
    ap.add_argument("--tokenizer", default="/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base")
    a = ap.parse_args()
    cfg = json.load(open(a.config))

    # --- training knobs (defaults = baseline; Exp 0 sets steps/batch) ---
    steps      = int(cfg.get("steps", 512))
    micro_b    = int(cfg.get("micro_batch", 32))
    grad_accum = int(cfg.get("grad_accum", 4))             # eff batch = micro_b*grad_accum*1024 tokens
    lr         = float(cfg.get("lr", 3e-4))
    warmup     = int(cfg.get("warmup", 32))
    cooldown   = int(cfg.get("cooldown", 96))
    min_ratio  = float(cfg.get("min_lr_ratio", 0.0))
    sched      = cfg.get("schedule", "wsd")
    wd         = float(cfg.get("weight_decay", 0.1))
    b1, b2     = float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.95))
    clip       = float(cfg.get("grad_clip", 1.0))
    seed       = int(cfg.get("seed", 444))
    compile_on = bool(cfg.get("torch_compile", True))
    grad_ckpt  = bool(cfg.get("grad_checkpointing", False))  # 1B fits without it → faster, consistent

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    device, dtype = "cuda:0", torch.bfloat16

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True); vocab = len(tok)
    config = build_config(cfg, vocab)
    assert config.max_position_embeddings == 1024, "block_size must be 1024 (C4)"
    model = ArgonneModel(config).to(device)
    if grad_ckpt: model.gradient_checkpointing_enable()
    if compile_on:
        try: model = torch.compile(model)
        except Exception as e: print("compile failed, eager:", e); compile_on = False
    n_params = sum(p.numel() for p in model.parameters())
    assert 0.8e9 <= n_params <= 3.0e9, f"param count {n_params} out of 1–3B band (C5)"

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(b1,b2), weight_decay=wd)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(warmup, steps, cooldown, min_ratio, sched))

    toks = load_tokens(a.data)
    train = Loader(toks, micro_b, BLOCK_SIZE, TRAIN_OFFSET)
    eff_tokens = micro_b*grad_accum*BLOCK_SIZE
    print(f"[probe] params={n_params:,} eff_batch={eff_tokens} steps={steps} "
          f"budget_tokens={steps*eff_tokens:,} lr={lr} sched={sched} block=1024 compile={compile_on}")

    model.train(); t0 = time.time(); status = "ok"; toks_done = 0; ema = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True); step_loss = 0.0
        for _ in range(grad_accum):
            x, y = train.batch(); x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=dtype):
                loss = model(x, labels=y).loss / grad_accum    # training loss MAY include aux terms
            loss.backward(); step_loss += loss.item(); toks_done += micro_b*BLOCK_SIZE
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step(); sch.step()
        ema = step_loss if ema is None else 0.9*ema + 0.1*step_loss
        if step % 25 == 0: print(f"step {step}/{steps} loss {step_loss:.4f} lr {sch.get_last_lr()[0]:.2e}")
        if time.time()-t0 > WALL_GUARD_S: status = "timeout_invalid"; break   # past soft window → re-run (C13)

    train_sec = time.time()-t0
    test_loss = evaluate(model, toks, micro_b, device, dtype)
    peak_gb = torch.cuda.max_memory_allocated()/1e9
    rec = {"id": cfg.get("id"), "slug": cfg.get("slug",""), "status": status,
           "params": int(n_params), "params_b": round(n_params/1e9,3),
           "steps_done": step+1, "tokens_trained": int(toks_done),
           "train_loss_ema": round(ema,4), "test_loss": round(test_loss,4),
           "test_ppl": round(math.exp(test_loss),3),
           "tokens_per_sec": round(toks_done/max(1e-9,train_sec),1),
           "train_seconds": round(train_sec,1), "peak_mem_gb": round(peak_gb,1),
           "compile": compile_on, "grad_ckpt": grad_ckpt, "seed": seed, "config": cfg}
    json.dump(rec, open(a.out, "w"), indent=2)
    with open(os.path.join(os.path.dirname(a.out), "results.jsonl"), "a") as f:
        f.write(json.dumps(rec)+"\n")
    print("[probe] RESULT " + json.dumps({k:rec[k] for k in
          ("id","status","params_b","test_loss","test_ppl","tokens_per_sec","train_seconds","peak_mem_gb")}))

if __name__ == "__main__":
    main()
```

> The skeleton is intentionally complete on the *tricky* parts (offsets, pure-CE eval, wall guard,
> param-band assert, no checkpointing). Finalize/QA it during Exp 0. Do **not** add checkpointing,
> sampling, or DDP.

### 8.3 `experiments/run_experiment.sh` — SLURM wrapper

This mirrors the production H200 launch pattern in **`continue.sh`** — same `--account`/`--partition`/
`--constraint=H200`, the same `--nodes=1 --ntasks=1`, the same module + `activate AI` block, the same
`--exclude` bad-node list — **adapted to 1 GPU, single-process** (`--gres=gpu:1` and plain `python`, not
`--gres=gpu:3` + `torchrun`/DDP, because each experiment is one H200 and `train_probe.py` is
single-process). The only additions are the 35-min cap (C2) and the `__pycache__` cleanup (C12).

```bash
#!/bin/bash
#SBATCH --job-name=argo-next
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --constraint=H200
#SBATCH --exclude=midway3-0423,midway3-[0298,0377-0378,0603-0606]   # bad-node list, verbatim from continue.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             # ONE H200 (continue.sh uses gpu:3; we are single-GPU, C1)
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:35:00          # hard kill = 30 min target + 5 min soft margin (C2)
set -eo pipefail
EXP="$1"                          # e.g. exp001_baseline
REPO=/home/youzhi/ArgonneAI
cd "$REPO/experiments"
module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Single-process, single-GPU — no torchrun/DDP (contrast continue.sh's `torchrun --nproc_per_node=3`).
python train_probe.py --config "configs/${EXP}.json" --out "results/${EXP}.json"
find "$REPO/experiments" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
```
Submit with logs routed into `results/logs/`:
```bash
SB=/software/slurm-current-el8-x86_64/bin/sbatch
$SB --output=experiments/results/logs/exp001_baseline.out \
    --error=experiments/results/logs/exp001_baseline.err \
    experiments/run_experiment.sh exp001_baseline
```

### 8.4 Config JSON schema (`configs/expNNN_<slug>.json`)

Only fields that differ from the running-best need to appear, but **write the full resolved config**
each time (so each run is self-describing and reproducible). Example baseline:
```json
{
  "id": "exp001", "slug": "baseline",
  "hidden_size": 2048, "num_hidden_layers": 16,
  "num_attention_heads": 16, "num_key_value_heads": 4,
  "intermediate_size": 5632,
  "qk_norm": true, "v_norm": true, "sandwich_norm": true,
  "interleaved_local_attention": true, "local_attention_window": 256,
  "logit_softcap": 15.0, "rope_theta": 1000000.0,
  "z_loss_weight": 0.0, "mtp_horizon": 1, "mtp_loss_weight": 0.0,
  "tie_word_embeddings": true,
  "steps": 512, "micro_batch": 32, "grad_accum": 4,
  "lr": 3e-4, "schedule": "wsd", "warmup": 32, "cooldown": 96, "min_lr_ratio": 0.0,
  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
  "torch_compile": true, "grad_checkpointing": false, "seed": 444
}
```

---

## 9. Experiment 0 — calibration (locks the global token budget)

**Goal:** measure steady-state tokens/sec for the baseline proxy on this exact H200, then pick `steps`
so the **baseline finishes in ≈ 20–22 min wall** (training + ~2 min compile + ~1 min eval). The
baseline is sized *short on purpose*: heavier experiments (deeper, wider-MLP, MHA, Phase-5 stacks) run
~1.3–1.7× slower at the **same** locked token budget, and every one of them must still complete inside
the 30-min target (35-min hard cap) to be valid (C13). Once chosen, **`steps`, `micro_batch`,
`grad_accum` are LOCKED for the headline comparison** across all 50 runs (only the batch-size
experiments deviate, and they're labeled).

**Procedure:**
1. Write `configs/exp000_calibration.json` = baseline but `steps: 64` (short).
2. Submit it. Confirm: flash-attn local-attention log line is correct (Section 2); param count ≈ 1.03B;
   `peak_mem_gb` (headroom for heavier configs — want < ~110 GB so MHA/4×-MLP/deep fit without ckpt);
   `tokens_per_sec` at steady state; compile time (wall − train_sec/util estimate).
3. Compute `steps = floor(target_train_seconds × tokens_per_sec / eff_batch)` with
   `target_train_seconds ≈ 1200` (≈ 20 min baseline) and `eff_batch = 131072`. Round to a clean number.
   - **Size for the heaviest config, not the baseline.** Estimate the max slowdown `R_max` over the
     curriculum (24-layer ≈ 1.5×; a Phase-5 stack of deeper + wider ≈ up to ~1.7×). Require
     `baseline_wall × R_max ≤ 30 min` (target), i.e. `baseline_wall ≤ 30 / R_max ≈ 18–20 min`; take the
     smaller of this and the 20–22 min target.
   - Worked estimate (Section 17): at ~48k tok/s, ~440 steps × 131072 ≈ 57.7M tokens ≈ 20 min train +
     ~3 min overhead ≈ **23 min** baseline; a 1.6× config ≈ **33 min ≤ 35** hard cap ✓. If Exp 0 shows
     slower tok/s, drop `steps` further. Fewer tokens = slightly more noise, but a conservative budget
     is what *guarantees every one of the 50 completes* (C13) — prioritize that over squeezing tokens.
   - Before submitting any unusually heavy run (Phase 5, or 24-layer), **predict its wall** from the
     logged `tokens_per_sec` of its nearest component and confirm `predicted_wall ≤ 33 min`; if not, the
     locked budget was set too high and must be reconsidered **here in Exp 0** — never mid-campaign
     (changing the budget later breaks iso-data with completed runs).
4. Record the locked `steps`/batch in `leaderboard.md` header. Exp 0 does **not** count as a search
   result (it's setup) but its loss is a sanity point.

If `peak_mem_gb` is too high to leave room for the heaviest Phase-3/4 configs, either lower
`micro_batch` (raise `grad_accum` to hold `eff_batch`) or accept enabling grad-checkpointing globally
(slower but uniform). Uniformity matters more than raw speed.

---

## 10. The 50-experiment curriculum

This is a **prior ordering** over a greedy adaptive search (Section 11), not a rigid script. Do the
high-transfer optimization/stability axes first (they make every later comparison fairer), then norms,
then attention shape, then capacity, then combine + confirm. Each entry states the **hypothesis** and
**what it controls for**. Numbers are the default; the adaptive rule may insert, drop, or reorder.

> Convention: each experiment changes **one** thing vs the **current running-best** (coordinate
> descent), unless it is an explicit interaction/combination test.

### Phase 0 — calibration & noise floor (Exp 0–3)
- **0 calibration** — lock `steps`/batch (Section 9).
- **1 baseline (seed 444)** — the reference test loss. Faithful down-scaled Argonne 3.0.
- **2 baseline (seed 445)**, **3 baseline (seed 446)** — establish noise floor **σ**. *Why first:* with
  no σ you cannot tell a real win from luck; everything downstream depends on it (Section 6).

### Phase 1 — optimization & schedule (Exp 4–18) — highest transfer at short horizon
- **4–9 LR sweep** {1e-4, 2e-4, 3e-4(ref), 5e-4, 8e-4, 1.2e-3}. *Why first among changes:* LR is the
  single largest lever and short under-trained runs are very LR-sensitive; fixing **LR\*** makes all
  later comparisons honest. Adopt LR\* (best within noise; prefer the lower LR on ties for stability).
- **10–12 warmup** {16, 32(ref), 64} at LR\*. Short budgets waste tokens on long warmups; find the knee.
- **13–14 schedule/cooldown:** cosine vs WSD; WSD cooldown fraction {10%, 20%(≈ref 96/512), 40%} and
  `min_lr_ratio` {0.0, 0.1}. *Why:* the end-of-run loss (the metric) is dominated by how aggressively LR
  decays at the end; a proper cooldown gives a lower, lower-variance final loss and better correlates
  with long runs.
- **15 beta2** {0.95(ref), 0.98, 0.99}. Short runs: higher β2 = slower second-moment adaptation; test it.
- **16 beta1** {0.9(ref), 0.95}. **17 weight_decay** {0.0, 0.05, 0.1(ref), 0.2}. **18 grad_clip**
  {0.4, 1.0(ref), 2.0}. Each cheap, each a known interaction with LR\*.

### Phase 2 — normalization & numerics (Exp 19–30)
These test whether the *production* stability features actually help at proxy scale, and tune them.
- **19 qk_norm off**, **20 v_norm off**, **21 sandwich_norm off** — ablate each production default.
  *Why:* confirm they earn their cost here; if one is neutral, prefer the simpler variant.
- **22 qk_norm+v_norm+sandwich all off** — interaction (do they only help together?).
- **23–25 logit_softcap** {0 (off), 15(ref), 30}. Softcap caps logits → changes CE; find the best.
- **26–27 z_loss_weight** {1e-4, 1e-3} (metric excludes z term — invariant #4). *Why:* z-loss stabilizes
  logit norms; may improve next-token CE indirectly.
- **28 rms_norm_eps** {1e-5 vs 1e-6(ref)}.
- **29 init scaling (code, in `experiments/model.py`):** residual-branch init `(2L)^-0.5` (current) vs a
  depth-aware alternative; **30 embedding init / √d scaling.** *Why:* init controls early-loss
  trajectory, which is most of a 67M-token run; flag as code-variant in the config.

### Phase 3 — attention shape (Exp 31–40)
- **31–34 GQA ratio:** `num_key_value_heads` ∈ {1 (MQA), 2, 4(ref), 8 (≈MHA at 16 heads)}. *Why:* KV
  heads trade quality vs KV-cache/throughput; production picked 4 — verify at this scale. Log tok/s too.
- **35–36 head_dim:** vary heads at fixed hidden 2048 → head_dim {64 (32 heads), 256 (8 heads)} vs 128
  ref. *Why:* production uses an unusually large head_dim 256; is it worth it, or is 128 better per FLOP?
- **37 interleaved local attention off** (all-global). **38–39 local window** {128, 512} vs 256(ref).
  *Why:* at block 1024 a 256 window covers ¼ context; the global/local mix is a real quality/speed knob.
  (Ensure flash-attn path — else window is ignored, Section 2.)
- **40 rope_theta** {1e4, 1e5} vs 1e6(ref). *Why:* θ=1e6 is sized for long context; at block 1024 a
  smaller θ may give better positional resolution and lower loss.

### Phase 4 — capacity / shape (Exp 41–47) — lower transfer confidence; label clearly
Hold params ≈ const where comparing *shape*; for capacity changes, also record params + a note that
the verdict is short-horizon (validate in Phase 5).
- **41–43 depth↔width at ~iso-params:** {12L/2304, 20L/1856, 24L/1664} vs 16L/2048(ref). *Why:* the
  depth/width frontier; pick the better-conditioned shape at equal capacity.
- **44–45 MLP ratio** (`intermediate_size`): {2.0×→4096, 3.5×→7168} vs 2.75×(ref). *Why:* SwiGLU width
  vs depth trade; capacity axis, short-horizon caveat applies.
- **46 untie embeddings** (`tie_word_embeddings:false`). *Why:* a separate lm_head can help at large
  vocab; but it adds 311M params — compare at iso-params by shrinking a layer, and note the param delta.
- **47 MTP** `mtp_horizon` 2, `mtp_loss_weight` 0.3 (metric still pure next-token CE). *Why:* multi-token
  prediction is a free auxiliary signal; does it improve the *single*-token test loss?

### Phase 5 — combine & confirm (Exp 48–50)
- **48 stacked best** — assemble all accepted changes into one config; verify the sum holds together
  (re-run vs running-best to catch negative interactions). *Why:* coordinate descent can over-credit
  changes that don't compose.
- **49 ablate-one-out of the stacked best** — drop the weakest accepted change; if loss doesn't rise
  past noise, keep the simpler model (Occam).
- **50 transfer/confirmation run** — re-run the winner at the **largest token budget whose predicted
  wall ≤ ~33 min** for the winner's measured speed (more steps, or a slightly smaller proxy to buy
  tokens), to check the advantage persists with more data. Pair it with a baseline re-run at that
  *same* enlarged budget for the comparison (that paired baseline is setup, like Exp 0 — it does not
  consume a numbered slot). *Why:* guards against short-horizon artifacts before recommending the
  recipe for the 2.88B port.

If an axis closes early (e.g. several norm ablations are all neutral), **reallocate** the freed slots to
the most promising open axis (finer LR×schedule grid, or a second combination test). Always end at 50.

---

## 11. Adaptive decision procedure ("after each job, plan and justify the next")

After **every** completed job, before submitting the next, do this and write it into
`results/decisions.md`:

1. **Parse** `results/expNNN.json` and check **validity** (C13): `status == "ok"`, finite `test_loss`,
   invariants intact.
   - **Invalid (operational failure** — timeout / OOM / crash / transient NaN): do **not** advance the
     `#valid/50` counter. Diagnose (Section 13), fix, and **re-submit the SAME experiment id** (bump an
     `attempt` field). Repeat until valid. A slot of the 50 is filled only by a valid result.
   - **Cleanly diverged because the *tested value itself* is unstable** (e.g. an intentionally-high LR
     NaNs on every attempt): that is a real finding, not an operational failure. Record
     `status="diverged"` with a sentinel (conclusively rejected, cannot be the winner), note the
     stability boundary in `decisions.md`, and **count the slot as valid** — do not loop forever.
   - **Valid:** proceed to step 2. (A completed run that is merely *worse* than baseline is still
     valid — a negative result answers its question and fills its slot.)
2. **Compare** `test_loss` to (a) the running-best and (b) the seed-noise band 2σ.
3. **Verdict:** `WIN` (beat running-best by > max(2σ, 0.003)) / `NEUTRAL` (within ±2σ) / `LOSS`.
   - On `WIN`: update running-best = this config; **confirm** with one extra seed before stacking more.
   - On `NEUTRAL`: keep the simpler/faster option; record and move on.
   - On `LOSS`: discard; note magnitude (a big loss is informative — it bounds the axis).
4. **Justify the next experiment** (this is the "why is this better than the previous exps" the mission
   asks for) — write 2–4 sentences answering:
   - *What did we just learn?* (the result and what it implies about the loss landscape)
   - *What is the next most informative single change, given the running-best?* (biggest expected
     test-loss reduction per the curriculum + observed trends; e.g. "LR\*=5e-4 won by 0.02 > 2σ=0.008,
     so re-center the schedule sweep around it; cooldown is now the largest untested lever because
     end-of-run LR decay dominates the metric.")
   - *Why this beats just repeating prior runs?* (it tests an unexplored, plausibly-large axis rather
     than re-confirming a settled one; or it re-validates a stack we suspect has interactions).
5. **Write** `configs/exp(NNN+1)_<slug>.json` as running-best + exactly one change (or a labeled
   combination). Then go to the Section 12 loop.

**Keep a running-best block at the top of `leaderboard.md`** so a cold restart (after context
compaction) can resume instantly: current running-best config, current σ, locked `steps`/batch,
last experiment id, and the next planned experiment.

---

## 12. The submit-one-job-and-wait-patiently loop (C6)

**Only one job in flight.** Never submit the next while one is queued or running.

**Advance only on validity (C13).** After a job ends, parse and validate it (Section 11.1). If it is an
operational failure, re-submit the **same** experiment id (don't write the next config, don't bump the
`#valid/50` counter). Only after a valid result is recorded do you plan + write + submit the next id.

```bash
SQ=/software/slurm-current-el8-x86_64/bin/squeue
SB=/software/slurm-current-el8-x86_64/bin/sbatch

# 1) guard: refuse to submit if an argo-next job is already queued/running
if "$SQ" -u youzhi -h -o "%j" | grep -q argo-next; then echo "a job is in flight; wait."; exit 0; fi

# 2) submit exactly one
JID=$("$SB" --parsable \
  --output=experiments/results/logs/exp004_lr5e4.out \
  --error=experiments/results/logs/exp004_lr5e4.err \
  experiments/run_experiment.sh exp004_lr5e4)
echo "submitted $JID"
```

**Wait patiently** for completion. Two robust options:
- *Polling:* every ~60–120 s, check `"$SQ" -j "$JID" -h` (empty output ⇒ finished) **and** that
  `results/exp004_lr5e4.json` exists. Queue waits can be minutes to hours; don't busy-spin — sleep
  between polls. (If the agent runtime supports `ScheduleWakeup`/`Monitor`, prefer a long-fallback wake
  ~20 min plus completion-driven re-invocation over tight polling.)
- *Backstop:* `--time=00:35:00` bounds any run at 35 min (30 target + 5 soft margin); the in-script
  33-min guard writes diagnostics just before that.

**Only after** the JSON exists and is parsed (Section 11) do you write the next config and submit it.

**Resume after context compaction:** read `leaderboard.md` (running-best, σ, locked budget, next
planned exp) and `results.jsonl` (everything done). The on-disk ledger is the source of truth — never
rely on in-context memory for campaign state.

**Total campaign size:** 50 runs × ~26 min ≈ 22 GPU-hours plus queue time — realistically a 1–2 day
campaign. That is expected; patience over one-at-a-time is the whole point of C6.

---

## 13. Guardrails

Failures fall into two classes (C13): **operational failures** → re-run the SAME experiment id until
valid (the slot is not filled, the `#valid/50` counter does not advance); **valid findings** (incl.
intentional divergence) → record and advance.

- **OOM** *(operational)*: `train_probe.py` halves micro-batch & doubles grad-accum (iso-data
  preserved), records `oom_fallback`; if still OOM, enables grad-checkpointing for that run (recorded)
  and re-runs the same id. Size `micro_batch` in Exp 0 so the heaviest planned config (MHA / 4×-MLP /
  24L / Phase-5 stack) fits — prefer headroom over speed.
- **Transient NaN** *(operational)*: `model.py` zeroes individual NaN-loss steps and warns; a few are
  fine. If the run still completes with a **finite** `test_loss`, it is **valid**. Re-run the same id
  only if the NaNs came from an infra glitch and the test loss ends up NaN/inf.
- **Intentional divergence** *(valid finding)*: if a config diverges *because the value under test is
  genuinely unstable* (e.g. top-of-sweep LR) and does so reproducibly, record `status="diverged"`
  (conclusively rejected, cannot win), note the stability boundary, and **count the slot** — do not
  loop forever or fold it into the running-best.
- **Timeout** *(operational)*: the 33-min abort guard writes `status="timeout_invalid"` — the config
  didn't finish the locked budget inside the soft window, so it is **invalid: re-run the same id** after
  diagnosing. If it is genuinely too slow at the locked budget, the global budget was set too high in
  Exp 0 (it should have been sized for this config). Prevent this in Exp 0 by predicting heavy runs'
  wall before submitting (Section 9); the budget cannot be lowered mid-campaign without breaking
  iso-data with completed runs.
- **Compile failure** *(operational)*: falls back to eager (recorded). Eager changes only speed, not
  loss; if eager threatens the soft window, treat as too-slow (re-run with compile fixed) rather than
  shrinking the budget.
- **flash-attn missing** *(blocker)*: local-attention experiments are void (window silently ignored).
  Verify the startup log in Exp 0 before trusting Phase-3 windows.

---

## 14. Bookkeeping

**`results/results.jsonl`** — append-only; one object per *run attempt* (including failed attempts, so
the history is auditable) with at least: `id, slug, attempt, status, valid, params, params_b,
steps_done, tokens_trained, train_loss_ema, test_loss, test_ppl, tokens_per_sec, train_seconds,
peak_mem_gb, compile, grad_ckpt, seed, config`. Failed attempts keep the same `id` with higher `attempt`.

**`results/leaderboard.md`** — top block = campaign state: running-best config, σ, locked steps/batch,
**`#valid/50`** (only valid results count toward 50, C13), and the next planned exp. Then a table
sorted by `test_loss` over **valid** runs only (one row per experiment id = its valid attempt), with
columns: id · slug · test_loss · Δ-vs-baseline · Δ-vs-best · verdict · params_b · tok/s · wall · seed.
Diverged findings are listed below the ranked table (rejected, not ranked) but still count as valid
slots.

**`results/decisions.md`** — the narrative log: for each experiment, the hypothesis, the result, the
verdict, and the justification for the next experiment (Section 11.4). This is the artifact that makes
the search legible and is part of the final deliverable.

---

## 15. Cleanup rules (per repo `CLAUDE.md` — "no artifacts")

- After every job: `find experiments -name __pycache__ -type d -prune -exec rm -rf {} +` (already in
  `run_experiment.sh`). Remove any stray `*.pyc`, `.pytest_cache`, etc.
- Assert no `*.pt` / `*.safetensors` / HF model dirs were created anywhere (C7). If found, delete and
  fix `train_probe.py`.
- `experiments/.gitignore` ignores `results/logs/`, `__pycache__/`, `*.pt`. Keep configs, `results.jsonl`,
  `leaderboard.md`, `decisions.md` (the durable record).
- Never write outside `experiments/`. Never modify codebase files.

---

## 16. Final deliverable

After Exp 50, append to `decisions.md` a **"Recommended Argonne-Next recipe"** section:
1. The winning proxy config, and the **delta vs the current production design** (which architecture
   fields and which optimizer/schedule settings changed, with the measured test-loss improvement vs the
   faithful-baseline and vs production-default settings, each ± its noise band).
2. **Confidence labels** per change: high (optimization/stability/norm axes) vs lower (capacity/shape),
   per Section 18.
3. The **port to 2.88B**: which changes are config-only (drop straight into `pretrain.py`'s constants /
   `ArgonneConfig`) vs which need the small `experiments/model.py` code change reproduced in a future
   codebase edit (out of scope here — recommend, don't apply).
4. The Exp-50 confirmation result (does the advantage persist at the larger token budget?).
5. Open questions / next search directions.

---

## 17. Timing/throughput math (worked example)

For a ~1.03B model at block 1024, fwd+bwd compute ≈ `6 × N_compute` per token where
`N_compute ≈ non-embed (721M) + tied lm_head matmul (311M) ≈ 1.03B` → ~6.2e9 FLOPs/token, +~15% for
attention ≈ **7.1e9 FLOPs/token** (no recompute; grad-checkpointing OFF for the 1B proxy).

H200 BF16 dense peak ≈ 989 TFLOPS. At a realistic ~40% MFU (1B + compile, no ckpt) → ~396 TFLOPS →
`396e12 / 7.1e9 ≈ 55.8k tok/s` ideal; budget conservatively at **~48k tok/s** to absorb dataloader and
eval. Then:
- Target **baseline ~20 min** so heavier configs fit: `steps ≈ 440 × eff_batch=131,072 ≈ 57.7M tokens`
  → `57.7e6 / 48e3 ≈ 1202 s ≈ 20 min` training + ~2 min compile + ~1 min eval ≈ **23 min** baseline.
- A 1.6× slower config at the **same** budget ≈ 32 min train + overhead ≈ **33 min ≤ 35** hard cap ✓.

**Do not trust this estimate — measure it in Exp 0 and set `steps` from the measured tok/s, sized for
the heaviest config** (Section 9). If Exp 0 shows ~38k tok/s, use `steps ≈ 350` (~46M tokens, ~20 min
baseline → ~32 min for a 1.6× config). The point of calibration is a budget that makes the *baseline*
short (~20 min) and the *slowest* config still finish inside the 30-min target / 35-min cap, so every
one of the 50 completes (C13). A conservative (smaller) budget that guarantees completion beats a
larger one that risks timeouts.

---

## 18. Risks and honest caveats

- **Short-horizon under-training.** 67M tokens for a 1B model is ≈ 0.065 tokens/param — far below
  Chinchilla-optimal (~20). Absolute loss is high; we compare *early-training* dynamics. Mitigations:
  WSD cooldown gives a clean low-variance final loss; the noise floor gates wins; Phase-5 confirms at a
  larger budget. **Conclusions transfer best for optimization/stability/norm changes; treat
  capacity/shape (Phase 4) verdicts as provisional.**
- **Proxy ≠ production.** A 1.03B proxy informs but does not prove 2.88B behavior. The deliverable
  recommends; a future full run validates.
- **Noise chasing.** Without σ, you will "discover" wins that are seed luck. Phases 0 and the 2σ
  acceptance rule exist precisely to prevent this. Re-measure σ mid-campaign.
- **Coordinate-descent local optima / interactions.** Greedy single-change search can miss
  interactions; Phase-5 stacking + ablation and the periodic re-validation catch the worst of it.
- **Speed-vs-data confound.** Capacity changes alter tok/s; the headline metric is iso-data (fixed
  tokens), with tok/s logged separately, so "slower but better per token" and "faster" stay
  distinguishable. Don't let a fast-but-worse config win on wall-clock — the metric is test loss.
- **Determinism.** `cudnn.benchmark` + tf32 + compile introduce tiny nondeterminism; this is *included*
  in the measured σ, so it doesn't bias comparisons — it just sets the resolution of "a real win."

---

### One-paragraph summary for a cold start
Create `experiments/`; copy `model.py`; build `train_probe.py` (single-GPU, block 1024 pinned, no
checkpoints, trains a fixed token budget from offset 0, evaluates pure next-token CE on a fixed 3.07M-
token tail of `train.bin` at offset 20.6B) and `run_experiment.sh` (1×H200, `--time=00:35:00`). Run
Exp 0 to lock `steps` (sized so even the heaviest config finishes inside the 30-min target / 35-min
hard cap). Run the baseline 3× to get σ. Then greedily test one change at a time against the
running-best — LR/schedule first, then norms, then attention shape, then capacity — accepting only
> max(2σ, 0.003 nats) improvements, **submitting one job at a time and waiting for each to finish**
before planning and justifying the next in `decisions.md`. A failed run is re-run as the *same*
experiment (never skipped); the campaign ends only when **50 *valid*** results exist. Then write the
recommended Argonne-Next recipe.
```
