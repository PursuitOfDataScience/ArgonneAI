# argonne4.5 probe campaign — 30 arms, 3 GPUs, ≤30 min each

Sequential. One job at a time. Each arm is designed **after** reading the previous arm's result;
nothing is pre-queued and nothing is auto-submitted.

## Setup

| | |
|---|---|
| Hardware | **3× H100** on `midway3-0426` / `0372` (`gold-6346,512g,H100`) — pinned |
| Harness | `exp/probe.py` (DDP), `exp/run.sh` (sbatch, untracked per repo policy) |
| Model | the decided a4.5 size: hidden 2560 / 24L / 10Q-2KV / head_dim 256 / 2.06B |
| Data | production mix `edu 50 / math 30 / code 20` from `/project/rcc/youzhi/data/argonne4_pretrain` |
| Metric | **TGT** = mean pure-CE over held-out `val_edu` / `val_math` / `val_code` (3M tokens each) |
| Batch | block 2048, micro 8 × accum 5 × 3 GPUs = **245,760 tok/step**, LR 6e-4 |

**Why H100 and not the H200 that was asked for.** Both usable H200 nodes (0600/0601 — the only two
that survive the node-exclusion policy) were **8/8 allocated** by other users' multi-day jobs; the
first free 3-GPU slot was ~7.5 h out and would have had to be re-won for each of 30 arms. H100
`midway3-0426` was fully idle. It is also **the exact hardware the argonne4.0 production run trained
on**, so throughput measured here is directly comparable to its measured 5.27 B tok/day — which the
H200 numbers would not have been. Every arm is pinned to the same two identical nodes.

## Invariants (each one is a scar from a previous campaign)

1. **Iso-token, never iso-wall**, for any quality comparison.
2. **Pure CE eval** — forward without labels, CE computed from logits, so MTP/z-loss cannot leak in.
3. **Pinned card + node recorded on every arm.** Cross-GPU comparison caused the KEY FINDING 14
   retraction and the Finding C retraction. Do not compare across hardware, ever.
4. Same source-sampling seed → every arm sees the same data in the same order.
5. A wall-guard trip still writes a record, marked `valid: false`.
6. **One job at a time**, submitted by hand, analysed before the next is written.

## Arms

| # | id | question | result |
|---|---|---|---|
| 01 | `e01_baseline` | Baseline: a4.5 size, pre-4.5 arch. What is tok/s, compile cost, HBM, TGT? | *running* |

---

### 01 — `e01_baseline`

**Hypothesis / purpose.** Nothing about the 2.06B config has ever been run. This arm exists to
produce the four numbers every later arm depends on: steady tok/s (sets the iso-token budget that
fits in 30 min), compile+startup cost (how much of the 30 min is not training), HBM headroom (how
much room the systems arms have to move), and the TGT baseline (the reference for every
architecture arm). Architecture is deliberately **pre-4.5** — SwiGLU 7040, legacy interleaved
attention, RoPE everywhere — so it is the true "do nothing" control.

**Result.** *(pending)*
