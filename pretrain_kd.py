"""Pretrain KD for the a5 recipe, shared by pretrain.py and continue_pretrain.py (argonne5.0 port).

Ported from exp/probe.py's KD branch, the code every a5 KD number was measured with (-0.538 tgt at
h6400 under tuned Muon, two draws; alpha 0.5 keeps only ~72% of it; K 2048 is the measured knee;
Qwen3-0.6B-Base beats the 1.7B as teacher). The math is the probe's, verbatim; exp/test_a5_trainer.py
checks loss AND gradient parity against a copy of the probe's lines. Only the plumbing is new: the
production DDP no_sync + sync micro-step branches, autocast, doc_mask, a compiled student beside an
eager teacher, and a gate that stops every rank together.

It lives in a module, not in pretrain.py, because continue_pretrain.py needs the same code and cannot
import pretrain.py (which parses argv at import time). One copy, so the stages cannot drift.

ALGEBRA (the probe's): an alpha-mixed KD objective equals a cross-entropy against
    p_target = (1-a)*onehot(y) + a*p_teacher,
so the soft term needs only the student's logits at the teacher's top-K ids plus its logsumexp:
    soft = -sum_k p_k*lg_k + (sum_k p_k)*lse,  and lse = ce + lg_y falls out of cross_entropy.
⛔ FULL-VOCAB NORMALISATION IS LOAD-BEARING. p_k is the teacher softmax over the WHOLE student vocab; the
top-K keep their actual mass and the tail is omitted, never redistributed, so sum_k p_k < 1 on purpose.
A K-slice-only softmax was measured WORSE THAN NO KD. Do not "fix" the missing mass.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_ckpt


def add_kd_args(parser):
    """The KD knobs. Default --kd_alpha 0 = KD off = a4.5 exactly (no teacher is ever loaded)."""
    parser.add_argument("--kd_teacher", type=str, default="", help="HF path of the pretrain-KD teacher (a5 recipe: /project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base; same tokenizer, its 151,936-wide logits are truncated to the student vocab). Only loaded when --kd_alpha > 0.")
    parser.add_argument("--kd_alpha", type=float, default=0.0, help="Loss = (1-alpha)*hard CE + alpha*soft CE against the teacher's top-K. 0 = KD off (a4.5). a5 recipe 1.0: -0.538 tgt at h6400 under tuned Muon (two draws); alpha 0.5 keeps only ~72%% of it.")
    parser.add_argument("--kd_topk", type=int, default=2048, help="Teacher top-K kept per position, normalised over the FULL vocab (a K-slice-only softmax measured WORSE than no KD). 2048 is the measured knee.")
    parser.add_argument("--kd_chunk", type=int, default=256, help="Student-side KD loss rows per checkpointed chunk (memory only, the objective is identical).")
    parser.add_argument("--kd_teacher_chunk", type=int, default=256, help="Teacher positions per logsumexp/top-K chunk (memory only, exact).")
    parser.add_argument("--kd_chunked_head", type=int, default=0, choices=[0, 1], help="1 = the student's output head runs per --kd_chunk rows inside the model forward, so its full (T, vocab) logits and their gradient never exist: needed for KD at long context (at block 13,568 they are 3.83 GiB each and KD missed a 40 GB card, A5_KD_RECIPE N+295). Same objective; costs one extra lm_head forward per chunk in backward. 0 = the measured block-1024 path (default).")


def check_kd_args(parser, args):
    if not (0.0 <= args.kd_alpha <= 1.0):
        parser.error("--kd_alpha must be in [0, 1]")
    if args.kd_alpha > 0 and not args.kd_teacher:
        parser.error("--kd_alpha > 0 needs --kd_teacher (a5 recipe: "
                     "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base)")
    if args.kd_topk < 1 or args.kd_chunk < 1 or args.kd_teacher_chunk < 1:
        parser.error("--kd_topk/--kd_chunk/--kd_teacher_chunk must be >= 1")


@torch.no_grad()
def kd_teacher_topk(teacher_logits, k, chunk):
    """Teacher top-K probabilities (normalised over the FULL vocab) and their ids, chunked over positions.

    Exact, not an approximation: logsumexp and top-K are per-position reductions over the vocab axis,
    so splitting the position axis changes nothing (probe: zero diff, identical indices), and top-K
    on the LOGITS equals top-K on the probabilities because softmax is monotone. The chunking only
    lowers the high-water mark: the probe measured it recovering ~2 MiB of PEAK at T=2048, because
    the fp32 [B,T,V] transient it avoids is freed under no_grad before the peak. Returns
    (probs fp32 [B,T,K], ids int64 [B,T,K]).
    """
    pv, pi = [], []
    for c0 in range(0, teacher_logits.size(1), chunk):
        tlc = teacher_logits[:, c0:c0 + chunk].float()
        lse_c = torch.logsumexp(tlc, dim=-1, keepdim=True)
        v_c, i_c = torch.topk(tlc, k, dim=-1)
        pv.append(torch.exp(v_c - lse_c)); pi.append(i_c)
        del tlc, lse_c, v_c, i_c
    return torch.cat(pv, dim=1), torch.cat(pi, dim=1)


def _kd_chunk_terms(lgc, yc, pc, ic):
    """(sum of hard CE, sum of soft CE) over one chunk of rows. The probe's `_chunk`, verbatim."""
    ce_c = F.cross_entropy(lgc, yc, reduction="none")
    lse_c = ce_c + lgc.gather(-1, yc.unsqueeze(-1)).squeeze(-1).float()
    lgk_c = lgc.gather(-1, ic).float()
    soft_c = -(pc * lgk_c).sum(-1) + pc.sum(-1) * lse_c
    return ce_c.sum(), soft_c.sum()


def kd_hard_soft(student_logits, labels, tkp, tki, chunk):
    """(hard, soft): mean hard CE and mean soft CE over all B*T positions.

    CHUNK PLUS RECOMPUTE, as measured in the probe (smoke job 54300220 OOM'd without it): chunking
    alone keeps every chunk's log_softmax for backward, which sums to the same fp32 [T, V] tensor as
    one pass (1.16 GiB at T=2048); checkpoint() keeps one chunk resident and recomputes the rest in
    backward, one extra CE forward. CE is a token-weighted sum, so the objective is identical.
    """
    V = student_logits.size(-1)
    flat, yf = student_logits.view(-1, V), labels.reshape(-1)
    K = tkp.size(-1)
    pf, idf = tkp.view(-1, K), tki.view(-1, K)
    N = yf.numel()
    h_s = f_s = None
    for c0 in range(0, N, chunk):
        sl = slice(c0, min(c0 + chunk, N))
        hc, fc = _torch_ckpt(_kd_chunk_terms, flat[sl], yf[sl], pf[sl], idf[sl], use_reentrant=False)
        h_s = hc if h_s is None else h_s + hc
        f_s = fc if f_s is None else f_s + fc
    return h_s / N, f_s / N


def kd_micro_loss(model, teacher, x, y, fwd_kw, vocab_size, kd_alpha, kd_topk, kd_chunk,
                  kd_teacher_chunk, want_gate=False, chunked_head=False):
    """One micro-step of pretrain KD, run under the CALLER's autocast (the probe's order of operations).

    The teacher sees plain `x`, as in the probe: no document ids, no mask; the student gets `fwd_kw`
    (document_ids under --doc_mask). Teacher logits are truncated to the student vocab before any
    normalisation, so the full-vocab softmax is over exactly the classes the student can emit. The
    student is called WITHOUT labels so it returns logits; model.py's fused chunked CE, z-loss and MTP
    are therefore not on this path (setup_kd_teacher refuses z-loss/MTP under KD).

    Returns (objective, hard, gate): objective = (1-alpha)*hard + alpha*soft (the caller divides by the
    accumulation count, exactly as the probe did), hard = plain next-token CE for logging, and gate =
    the probe's KD-GATE statistics when want_gate, else None.
    """
    with torch.no_grad():
        # use_cache=False: identical logits; the probe's default call also built a KV cache it never read.
        tl = teacher(x, use_cache=False).logits[..., :vocab_size]
        tkp, tki = kd_teacher_topk(tl, kd_topk, kd_teacher_chunk)
        del tl
    if chunked_head:
        # --kd_chunked_head: the model computes the same two means from its hidden states, chunk by chunk
        # (model.py _chunked_kd_terms), so the student's full logits never exist. Inside the model's
        # forward, so DDP sees the head's use like any other parameter's.
        lg = None
        hard, soft = model(x, labels=y, kd_targets=(tkp, tki, kd_chunk), **fwd_kw).loss.unbind(0)
    else:
        lg = model(x, **fwd_kw).logits
        hard, soft = kd_hard_soft(lg, y, tkp, tki, kd_chunk)
    gate = None
    if want_gate:
        # TEACHER-CE GATE (probe): a vocab-misaligned or broken teacher yields a clean, believable
        # NULL rather than an error, so it is checked before the run can report anything.
        gate = {"mass": tkp.sum(-1).mean().detach(),
                "gold": (tki == y.unsqueeze(-1)).any(-1).float().mean().detach(),
                "hard": hard.detach(), "soft": soft.detach()}
    objective = (1.0 - kd_alpha) * hard + kd_alpha * soft
    del tkp, tki, lg
    return objective, hard, gate


def param_connected_zero(base_model, device):
    """A scalar 0 wired to every trainable parameter directly, never through the activations.

    model.py's NaN guard on the non-KD path: backward through it delivers an exact zero gradient to
    each parameter, so every DDP hook fires (no collective desync) and the step is a no-op rather
    than a NaN that poisons the weights and the Muon/AdamW state. The KD loss is computed outside the
    model, so the KD path needs its own copy of the same contract.
    """
    zero = torch.zeros((), device=device, dtype=torch.float32)
    for p in base_model.parameters():
        if p.requires_grad:
            zero = zero + torch.nan_to_num(p).sum().float() * 0.0
    return zero


def load_kd_teacher(path, device):
    """The probe's teacher load: bf16, eval, frozen. Never compiled (the student is).

    transformers renamed `torch_dtype` to `dtype`; midway3's 5.x takes `dtype`, ALCF's 4.51 raised
    "unexpected keyword argument 'dtype'", so try the new name and fall back (probe, verbatim).
    """
    from transformers import AutoModelForCausalLM
    try:
        teacher = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True)
    except TypeError:
        teacher = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                       trust_remote_code=True)
    teacher = teacher.to(device).eval()
    teacher.requires_grad_(False)
    return teacher


def setup_kd_teacher(args, vocab_size, device, is_main, zloss_or_mtp=False, loss_chunk_size=0):
    """Load and check the teacher when --kd_alpha > 0, else return None (a4.5 never sees a teacher).

    Every refusal runs on every rank with identical inputs, so all ranks exit together.
    """
    if args.kd_alpha <= 0:
        if args.kd_teacher and is_main:
            print(f"[kd] KD OFF (--kd_alpha 0): --kd_teacher {args.kd_teacher} is ignored", flush=True)
        return None
    if zloss_or_mtp:
        raise SystemExit("FATAL: the KD path computes its loss from the student's logits, so z-loss "
                         "and MTP would be silently dropped (the probe recorded z-loss as inert on "
                         "this path). Turn them off or turn KD off.")
    if args.kd_topk > vocab_size:
        raise SystemExit(f"FATAL: --kd_topk {args.kd_topk} exceeds the student vocab {vocab_size}")
    teacher = load_kd_teacher(args.kd_teacher, device)
    t_vocab = int(teacher.config.vocab_size)
    if t_vocab < vocab_size:
        raise SystemExit(f"FATAL: teacher vocab {t_vocab} < student vocab {vocab_size}: cannot truncate")
    if is_main:
        print(f"[kd] KD ON: alpha={args.kd_alpha} topk={args.kd_topk} chunk={args.kd_chunk} "
              f"teacher_chunk={args.kd_teacher_chunk} teacher={args.kd_teacher} (vocab {t_vocab} "
              f"-> truncated to the student's {vocab_size}); teacher bf16, eval, frozen, eager "
              f"({sum(p.numel() for p in teacher.parameters()):,} params)", flush=True)
        if loss_chunk_size > 0:
            print("[kd] note: --loss_chunk_size is inert under KD: the student returns logits and "
                  "the KD loss is chunked by --kd_chunk instead.", flush=True)
    return teacher


class KDStep:
    """Per-process KD state around kd_micro_loss: the first-micro-step gate and the NaN guard.

    Call it inside the trainer's micro-step, under autocast and inside/outside DDP no_sync exactly
    where the model forward used to be; it returns (objective, hard) and the trainer divides the
    objective by its accumulation count and calls backward, as for the plain CE.
    """

    def __init__(self, teacher, base_model, vocab_size, args, device, world_size, is_main):
        self.teacher, self.base_model, self.vocab_size = teacher, base_model, vocab_size
        self.alpha, self.topk = args.kd_alpha, args.kd_topk
        self.chunk, self.teacher_chunk = args.kd_chunk, args.kd_teacher_chunk
        self.chunked_head = bool(getattr(args, "kd_chunked_head", 0))
        self.device, self.world_size, self.is_main = device, world_size, is_main
        self.gate_pending = True
        self.nan_count = 0

    def __call__(self, model, x, y, fwd_kw, global_step):
        objective, hard, gate = kd_micro_loss(
            model, self.teacher, x, y, fwd_kw, self.vocab_size, self.alpha, self.topk,
            self.chunk, self.teacher_chunk, want_gate=self.gate_pending, chunked_head=self.chunked_head)
        if gate is not None:
            self._gate(gate, global_step)
        if torch.isnan(objective):
            self.nan_count += 1
            if self.nan_count <= 5 or self.nan_count % 100 == 0:
                print(f"WARNING: NaN KD loss detected (occurrence {self.nan_count}); "
                      "zeroing this micro-step's loss.", flush=True)
            zero = param_connected_zero(self.base_model, objective.device)
            return zero, zero.detach()
        return objective, hard

    def _gate(self, gate, global_step):
        """The probe's KD GATE, on the first micro-step of EVERY process: a resumed slice reloads the
        teacher, so a wrong --kd_teacher on slice 40 is as silent as on slice 1. Each rank checks its
        own batch and the verdict is all-reduced, so a failure stops every rank instead of leaving the
        others blocked in the next collective. Uniform over 151,680 classes is CE 11.93; a
        vocab-misaligned teacher lands near gold_in_topk = K/V.
        """
        self.gate_pending = False
        stats = torch.stack([gate["mass"].float(), gate["gold"].float(),
                             gate["hard"].float(), gate["soft"].float()]).tolist()
        # NOT(>= 0.05) rather than the probe's (< 0.05), so a NaN statistic (a teacher with NaN
        # weights) FAILS the gate instead of slipping through every comparison.
        fail = torch.tensor([0.0 if (stats[0] >= 0.05 and stats[1] >= 0.05) else 1.0], device=self.device)
        if self.world_size > 1:
            dist.all_reduce(fail, op=dist.ReduceOp.MAX)
        if self.is_main:
            print(f"[kd] KD GATE (first micro-step of this process, global_step {global_step}): "
                  f"hard_ce={stats[2]:.4f} topk_mass={stats[0]:.4f} gold_in_topk={stats[1]:.4f} "
                  f"soft={stats[3]:.4f} -> {'FAILED' if fail.item() > 0 else 'ok on every rank'}", flush=True)
        if fail.item() > 0:
            if dist.is_initialized():
                dist.destroy_process_group()
            raise SystemExit("KD GATE FAILED: teacher top-K mass or gold-hit rate implausible on at least "
                             "one rank; refusing to train a run whose null would be indistinguishable "
                             "from a real one")
