"""argonne4.5 probe harness: 3x H200 DDP, <=30 min per arm.

One arm = one JSON config. Trains the argonne4.5 candidate on the production 50/30/20
edu/math/code mixture for a FIXED TOKEN BUDGET, then evaluates pure next-token CE on the
production held-out val sets. Emits throughput + HBM so systems arms and quality arms use the
same harness.

Invariants (each one is a lesson from a previous Argonne campaign):
  I1  Quality arms are compared ISO-TOKEN, never iso-wall. Wall time is a resource, not a metric.
  I2  Eval is PURE next-token CE: forward WITHOUT labels, compute CE from logits. MTP / z-loss
      must never leak into the reported number (arch-sweep invariant #4).
  I3  The reported node name is part of the record. Cross-GPU comparison is what forced the
      KEY FINDING 14 retraction and the Finding C retraction, and this cluster mixes H100/H200.
  I4  The source-sampling RNG is seeded identically on every rank and is independent of the
      micro-batch, so every arm sees the same source sequence.
  I5  A wall-guard trip still writes a record, marked invalid. A missing result is
      indistinguishable from a crash otherwise.
"""
import gc
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ArgonneConfig, ArgonneModel

VOCAB = 151680                      # len(Qwen3 tokenizer) padded to a mult of 128 for fp8 lm_head
EOS = 151643                        # Qwen3 <|endoftext|>; the flat bins use it as the doc separator.
# Measured density in the first 20M tokens of each source: edu 1 doc / 1,008 tok,
# finemath 1 / 1,505, code 1 / 1,875 -> a block-2048 window spans ~1.1-2.0 documents, so the
# packed batch really is letting tokens attend across document boundaries.
# Cluster-portable: midway3 default, overridden by A45_DATA on ALCF (where /eagle holds
# the same bins). Hardcoding this cost a round-trip -- a remote-only edit was silently
# clobbered the next time this file was rsynced out.
DATA = os.environ.get("A45_DATA", "/project/rcc/youzhi/data/argonne4_pretrain")
SOURCES = [("edu_flat.bin", 50), ("finemath_flat.bin", 30), ("code_flat.bin", 20)]
VALS = [("edu", "val_edu.bin"), ("math", "val_math.bin"), ("code", "val_code.bin")]
# EXTRA_VALS are measured and reported in `ce` but DELIBERATELY EXCLUDED from `tgt`.
# tgt = mean(ces.values()) over VALS, so adding a 4th entry to VALS would silently redefine tgt and
# make every new arm incomparable to the ~170 already run. This keeps tgt fixed while still
# reporting the new metric.
# code2: a SECOND held-out code slice from code_flat.bin at 70% offset, MEASURED zero 16-token
# shingle overlap with val_code.bin. Added because val_code.bin is 79% DUPLICATED at the shingle
# level (636,248 distinct windows per 3M tokens, vs 2,190,504 for val_code2) -- a repetitive eval is
# easier to fit and far more memorisation-sensitive, so a large code gain must replicate here.
EXTRA_VALS = [("code2", "val_code2.bin"), ("codetask", "val_codetask.bin")]
# codetask added 2026-08-27: HumanEval (164) + MBPP (964) = 1,128 items, 102,136 Qwen3 tokens.
# It closes the campaign's named gap -- val_code and val_code2 are BOTH carved from code_flat.bin,
# so 'code' has only ever meant 'models the pretrain code distribution better'. This is a real
# code-TASK distribution (signature + docstring + reference solution).
# ⚠️ TWO LIMITS, both structural:
#  1. CONTAMINATION IS CERTAIN, not suspected -- these benchmarks are in every public corpus, and
#     ~1% verbatim overlap was already measured between val_code and both code_flat and the anneal
#     corpus. ABSOLUTE CE here is meaningless; only the RELATIVE ranking of two arms that both see
#     the same leakage is informative.
#  2. It is 102k tokens vs 3M for the other val sets -- 30x less data, so a NOISIER estimate. Its
#     run-to-run floor must be measured in-cell before any claim, exactly as code2's was (code2
#     turned out to be 8.5-11.8x the code floor).
# RHO-1 reference: the argonne4.0 1.04B checkpoint, trained on this exact 50/30/20 mixture and the
# same tokenizer, so its per-token loss is a meaningful "what a competent model already finds easy".
RHO1_REF = "/project/rcc/youzhi/models/a4_dose/a4_base_59723"


def load_bin(path):
    with open(path, "rb") as f:
        if np.frombuffer(f.read(256 * 4), dtype=np.int32)[0] != 20240801:
            raise ValueError(f"bad magic: {path}")
    return np.memmap(path, dtype=np.uint32, mode="r", offset=256 * 4)


class Src:
    """One flat .bin, sharded across ranks (offset rank*B*T, stride B*T*world)."""

    def __init__(self, path, B, T, rank, world, start=0):
        self.t, self.B, self.T = load_bin(path), B, T
        self.rank, self.world = rank, world
        self.base = start
        self.pos = start + rank * B * T

    def state(self):
        return {"pos": int(self.pos), "base": int(self.base)}

    def load_state(self, st):
        self.pos, self.base = int(st["pos"]), int(st["base"])

    def next(self):
        n = self.B * self.T + 1
        if self.pos + n > len(self.t):
            self.pos = self.base + self.rank * self.B * self.T
        buf = torch.from_numpy(self.t[self.pos:self.pos + n].astype(np.int64))
        self.pos += self.B * self.T * self.world
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)


class Mix:
    """Weighted multi-source sampler. I4: rng carries NO rank term, so all ranks agree."""

    def __init__(self, B, T, rank, world, seed=1337):
        self.s = [Src(os.path.join(DATA, p), B, T, rank, world) for p, _ in SOURCES]
        w = np.array([w for _, w in SOURCES], dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)
        # length follows SOURCES, not a hardcoded 3 -- a 4-source mix (e.g. the anneal corpus
        # added via cfg["sources"]) indexed past the end of [0,0,0] and raised IndexError at
        # self.counts[i] += 1 in next().
        self.counts = [0] * len(SOURCES)

    def state(self):
        # The sampler's ENTIRE state is 3 byte-offsets + the bit-generator state + counts. That is
        # why resume is cheap here: there is no shuffle buffer, no worker state, nothing implicit.
        return {"srcs": [x.state() for x in self.s],
                "rng": self.rng.bit_generator.state,
                "counts": list(self.counts)}

    def load_state(self, st):
        for x, sub in zip(self.s, st["srcs"]):
            x.load_state(sub)
        self.rng.bit_generator.state = st["rng"]
        self.counts = list(st["counts"])

    def next(self):
        i = int(self.rng.choice(len(self.s), p=self.p))
        self.counts[i] += 1
        return self.s[i].next()


# Max rows (sequences x tokens) in one eval forward. 4096 keeps the 151,680-wide logit tensor
# near 1.2 GiB in bf16, which fits beside the resident weights on a 44 GiB A40.
EVAL_ROW_CAP = int(os.environ.get("PROBE_EVAL_ROW_CAP", "4096"))

@torch.no_grad()
def eval_ce(model, path, B, T, device, windows=64):
    """I2: pure next-token CE. Forward WITHOUT labels so no auxiliary term can leak in.

    Called at MULTIPLE T (see eval_lengths). Evaluating past the training block is the only way
    to see length generalisation, which is the entire reason the NoPE-global layout was proposed:
    CE at the training length is blind to it. Batch is chosen so B*T is constant, keeping the
    fp32 logit transient the same size at every length.
    """
    model.eval()
    toks = load_bin(path)
    tot, n = 0.0, 0
    for w in range(windows):
        off = w * B * T
        if off + B * T + 1 > len(toks):
            break
        buf = torch.from_numpy(toks[off:off + B * T + 1].astype(np.int64))
        xw = buf[:-1].view(B, T)
        yw = buf[1:].view(B, T)
        # CHUNK THE EVAL BATCH. Measured 2026-08-22 (job 54266418): eval is far MORE memory-hungry
        # than training here. Training runs micro 1 x 2048 = 2,048 rows, but eval was called with
        # B=8 at T=2048 = 16,384 rows -- an 8x larger logit tensor (151,680 wide), 4.63 GiB in bf16
        # alone. It OOM'd AFTER a full 195-step run: whole compute bill, no tgt.
        # Chunking is exactly measurement-preserving: the CE below is TOKEN-WEIGHTED and the windows
        # read consecutive offsets, so splitting a window's batch covers identical tokens and yields
        # an identical average. Only the transient shrinks. Do not "fix" this by lowering B or
        # windows instead -- that would change which tokens are scored and make arms incomparable.
        mb = max(1, EVAL_ROW_CAP // T)
        for i in range(0, B, mb):
            x = xw[i:i + mb].to(device)
            y = yw[i:i + mb].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).logits
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
            tot += ce.item() * y.numel(); n += y.numel()
            del logits, x, y
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    main_proc = rank == 0
    t_job = time.time()
    deadline = t_job + float(cfg.get("wall_guard_sec", 1500))

    torch.manual_seed(cfg.get("seed", 444))
    np.random.seed(cfg.get("seed", 444))

    T = int(cfg.get("block_size", 2048))
    B = int(cfg["micro_batch"])
    accum = int(cfg["grad_accum"])
    eff = B * accum * T * world
    budget = int(cfg["train_tokens"])
    steps = max(1, budget // eff)

    arch = {k: v for k, v in cfg.items() if k in {
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "qk_norm", "v_norm", "sandwich_norm", "z_loss_weight",
        "logit_softcap", "rope_theta", "tie_word_embeddings", "mtp_horizon", "mtp_loss_weight",
        "interleaved_local_attention", "local_attention_window",
        "attn_pattern", "sliding_window_size", "nope_global", "attn_gate", "mlp_type",
        "mtp_module_layers", "doc_mask",
        # added 2026-08-26: these are real ArgonneConfig fields that were NEVER reachable from a
        # probe config, so four arch knobs had never been measured. Adding them to the passthrough
        # changes nothing for existing arms (absent key -> ArgonneConfig default).
        # VERIFIED WIRED before adding: attention_bias -> q/k/v/o nn.Linear (model.py:294-309),
        # mlp_bias -> gate/up/down (557-567), attention_dropout -> SDPA dropout_p + F.dropout
        # (426,472,482,493,535). rms_norm_eps is a real RMSNorm arg.
        # hidden_dropout -> nn.Dropout in the MLP/block (570,596).
        # NOT added: head_dim, which is DERIVED (hidden_size//num_heads, model.py:285) and is NOT
        # a config field -- passing it would be silently ignored and would imply a knob that does
        # not exist. Vary num_attention_heads instead.
        "attention_bias", "mlp_bias", "attention_dropout", "hidden_dropout", "rms_norm_eps"}}
    mc = ArgonneConfig(vocab_size=VOCAB, max_position_embeddings=T, use_flash_attention=True,
                       loss_chunk_size=int(cfg.get("loss_chunk_size", 0)), **arch)
    mc._keep_in_fp32_modules = []
    model = ArgonneModel(mc).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    if cfg.get("grad_checkpointing", 1):
        model.set_gradient_checkpointing(True)
        model.checkpoint_stride = int(cfg.get("checkpoint_stride", 1))

    if cfg.get("fp8", 0):
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        f8 = Float8LinearConfig(enable_fsdp_float8_all_gather=False)
        convert_to_float8_training(
            model, config=f8,
            module_filter_fn=lambda m, fqn: ("lm_head" not in fqn or cfg.get("fp8_lm_head", 1))
            and getattr(m, "in_features", 16) % 16 == 0 and getattr(m, "out_features", 16) % 16 == 0)
    t_compile0 = time.time()
    if cfg.get("torch_compile", 1):
        model = torch.compile(model, mode=cfg.get("compile_mode", "default"))
    model = DDP(model, device_ids=[local])
    # bf16 gradient compression, so the CE cost of halving the wire bytes is measurable here.
    # MEASURED 2026-08-28 on 2 nodes at 11 micro-steps: +14.5% throughput (31,485 -> 36,050
    # tok/s), which implies comms is ~25% of step time and half of it was recovered. That is a
    # THROUGHPUT result only -- cross-rank gradient summation moves to bf16, which is a
    # numerical change, so it needs a CE arm before adoption.
    # NOTE: only has any effect with world_size > 1. The standard session runs four INDEPENDENT
    # 1-GPU arms, where there is no all-reduce at all and this is a no-op -- the CE arm must be
    # a dedicated multi-rank probe job.
    # ⚠️ SYNCED BACK FROM /eagle 2026-09-04. This block existed ONLY on the ALCF copy and was
    # missing from the repo, i.e. the tracked instrument was not the one running the arms. That is
    # exactly the failure the DATA comment above warns about ("a remote-only edit was silently
    # clobbered the next time this file was rsynced out"). Keep the repo the SUPERSET.
    if str(cfg.get("ddp_comm_hook", "none")).lower() == "bf16" and world > 1:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as _hooks
        model.register_comm_hook(None, _hooks.bf16_compress_hook)
        if rank == 0:
            print("[probe] DDP comm hook: bf16_compress (wire bytes halved)", flush=True)

    # OPTIMIZER SELECTION (added 2026-08-23). `optimizer` defaults to "adamw", so every config
    # written before this date takes the identical code path it always did -- the recorded lesson is
    # that three of this campaign's first four arms died to code changes, so this one is additive.
    _optname = str(cfg.get("optimizer", "adamw")).lower()
    if _optname == "muon":
        # Recorded constraints, not choices: embeddings NEVER on Muon (+1.257); muon_wd must be
        # non-zero (wd=0 is +0.065 because the coupled decay is an effective-LR ramp); and the AdamW
        # group is now embeddings+norms only (~30% of params), for which 6e-4 was never tuned --
        # raising it to 2e-3 is recorded at -0.0914 (65 sigma). So `adam_lr` is a separate knob.
        from muon import MuonAdamW, split_params
        # Unwrap DDP (+ torch.compile) HERE rather than reusing `base`: that variable is not bound
        # until line ~423, and referencing it from here raised UnboundLocalError -- caught by the
        # 30-second smoke (job 54437676) instead of by a 24-minute arm. `ast.parse` accepts an
        # unbound name because it is a RUNTIME error, which is the same trap the campaign recorded
        # for a missing `import gc`: asserting the IMPORT landed did not assert the VARIABLES exist.
        _root = model.module if hasattr(model, "module") else model
        _root = _root._orig_mod if hasattr(_root, "_orig_mod") else _root
        _mp, _ap, _rep, _mn, _an = split_params(_root)
        assert _mp, "Muon requested but the split found no 2-D block weights"
        assert not any(("embed" in n.lower() or "lm_head" in n.lower()) for n in _mn), \
            "an embedding/lm_head leaked into the Muon group -- recorded at +1.257, refusing to run"
        if main_proc:
            print(_rep, flush=True)
        opt = MuonAdamW(_mp, _ap,
                        muon_lr=float(cfg.get("muon_lr", 0.04)),
                        adam_lr=float(cfg.get("adam_lr", 2e-3)),
                        momentum=float(cfg.get("muon_momentum", 0.95)),
                        ns_steps=int(cfg.get("muon_ns_steps", 5)),
                        muon_wd=float(cfg.get("muon_wd", 0.01)),
                        nesterov=bool(int(cfg.get("muon_nesterov", 1))),
                        batched=bool(int(cfg.get("muon_batched", 0))),
                        batch_size=int(cfg.get("muon_batch_size", 8)))
    elif int(cfg.get("optimizer_offload", 0)):
        # ⭐ HOST-RESIDENT AdamW MOMENTS (INFRA M+179a). fp32 AdamW at 2.0637 B needs m/v = 15.38 GiB
        # on top of params 7.69 + grads 7.69, and cell 28's a4.5 PRODUCTION BASELINE arm OOM'd on the
        # A100-40G with 38.30 GiB allocated of 39.39. Production only fits by sharding AdamW across 24
        # ranks; at world=1 there is nothing to shard, so this was the one comparison the rig could not
        # run. Holding m/v on the host frees that 15.38 GiB for one round trip per OPTIMISER step
        # (grads out, params back) = ~1-2 s/step, ~3-5 min over an h146 arm.
        # ⚠️ NOT bit-identical to the branch below, which uses the FUSED CUDA kernel; this path is CPU
        # elementwise AdamW. Equivalent arithmetic, possibly different last bits -- irrelevant against
        # an h146 floor of 0.019. adamw_offload.selftest() shows 0.000e+00 delta vs non-fused AdamW.
        # ⛔ OPT-IN ONLY. Absent `optimizer_offload`, this branch is skipped and every existing arm
        # takes the untouched fused path below, byte-for-byte.
        from adamw_offload import CPUOffloadAdamW
        opt = CPUOffloadAdamW(model.parameters(), lr=cfg.get("lr", 6e-4),
                              betas=(0.9, 0.95), weight_decay=0.1)
        print("[probe] AdamW moments OFFLOADED to host: %.2f GiB of HBM not used by optimiser state"
              % (opt.host_bytes() / 2**30), flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 6e-4),
                                betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    warm, cd = int(cfg.get("warmup", 0.02 * steps)) or 1, cfg.get("cooldown_frac", 0.15)
    cds = max(1, int(cd * steps))

    def lr_l(s):
        if s < warm:
            return s / warm
        if s < steps - cds:
            return 1.0
        return 1.0 - min(1.0, (s - (steps - cds)) / cds) * 0.9
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_l)

    # Data mixture is a config knob now. It is the one axis the a4 campaign showed to be BOTH
    # scale-robust AND growing with tokens ("the composition win is scale-robust and GROWING"),
    # which is exactly the property the size question turned out to lack -- so unlike size, a
    # probe-scale mixture result has a defensible claim to transfer.
    if cfg.get("mix") or cfg.get("sources"):
        global SOURCES
        # `mix` reweights the three default pretrain bins. `sources` (added 2026-08-26) overrides the
        # FILENAMES too, which is what makes the midtraining/anneal corpora testable at probe scale --
        # before this, line 269 hardcoded edu/finemath/code and no config could reference
        # reasoning_anneal_flat.bin even though it sits in the same data dir.
        # Format: "sources": [["anneal/reasoning_anneal_flat.bin", 30], ["edu_flat.bin", 35], ...]
        # Weights are relative; Mix normalises them.
        if cfg.get("sources"):
            SOURCES = [(str(p_), float(w_)) for p_, w_ in cfg["sources"]]
        else:
            SOURCES = [(p_, w_) for p_, w_ in zip(
                ["edu_flat.bin", "finemath_flat.bin", "code_flat.bin"], cfg["mix"])]
        if main_proc:
            print(f"[probe] sources = {SOURCES}", flush=True)
    mix = Mix(B, T, rank, world, seed=cfg.get("data_seed", 1337))
    use_doc = bool(cfg.get("doc_mask", False))
    # ---- RHO-1 selective language modelling -------------------------------------------------
    # Backprop only the top-k% of tokens by EXCESS loss (student CE - reference CE). The claim is
    # 2-10x data efficiency, which is the single biggest lever on the tokens/param constraint that
    # arms 10/11 measured as dominant. Costs one frozen forward of a 1.04B reference per micro-step.
    rho1_topk = float(cfg.get("rho1_topk", 0.0))
    # ---- KD from a public teacher -------------------------------------------------------
    # ALGEBRA FIRST (this is the lesson that cost two forward-only memory errors): an alpha-mixed
    # KD objective is IDENTICAL to a cross-entropy against the mixed target distribution
    #     p_target = (1-a)*onehot(y) + a*p_teacher
    # so it needs ONE softmax over the student, not a separate KL term with its own temporaries.
    #
    # FULL-VOCAB NORMALIZATION IS LOAD-BEARING. The campaign measured that normalizing over only
    # the retained top-K slice is WORSE THAN NO KD AT ALL. So p_teacher is normalized over the
    # full vocab via logsumexp and the top-K entries keep their ACTUAL mass -- the dropped tail
    # mass is simply omitted from the sum, never redistributed. Do not "fix" the fact that the
    # weights sum to <1; that is the correct objective.
    #
    # Top-K is taken on the LOGITS, which is exact: softmax is monotone, so argtop-K agrees.
    # This avoids materialising a full fp32 softmax (a [1,2048,151680] fp32 tensor is 1.24 GiB).
    kd_alpha = float(cfg.get("kd_alpha", 0.0))
    CH_T = int(cfg.get("kd_teacher_chunk", 256))   # positions per teacher-side logsumexp/top-K chunk
    kd_topk = int(cfg.get("kd_topk", 32))
    teacher = None
    if kd_alpha > 0:
        from transformers import AutoModelForCausalLM
        # Cluster-portable, same pattern as A45_DATA: config key wins, then $A45_TEACHER, then
        # the midway3 default. Hardcoding this bit twice (once per new config batch).
        tp_path = cfg.get("kd_teacher", os.environ.get("A45_TEACHER",
                          "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base"))
        # transformers renamed this kwarg: <=4.5x wants `torch_dtype`, newer accepts `dtype`.
        # midway3's env took `dtype`; ALCF's transformers 4.51.0 raises
        # "Qwen3ForCausalLM.__init__() got an unexpected keyword argument 'dtype'".
        # Try the new name, fall back to the old one, so the same file runs on both.
        try:
            teacher = AutoModelForCausalLM.from_pretrained(
                tp_path, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher = AutoModelForCausalLM.from_pretrained(
                tp_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
        teacher.requires_grad_(False)
        if main_proc:
            print(f"[probe] KD on: alpha={kd_alpha} topk={kd_topk} teacher={os.path.basename(tp_path)} "
                  f"(vocab {teacher.config.vocab_size} -> truncated to {VOCAB})", flush=True)
        if teacher.config.vocab_size < VOCAB:
            raise SystemExit(f"teacher vocab {teacher.config.vocab_size} < ours {VOCAB}: cannot truncate")
    ref_model = None
    if rho1_topk > 0:
        ref_model = ArgonneModel.from_pretrained(RHO1_REF, dtype=torch.bfloat16).to(dev).eval()
        for prm in ref_model.parameters():
            prm.requires_grad_(False)
        if main_proc:
            print(f"[probe] RHO-1 on: keep top {rho1_topk:.0%} by excess loss, ref={RHO1_REF}", flush=True)
    if main_proc:
        print(f"[probe] {cfg['id']} params={nparams:,} steps={steps} eff_batch={eff:,} "
              f"block={T} micro={B}x{accum} world={world} node={socket.gethostname()}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    losses, t_first, t_steady, tok_steady, aborted = [], None, None, 0, False
    STEADY_FROM = min(10, max(2, steps // 10))
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for m in range(accum):
            x, y = mix.next()
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            # Document ids for intra-document masking: doc_id[i] = how many EOS tokens occur
            # strictly BEFORE position i, so the separator belongs to the document it terminates.
            # Ids only need to be unique within a row; each row is an independent window.
            kw = {}
            if use_doc:
                is_eos = (x == EOS)
                kw["document_ids"] = torch.cumsum(is_eos.long(), dim=1) - is_eos.long()
            sync = (m == accum - 1)
            with model.no_sync() if not sync else torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if ref_model is not None:
                        with torch.no_grad():
                            # Read the vocab off the TENSOR: the exported a4 reference has the fp8
                            # lm_head padding trimmed (151,669) while the student trains padded
                            # (151,680). Targets max at 151,643 so CE is valid against either.
                            rl = ref_model(x).logits
                            ref_ce = F.cross_entropy(
                                rl.reshape(-1, rl.size(-1)), y.view(-1), reduction="none")
                        lg = model(x, **kw).logits
                        tok_ce = F.cross_entropy(
                            lg.reshape(-1, lg.size(-1)), y.view(-1), reduction="none")
                        keep = max(1, int(rho1_topk * tok_ce.numel()))
                        sel = torch.topk(tok_ce.detach() - ref_ce, keep).indices
                        loss = tok_ce[sel].mean() / accum
                    elif teacher is not None:
                        with torch.no_grad():
                            tl = teacher(x).logits[...,:VOCAB]
                            # Chunked teacher-side logsumexp/top-K.
                            # ⚠️ HONEST NOTE: this was added expecting to recover ~1.0 GiB, because
                            # logsumexp(tl.float()) materialises a [B,T,151936] FP32 tensor (1.16 GiB
                            # at T=2048). It recovered ~2 MiB. Job 54302190 came back with numbers
                            # IDENTICAL to job 54300993 (43.77 GiB in use, 594 MiB requested, 589.81
                            # free vs 591.81). Reason: that transient lives under no_grad and is freed
                            # immediately, so it was never part of the PEAK -- the peak is persistent
                            # state plus `lg`. Optimising a transient does nothing for a peak.
                            # Kept anyway: it is exact (verified zero diff, identical indices) and
                            # strictly lowers the high-water mark on smaller-HBM cards.
                            # Exact, not an approximation: logsumexp and top-K are both per-position
                            # reductions over the vocab axis, so splitting the position axis changes
                            # nothing. Full-vocab normalization is still per-token over all 151,680
                            # classes -- the k-slice-only variant was measured WORSE THAN NO KD.
                            _pv, _pi = [], []
                            for c0 in range(0, tl.size(1), CH_T):
                                tlc = tl[:, c0:c0 + CH_T].float()
                                lse_c = torch.logsumexp(tlc, dim=-1, keepdim=True)
                                v_c, i_c = torch.topk(tlc, kd_topk, dim=-1)
                                _pv.append(torch.exp(v_c - lse_c)); _pi.append(i_c)
                                del tlc, lse_c, v_c, i_c
                            tkp = torch.cat(_pv, dim=1); tki = torch.cat(_pi, dim=1)
                            del tl, _pv, _pi
                        lg = model(x, **kw).logits
                        # MEMORY, measured not guessed (smoke job 54300220 OOM'd here):
                        #   "Tried to allocate 1.16 GiB" == [2048, 151680] fp32 == the log_softmax
                        # that F.cross_entropy saves for backward. The no-KD path never pays this
                        # because it uses the model's FUSED chunked CE (loss_chunk_size=0) and never
                        # materialises logits at all; this branch needs logits, so it does.
                        #
                        # Two fixes, both required:
                        #  (1) ALGEBRA (kept from before): log q_v = lg_v - lse, so
                        #      soft = -sum_k p_k lg_k + (sum_k p_k)*lse needs only [N,K] and [N].
                        #      lse falls out of cross_entropy exactly, since ce = lse - lg_y.
                        #  (2) CHUNK **PLUS RECOMPUTE**. Chunking alone does NOT help: every chunk's
                        #      log_softmax is retained for backward, summing to the same 1.24 GiB.
                        #      Wrapping each chunk in checkpoint() keeps one chunk resident and
                        #      recomputes the rest in backward -- one extra CE forward, negligible
                        #      against the 2.06B model.
                        # Exact same objective either way: CE is a token-weighted sum, so summing
                        # over chunks of consecutive tokens is identical to one pass.
                        V = lg.size(-1)
                        flat, yf = lg.view(-1, V), y.reshape(-1)
                        pf, idf = tkp.view(-1, kd_topk), tki.view(-1, kd_topk)
                        N, CH = yf.numel(), int(cfg.get("kd_chunk", 256))

                        def _chunk(lgc, yc, pc, ic):
                            ce_c = F.cross_entropy(lgc, yc, reduction="none")
                            lse_c = ce_c + lgc.gather(-1, yc.unsqueeze(-1)).squeeze(-1).float()
                            lgk_c = lgc.gather(-1, ic).float()
                            soft_c = -(pc * lgk_c).sum(-1) + pc.sum(-1) * lse_c
                            return ce_c.sum(), soft_c.sum()

                        h_s = f_s = None
                        for c0 in range(0, N, CH):
                            sl = slice(c0, min(c0 + CH, N))
                            hc, fc = ckpt(_chunk, flat[sl], yf[sl], pf[sl], idf[sl],
                                          use_reentrant=False)
                            h_s = hc if h_s is None else h_s + hc
                            f_s = fc if f_s is None else f_s + fc
                        hard, soft = h_s / N, f_s / N
                        if s == 0 and m == 0 and main_proc:
                            # TEACHER-CE GATE, run BEFORE the arm can report anything. A
                            # vocab-misaligned teacher yields a clean believable NULL, not an error.
                            # Uniform over 151,680 classes is CE 11.93.
                            gold_hit = (tki == y.unsqueeze(-1)).any(-1).float().mean().item()
                            print(f"[probe] KD GATE step0: hard_ce={hard.item():.4f} "
                                  f"topk_mass={tkp.sum(-1).mean().item():.4f} "
                                  f"gold_in_topk={gold_hit:.4f} soft={soft.item():.4f}", flush=True)
                            if tkp.sum(-1).mean().item() < 0.05 or gold_hit < 0.05:
                                raise SystemExit("KD GATE FAILED: teacher top-K mass or gold-hit rate "
                                                 "implausible -- refusing to run an arm whose null "
                                                 "would be indistinguishable from a real one")
                        loss = ((1.0 - kd_alpha) * hard + kd_alpha * soft) / accum
                        del tkp, tki, lg, flat, pf, idf, h_s, f_s
                    else:
                        loss = model(x, labels=y, **kw).loss / accum
                loss.backward()
            acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 0.4))
        opt.step(); sched.step()
        losses.append(acc)
        if s == 0:
            t_first = time.time()
        if s == STEADY_FROM:
            torch.cuda.synchronize(); t_steady = time.time(); tok_steady = 0
        elif t_steady is not None:
            tok_steady += eff
        # Live progress, so a stalled or slower-than-budget arm is visible from squeue-time rather
        # than only at the end. Also the only place the wall-guard ETA is observable mid-run.
        if main_proc and (s % 25 == 0 or s == steps - 1) and t_steady is not None and s > STEADY_FROM:
            el = time.time() - t_steady
            tps_now = tok_steady / max(1e-9, el)
            eta = (steps - 1 - s) * eff / max(1.0, tps_now)
            print(f"[probe] step {s}/{steps} loss={acc:.4f} tok/s={tps_now:,.0f} "
                  f"eta={eta/60:.1f}min guard_left={(deadline - time.time())/60:.1f}min "
                  f"hbm={torch.cuda.max_memory_allocated()/torch.cuda.get_device_properties(local).total_memory:.1%}",
                  flush=True)
        if time.time() > deadline:
            aborted = True
            if main_proc:
                print(f"[probe] WALL GUARD at step {s}/{steps}", flush=True)
            break
    torch.cuda.synchronize()
    t_train_end = time.time()
    done_steps = len(losses)
    tps = tok_steady / max(1e-9, t_train_end - t_steady) if t_steady else 0.0
    hbm = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(local).total_memory

    base = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module

    # FREE THE TRAINING STATE BEFORE EVAL. Measured 2026-08-22 (job 54258539): the arm trained all
    # 195 steps at 94.6% HBM and then OOM'd in the FIRST eval forward, wanting 1.25 GiB with 1.18 GiB
    # free and 42.05 GiB still allocated -- because AdamW's m/v (15.38 GiB) and the gradients
    # (7.69 GiB) are still resident when eval starts. Training succeeded and only the MEASUREMENT
    # died, which is the worst way to lose an arm: full compute cost, no tgt.
    # The recorded arms never hit this because they ran fp8=1 on H100/H200, which both saved memory
    # and had more of it. On A40/A100 with fp8=0 this is mandatory.
    # Eval is @torch.no_grad and only needs the weights, so dropping the optimizer and grads is free.
    try:
        opt.zero_grad(set_to_none=True)
        del opt
    except Exception:
        pass
    for _p in base.parameters():
        _p.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    if main_proc:
        print(f"[probe] freed optimizer+grads before eval: "
              f"{torch.cuda.memory_allocated()/2**30:.2f} GiB allocated, "
              f"{torch.cuda.mem_get_info()[0]/2**30:.2f} GiB free", flush=True)

    ces, len_ces = {}, {}
    # Evaluate even after a wall-guard trip. SLURM gives ~35 min and the guard fires at 27, so
    # there is room, and a partial run WITH a TGT is usable -- compared at equal wall time it is
    # exactly the iso-compute question. Without one it is a wasted arm; a05 and a11 both died here.
    if main_proc:
        # skip_eval: for SMOKE TESTS only. Eval is 3 domains x 64 windows x 8 rows x T, ~411s at
        # T=2048 -- as expensive as 1.5M training tokens. A smoke test exists to prove a code path
        # runs on the GPU and to measure real tok/s; paying 411s to score a 9-step model tells us
        # nothing. Any arm that reports a comparable number must leave this at 0.
        for name, f in VALS:
            if int(cfg.get("skip_eval", 0)):
                ces[name] = float("nan"); continue
            ces[name] = eval_ce(base, os.path.join(DATA, f), max(1, 8), T, dev)
        tgt_names = [n for n, _ in VALS]          # freeze the tgt basis BEFORE extras are added
        for name, f in EXTRA_VALS:
            if int(cfg.get("skip_eval", 0)):
                ces[name] = float("nan"); continue
            try:
                ces[name] = eval_ce(base, os.path.join(DATA, f), max(1, 8), T, dev)
            except Exception as e:
                ces[name] = float("nan")          # a missing extra must never fail an arm
        # Length generalisation: same held-out text, longer windows. Anything past the training
        # block T is extrapolation. B*T held at 16,384 so the logit transient does not grow.
        for L in cfg.get("eval_lengths", []):
            if L == T:
                len_ces[str(L)] = float(np.mean([ces[n] for n in tgt_names]))
                continue
            try:
                vals = [eval_ce(base, os.path.join(DATA, f), max(1, 16384 // L), L, dev, windows=16)
                        for _, f in VALS]
                len_ces[str(L)] = float(np.mean(vals))
            except Exception as e:
                len_ces[str(L)] = f"ERR {type(e).__name__}"

    if main_proc:
        rec = dict(
            id=cfg["id"], status="wall_guard" if aborted else "ok", valid=not aborted,
            node=socket.gethostname(), gpu=torch.cuda.get_device_name(local), world=world,
            params=nparams, steps_planned=steps, steps_done=done_steps,
            tokens=done_steps * eff, eff_batch=eff, block=T, micro_batch=B, grad_accum=accum,
            train_loss_ema=float(np.mean(losses[-20:])) if losses else None,
            ce=ces,
            # tgt averages ONLY the original VALS names, never EXTRA_VALS -- see the note at VALS.
            tgt=float(np.mean([ces[n] for n in tgt_names])) if ces else None,
            tgt_by_length=len_ces,
            tokens_per_sec=round(tps, 1), hbm_frac=round(hbm, 4),
            startup_sec=round(t_first - t_job, 1) if t_first else None,
            compile_and_first_step_sec=round(t_first - t_compile0, 1) if t_first else None,
            train_sec=round(t_train_end - (t_steady or t_job), 1),
            job_sec=round(time.time() - t_job, 1), src_counts=mix.counts, config=cfg)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)
        print("[probe] RESULT " + json.dumps({k: rec[k] for k in
              ("id", "status", "params", "tokens", "tgt", "ce", "tokens_per_sec",
               "hbm_frac", "startup_sec", "job_sec")}), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
