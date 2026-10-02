"""Muon (Momentum Orthogonalised by Newton-Schulz): the PRODUCTION copy used by pretrain.py and
continue_pretrain.py.

PROVENANCE AND WHAT DIFFERS FROM exp/muon.py (argonne5.0 port, 2026-09-29)
--------------------------------------------------------------------------
Copied from exp/muon.py (md5 0dd04869..., the file every a5 probe number was measured with). The
update math is unchanged line for line: `zeropower_via_newtonschulz5`, `zeropower_batched`, `Muon`,
`split_params` and every arithmetic line of `MuonAdamW.step` are verbatim. Three deliberate changes,
plus a trainer-integration section at the bottom (flags, the checked build, the checkpoint layout)
that both trainers import so their Muon handling cannot drift:

1. The batched path's per-shape cache no longer lives INSIDE the param group. exp/muon.py stored it as
   `group["_shape_groups"]`, and `Optimizer.state_dict()` packs every group key except "params", so a
   checkpoint pickled the live weight tensors; `load_state_dict()` then rebuilt the group FROM the
   checkpoint, cache included, and every later batched step updated those deserialized copies instead
   of the model. Reproduced on CPU 2026-09-29: after save -> load, 0 of 2 Muon matrices moved on the
   next step. The probe never resumed, so it never hit this; production resumes every slice. The cache
   is now held on the optimizer and keyed by parameter identity, which survives a load.
2. `muon_param_groups()` / `muon_defaults()` build the two tagged groups and the defaults dict, and
   `MuonAdamW.__init__` now calls them, so the groups a ZeRO-sharded run builds cannot drift from the
   ones the unsharded optimizer builds (exp/test_a5_trainer.py compares both against exp/muon.py).
3. `MuonAdamWGroups` is the constructor ZeroRedundancyOptimizer needs: it is called as
   `optimizer_class(params=<this rank's slice of every group>, **defaults)`.

The defaults below (muon_lr 0.04, adam_lr 2e-3) are the PROBE's h500-era values. pretrain.py
deliberately has NO default for either: the optimum moves with horizon (muon_lr 0.08 at 500 steps,
~0.016-0.02 at 6,400 steps, LR ~ horizon^-0.6; adam_lr 2e-3 short, 5e-4 at 6,400), so a production
run must state both.

WHY THIS FILE EXISTS
--------------------
The recorded a5 recipe audit found that at production shape (2.06B) the recipe is ~97% the optimizer
swap: a4.5 6.5720 -> Muon+ReLU^2 5.3049, i.e. **-1.267 at 43 sigma**. Every arm of the 2026-08-22/23
KD campaign ran plain `AdamW(model.parameters())` because there is no Muon anywhere in this repo, so
34 arms optimised a second-order term (the whole KD stack is -0.21) in a regime the production model
will not use.

SECOND PAYOFF, which is why this is worth more than another knob: Muon keeps ONE state tensor
(momentum) per hidden weight instead of AdamW's two (m and v). At 2.06B with ~70% of params on Muon
that is ~5.8 GiB of optimiser state freed -- **50x the 110 MiB by which the 1.7B teacher OOM'd**
(n322). So this build also unlocks the biggest recorded non-optimizer lever.

DESIGN CONSTRAINTS TAKEN FROM THE RECORD, NOT INVENTED HERE
-----------------------------------------------------------
- ⛔ **Embeddings must NOT go on Muon** -- recorded at +1.257, the single worst result in the campaign.
  With `tie_word_embeddings=True` the lm_head *is* the embedding, so it stays on AdamW too.
- ⛔ **`muon_wd = 0` is +0.065 (47 sigma).** The coupled decay is an effective-LR ramp, not a
  regulariser: measured hidden-weight RMS 1.041x without it vs 0.420x with. Keep it on.
- ⚠️ **`muon_lr` MUST be tuned and prefers the high side** -- undershooting costs ~8x what
  overshooting does. Recorded starting point ~0.04.
- ⭐ **The AdamW group's LR is NOT 6e-4 any more.** Under Muon that group is embeddings + norms only
  (~30% of params), and 6e-4 was AdamW-everything's tuned value. Raising it 6e-4 -> 2e-3 is recorded
  at **-0.0914 (65 sigma)**. Swapping the optimizer for 70% of the params silently untests every
  hyperparameter on the other 30%, so `adam_lr` is exposed separately and defaults to 2e-3 here.
- ⚠️ **Wall-clock parity FAILS at 2.06B** (Muon -1.6%): Newton-Schulz is O(n^3) in width, and the
  recorded parity result was measured at 1.03B. So any Muon arm must be scored ISO-COMPUTE.
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration -> approximately orthogonalised G (same shape).

    Runs in bfloat16 on purpose: the iteration is contractive and the coefficients below are the
    standard quintic tuned for fast convergence of the singular values toward 1, not for numerical
    exactness. Transposing so the short side is last keeps the A = X X^T product small.
    """
    assert G.ndim == 2, f"Newton-Schulz needs a matrix, got {G.ndim}D"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters only. Pair it with AdamW for embeddings / norms / biases.

    Update: buf = mu*buf + g ; (Nesterov: g <- g + mu*buf) ; step -= lr * NS5(g) * sqrt(max(1, r/c)).
    The sqrt(fan-out/fan-in) factor keeps the update RMS comparable across differently-shaped
    matrices, which is what lets ONE `muon_lr` serve every hidden weight in the model.
    """

    def __init__(self, params, lr=0.04, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                           ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu, nesterov = group["lr"], group["momentum"], group["nesterov"]
            ns_steps, wd = group["ns_steps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert p.ndim == 2, f"Muon got a {p.ndim}D param; route it to AdamW instead"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                upd = g.add(buf, alpha=mu) if nesterov else buf
                upd = zeropower_via_newtonschulz5(upd, steps=ns_steps)
                # DECOUPLED weight decay, applied to the parameter. Recorded as load-bearing:
                # muon_wd=0 is +0.065 because without it the effective LR ramps as ||W|| grows.
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(upd, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)
        return loss


def split_params(model):
    """(muon_params, adam_params, report) -- the split is the load-bearing decision, so it is
    explicit and reported rather than inferred from a name heuristic at the call site.

    Muon: 2-D parameters inside the transformer blocks.
    AdamW: everything else -- embeddings (recorded +1.257 if put on Muon; with tied weights this is
    also the lm_head), every norm and bias (1-D), and any 2-D parameter outside the blocks.
    """
    muon, adam, muon_names, adam_names = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = ("embed" in name.lower()) or ("wte" in name.lower()) or ("lm_head" in name.lower())
        if p.ndim == 2 and not is_embed:
            muon.append(p); muon_names.append(name)
        else:
            adam.append(p); adam_names.append(name)
    nm, na = sum(p.numel() for p in muon), sum(p.numel() for p in adam)
    report = (f"[muon] split: {len(muon)} tensors / {nm:,} params ({100*nm/(nm+na):.1f}%) on Muon, "
              f"{len(adam)} tensors / {na:,} params ({100*na/(nm+na):.1f}%) on AdamW "
              f"(embeddings+norms held out on purpose)")
    return muon, adam, report, muon_names, adam_names


@torch.no_grad()
def zeropower_batched(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz on a STACK of identically-shaped matrices: G is (N, r, c).

    Mathematically identical to calling `zeropower_via_newtonschulz5` on each slice -- same
    coefficients, same per-matrix normalisation, same transpose rule (which is shared, because the
    slices share a shape). The only difference is that the matmuls are batched, so the 168 Muon
    tensors become 4 shape-groups: ~80 kernel launches per step instead of ~3,360, a 42x reduction.

    WHY THIS AND NOT FEWER ITERATIONS: n344 measured `ns_steps` 5 -> 3 at **+0.2533 tgt (7.4x bar)**
    for only **+5.0% throughput** -- a terrible trade, and it proves the 14.2% wall penalty is not the
    NS arithmetic. Two fewer iterations of the actual math buys 5%; the rest is per-tensor loop and
    launch overhead. Batching removes the overhead without touching the math, so it is the free
    version of the rebate that n344 tried to buy with quality.
    """
    assert G.ndim == 3, f"expected a stacked (N,r,c) batch, got {G.ndim}D"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    # per-matrix Frobenius norm, kept as (N,1,1) so the divide broadcasts exactly as the loop version
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """ONE real `torch.optim.Optimizer` with two tagged param groups: Muon for the 2-D block weights,
    AdamW for embeddings/norms. `step()` dispatches on `group["use_muon"]`.

    WHY NOT A WRAPPER HOLDING TWO OPTIMISERS: that was the first design and it failed in 1 minute --
    `LambdaLR` does an `isinstance(optimizer, Optimizer)` check, so a duck-typed shim raises
    "MultiOptimizer is not an Optimizer" no matter how faithfully it mimics the interface. Two param
    groups on one optimiser is also strictly better: the scheduler scales each group's `lr` by the
    same warmup/cooldown SHAPE while each keeps its own base (muon 0.04 vs adam 2e-3, a 20x gap), and
    there is only one place for that shape to live.
    """

    def __init__(self, muon_params, adam_params, muon_lr=0.04, adam_lr=2e-3, momentum=0.95,
                 nesterov=True, ns_steps=5, muon_wd=0.01, adam_betas=(0.9, 0.95), adam_wd=0.1,
                 eps=1e-8, batched=False, batch_size=8):
        hp = dict(muon_lr=muon_lr, adam_lr=adam_lr, momentum=momentum, nesterov=nesterov,
                  ns_steps=ns_steps, muon_wd=muon_wd, adam_betas=adam_betas, adam_wd=adam_wd,
                  eps=eps, batched=batched, batch_size=batch_size)
        super().__init__(muon_param_groups(muon_params, adam_params, **hp), muon_defaults(**hp))

    def _shape_groups_for(self, group):
        """Same-shape parameter lists for the batched path, built once per group.

        Cached on the OPTIMIZER, keyed by the identity of the group's parameters, never in the group
        dict: a group-dict entry is written into state_dict() and restored by load_state_dict(), which
        pointed the batched update at deserialized copies after every resume (see the module header).
        Build order is identical to exp/muon.py (first appearance of each shape, params in group order).
        """
        cache = self.__dict__.setdefault("_shape_group_cache", {})
        key = tuple(id(p) for p in group["params"])
        shape_groups = cache.get(key)
        if shape_groups is None:
            byshape = {}
            for p in group["params"]:
                byshape.setdefault(tuple(p.shape), []).append(p)
            shape_groups = cache[key] = list(byshape.values())
        return shape_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"] and group.get("batched"):
                mu, nesterov, ns = group["momentum"], group["nesterov"], group["ns_steps"]
                # group by shape once, cache on the optimiser (NOT in the group: see _shape_groups_for)
                shape_groups = self._shape_groups_for(group)
                # SUB-BATCH the shape group. Batching all 48 same-shape tensors at once OOM'd
                # (job 54490811, 15:06): the (7040,2560) group is gate+up across 24 layers = 48
                # tensors, whose stacked bf16 X alone is 1.61 GiB and whose peak with A=X@X^T and
                # A@A is ~2.78 GiB -- against ~1.2 GiB free at HBM 0.818 with the teacher resident.
                # A cap of `batch_size` keeps the launch reduction (48/8 = 6x on the worst group,
                # ~8x overall) while bounding the intermediate to ~0.5 GiB.
                bs = int(group.get("batch_size", 8)) or 8
                for plist_full in shape_groups:
                  for i0 in range(0, len(plist_full), bs):
                    plist = plist_full[i0:i0 + bs]
                    live = [p for p in plist if p.grad is not None]
                    if not live:
                        continue
                    ups = []
                    for p in live:
                        st = self.state[p]
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(p.grad)
                        buf = st["momentum_buffer"]
                        buf.mul_(mu).add_(p.grad)
                        ups.append(p.grad.add(buf, alpha=mu) if nesterov else buf)
                    stacked = zeropower_batched(torch.stack(ups), steps=ns)
                    for p, upd in zip(live, stacked):
                        if wd:
                            p.mul_(1 - lr * wd)
                        p.add_(upd, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)
            elif group["use_muon"]:
                mu, nesterov, ns = group["momentum"], group["nesterov"], group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    assert p.ndim == 2, f"Muon group got a {p.ndim}D param; route it to AdamW"
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p.grad)
                    buf = st["momentum_buffer"]
                    buf.mul_(mu).add_(p.grad)
                    upd = p.grad.add(buf, alpha=mu) if nesterov else buf
                    upd = zeropower_via_newtonschulz5(upd, steps=ns)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(upd, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)
            else:
                b1, b2 = group["betas"]; eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "exp_avg" not in st:
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                        st["step"] = 0
                    st["step"] += 1
                    m, v, t = st["exp_avg"], st["exp_avg_sq"], st["step"]
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    denom = (v / (1 - b2 ** t)).sqrt_().add_(eps)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(m / (1 - b1 ** t), denom, value=-lr)
        return loss


class MuonAdamWGroups(MuonAdamW):
    """MuonAdamW built from ALREADY-TAGGED param groups: the entry point ZeroRedundancyOptimizer needs.

    ZeRO constructs its local optimizer as `optimizer_class(params=<this rank's slice of every group>,
    **defaults)`, i.e. a list of group dicts that already carry `use_muon`, `momentum`, `betas` and so
    on, plus the defaults dict. MuonAdamW's own constructor takes (muon_params, adam_params) and would
    re-tag them, so this subclass hands the groups straight to torch.optim.Optimizer. A rank can
    receive an EMPTY slice of a group (ZeRO balances by size across all groups); step() then simply
    has nothing to do for that group. Build the groups with muon_param_groups() and the defaults with
    muon_defaults() so they are exactly what an unsharded MuonAdamW would hold.
    """

    def __init__(self, params, **defaults):
        torch.optim.Optimizer.__init__(self, params, defaults)


def muon_param_groups(muon_params, adam_params, muon_lr=0.04, adam_lr=2e-3, momentum=0.95,
                      nesterov=True, ns_steps=5, muon_wd=0.01, adam_betas=(0.9, 0.95), adam_wd=0.1,
                      eps=1e-8, batched=False, batch_size=8):
    """The two tagged groups, Muon first, exactly as MuonAdamW builds them (it calls this).

    ORDER IS PART OF THE CHECKPOINT FORMAT: torch.optim numbers params by their position across
    param_groups, so canonical order = every Muon param (split_params order), then every AdamW param.
    ZeroRedundancyOptimizer's global index is the same concatenation, which is what lets one
    consolidated state_dict load into both the sharded and the unsharded optimizer.
    """
    groups = []
    if muon_params:
        groups.append(dict(params=list(muon_params), lr=muon_lr, use_muon=True,
                           momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                           weight_decay=muon_wd, batched=batched,
                           batch_size=batch_size))
    if adam_params:
        groups.append(dict(params=list(adam_params), lr=adam_lr, use_muon=False,
                           betas=adam_betas, weight_decay=adam_wd, eps=eps))
    return groups


def muon_defaults(muon_lr=0.04, adam_lr=None, momentum=0.95, nesterov=True, ns_steps=5,
                  muon_wd=0.01, adam_betas=(0.9, 0.95), adam_wd=None, eps=1e-8, batched=None,
                  batch_size=None):
    """The optimizer-level defaults dict MuonAdamW passes to torch.optim.Optimizer.

    Accepts the same keywords as muon_param_groups() so one hyperparameter dict can feed both;
    adam_lr, adam_wd, batched and batch_size are per-group only and deliberately not defaults (that
    is how exp/muon.py builds it, and torch copies any missing group key from here).
    """
    return dict(lr=muon_lr, use_muon=True, momentum=momentum, nesterov=nesterov,
                ns_steps=ns_steps, weight_decay=muon_wd, betas=adam_betas, eps=eps)


# ---------------------------------------------------------------------------------------------------
# TRAINER INTEGRATION (argonne5.0). Shared by pretrain.py and continue_pretrain.py, which cannot import
# each other (pretrain.py parses argv at import time; that is why continue_pretrain.py carries copies
# of the ZeRO helpers). Flags, validation, the build and the checkpoint layout live here ONCE, so the
# two stages cannot drift apart the way the sliding window and the --cooldown 0 fix did.
# ---------------------------------------------------------------------------------------------------
def add_muon_args(parser):
    """--optimizer and the Muon knobs. Defaults reproduce a4.5 (AdamW) exactly."""
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"], help="adamw = a4.5 (AdamW on every param, LR --lr). muon = a5: Muon on the 2-D block weights + AdamW on embeddings/norms, one optimizer with two tagged param groups (muon.py). Recorded at production shape as ~97%% of the a5 recipe; at h6400 tuned-vs-tuned it beats AdamW by 0.247 tgt CE on two draws.")
    parser.add_argument("--muon_lr", type=float, default=None, help="REQUIRED with --optimizer muon, deliberately no default: the optimum moves with horizon (probe: 0.08 at 500 steps, ~0.016-0.02 at 6,400 steps, LR ~ horizon^-0.6). Derive it for the run's horizon; never copy 0.08.")
    parser.add_argument("--adam_lr", type=float, default=None, help="REQUIRED with --optimizer muon: LR of the AdamW group (embeddings + norms only under Muon). Probe: 2e-3 at short horizons, 5e-4 at 6,400 steps. --lr stays the AdamW-everything LR and is unused under Muon.")
    parser.add_argument("--muon_wd", type=float, default=0.01, help="Muon decoupled weight decay. Must be > 0: 0 measured +0.065 (47 sigma), the decay is an effective-LR control, not a regulariser.")
    parser.add_argument("--muon_momentum", type=float, default=0.95, help="Muon momentum (Nesterov on). 0.90 measured code +0.037 worse.")
    parser.add_argument("--muon_ns_steps", type=int, default=5, help="Newton-Schulz iterations. 3 measured +0.2533 tgt for +5%% throughput.")
    parser.add_argument("--muon_batched", type=int, default=0, choices=[0, 1], help="Batch the Newton-Schulz matmuls per shape. Default 0 = the measured a5 recipe: batched was 2.8-4.0%% SLOWER (two cells) and +0.0089 tgt, and no a5 arm ran it. Same math, different fp accumulation order.")


def check_muon_args(parser, args):
    """Refuse, at parse time and on every rank alike, what the record rules out."""
    if args.optimizer == "muon":
        if args.muon_lr is None or args.adam_lr is None:
            parser.error("--optimizer muon needs BOTH --muon_lr and --adam_lr. There are no defaults on "
                         "purpose: the probe optimum moved from 0.08/2e-3 at 500 steps to ~0.02/5e-4 at "
                         "6,400 steps (LR ~ horizon^-0.6), so derive both for this run's horizon.")
        if args.muon_lr <= 0 or args.adam_lr <= 0:
            parser.error("--muon_lr and --adam_lr must be > 0")
        if args.muon_wd <= 0:
            parser.error("--muon_wd must be > 0: muon_wd 0 measured +0.065 (47 sigma); the decay is what "
                         "keeps the effective LR from ramping as ||W|| grows.")
        if args.muon_ns_steps < 1 or not (0.0 <= args.muon_momentum < 1.0):
            parser.error("--muon_ns_steps must be >= 1 and --muon_momentum in [0, 1)")
    elif args.muon_lr is not None or args.adam_lr is not None:
        parser.error("--muon_lr/--adam_lr only apply to --optimizer muon; under adamw the LR is --lr. "
                     "A flag that silently does nothing is how LOCAL_ATTENTION_WINDOW stayed dead.")


def muon_hparams(args):
    """One hyperparameter dict for both muon_param_groups() and muon_defaults().

    The AdamW group's betas and weight decay come from the trainers' existing --adam_beta1/--adam_beta2/
    --weight_decay flags, whose defaults (0.9, 0.95, 0.1) equal the probe's MuonAdamW defaults.
    Nesterov is fixed on (off measured +0.065 tgt, +0.082 code) and eps is the probe's 1e-8.
    """
    return dict(muon_lr=args.muon_lr, adam_lr=args.adam_lr, momentum=args.muon_momentum,
                nesterov=True, ns_steps=args.muon_ns_steps, muon_wd=args.muon_wd,
                adam_betas=(args.adam_beta1, args.adam_beta2), adam_wd=args.weight_decay,
                eps=1e-8, batched=bool(args.muon_batched), batch_size=8)


def build_muon_optimizer(base_model, args, zero_sharded=False, is_main=True):
    """MuonAdamW over split_params(base_model), exactly as exp/probe.py builds it; ZeRO-sharded on request.

    ⛔ Embeddings (and the tied lm_head, which IS the embedding) must never be on Muon: recorded at
    +1.257, the worst result of the campaign. split_params already routes them to AdamW; this checks it
    by name AND by tensor identity, because a name heuristic is exactly what fails silently.

    `_canonical_groups` is recorded on the returned optimizer: the trainers' ZeRO save path uses it to
    write the same two-group state_dict an unsharded MuonAdamW emits (see muon_canonical_index).
    """
    muon_p, adam_p, report, muon_names, _adam_names = split_params(base_model)
    if not muon_p:
        raise SystemExit("FATAL: --optimizer muon but split_params found no 2-D block weights")
    leaked = [n for n in muon_names if any(k in n.lower() for k in ("embed", "lm_head", "wte"))]
    muon_ids = {id(p) for p in muon_p}
    emb = base_model.get_input_embeddings().weight
    head = base_model.get_output_embeddings()
    if leaked or id(emb) in muon_ids or (head is not None and id(head.weight) in muon_ids):
        raise SystemExit(f"FATAL: an embedding/lm_head reached the Muon group ({leaked or 'by identity'}); "
                         f"recorded at +1.257, refusing to train")
    every = [p for p in base_model.parameters() if p.requires_grad]
    if len(muon_p) + len(adam_p) != len(every) or \
            {id(p) for p in every} != muon_ids | {id(p) for p in adam_p}:
        raise SystemExit("FATAL: the Muon/AdamW split does not cover every trainable parameter exactly once")
    if is_main:
        print(report, flush=True)
        print(f"[muon] muon_lr={args.muon_lr} adam_lr={args.adam_lr} muon_wd={args.muon_wd} "
              f"momentum={args.muon_momentum} nesterov=True ns_steps={args.muon_ns_steps} "
              f"batched={bool(args.muon_batched)} zero_sharded={bool(zero_sharded)} | AdamW group: "
              f"betas=({args.adam_beta1}, {args.adam_beta2}) wd={args.weight_decay} | --lr {args.lr} "
              f"is unused under Muon", flush=True)
    hp = muon_hparams(args)
    if zero_sharded:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(muon_param_groups(muon_p, adam_p, **hp),
                                      optimizer_class=MuonAdamWGroups, **muon_defaults(**hp))
    else:
        opt = MuonAdamW(muon_p, adam_p, **hp)
    opt._canonical_groups = [list(muon_p), list(adam_p)]
    return opt


def muon_canonical_index(optimizer):
    """{id(param): index} in the unsharded MuonAdamW numbering, or None if not built by build_muon_optimizer.

    Numbering = every Muon param, then every AdamW param (param_groups order), which is also
    ZeroRedundancyOptimizer's own global index. Refuses a group layout that no longer matches the
    recorded one; deterministic, so every rank raises together, before any shard is written.
    """
    canon = getattr(optimizer, "_canonical_groups", None)
    if canon is None:
        return None
    if len(optimizer.param_groups) != len(canon) or any(
            [id(p) for p in g["params"]] != [id(p) for p in c]
            for g, c in zip(optimizer.param_groups, canon)):
        raise RuntimeError("optimizer param_groups no longer match _canonical_groups; refusing to "
                           "write a checkpoint with a scrambled index")
    return {id(p): i for i, p in enumerate(p for group_params in canon for p in group_params)}


def muon_canonical_param_groups(optimizer):
    """state_dict()['param_groups'] as an unsharded MuonAdamW would write it: one entry per group.

    Hyperparameters come from the optimizer's GLOBAL param_groups, the ones LambdaLR writes, so `lr` is
    the one the NEXT step will use. ZeroRedundancyOptimizer's local shard copy is refreshed only inside
    step(), i.e. one step stale; saving it would make every resume run one step at the previous LR
    (measured on the unchanged AdamW ZeRO path: saved 2.5e-4 where the scheduler expects 5e-4).
    """
    out, start = [], 0
    for g, group_params in zip(optimizer.param_groups, optimizer._canonical_groups):
        entry = {k: v for k, v in g.items() if k != "params"}
        entry["params"] = list(range(start, start + len(group_params)))
        start += len(group_params)
        out.append(entry)
    return out


def check_resume_optimizer(optimizer_state, optimizer_name, where):
    """Refuse to load AdamW state into Muon or the reverse (a different layout AND update rule).

    An AdamW checkpoint has one untagged group, a MuonAdamW one has two tagged use_muon. Mixing them
    fails deep inside load_state_dict or, with a lucky group count, loads moments under the wrong rule.
    Moving an a4.5 (AdamW) checkpoint onto Muon is a FRESH optimizer: --reset_schedule 1.
    """
    ck_is_muon = any(bool(g.get("use_muon")) for g in optimizer_state.get("param_groups", []))
    if ck_is_muon != (optimizer_name == "muon"):
        raise SystemExit(
            f"FATAL: {where} holds {'MuonAdamW' if ck_is_muon else 'AdamW'} optimizer state but this "
            f"run is --optimizer {optimizer_name}. Resume with the matching --optimizer, or pass "
            f"--reset_schedule 1 to start a fresh optimizer from these weights.")
