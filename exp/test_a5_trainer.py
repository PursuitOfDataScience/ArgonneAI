"""argonne5.0 port of the a5 recipe (Muon + pretrain KD) into pretrain.py and continue_pretrain.py, on CPU.

    python exp/test_a5_trainer.py unit    # ~1 min, fine on a login node
    python exp/test_a5_trainer.py e2e     # the REAL main() via exp/a5_cpu_harness.py, 1 and 2 ranks (gloo)
    python exp/test_a5_trainer.py real    # real Qwen3-0.6B-Base teacher + Qwen3 tokenizer on real text
    python exp/test_a5_trainer.py all

What each part proves, lettered as in the port's task list:
  (a) the DEFAULT path is the a4.5 path: argument defaults, the AdamW build_optimizer returns, and an
      end-to-end run whose checkpoint is BIT-IDENTICAL to the pre-port pretrain.py (git b4e11eb), both
      unsharded and ZeRO on 2 ranks;
  (b) KD loss AND gradient parity against a verbatim copy of exp/probe.py's KD lines;
  (c) Muon save -> resume on 1 rank is bit-identical to an uninterrupted run (with KD and doc_mask on);
  (d) ZeRO + Muon on 2 gloo ranks: the shard-merged checkpoint equals the state_dict an unsharded
      MuonAdamW writes, and resuming it sharded (2 ranks) or unsharded (1 rank) is bit-identical;
  (e) the Muon split at the real a4.5 shape, built on the meta device so nothing is allocated;
  (f) continue_pretrain.py, which imports the same muon.py / pretrain_kd.py: default path bit-identical
      to its pre-port version, and Muon + KD save -> resume bit-identical on 1 rank and ZeRO 2 ranks.

WHY 1-RANK AND 2-RANK RUNS CAN BE COMPARED BIT FOR BIT: the fixture corpus is PERIODIC with period
B*T, so every micro-batch on every rank is the same window. A 2-rank step at accum 1 then averages two
identical gradients ((g+g)/2 == g exactly) and a 1-rank step at accum 2 sums two halves (g/2+g/2 == g
exactly, scaling by a power of two is exact), so the trajectories agree to the last bit and any
difference is a real defect in the sharding or checkpoint code. OMP/MKL threads are pinned to 1 so
CPU matmul blocking cannot differ between a 1-process and a 2-process run.
"""
import importlib.util
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True          # never leave .pyc beside a file this test imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(REPO, "exp", "a5_cpu_harness.py")
BASE_REV = os.environ.get("A5_BASE_REV", "b4e11eb")              # the branch point: pre-port trainer
ORIG_MUON = os.environ.get("A5_ORIG_MUON", "/home/youzhi/ArgonneAI/exp/muon.py")   # probe's muon.py
QWEN = os.environ.get("A5_QWEN", "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base")
REAL_BIN = os.environ.get("A5_REAL_BIN", "/project/rcc/youzhi/data/argonne4_pretrain/val_edu.bin")
TOY_ARCH = {"HIDDEN_SIZE": 64, "NUM_LAYERS": 2, "NUM_HEADS": 4, "NUM_KV_HEADS": 2,
            "INTERMEDIATE_SIZE": 128}
# continue_pretrain.py derives its architecture from the checkpoint it resumes, assuming head_dim 256
# (its own invariant), so its seeds use a toy shape with 256-wide heads.
TOY_ARCH_CP = {"HIDDEN_SIZE": 256, "NUM_LAYERS": 2, "NUM_HEADS": 1, "NUM_KV_HEADS": 1,
               "INTERMEDIATE_SIZE": 256}
FAKE_ARGV = ["pretrain.py", "--tokenizer_path", "/dev/null", "--data_path", "/dev/null",
             "--checkpoint_dir", "/dev/null", "--batch_size", "1", "--block_size", "8",
             "--total_batch_size", "8"]

failures, notes = [], []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""), flush=True)
    if not cond:
        failures.append(f"{label} {detail}")
    return cond


def load_module(path, name, argv=None):
    """Import a trainer file under `name` with a fake argv (it parses args at import time)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    saved = sys.argv
    sys.argv = list(argv or FAKE_ARGV)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def base_rev_file(relpath, workdir):
    """Write <relpath> as of BASE_REV into workdir (the pre-port file), return its path."""
    out = os.path.join(workdir, os.path.basename(relpath))
    blob = subprocess.run(["git", "-C", REPO, "show", f"{BASE_REV}:{relpath}"], check=True,
                          capture_output=True).stdout
    with open(out, "wb") as f:
        f.write(blob)
    return out


# ------------------------------------------------------------------------------------------------
# Fixtures: a 250-token tokenizer, tiny random Qwen3 teachers, a periodic corpus
# ------------------------------------------------------------------------------------------------
V_TOY = 250          # student vocab = len(tokenizer)
EOS_TOY = 249
PERIOD_PATTERN = None


def build_fixtures(root):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    fx = {}
    vocab = {f"t{i}": i for i in range(V_TOY - 1)}
    vocab["<eos>"] = EOS_TOY
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", unk_token="t0")
    fx["tok"] = os.path.join(root, "tok")
    fast.save_pretrained(fx["tok"])

    cfg = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                      max_position_embeddings=128, tie_word_embeddings=True)
    torch.manual_seed(1)
    teacher = Qwen3ForCausalLM(cfg)
    fx["teacher"] = os.path.join(root, "teacher")
    teacher.save_pretrained(fx["teacher"])
    # A BROKEN teacher: final norm zeroed => all-zero logits => uniform => top-K mass K/V. The gate
    # must refuse it (K=4 of 250 => mass 0.016 < 0.05).
    with torch.no_grad():
        teacher.model.norm.weight.zero_()
    fx["teacher_uniform"] = os.path.join(root, "teacher_uniform")
    teacher.save_pretrained(fx["teacher_uniform"])

    # Periodic corpus: period 16 == B*T, two EOS separators so doc_mask has documents to separate.
    global PERIOD_PATTERN
    rng = np.random.default_rng(7)
    pat = rng.integers(0, V_TOY - 1, size=16).astype(np.uint32)
    pat[5] = EOS_TOY
    pat[11] = EOS_TOY
    PERIOD_PATTERN = pat
    for name in ("a.bin", "b.bin"):
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240801
        with open(os.path.join(root, name), "wb") as f:
            f.write(header.tobytes())
            f.write(np.tile(pat, 4096).tobytes())
    fx["sources"] = f"{os.path.join(root, 'a.bin')}:0.6,{os.path.join(root, 'b.bin')}:0.4"
    return fx


# ------------------------------------------------------------------------------------------------
# UNIT TESTS
# ------------------------------------------------------------------------------------------------
def probe_kd_reference(tl, lg, y, kd_topk, CH_T, kd_chunk, kd_alpha, accum=1):
    """exp/probe.py's KD lines (448-507 at b4e11eb), VERBATIM apart from being a function."""
    with torch.no_grad():
        _pv, _pi = [], []
        for c0 in range(0, tl.size(1), CH_T):
            tlc = tl[:, c0:c0 + CH_T].float()
            lse_c = torch.logsumexp(tlc, dim=-1, keepdim=True)
            v_c, i_c = torch.topk(tlc, kd_topk, dim=-1)
            _pv.append(torch.exp(v_c - lse_c)); _pi.append(i_c)
            del tlc, lse_c, v_c, i_c
        tkp = torch.cat(_pv, dim=1); tki = torch.cat(_pi, dim=1)
        del _pv, _pi
    V = lg.size(-1)
    flat, yf = lg.view(-1, V), y.reshape(-1)
    pf, idf = tkp.view(-1, kd_topk), tki.view(-1, kd_topk)
    N, CH = yf.numel(), int(kd_chunk)

    def _chunk(lgc, yc, pc, ic):
        ce_c = F.cross_entropy(lgc, yc, reduction="none")
        lse_c = ce_c + lgc.gather(-1, yc.unsqueeze(-1)).squeeze(-1).float()
        lgk_c = lgc.gather(-1, ic).float()
        soft_c = -(pc * lgk_c).sum(-1) + pc.sum(-1) * lse_c
        return ce_c.sum(), soft_c.sum()

    h_s = f_s = None
    for c0 in range(0, N, CH):
        sl = slice(c0, min(c0 + CH, N))
        hc, fc = ckpt(_chunk, flat[sl], yf[sl], pf[sl], idf[sl], use_reentrant=False)
        h_s = hc if h_s is None else h_s + hc
        f_s = fc if f_s is None else f_s + fc
    hard, soft = h_s / N, f_s / N
    loss = ((1.0 - kd_alpha) * hard + kd_alpha * soft) / accum
    return loss, hard, soft, tkp, tki


def unit_kd_parity(KD):
    print("\n=== (b) KD loss + gradient parity vs exp/probe.py's lines (pretrain_kd.py) ===")
    cases = [  # B, T, V_student, V_teacher, K, CH_T, CH, alpha, accum, autocast_bf16
        (2, 13, 97, 101, 17, 5, 7, 1.0, 1, False),
        (1, 16, 250, 256, 128, 3, 5, 0.5, 4, True),
        (3, 8, 64, 64, 64, 8, 24, 0.25, 2, False),
        (1, 37, 311, 400, 64, 256, 256, 1.0, 20, True),
    ]
    for (B, T, Vs, Vt, K, CH_T, CH, a, accum, amp) in cases:
        g = torch.Generator().manual_seed(B * 1000 + T)
        tl_full = torch.randn(B, T, Vt, generator=g) * 3
        tl = tl_full[..., :Vs]                           # the teacher truncation both paths receive
        y = torch.randint(0, Vs, (B, T), generator=g)
        base = torch.randn(B, T, Vs, generator=g) * 2
        lg_ref = base.clone().requires_grad_(True)
        lg_new = base.clone().requires_grad_(True)
        tag = f"B{B} T{T} V{Vs} K{K} chT{CH_T} ch{CH} a{a} accum{accum}{' bf16-autocast' if amp else ''}"
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            src_ref = lg_ref.bfloat16() if amp else lg_ref
            src_new = lg_new.bfloat16() if amp else lg_new
            loss_r, hard_r, soft_r, tkp_r, tki_r = probe_kd_reference(
                tl.bfloat16() if amp else tl, src_ref, y, K, CH_T, CH, a, accum)
            tkp_n, tki_n = KD.kd_teacher_topk(tl.bfloat16() if amp else tl, K, CH_T)
            hard_n, soft_n = KD.kd_hard_soft(src_new, y, tkp_n, tki_n, CH)
            loss_n = ((1.0 - a) * hard_n + a * soft_n) / accum
        loss_r.backward()
        loss_n.backward()
        check(f"[{tag}] teacher top-K probs+ids identical",
              torch.equal(tkp_r, tkp_n) and torch.equal(tki_r, tki_n))
        check(f"[{tag}] loss identical", torch.equal(loss_r, loss_n),
              f"{loss_r.item():.9f} vs {loss_n.item():.9f}")
        check(f"[{tag}] d loss / d student logits identical", torch.equal(lg_ref.grad, lg_new.grad),
              f"max|diff| {(lg_ref.grad - lg_new.grad).abs().max().item():.3e}")
        if not amp:
            # Independent check of the ALGEBRA against a dense formula (full-vocab softmax, top-K kept
            # at their true mass, tail dropped): proves the probe's lse trick computes that objective.
            with torch.no_grad():
                p_full = torch.softmax(tl.float(), -1)
                logq = torch.log_softmax(base.float(), -1)
                topi = torch.topk(tl.float(), K, dim=-1).indices
                soft_dense = -(p_full.gather(-1, topi) * logq.gather(-1, topi)).sum(-1).mean()
                hard_dense = F.cross_entropy(base.view(-1, Vs), y.view(-1))
            check(f"[{tag}] soft == dense -sum_k p_k log q_k (full-vocab p)",
                  torch.allclose(soft_n.detach(), soft_dense, rtol=1e-5, atol=1e-5),
                  f"{soft_n.item():.6f} vs {soft_dense.item():.6f}")
            check(f"[{tag}] hard == plain cross_entropy",
                  torch.allclose(hard_n.detach(), hard_dense, rtol=1e-6, atol=1e-6))
            check(f"[{tag}] top-K mass < 1 when K < V (tail omitted, not renormalised)",
                  (K == Vs) or bool((tkp_n.sum(-1) < 1).all()))

    # kd_micro_loss end to end: a toy teacher with a WIDER vocab (sliced inside), a toy student.
    class Toy(nn.Module):
        def __init__(self, V, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.emb = nn.Embedding(V, 16)
            self.out = nn.Linear(16, V, bias=False)

        def forward(self, x, use_cache=None, document_ids=None, labels=None):
            h = self.emb(x.clamp_max(self.emb.num_embeddings - 1))
            if document_ids is not None:
                h = h + 0.01 * document_ids.unsqueeze(-1).float()
            class O:
                pass
            o = O()
            o.logits = self.out(h)
            return o

    Vs, Vt = 50, 70
    teacher, student = Toy(Vt, 3), Toy(Vs, 4)
    student_ref = Toy(Vs, 4)
    x = torch.randint(0, Vs, (2, 11))
    y = torch.randint(0, Vs, (2, 11))
    kw = {"document_ids": torch.cumsum((x == 7).long(), 1)}
    obj, hard, gate = KD.kd_micro_loss(student, teacher, x, y, kw, Vs, 0.7, 9, 4, 3, want_gate=True)
    obj.backward()
    with torch.no_grad():
        tl = teacher(x).logits[..., :Vs]
    lg = student_ref(x, **kw).logits
    loss_r, hard_r, soft_r, tkp_r, tki_r = probe_kd_reference(tl, lg, y, 9, 3, 4, 0.7)
    loss_r.backward()
    check("kd_micro_loss == probe reference (teacher sliced to the student vocab, doc ids to student)",
          torch.equal(obj, loss_r) and torch.equal(hard, hard_r))
    check("kd_micro_loss gradients reach the student exactly as the reference's",
          all(torch.equal(p.grad, q.grad) for p, q in zip(student.parameters(), student_ref.parameters())))
    check("teacher receives no gradient", all(p.grad is None for p in teacher.parameters()))
    gold = (tki_r == y.unsqueeze(-1)).any(-1).float().mean()
    check("gate stats = the probe's (top-K mass, gold-in-top-K)",
          torch.equal(gate["mass"], tkp_r.sum(-1).mean()) and torch.equal(gate["gold"], gold))

    # param_connected_zero: the NaN guard's zero gives EXACT zero, finite gradients even when a
    # parameter itself holds a NaN (nan_to_num), and reaches every parameter.
    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    with torch.no_grad():
        m[0].weight[0, 0] = float("nan")
    z = KD.param_connected_zero(m, "cpu")
    z.backward()
    check("NaN guard: param_connected_zero -> every grad exactly 0 and finite",
          z.item() == 0.0 and all(p.grad is not None and torch.equal(p.grad, torch.zeros_like(p))
                                   for p in m.parameters()))


def unit_muon_equivalence(pt, workdir):
    print("\n=== muon.py (production copy) vs exp/muon.py (the probe's) ===")
    import muon as prod
    if not os.path.exists(ORIG_MUON):
        notes.append(f"exp/muon.py original not found at {ORIG_MUON}: equivalence vs the probe copy skipped")
        print(f"  SKIP  original {ORIG_MUON} missing")
        return
    local = os.path.join(workdir, "muon_orig_copy.py")
    shutil.copyfile(ORIG_MUON, local)                  # import a COPY: never touch the main tree
    orig = load_module(local, "muon_orig_copy")

    def net(seed):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(24, 48), nn.Linear(48, 24), nn.Linear(24, 48), nn.LayerNorm(48),
                             nn.Embedding(10, 48))

    for batched in (False, True):
        a, b = net(0), net(0)
        ma, aa, *_ = orig.split_params(a)
        mb, ab, *_ = prod.split_params(b)
        hp = dict(muon_lr=0.02, adam_lr=5e-4, batched=batched, batch_size=2)
        oa, ob = orig.MuonAdamW(ma, aa, **hp), prod.MuonAdamW(mb, ab, **hp)
        strip = lambda gs: [{k: v for k, v in g.items() if k != "params"} for g in gs]
        check(f"[batched={batched}] param_groups hyperparameters identical to exp/muon.py",
              strip(oa.param_groups) == strip(ob.param_groups))
        check(f"[batched={batched}] defaults identical", oa.defaults == ob.defaults)
        x = torch.randn(5, 24)
        for _ in range(4):
            for m, o in ((a, oa), (b, ob)):
                o.zero_grad()
                h = m[2](m[1](m[0](x)))
                (m[3](h).pow(2).mean() + m[4].weight.sum() * 1e-3).backward()
                o.step()
        check(f"[batched={batched}] 4 steps bit-identical to exp/muon.py",
              all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))

        # ZeRO entry point: same groups/defaults, same trajectory
        c = net(0)
        mc, ac, *_ = prod.split_params(c)
        oc = prod.MuonAdamWGroups(prod.muon_param_groups(mc, ac, **hp), **prod.muon_defaults(**hp))
        check(f"[batched={batched}] MuonAdamWGroups(muon_param_groups, muon_defaults) == MuonAdamW",
              strip(oc.param_groups) == strip(ob.param_groups) and oc.defaults == ob.defaults)

    # The resume hazard the production copy fixes (batched path).
    def run(mod, resume):
        m = net(1)
        mp, ap, *_ = mod.split_params(m)
        o = mod.MuonAdamW(mp, ap, muon_lr=0.02, adam_lr=5e-4, batched=True, batch_size=2)
        x = torch.randn(5, 24, generator=torch.Generator().manual_seed(3))

        def step(mm, oo):
            oo.zero_grad()
            h = mm[2](mm[1](mm[0](x)))
            mm[3](h).pow(2).mean().backward()
            oo.step()
        for _ in range(2):
            step(m, o)
        if resume:
            buf = io.BytesIO()
            torch.save({"m": m.state_dict(), "o": o.state_dict()}, buf)
            buf.seek(0)
            ck = torch.load(buf, weights_only=False)
            m = net(99)
            m.load_state_dict(ck["m"])
            mp, ap, *_ = mod.split_params(m)
            o = mod.MuonAdamW(mp, ap, muon_lr=0.02, adam_lr=5e-4, batched=True, batch_size=2)
            o.load_state_dict(ck["o"])
        for _ in range(2):
            step(m, o)
        return m, o

    straight, _ = run(prod, False)
    resumed, o_res = run(prod, True)
    check("production muon.py: batched save -> load -> 2 steps == uninterrupted, bit for bit",
          all(torch.equal(p, q) for p, q in zip(straight.parameters(), resumed.parameters())))
    check("production muon.py: state_dict param_groups hold no cache and no tensors",
          all(k != "_shape_groups" and not isinstance(v, torch.Tensor)
              for g in o_res.state_dict()["param_groups"] for k, v in g.items()))
    o_straight, _ = run(orig, False)
    o_resumed, _ = run(orig, True)
    check("exp/muon.py reproduces the hazard (documented reason for the change): resumed != straight",
          not all(torch.equal(p, q) for p, q in zip(o_straight.parameters(), o_resumed.parameters())))


def unit_defaults_and_optimizer(pt, old):
    print("\n=== (a) default arguments and optimizer == the pre-port pretrain.py ===")
    va, vb = vars(old.args), vars(pt.args)
    same = {k: (va[k], vb.get(k)) for k in va if va[k] != vb.get(k)}
    check("every pre-port flag keeps its default", not same, str(same))
    expect_new = {"optimizer": "adamw", "muon_lr": None, "adam_lr": None, "muon_wd": 0.01,
                  "muon_momentum": 0.95, "muon_ns_steps": 5, "muon_batched": 0, "kd_teacher": "",
                  "kd_alpha": 0.0, "kd_topk": 2048, "kd_chunk": 256, "kd_teacher_chunk": 256,
                  "kd_chunked_head": 0, "arch": "a45", "max_steps": 0}
    added = {k: vb[k] for k in vb if k not in va}
    check("new flags are exactly the expected set with the expected defaults", added == expect_new,
          str(added))

    import model as M
    M._attention_path_logged = True

    def tiny():
        torch.manual_seed(0)
        cfg = M.ArgonneConfig(vocab_size=120, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=1, intermediate_size=64, max_position_embeddings=32,
                              tie_word_embeddings=True)
        return M.ArgonneModel(cfg)
    ma, mb = tiny(), tiny()
    oa = old.build_optimizer(ma.parameters(), old.args, 1)
    ob = pt.build_optimizer(mb, pt.args, 1)
    check("default build_optimizer returns torch.optim.AdamW", type(ob) is torch.optim.AdamW, type(ob).__name__)
    strip = lambda gs: [{k: v for k, v in g.items() if k != "params"} for g in gs]
    check("same AdamW hyperparameters (lr, betas, weight_decay, fused=True, ...)",
          strip(oa.param_groups) == strip(ob.param_groups), str(strip(ob.param_groups)))
    check("same parameter order (model.parameters())",
          [tuple(p.shape) for p in oa.param_groups[0]["params"]] ==
          [tuple(p.shape) for p in ob.param_groups[0]["params"]])
    check("fused AdamW path taken (fused=True, supported on this CPU build)",
          ob.param_groups[0].get("fused") is True)
    x = torch.randint(0, 120, (2, 16))
    for _ in range(3):
        for m, o in ((ma, oa), (mb, ob)):
            o.zero_grad()
            m.train()
            m(x, labels=x).loss.backward()
            o.step()
    check("3 AdamW steps bit-identical, old vs new build_optimizer",
          all(torch.equal(p, q) for p, q in zip(ma.parameters(), mb.parameters())))


def unit_validation(pt_path):
    print("\n=== flag validation refuses configurations the record rules out ===")
    base = FAKE_ARGV
    cases = [
        (["--optimizer", "muon"], "needs BOTH --muon_lr and --adam_lr"),
        (["--optimizer", "muon", "--muon_lr", "0.02"], "needs BOTH"),
        (["--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "5e-4", "--muon_wd", "0"], "--muon_wd must be > 0"),
        (["--muon_lr", "0.02"], "only apply to --optimizer muon"),
        (["--kd_alpha", "1.0"], "needs --kd_teacher"),
        (["--kd_alpha", "1.5", "--kd_teacher", "x"], "must be in [0, 1]"),
    ]
    for i, (extra, msg) in enumerate(cases):
        r = subprocess.run([sys.executable, "-c",
                            "import sys, importlib.util; sys.dont_write_bytecode=True; "
                            f"sys.argv={base + extra!r}; "
                            f"s=importlib.util.spec_from_file_location('p', {pt_path!r}); "
                            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"],
                           capture_output=True, text=True, timeout=300)
        check(f"refused: {' '.join(extra)}", r.returncode == 2 and msg in r.stderr,
              f"rc={r.returncode} {r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else ''}")
    ok = subprocess.run([sys.executable, "-c",
                         "import sys, importlib.util; sys.dont_write_bytecode=True; "
                         f"sys.argv={base + ['--optimizer', 'muon', '--muon_lr', '0.02', '--adam_lr', '5e-4', '--kd_alpha', '1', '--kd_teacher', 'x']!r}; "
                         f"s=importlib.util.spec_from_file_location('p', {pt_path!r}); "
                         "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('PARSED')"],
                        capture_output=True, text=True, timeout=300)
    good = ok.returncode == 0 and "PARSED" in ok.stdout
    check("accepted: a valid a5 command line", good, "" if good else ok.stderr.strip()[-200:])


def unit_schedule_two_groups():
    print("\n=== LambdaLR scales BOTH Muon groups by one shape ===")
    import muon as prod
    net = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    mp, ap, *_ = prod.split_params(net)
    opt = prod.MuonAdamW(mp, ap, muon_lr=0.02, adam_lr=5e-4)
    warm, est, cd, floor = 4, 20, 8, 0.1

    def lam(s):                                       # pretrain.py's WSD shape
        if s < warm:
            return s / max(1, warm)
        start = max(warm, est - cd)
        if s < start:
            return 1.0
        return 1.0 - min(1.0, (s - start) / max(1, cd)) * (1.0 - floor)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    ok = True
    for s in range(est + 2):
        lm, la = opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]
        ok &= math.isclose(lm, 0.02 * lam(s), rel_tol=0, abs_tol=1e-15) and \
              math.isclose(la, 5e-4 * lam(s), rel_tol=0, abs_tol=1e-15)
        opt.step()
        sch.step()
    check("every step: muon lr = 0.02*lambda(s) and adam lr = 5e-4*lambda(s)", ok)
    check("scheduler base_lrs are the two group bases", sch.base_lrs == [0.02, 5e-4], str(sch.base_lrs))


def unit_production_split(pt):
    print("\n=== (e) Muon split at the real a4.5 shape (meta device, nothing allocated) ===")
    import model as M
    M._attention_path_logged = True
    cfg = M.ArgonneConfig(
        vocab_size=151680, hidden_size=pt.HIDDEN_SIZE, num_hidden_layers=pt.NUM_LAYERS,
        num_attention_heads=pt.NUM_HEADS, num_key_value_heads=pt.NUM_KV_HEADS,
        intermediate_size=pt.INTERMEDIATE_SIZE, max_position_embeddings=1024, rope_theta=pt.ROPE_THETA,
        qk_norm=pt.ENABLE_QK_NORM, v_norm=pt.ENABLE_V_NORM, sandwich_norm=pt.ENABLE_SANDWICH_NORM,
        z_loss_weight=pt.Z_LOSS_WEIGHT, interleaved_local_attention=pt.ENABLE_INTERLEAVED_LOCAL_ATTENTION,
        local_attention_window=None, attn_pattern=pt.ATTN_PATTERN, mlp_type=pt.MLP_TYPE,
        logit_softcap=pt.LOGIT_SOFTCAP, tie_word_embeddings=True)
    with torch.device("meta"):
        m = M.ArgonneModel(cfg)
    total = sum(p.numel() for p in m.parameters())
    print(f"  shape: hidden {pt.HIDDEN_SIZE}, layers {pt.NUM_LAYERS}, heads {pt.NUM_HEADS}/{pt.NUM_KV_HEADS}, "
          f"intermediate {pt.INTERMEDIATE_SIZE} ({pt.MLP_TYPE}), vocab 151,680 tied: {total:,} params")
    check("a4.5 production parameter count reproduced (2,063,667,712)", total == 2_063_667_712, f"{total:,}")
    import argparse
    a = argparse.Namespace(**{**vars(pt.args), "optimizer": "muon", "muon_lr": 0.02, "adam_lr": 5e-4})
    import muon as prod
    opt = prod.build_muon_optimizer(m, a, zero_sharded=False, is_main=True)   # prints the split report
    g_m, g_a = opt.param_groups
    n_m = sum(p.numel() for p in g_m["params"]); n_a = sum(p.numel() for p in g_a["params"])
    names = {id(p): n for n, p in m.named_parameters()}
    muon_names = [names[id(p)] for p in g_m["params"]]
    check("168 Muon tensors (7 matrices x 24 layers)", len(g_m["params"]) == 168, str(len(g_m["params"])))
    check("every Muon tensor is 2-D and inside a block",
          all(p.ndim == 2 for p in g_m["params"]) and all(n.startswith("blocks.") for n in muon_names))
    check("no embedding / lm_head on Muon (by name and by identity)",
          not any(("embed" in n or "lm_head" in n) for n in muon_names)
          and all(p is not m.embed_tokens.weight for p in g_m["params"]))
    check("tied embedding (== lm_head) is in the AdamW group",
          any(p is m.embed_tokens.weight for p in g_a["params"]) and m.lm_head.weight is m.embed_tokens.weight)
    check("groups cover every parameter exactly once", n_m + n_a == total)
    gib = 2 ** 30
    print(f"  Muon group:  {len(g_m['params'])} tensors, {n_m:,} params ({100*n_m/total:.1f}%)")
    print(f"  AdamW group: {len(g_a['params'])} tensors, {n_a:,} params ({100*n_a/total:.1f}%): "
          f"embedding {m.embed_tokens.weight.numel():,} + {len(g_a['params'])-1} norm vectors")
    print(f"  fp32 optimizer state: Muon momentum {4*n_m/gib:.2f} GiB + AdamW m,v {8*n_a/gib:.2f} GiB = "
          f"{(4*n_m+8*n_a)/gib:.2f} GiB, vs AdamW-everything {8*total/gib:.2f} GiB "
          f"({(8*total-4*n_m-8*n_a)/gib:.2f} GiB freed)")


def unit_shared_wiring(pt, cp_new, cp_old, KD):
    print("\n=== both trainers use ONE copy of the a5 code; continue_pretrain.py default optimizer unchanged ===")
    import muon as prod
    check("pretrain.py and continue_pretrain.py share pretrain_kd.KDStep / setup_kd_teacher",
          pt.KDStep is KD.KDStep and cp_new.KDStep is KD.KDStep and cp_new.setup_kd_teacher is KD.setup_kd_teacher)
    check("... and muon.build_muon_optimizer / canonical checkpoint helpers",
          pt.build_muon_optimizer is prod.build_muon_optimizer and cp_new.build_muon_optimizer is prod.build_muon_optimizer
          and cp_new.muon_canonical_param_groups is prod.muon_canonical_param_groups)
    import argparse
    import model as M
    M._attention_path_logged = True

    def tiny():
        torch.manual_seed(0)
        return M.ArgonneModel(M.ArgonneConfig(vocab_size=120, hidden_size=32, num_hidden_layers=2,
                                              num_attention_heads=2, num_key_value_heads=1,
                                              intermediate_size=64, max_position_embeddings=32,
                                              tie_word_embeddings=True))
    ns = argparse.Namespace(lr=3e-4, adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.1, zero_optimizer=0,
                            optimizer="adamw")
    ma, mb = tiny(), tiny()
    oa = cp_old.build_optimizer(ma.parameters(), ns, 1)
    ob = cp_new.build_optimizer(mb, ns, 1)
    strip = lambda gs: [{k: v for k, v in g.items() if k != "params"} for g in gs]
    check("continue_pretrain.py default build_optimizer == pre-port (fused AdamW, same groups)",
          type(ob) is torch.optim.AdamW and strip(oa.param_groups) == strip(ob.param_groups))


def unit_kd_chunked_head(KD):
    """--kd_chunked_head 1 (model.py _chunked_kd_terms) against the default path on a real ArgonneModel: same
    objective, same hard CE, same gradients on every student parameter (tied head included), for a chunk
    size that does not divide the rows and one that does; and the default path is untouched."""
    print("\n=== KD chunked head: model-side chunked logits == the full-logit path (CPU, fp32) ===")
    import model as M
    M._attention_path_logged = True

    def tiny(seed):
        torch.manual_seed(seed)
        return M.ArgonneModel(M.ArgonneConfig(vocab_size=120, hidden_size=32, num_hidden_layers=2,
                                              num_attention_heads=2, num_key_value_heads=1,
                                              intermediate_size=64, max_position_embeddings=32,
                                              tie_word_embeddings=True))

    class Teacher(nn.Module):                      # wider vocab, sliced to the student's inside kd_micro_loss
        def __init__(self, V):
            super().__init__()
            torch.manual_seed(7)
            self.emb = nn.Embedding(V, 16)
            self.out = nn.Linear(16, V, bias=False)

        def forward(self, x, use_cache=None, **kw):
            class O:
                pass
            o = O()
            o.logits = self.out(self.emb(x))
            return o
    V = 120
    teacher = Teacher(140)
    x = torch.randint(0, V, (2, 13))
    y = torch.randint(0, V, (2, 13))
    for chunk in (5, 13):
        ma, mb = tiny(0), tiny(0)
        ma.train(); mb.train()
        oa, ha, _ = KD.kd_micro_loss(ma, teacher, x, y, {}, V, 0.7, 9, chunk, 4)
        ob, hb, _ = KD.kd_micro_loss(mb, teacher, x, y, {}, V, 0.7, 9, chunk, 4, chunked_head=True)
        oa.backward(); ob.backward()
        gmax = max((p.grad - q.grad).abs().max().item() for p, q in zip(ma.parameters(), mb.parameters()))
        check(f"[chunk {chunk}] chunked head objective == full-logit objective",
              torch.allclose(oa, ob, rtol=1e-6, atol=1e-6), f"{oa.item():.7f} vs {ob.item():.7f}")
        check(f"[chunk {chunk}] chunked head hard CE == full-logit hard CE",
              torch.allclose(ha, hb, rtol=1e-6, atol=1e-6))
        check(f"[chunk {chunk}] every student gradient matches (tied head + body)",
              all(p.grad is not None and q.grad is not None for p, q in zip(ma.parameters(), mb.parameters()))
              and gmax < 1e-6, f"max |diff| {gmax:.2e}")
    mc = tiny(0); mc.train()
    out = mc(x, labels=y, kd_targets=None)
    check("without kd_targets the forward is the default path (a loss, no KD branch)",
          out.loss is not None and out.loss.dim() == 0)


def unit_kdstep(KD):
    print("\n=== KDStep: gate refusal, gate pass, NaN guard (1 process) ===")
    import argparse

    class Tiny(nn.Module):
        def __init__(self, V, seed, zero=False):
            super().__init__()
            torch.manual_seed(seed)
            self.emb = nn.Embedding(V, 16)
            self.out = nn.Linear(16, V, bias=False)
            if zero:
                with torch.no_grad():
                    self.out.weight.zero_()          # uniform teacher: top-K mass = K/V

        def forward(self, x, use_cache=None, **kw):
            class O:
                pass
            o = O()
            o.logits = self.out(self.emb(x))
            return o
    V = 200
    x = torch.randint(0, V, (1, 12))
    y = torch.randint(0, V, (1, 12))
    student = Tiny(V, 1)
    a = argparse.Namespace(kd_alpha=1.0, kd_topk=4, kd_chunk=5, kd_teacher_chunk=3)
    step = KD.KDStep(Tiny(V, 2, zero=True), student, V, a, "cpu", 1, True)
    try:
        step(student, x, y, {}, 0)
        refused = False
    except SystemExit as e:
        refused = "KD GATE FAILED" in str(e)
    check("uniform teacher (mass 4/200 = 0.02 < 0.05) -> SystemExit('KD GATE FAILED')", refused)
    a2 = argparse.Namespace(kd_alpha=1.0, kd_topk=150, kd_chunk=5, kd_teacher_chunk=3)
    ok = KD.KDStep(Tiny(V, 3), student, V, a2, "cpu", 1, True)
    obj, hard = ok(student, x, y, {}, 0)
    check("sane teacher passes the gate once, then the gate is off", not ok.gate_pending and torch.isfinite(obj))
    nan_teacher = Tiny(V, 4)
    with torch.no_grad():
        nan_teacher.out.weight.fill_(float("nan"))
    nan_step = KD.KDStep(nan_teacher, student, V, a2, "cpu", 1, True)
    try:
        nan_step(student, x, y, {}, 0)
        nan_refused = False
    except SystemExit:
        nan_refused = True
    check("NaN teacher FAILS the gate (the probe's `< 0.05` test would have let NaN through)", nan_refused)
    nan_step2 = KD.KDStep(nan_teacher, student, V, a2, "cpu", 1, True)
    nan_step2.gate_pending = False                  # as if the NaN appeared after the gate
    student.zero_grad()
    z, h = nan_step2(student, x, y, {}, 5)
    z.backward()
    check("NaN objective after the gate -> param-connected zero: loss 0, every grad exactly 0",
          z.item() == 0.0 and nan_step2.nan_count == 1 and
          all(torch.equal(p.grad, torch.zeros_like(p)) for p in student.parameters()))


def unit_validation_cp(cp_path):
    print("\n=== continue_pretrain.py refuses the same configurations (shared validation) ===")
    req = ["--tokenizer_path", "/dev/null", "--data_path", "/dev/null", "--checkpoint_dir", "/dev/null",
           "--lr", "1e-4", "--batch_size", "1", "--total_batch_size", "8", "--block_size", "8"]
    for extra, msg in ((["--optimizer", "muon"], "needs BOTH --muon_lr and --adam_lr"),
                       (["--kd_alpha", "1.0"], "needs --kd_teacher"),
                       (["--adam_lr", "1e-3"], "only apply to --optimizer muon")):
        r = subprocess.run([sys.executable, cp_path] + req + extra, capture_output=True, text=True,
                           timeout=300, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        check(f"continue_pretrain.py refused: {' '.join(extra)}", r.returncode == 2 and msg in r.stderr,
              f"rc={r.returncode}")


# ------------------------------------------------------------------------------------------------
# END-TO-END through the real main()
# ------------------------------------------------------------------------------------------------
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
           TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           A5_TOY_ARCH=json.dumps(TOY_ARCH))


def run_trainer(trainer, args, ranks=1, timeout=900, log=None, env_extra=None):
    env = dict(ENV, **(env_extra or {}))
    if ranks == 1:
        cmd = [sys.executable, HARNESS, trainer] + args
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={ranks}", HARNESS, trainer] + args
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += "\n[TIMEOUT]"
        rc = "timeout"
    if log:
        with open(log, "w") as f:
            f.write(" ".join(cmd) + "\n\n" + out)
    return rc, out, time.time() - t0


def step_losses(out):
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"^Step (\d+) \| Loss: ([0-9.naninf]+)", out, re.M)}


def lr_pairs(out):
    return [(float(a), float(b)) for a, b in re.findall(r"LR: ([0-9.e+-]+) \(adam ([0-9.e+-]+)\)", out)]


def load_ckpt(d, step):
    return torch.load(os.path.join(d, f"checkpoint_step_{step}.pt"), map_location="cpu", weights_only=False)


def same_tree(a, b):
    """Exact structural equality with torch.equal on tensors."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype \
            and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def diff_ckpt(ca, cb, parts=("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")):
    bad = []
    for k in parts:
        if not same_tree(ca[k], cb[k]):
            if k == "model_state_dict":
                worst = max((ca[k][n].float() - cb[k][n].float()).abs().max().item() for n in ca[k])
                bad.append(f"{k} (max|diff| {worst:.3e})")
            else:
                bad.append(k)
    for k in ("global_step", "tokens_processed"):
        if ca[k] != cb[k]:
            bad.append(f"{k} {ca[k]} vs {cb[k]}")
    return bad


def e2e(fx, work, old_trainer, new_trainer):
    common = ["--tokenizer_path", fx["tok"], "--train_sources", fx["sources"], "--train_tokens", "256",
              "--batch_size", "1", "--block_size", "16", "--total_batch_size", "32", "--lr", "1e-3",
              "--warmup_steps", "4", "--schedule", "wsd", "--cooldown_frac", "0.5", "--min_lr_ratio", "0.1",
              "--grad_clip", "0.4", "--precision", "bf16", "--torch_compile", "0",
              "--gradient_checkpointing", "1", "--checkpoint_interval", "999999", "--log_interval", "1",
              "--seed", "444", "--doc_shuffle_seed", "1337"]
    a5 = ["--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "5e-4", "--doc_mask", "1",
          "--kd_teacher", fx["teacher"], "--kd_alpha", "1.0", "--kd_topk", "128", "--kd_chunk", "5",
          "--kd_teacher_chunk", "3"]
    logs = os.path.join(work, "logs")
    os.makedirs(logs, exist_ok=True)

    def D(name):
        return os.path.join(work, name)

    def go(name, trainer, extra, ranks=1, resume_from_dir=None, expect_fail=False, env_extra=None):
        d = D(name)
        if resume_from_dir:
            shutil.copytree(resume_from_dir, d, symlinks=True)   # keep checkpoint_last.pt a symlink
        rc, out, dt = run_trainer(trainer, common + ["--checkpoint_dir", d] + extra, ranks,
                                  log=os.path.join(logs, name + ".log"), env_extra=env_extra)
        print(f"    ran {name:<16} ranks={ranks} rc={rc} {dt:5.1f}s", flush=True)
        if rc != 0 and not expect_fail:
            print("\n".join(out.splitlines()[-25:]))
        return rc, out

    # ---- (a) default path, pre-port vs post-port, to the epoch end (8 steps + final export) -------
    print("\n=== (a) end to end: default flags, pre-port pretrain.py vs this one (bit-identical) ===")
    for ranks, extra in ((1, []), (2, ["--zero_optimizer", "1"])):
        tag = "ZeRO 2 ranks" if ranks == 2 else "1 rank"
        r1, _ = go(f"def_old_{ranks}", old_trainer, extra, ranks)
        r2, out2 = go(f"def_new_{ranks}", new_trainer, extra, ranks)
        if check(f"[{tag}] both runs complete", r1 == 0 and r2 == 0):
            bad = diff_ckpt(load_ckpt(D(f"def_old_{ranks}"), 8), load_ckpt(D(f"def_new_{ranks}"), 8))
            check(f"[{tag}] step-8 checkpoint (weights, AdamW state, scheduler) bit-identical", not bad, str(bad))
            check(f"[{tag}] HF export written by both",
                  os.path.exists(os.path.join(D(f"def_new_{ranks}"), "final_model_complete", "config.json")))
            check(f"[{tag}] default run prints no Muon/KD lines", "[muon]" not in out2 and "[kd]" not in out2)

    # ---- (c) Muon + KD + doc_mask, 1 rank: uninterrupted vs save@2 -> resume -> 6 ----------------
    print("\n=== (c) Muon + KD + doc_mask, 1 rank: save -> resume == uninterrupted ===")
    for batched in ("0", "1"):
        ex = a5 + ["--muon_batched", batched]
        ru, outu = go(f"c_u_b{batched}", new_trainer, ex + ["--max_steps", "6"])
        rs, _ = go(f"c_s_b{batched}", new_trainer, ex + ["--max_steps", "2"])
        rr, outr = go(f"c_r_b{batched}", new_trainer, ex + ["--max_steps", "6"],
                      resume_from_dir=D(f"c_s_b{batched}"))
        if check(f"[batched={batched}] all three runs complete", ru == rs == rr == 0):
            bad = diff_ckpt(load_ckpt(D(f"c_u_b{batched}"), 6), load_ckpt(D(f"c_r_b{batched}"), 6))
            check(f"[batched={batched}] step-6 weights + Muon/AdamW state + scheduler bit-identical", not bad, str(bad))
            lu, lr_ = step_losses(outu), step_losses(outr)
            check(f"[batched={batched}] logged losses of steps 3-6 identical after the resume",
                  all(lu.get(s) == lr_.get(s) for s in (3, 4, 5, 6)) and len(lr_) == 4,
                  f"{[lu.get(s) for s in (3,4,5,6)]} vs {[lr_.get(s) for s in (3,4,5,6)]}")
            check(f"[batched={batched}] resumed run logs hard_CE and both LRs",
                  "hard_CE:" in outr and "(adam " in outr)
            # The log prints LRs at 3 significant digits (:.2e), so the printed ratio is only good to
            # ~1%; the checkpoint's own group lrs give the exact ratio.
            pairs = lr_pairs(outu)
            check(f"[batched={batched}] logged muon/adam LR ratio = 40 at print precision, warmup through cooldown",
                  len(pairs) == 6 and all(abs(m / a - 40.0) < 0.4 for m, a in pairs if a > 0), str(pairs))
            g6 = load_ckpt(D(f"c_u_b{batched}"), 6)["optimizer_state_dict"]["param_groups"]
            check(f"[batched={batched}] checkpoint group lrs: muon/adam = 40 exactly, both at the same lambda",
                  math.isclose(g6[0]["lr"] / g6[1]["lr"], 40.0, rel_tol=1e-9)
                  and math.isclose(g6[0]["lr"] / g6[0]["initial_lr"], g6[1]["lr"] / g6[1]["initial_lr"], rel_tol=1e-12),
                  f"{g6[0]['lr']} / {g6[1]['lr']}")
            check(f"[batched={batched}] KD gate ran and passed in both slices",
                  outu.count("ok on every rank") == 1 and outr.count("ok on every rank") == 1)
            ck = load_ckpt(D(f"c_r_b{batched}"), 6)
            groups = ck["optimizer_state_dict"]["param_groups"]
            check(f"[batched={batched}] checkpoint is the two-group MuonAdamW format with no tensors in groups",
                  [g["use_muon"] for g in groups] == [True, False]
                  and not any(isinstance(v, torch.Tensor) or k == "_shape_groups" for g in groups for k, v in g.items()))
        if batched == "0":
            outu_c = outu

    # ---- (d) ZeRO + Muon + KD + doc_mask on 2 gloo ranks --------------------------------------
    print("\n=== (d) ZeRO + Muon + KD + doc_mask, 2 gloo ranks ===")
    z = ["--zero_optimizer", "1"]
    r_u2, out_u2 = go("d_u2", new_trainer, a5 + z + ["--max_steps", "6"], ranks=2)
    r_u1, _ = go("d_u1", new_trainer, a5 + ["--max_steps", "6"], ranks=1)
    r_s2, out_s2 = go("d_s2", new_trainer, a5 + z + ["--max_steps", "2"], ranks=2)
    r_s1, _ = go("d_s1", new_trainer, a5 + ["--max_steps", "2"], ranks=1)
    r_r22, out_r22 = go("d_r22", new_trainer, a5 + z + ["--max_steps", "6"], ranks=2, resume_from_dir=D("d_s2"))
    r_r21, _ = go("d_r21", new_trainer, a5 + ["--max_steps", "6"], ranks=1, resume_from_dir=D("d_s2"))
    r_r12, _ = go("d_r12", new_trainer, a5 + z + ["--max_steps", "6"], ranks=2, resume_from_dir=D("d_s1"))
    ok_runs = check("all seven runs complete",
                    all(r == 0 for r in (r_u2, r_u1, r_s2, r_s1, r_r22, r_r21, r_r12)),
                    str([r_u2, r_u1, r_s2, r_s1, r_r22, r_r21, r_r12]))
    check("ZeRO run really sharded MuonAdamW over 2 ranks (accum 1: sync branch)",
          "Using 2 GPUs with DistributedDataParallel" in out_u2 and "zero_sharded=True" in out_u2)
    if ok_runs:
        c_s2, c_s1 = load_ckpt(D("d_s2"), 2), load_ckpt(D("d_s1"), 2)
        osd2, osd1 = c_s2["optimizer_state_dict"], c_s1["optimizer_state_dict"]
        check("shard-merged step-2 optimizer state == unsharded MuonAdamW.state_dict() (structure AND values)",
              same_tree(osd2, osd1),
              f"groups {len(osd2['param_groups'])} vs {len(osd1['param_groups'])}, "
              f"state {len(osd2['state'])} vs {len(osd1['state'])}")
        check("  ... its param_groups: 2 entries, Muon first, contiguous canonical indices",
              [g["use_muon"] for g in osd2["param_groups"]] == [True, False]
              and osd2["param_groups"][0]["params"] + osd2["param_groups"][1]["params"] == list(range(len(osd2["state"]))))
        check("  ... Muon entries hold only momentum_buffer, AdamW entries exp_avg/exp_avg_sq/int step",
              all(set(osd2["state"][i]) == {"momentum_buffer"} for i in osd2["param_groups"][0]["params"])
              and all(set(osd2["state"][i]) == {"exp_avg", "exp_avg_sq", "step"}
                      and isinstance(osd2["state"][i]["step"], int) for i in osd2["param_groups"][1]["params"]))
        check("  ... saved lr is the one the NEXT step uses (not the local shard's stale copy)",
              osd2["param_groups"][0]["lr"] == osd1["param_groups"][0]["lr"]
              and c_s2["scheduler_state_dict"]["_last_lr"][0] == osd2["param_groups"][0]["lr"])
        c_u2 = load_ckpt(D("d_u2"), 6)
        for name, label in (("d_u1", "1-rank uninterrupted (unsharded)"),
                            ("d_r22", "2-rank ZeRO resume of the ZeRO checkpoint"),
                            ("d_r21", "1-rank UNSHARDED resume of the ZeRO checkpoint"),
                            ("d_r12", "2-rank ZeRO resume of an UNSHARDED checkpoint")):
            bad = diff_ckpt(c_u2, load_ckpt(D(name), 6))
            check(f"step 6: {label} == 2-rank ZeRO uninterrupted, bit for bit", not bad, str(bad))
        check("ZeRO continuation logs the same losses as the uninterrupted ZeRO run",
              all(step_losses(out_u2).get(s) == step_losses(out_r22).get(s) for s in (3, 4, 5, 6)))
        # torch's ZeroRedundancyOptimizer.load_state_dict leaves a second copy of each rank's shard in the
        # wrapper's own `state`; the trainer drops it (drop_zero_duplicate_state). The bit-for-bit checks
        # above are what prove dropping it changes nothing; this proves the copy existed and was released.
        check("ZeRO resume releases the duplicate optimizer state a ZeRO load leaves behind",
              bool(re.search(r"\[zero\] released the duplicate optimizer state .*: [0-9.]+ GiB", out_r22)))
        check("ZeRO 2-rank trajectory == 1-rank trajectory (periodic corpus), logged losses",
              all(step_losses(out_u2).get(s) == step_losses(outu_c).get(s) for s in range(1, 7)))

    # ---- the DDP no_sync branch: 2 ranks at accum 2 (--total_batch_size 64) -----------------------
    print("\n=== (d) no_sync branch: ZeRO + Muon + KD + doc_mask, 2 ranks x accum 2 ===")
    tb = ["--total_batch_size", "64", "--train_tokens", "512"]     # same 8-step schedule, 64 tok/step
    r_nu, out_nu = go("ns_u2", new_trainer, a5 + z + tb + ["--max_steps", "6"], ranks=2)
    r_ns, _ = go("ns_s2", new_trainer, a5 + z + tb + ["--max_steps", "2"], ranks=2)
    r_nr, out_nr = go("ns_r22", new_trainer, a5 + z + tb + ["--max_steps", "6"], ranks=2, resume_from_dir=D("ns_s2"))
    r_n1, out_n1 = go("ns_u1", new_trainer, a5 + tb + ["--max_steps", "6"], ranks=1)
    if check("no_sync runs complete (2-rank x accum 2, and the 1-rank x accum 4 cross-check)",
             r_nu == r_ns == r_nr == r_n1 == 0):
        check("2 ranks x 2 accumulation steps really configured (micro-step 0 runs under no_sync)",
              "GPUs: 2" in out_nu and "Gradient accumulation steps: 2" in out_nu)
        bad = diff_ckpt(load_ckpt(D("ns_u2"), 6), load_ckpt(D("ns_r22"), 6))
        check("no_sync path: save@2 -> resume -> 6 == uninterrupted, bit for bit", not bad, str(bad))
        lu, l1 = step_losses(out_nu), step_losses(out_n1)
        check("no_sync path agrees with 1 rank x accum 4 at print precision (steps 1-6)",
              len(lu) == 6 and all(abs(float(lu[k]) - float(l1[k])) <= 2e-4 for k in lu),
              f"{[lu.get(k) for k in range(1, 7)]} vs {[l1.get(k) for k in range(1, 7)]}")

    # ---- torch.compile on the student (production default), teacher eager -------------------------
    print("\n=== torch.compile(DDP(student)) + eager teacher, 2 ranks x accum 2, ZeRO + Muon + KD ===")
    rc, outc = go("compiled2", new_trainer, a5 + z + tb + ["--torch_compile", "1", "--max_steps", "3"], ranks=2,
                  env_extra={"TORCHINDUCTOR_CACHE_DIR": os.path.join(work, "inductor")})
    if check("compiled run completes (compile + no_sync + KD gate + sharded save at step 3)",
             rc == 0 and "Compiling model with torch.compile" in outc and "ok on every rank" in outc
             and os.path.exists(os.path.join(D("compiled2"), "checkpoint_step_3.pt"))):
        lc, le = step_losses(outc), step_losses(out_nu)
        check("compiled losses match the eager run to 1e-2 (inductor reorders bf16 math)",
              len(lc) == 3 and all(abs(float(lc[k]) - float(le[k])) < 1e-2 for k in lc),
              f"{[lc.get(k) for k in (1, 2, 3)]} vs eager {[le.get(k) for k in (1, 2, 3)]}")

    # ---- refusals: broken teacher, optimizer mismatch on resume ----------------------------------
    print("\n=== refusals ===")
    bad_teacher = [a if a != fx["teacher"] else fx["teacher_uniform"] for a in a5]
    bad_teacher[bad_teacher.index("--kd_topk") + 1] = "4"
    for ranks in (1, 2):
        rc, out = go(f"gate_{ranks}", new_trainer, bad_teacher + ["--max_steps", "3"], ranks=ranks,
                     expect_fail=True)
        check(f"[{ranks} rank(s)] uniform teacher: KD GATE FAILED, every rank exits nonzero, no hang",
              rc not in (0, "timeout") and "KD GATE FAILED" in out and "Step 1 |" not in out,
              f"rc={rc}")
    rc, out = go("mismatch_a", new_trainer, ["--max_steps", "3"], resume_from_dir=D("d_s1"),
                 expect_fail=True)
    check("Muon checkpoint resumed with --optimizer adamw is refused with the fix in the message",
          rc not in (0, "timeout") and "holds MuonAdamW optimizer state" in out and "--reset_schedule 1" in out)
    rc, out = go("mismatch_m", new_trainer, a5 + ["--max_steps", "10"], resume_from_dir=D("def_new_1"),
                 expect_fail=True)
    check("AdamW checkpoint resumed with --optimizer muon is refused", rc not in (0, "timeout") and "holds AdamW optimizer state" in out)
    rc, out = go("reset_m", new_trainer, a5 + ["--max_steps", "2", "--reset_schedule", "1"],
                 resume_from_dir=D("def_new_1"))
    check("... and with --reset_schedule 1 it starts a fresh Muon optimizer from those weights",
          rc == 0 and "Reset schedule mode" in out and "[muon] split" in out)

    # ---- diagnostic, not a gate: the PRE-EXISTING ZeRO AdamW save path, same resume test ----------
    print("\n=== diagnostic: ZeRO AdamW (a4.5 path, unchanged) save -> resume ===")
    ra, _ = go("adz_u", new_trainer, z + ["--max_steps", "6"], ranks=2)
    rb, _ = go("adz_s", new_trainer, z + ["--max_steps", "2"], ranks=2)
    rc_, _ = go("adz_r", new_trainer, z + ["--max_steps", "6"], ranks=2, resume_from_dir=D("adz_s"))
    if ra == rb == rc_ == 0:
        bad = diff_ckpt(load_ckpt(D("adz_u"), 6), load_ckpt(D("adz_r"), 6))
        saved = load_ckpt(D("adz_s"), 2)
        lr_saved = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
        lr_sched = saved["scheduler_state_dict"]["_last_lr"][0]
        msg = (f"ZeRO AdamW resume {'is' if not bad else 'is NOT'} bit-identical ({bad or 'exact'}); "
               f"step-2 checkpoint saved lr {lr_saved:.3e} while the scheduler says the next step uses "
               f"{lr_sched:.3e}")
        print("  NOTE  " + msg)
        notes.append("pre-existing, AdamW ZeRO path (left unchanged as instructed): " + msg)


def e2e_continue(fx, work, old_cp, new_cp, new_pt):
    print("\n=== continue_pretrain.py end to end (seeded by pretrain.py; its arch is derived from the seed) ===")
    logs = os.path.join(work, "logs")
    os.makedirs(logs, exist_ok=True)

    def D(name):
        return os.path.join(work, name)

    def go(name, trainer, args, ranks=1, resume_from_dir=None, env_extra=None, expect_fail=False):
        d = D(name)
        if resume_from_dir:
            shutil.copytree(resume_from_dir, d, symlinks=True)
        rc, out, dt = run_trainer(trainer, args + ["--checkpoint_dir", d], ranks,
                                  log=os.path.join(logs, name + ".log"), env_extra=env_extra)
        print(f"    ran {name:<16} ranks={ranks} rc={rc} {dt:5.1f}s", flush=True)
        if rc != 0 and not expect_fail:
            print("\n".join(out.splitlines()[-25:]))
        return rc, out

    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240801
    short = D("short.bin")                 # 161 periodic tokens: one epoch = 10 windows = 5 steps
    with open(short, "wb") as f:
        f.write(header.tobytes())
        f.write(np.tile(PERIOD_PATTERN, 11)[:161].tobytes())
    longbin = D("a.bin")
    arch = {"A5_TOY_ARCH": json.dumps(TOY_ARCH_CP)}
    seed_args = ["--tokenizer_path", fx["tok"], "--train_sources", fx["sources"], "--train_tokens", "256",
                 "--batch_size", "1", "--block_size", "16", "--total_batch_size", "32", "--lr", "1e-3",
                 "--warmup_steps", "1", "--precision", "bf16", "--torch_compile", "0",
                 "--checkpoint_interval", "999999", "--log_interval", "1", "--max_steps", "2"]
    ra, _ = go("cp_seed_adamw", new_pt, seed_args, env_extra=arch)
    rm, _ = go("cp_seed_muon", new_pt, seed_args + ["--optimizer", "muon", "--muon_lr", "0.02",
                                                     "--adam_lr", "5e-4"], env_extra=arch)
    if not check("seeds written by pretrain.py (AdamW and Muon, head_dim-256 toy shape)", ra == rm == 0):
        return
    seed_a = os.path.join(D("cp_seed_adamw"), "checkpoint_step_2.pt")
    seed_m = os.path.join(D("cp_seed_muon"), "checkpoint_step_2.pt")
    common = ["--tokenizer_path", fx["tok"], "--batch_size", "1", "--block_size", "16",
              "--total_batch_size", "32", "--lr", "1e-3", "--warmup_steps", "2", "--grad_clip", "0.4",
              "--precision", "bf16", "--torch_compile", "0", "--gradient_checkpointing", "1",
              "--checkpoint_interval", "999999", "--schedule", "wsd"]

    # (a) default path: pre-port vs post-port continue_pretrain.py, new phase from the AdamW seed,
    # to the epoch end (5 steps) + HF export
    for ranks, extra in ((1, []), (2, ["--zero_optimizer", "1"])):
        tag = "ZeRO 2 ranks" if ranks == 2 else "1 rank"
        args = common + ["--data_path", short, "--cooldown", "3", "--reset_schedule", "1",
                         "--resume_from", seed_a] + extra
        r1, _ = go(f"cpdef_old_{ranks}", old_cp, args, ranks)
        r2, out2 = go(f"cpdef_new_{ranks}", new_cp, args, ranks)
        if check(f"[continue, {tag}] pre-port and post-port runs complete the epoch", r1 == 0 and r2 == 0):
            bad = diff_ckpt(load_ckpt(D(f"cpdef_old_{ranks}"), 7), load_ckpt(D(f"cpdef_new_{ranks}"), 7))
            check(f"[continue, {tag}] default path: step-7 checkpoint bit-identical to the pre-port trainer",
                  not bad, str(bad))
            check(f"[continue, {tag}] default run prints no Muon/KD lines", "[muon]" not in out2 and "[kd]" not in out2)

    # (c)/(d) Muon + KD: new phase from the Muon seed, save at 6, resume to 12; 1 rank and ZeRO 2 ranks
    mk = ["--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "5e-4", "--kd_teacher", fx["teacher"],
          "--kd_alpha", "1.0", "--kd_topk", "128", "--kd_chunk", "5", "--kd_teacher_chunk", "3",
          "--doc_mask", "1"]
    base = common + ["--data_path", longbin, "--cooldown", "2044"] + mk
    first = ["--reset_schedule", "1", "--resume_from", seed_m]
    z = ["--zero_optimizer", "1"]
    r_u1, out_u1 = go("cpm_u1", new_cp, base + first + ["--max_steps", "12"])
    r_s1, _ = go("cpm_s1", new_cp, base + first + ["--max_steps", "6"])
    r_r1, out_r1 = go("cpm_r1", new_cp, base + ["--max_steps", "12"], resume_from_dir=D("cpm_s1"))
    r_u2, out_u2 = go("cpm_u2", new_cp, base + first + z + ["--max_steps", "12"], ranks=2)
    r_s2, _ = go("cpm_s2", new_cp, base + first + z + ["--max_steps", "6"], ranks=2)
    r_r22, out_r22 = go("cpm_r22", new_cp, base + z + ["--max_steps", "12"], ranks=2, resume_from_dir=D("cpm_s2"))
    r_r21, _ = go("cpm_r21", new_cp, base + ["--max_steps", "12"], ranks=1, resume_from_dir=D("cpm_s2"))
    if check("[continue] all Muon+KD runs complete", all(r == 0 for r in (r_u1, r_s1, r_r1, r_u2, r_s2, r_r22, r_r21)),
             str([r_u1, r_s1, r_r1, r_u2, r_s2, r_r22, r_r21])):
        check("[continue] new phase from the Muon seed builds a FRESH MuonAdamW and passes the KD gate",
              "Reset-schedule (NEW PHASE)" in out_u1 and "[muon] split" in out_u1 and "ok on every rank" in out_u1)
        check("[continue] Step 10 line carries hard_CE and both LRs",
              bool(re.search(r"^Step 10 \| Loss: [0-9.]+ \| hard_CE: [0-9.]+ .*\(adam ", out_u1, re.M)))
        bad = diff_ckpt(load_ckpt(D("cpm_u1"), 12), load_ckpt(D("cpm_r1"), 12))
        check("[continue, 1 rank] save@6 -> resume -> 12 == uninterrupted, bit for bit", not bad, str(bad))
        check("[continue] ZeRO shard-merged step-6 optimizer state == unsharded MuonAdamW state_dict",
              same_tree(load_ckpt(D("cpm_s2"), 6)["optimizer_state_dict"],
                        load_ckpt(D("cpm_s1"), 6)["optimizer_state_dict"]))
        c_u2 = load_ckpt(D("cpm_u2"), 12)
        for name, label in (("cpm_u1", "1-rank uninterrupted"), ("cpm_r22", "2-rank ZeRO resume"),
                            ("cpm_r21", "1-rank UNSHARDED resume of the ZeRO checkpoint")):
            bad = diff_ckpt(c_u2, load_ckpt(D(name), 12))
            check(f"[continue] step 12: {label} == 2-rank ZeRO uninterrupted, bit for bit", not bad, str(bad))
        check("[continue] ZeRO resume releases the duplicate optimizer state a ZeRO load leaves behind",
              bool(re.search(r"\[zero\] released the duplicate optimizer state .*: [0-9.]+ GiB", out_r22)))
    # the DDP no_sync branch in continue_pretrain.py: 2 ranks x accum 2
    tb = ["--total_batch_size", "64"]
    r_a, out_a = go("cpns_u2", new_cp, base + first + z + tb + ["--max_steps", "8"], ranks=2)
    r_b, _ = go("cpns_s2", new_cp, base + first + z + tb + ["--max_steps", "5"], ranks=2)
    r_c, _ = go("cpns_r22", new_cp, base + z + tb + ["--max_steps", "8"], ranks=2, resume_from_dir=D("cpns_s2"))
    if check("[continue] no_sync runs complete (2 ranks x accum 2)", r_a == r_b == r_c == 0):
        check("[continue] 2 ranks x 2 accumulation steps really configured",
              "GPUs: 2" in out_a and "Gradient accumulation steps: 2" in out_a)
        bad = diff_ckpt(load_ckpt(D("cpns_u2"), 8), load_ckpt(D("cpns_r22"), 8))
        check("[continue] no_sync path: save@5 -> resume -> 8 == uninterrupted, bit for bit", not bad, str(bad))
    # doc_mask is LIVE in continue_pretrain.py: same seed and data, mask off vs on, to step 10 (this trainer
    # prints a Step line every 10 steps). The fixture windows hold EOS separators, so a live mask must change
    # the logged loss; an unchanged loss means the flag reached the config but the ids never reached the
    # forward (pretrain.py's pre-2026-09-01 defect).
    dm = common + ["--data_path", longbin, "--cooldown", "2044", "--optimizer", "muon", "--muon_lr", "0.02",
                   "--adam_lr", "5e-4", "--reset_schedule", "1", "--resume_from", seed_m, "--max_steps", "10"]
    r0, o0 = go("cpdm_off", new_cp, dm + ["--doc_mask", "0"])
    r1, o1 = go("cpdm_on", new_cp, dm + ["--doc_mask", "1"])
    if check("[continue] doc_mask off/on runs complete", r0 == 0 and r1 == 0):
        check("[continue] the effective doc_mask state is printed", "doc_mask: False" in o0 and "doc_mask: True" in o1)
        l0, l1 = step_losses(o0), step_losses(o1)
        first = min(set(l0) & set(l1)) if set(l0) & set(l1) else None
        check("[continue] doc_mask 1 changes the logged loss (the mask reaches the forward)",
              first is not None and l0[first] != l1[first], f"step {first}: off {l0.get(first)} on {l1.get(first)}")
    rc, out = go("cp_mismatch", new_cp, base + ["--max_steps", "12"], resume_from_dir=D("cpdef_new_1"),
                 expect_fail=True)
    check("[continue] AdamW checkpoint resumed with --optimizer muon (no reset) is refused",
          rc not in (0, "timeout") and "holds AdamW optimizer state" in out)


def real_teacher(work, new_trainer):
    print("\n=== real teacher: Qwen3-0.6B-Base + Qwen3 tokenizer on real text (1 rank, toy student) ===")
    if not (os.path.isdir(QWEN) and os.path.exists(REAL_BIN)):
        print("  SKIP  teacher or data missing")
        notes.append("real-teacher test skipped: teacher or data missing")
        return
    for tag, extra in (("unpadded vocab", []), ("padded vocab (--pad_vocab_multiple 128)",
                                                ["--pad_vocab_multiple", "128"])):
        d = os.path.join(work, "real_" + ("pad" if extra else "nopad"))
        args = ["--tokenizer_path", QWEN, "--data_path", REAL_BIN, "--checkpoint_dir", d,
                "--batch_size", "1", "--block_size", "64", "--total_batch_size", "128", "--lr", "1e-3",
                "--warmup_steps", "1", "--precision", "bf16", "--torch_compile", "0",
                "--checkpoint_interval", "999999", "--log_interval", "1", "--max_steps", "2",
                "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "5e-4", "--doc_mask", "1",
                "--kd_teacher", QWEN, "--kd_alpha", "1.0"] + extra
        rc, out, dt = run_trainer(new_trainer, args, 1, timeout=1800,
                                  log=os.path.join(work, "real_" + ("pad" if extra else "nopad") + ".log"))
        print(f"    ran real ({tag}) rc={rc} {dt:.1f}s")
        gate = re.search(r"KD GATE .*?hard_ce=([0-9.]+) topk_mass=([0-9.]+) gold_in_topk=([0-9.]+) soft=([0-9.]+)", out)
        trunc = re.search(r"\(vocab (\d+) -> truncated to the student's (\d+)\)", out)
        print("    " + (gate.group(0) if gate else "no gate line"))
        print("    " + (trunc.group(0) if trunc else "no truncation line"))
        want = "151680" if extra else "151669"
        check(f"[{tag}] run completes (2 steps, save at step 2)", rc == 0 and
              os.path.exists(os.path.join(d, "checkpoint_step_2.pt")))
        check(f"[{tag}] teacher 151936 truncated to the student's {want}",
              bool(trunc) and trunc.group(1) == "151936" and trunc.group(2) == want)
        check(f"[{tag}] gate passes on real text: top-K mass > 0.9 and gold-in-top-K > 0.5",
              bool(gate) and float(gate.group(2)) > 0.9 and float(gate.group(3)) > 0.5)
        losses = step_losses(out)
        check(f"[{tag}] finite losses logged with hard_CE", len(losses) == 2 and "hard_CE:" in out
              and all(math.isfinite(float(v)) for v in losses.values()))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "unit"
    work = tempfile.mkdtemp(prefix="a5test_")
    print(f"workdir: {work}")
    sys.path.insert(0, REPO)
    old_dir = os.path.join(work, "prepost")
    os.makedirs(old_dir)
    old_trainer = base_rev_file("pretrain.py", old_dir)
    old_cp = base_rev_file("continue_pretrain.py", old_dir)
    base_rev_file("model.py", old_dir)       # the pre-port trainers import model.py from their own dir
    new_trainer = os.path.join(REPO, "pretrain.py")
    new_cp = os.path.join(REPO, "continue_pretrain.py")
    try:
        if what in ("unit", "all"):
            pt = load_module(new_trainer, "pt_new")
            old = load_module(old_trainer, "pt_old")
            cp_new = load_module(new_cp, "cp_new")       # argparse is in its __main__ block: no argv needed
            cp_old = load_module(old_cp, "cp_old")
            import pretrain_kd as KD
            unit_defaults_and_optimizer(pt, old)
            unit_kd_parity(KD)
            unit_kd_chunked_head(KD)
            unit_kdstep(KD)
            unit_muon_equivalence(pt, work)
            unit_schedule_two_groups()
            unit_production_split(pt)
            unit_shared_wiring(pt, cp_new, cp_old, KD)
            unit_validation(new_trainer)
            unit_validation_cp(new_cp)
        if what in ("e2e", "all"):
            fx = build_fixtures(work)
            e2e(fx, work, old_trainer, new_trainer)
            e2e_continue(fx, work, old_cp, new_cp, new_trainer)
        if what in ("real", "all"):
            real_teacher(work, new_trainer)
    except BaseException as e:            # an uncaught error is a failure, never "ALL PASS"
        import traceback
        traceback.print_exc()
        failures.append(f"uncaught {type(e).__name__}: {e}")
    finally:
        print("\n" + "=" * 72)
        for n in notes:
            print("NOTE: " + n)
        if failures:
            print(f"FAILED ({len(failures)}):")
            for f in failures:
                print("  - " + f)
            print(f"(workdir kept for inspection: {work})")
        else:
            print(f"ALL PASS ({what})")
            shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
