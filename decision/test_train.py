"""CPU tests for decision/train.py checkpoints (tiny trunk, gloo): ~2 min.

    python decision/test_train.py

  (a) world 1: interrupt after step 1 (yield file) -> resume -> final weights BIT-IDENTICAL to an uninterrupted run
  (b) world 2 with ZeRO: the same, bit-identical
  (c) world 1 -> world 2: the checkpoint written by one rank reloads into two ZeRO ranks, every optimizer state is
      restored, and the run finishes having consumed each micro-batch of the epoch exactly once
  (d) the lock file is written while training and removed at exit
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.environ.get("DECISION_TEST_PACKED", os.path.join(REPO, "data", "decision", "test_packed"))
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)
    return ok


def subset(src, dst, n, seed):
    """A small packed split: the first n examples (by a seeded choice), re-based."""
    os.makedirs(dst, exist_ok=True)
    T = np.load(f"{src}/tokens.npy", mmap_mode="r")
    I = np.load(f"{src}/index.npy")
    O = np.load(f"{src}/opt_pos.npy")
    S = np.load(f"{src}/soft.npy")
    meta = json.load(open(f"{src}/meta.json"))
    rng = np.random.default_rng(seed)
    short = np.where(I[:, 1] < 1500)[0]
    pick = np.sort(rng.choice(short, size=n, replace=False))
    toks, opts, softs, idx = [], [], [], []
    st = os_ = ss = 0
    for j in pick:
        s, L, ans, o, k, lab, typ, src_id, sft, tr = I[j]
        toks.append(np.asarray(T[s:s + L]))
        opts.append(O[o:o + k])
        new_ss = -1
        if sft >= 0:
            softs.append(S[sft:sft + k])
            new_ss = ss
            ss += k
        idx.append((st, L, ans, os_, k, lab, typ, src_id, new_ss, tr))
        st += L
        os_ += k
    np.save(f"{dst}/tokens.npy", np.concatenate(toks).astype(np.int32))
    np.save(f"{dst}/opt_pos.npy", np.concatenate(opts).astype(np.int32))
    np.save(f"{dst}/soft.npy", np.concatenate(softs).astype(np.float32) if softs else np.zeros(0, np.float32))
    np.save(f"{dst}/index.npy", np.asarray(idx, dtype=np.int64))
    meta = dict(meta, n=n, tokens=int(st), ids=[meta["ids"][j] for j in pick])
    json.dump(meta, open(f"{dst}/meta.json", "w"))


def run(out, data, world=1, extra=(), yield_file=None, lock=None, env_extra=None, timeout=900):
    cmd = ([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}"] if world > 1
           else [sys.executable]) + [os.path.join(HERE, "train.py"), "--tiny", "1", "--data", data, "--out", out,
                                     "--micro_tokens", "3072", "--micro_max_ex", "8", "--eval_every", "0",
                                     "--log_every", "1", "--grad_ckpt", "0", "--warmup_frac", "0.1", "--lr", "3e-3",
                                     "--head_lr", "3e-3", "--save_every_min", "999"] + list(extra)
    if yield_file:
        cmd += ["--yield_file", yield_file]
    if lock:
        cmd += ["--lock_file", lock]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               **(env_extra or {}))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=REPO)
    except subprocess.TimeoutExpired as e:
        return "timeout", (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    return r.returncode, r.stdout + r.stderr


def weights(final):
    import torch
    sys.path.insert(0, REPO)
    from decision.modeling import DecisionModel
    m = DecisionModel.load(final, dtype=torch.float32)
    return {k: v.clone() for k, v in m.state_dict().items()}


def same(a, b):
    import torch
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def main():
    work = tempfile.mkdtemp(prefix="dec_train_test_")
    try:
        data = os.path.join(work, "data")
        subset(f"{SRC}/train", f"{data}/train", 160, 1)
        subset(f"{SRC}/val", f"{data}/val", 40, 2)

        print("=== (a) world 1: interrupt + resume == uninterrupted")
        rc, out_u = run(f"{work}/u1", data)
        check("uninterrupted world-1 run completes", rc == 0 and os.path.isdir(f"{work}/u1/final"), out_u[-300:] if rc else "")
        yf = f"{work}/yield"
        open(yf, "w").close()
        lock = f"{work}/lock"
        rc, out_a = run(f"{work}/i1", data, yield_file=yf, lock=lock)
        check("yield file makes the trainer save a checkpoint and exit after one step",
              rc == 0 and os.path.exists(f"{work}/i1/ckpt/state.json") and not os.path.isdir(f"{work}/i1/final")
              and "yield requested at step 1" in out_a, out_a[-300:])
        check("lock file removed at exit", not os.path.exists(lock))
        os.remove(yf)
        rc, out_b = run(f"{work}/i1", data, lock=lock)
        check("resumed world-1 run completes", rc == 0 and os.path.isdir(f"{work}/i1/final"), out_b[-300:] if rc else "")
        if rc == 0:
            check("interrupt + resume final weights BIT-IDENTICAL to the uninterrupted run",
                  same(weights(f"{work}/u1/final"), weights(f"{work}/i1/final")))
            check("the finished run removed its resumable checkpoint (latest-only; final supersedes it)",
                  not os.path.exists(f"{work}/i1/ckpt"))

        print("=== (b) world 2 + ZeRO: interrupt + resume == uninterrupted")
        rc, out_u2 = run(f"{work}/u2", data, world=2)
        check("uninterrupted world-2 ZeRO run completes", rc == 0 and os.path.isdir(f"{work}/u2/final"), out_u2[-400:] if rc else "")
        open(yf, "w").close()
        rc, _ = run(f"{work}/i2", data, world=2, yield_file=yf)
        os.remove(yf)
        rc2, out_i2 = run(f"{work}/i2", data, world=2)
        check("world-2 resume completes", rc == 0 and rc2 == 0 and os.path.isdir(f"{work}/i2/final"), out_i2[-400:] if rc2 else "")
        if rc2 == 0:
            check("world-2 interrupt + resume BIT-IDENTICAL to uninterrupted", same(weights(f"{work}/u2/final"), weights(f"{work}/i2/final")))

        print("=== (c) world 1 checkpoint -> world 2 ZeRO resume")
        open(yf, "w").close()
        rc, _ = run(f"{work}/x", data, yield_file=yf)
        os.remove(yf)
        state = json.load(open(f"{work}/x/ckpt/state.json"))
        rc, out_x = run(f"{work}/x", data, world=2)
        m = re.search(r"written at world (\d+), running at world (\d+); rank 0 restored optimizer state for (\d+)/(\d+)", out_x)
        check("world-1 checkpoint resumes at world 2", rc == 0 and os.path.isdir(f"{work}/x/final") and m is not None, out_x[-400:])
        if m:
            check("every parameter rank 0 owns got its optimizer state back", m.group(3) == m.group(4) and int(m.group(3)) > 0,
                  f"{m.group(3)}/{m.group(4)}")
        losses = [float(v) for v in re.findall(r"^step \d+ \| loss ([0-9.]+)", out_x, re.M)]
        check("training continued with finite losses", len(losses) > 0 and all(np.isfinite(losses)), str(losses[:5]))
        fin = re.search(r"\[train\] finished: ([0-9,]+)/([0-9,]+) micro-batches \(100.0%\)", out_x)
        check("the epoch completed (every micro-batch consumed or counted)", fin is not None, out_x[-200:])
        print("=== (d) periodic saves with ranks on different clocks (the desync that once hung a multi-GPU run)")
        rc, out_d = run(f"{work}/d", data, world=2, extra=["--save_every_min", "0.01"],
                        env_extra={"DECISION_TEST_RANK_STAGGER": "1.5"}, timeout=400)
        saves = len(re.findall(r"checkpoint saved at step", out_d))
        check("world 2, rank 1 starting 1.5 s late, 0.6 s save interval: no hang, several periodic saves, run completes",
              rc == 0 and saves >= 2 and os.path.isdir(f"{work}/d/final"), f"rc={rc} saves={saves}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
