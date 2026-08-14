"""Prove WeightedMultiLoader resumes EXACTLY, across ranks and across a micro-batch change.

Why this matters more than it looks: weekend.sh runs pretraining as ~576 chained one-hour
slices, so every slice after the first resumes. A resume that silently restarts a rank's cursor,
or lets two ranks read the same window, costs nothing visible -- no crash, no loss spike, just a
model trained on a fraction of the corpus it was supposed to see. And the production worker picks
the micro-batch from the card it lands on (22 on an H200, 16 on a 94G H100, 11 on an 80G H100),
so a resume onto a different card changes B mid-run. That path has to be exact too.

    python exp/test_loader_resume.py

CPU-only, a few seconds, no GPU needed.
"""
import importlib.util
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

# next_batch() calls .pin_memory(), which needs a CUDA driver. Pinning is a host-memory
# placement detail with no effect on WHICH tokens come back, so stubbing it lets this run on a
# login node without changing anything the test actually measures.
torch.Tensor.pin_memory = lambda self, *a, **k: self

TREES = ["/home/youzhi/ArgonneAI-4.5", "/home/youzhi/ArgonneAI"]
WORLD = 3
T = 16          # block size
B = 4           # micro batch

failures = []


def check(label, cond, detail=""):
    print(f"    {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(f"{label} {detail}")


def load(tree):
    path = os.path.join(tree, "pretrain.py")
    spec = importlib.util.spec_from_file_location(f"pt_{abs(hash(tree)) % 10**8}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    saved = sys.argv
    sys.argv = ["pretrain.py", "--tokenizer_path", "/dev/null", "--data_path", "/dev/null",
                "--checkpoint_dir", "/dev/null", "--batch_size", "1", "--block_size", "8",
                "--total_batch_size", "8"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def make_bin(path, ntok, base):
    """llm.c shard: 256 int32 header (magic 20240801) then uint32 tokens.

    Token values are base+i so any batch identifies exactly which window it came from --
    that is what makes 'did rank 1 read rank 0's data' a decidable question.
    """
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240801
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write((base + np.arange(ntok, dtype=np.uint32)).tobytes())


for tree in TREES:
    print(f"\n=== {tree.split('/')[-1]} / WeightedMultiLoader ===")
    mod = load(tree)
    d = tempfile.mkdtemp(prefix="loaderres_")
    try:
        make_bin(os.path.join(d, "a.bin"), 60000, 1_000_000)
        make_bin(os.path.join(d, "b.bin"), 60000, 5_000_000)
        srcs = [(os.path.join(d, "a.bin"), 0.7), (os.path.join(d, "b.bin"), 0.3)]

        def fresh(b=B):
            return [mod.WeightedMultiLoader(srcs, b, T, rank=r, world_size=WORLD, seed=1337)
                    for r in range(WORLD)]

        def draw(loaders, n):
            """One global step = every rank draws once, in rank order (lockstep, as DDP runs)."""
            out = []
            for _ in range(n):
                out.append([tuple(l.next_batch()[0].flatten().tolist()) for l in loaders])
            return out

        # --- reference run: 20 steps, snapshot, then 8 more -------------------------------
        ref = fresh()
        draw(ref, 20)
        state = ref[0].get_position()                 # save_checkpoint calls this on rank 0 only
        tokens_at_save = ref[0].drawn_tokens()        # == pretrain.py's tokens_processed
        after = draw(ref, 8)

        check("all 3 ranks stayed in sync on the source choice",
              all(l.counts == ref[0].counts for l in ref), str([l.counts for l in ref]))

        # --- ranks must not overlap -------------------------------------------------------
        step0 = after[0]
        overlap = set(step0[0]) & set(step0[1]) | set(step0[0]) & set(step0[2]) | set(step0[1]) & set(step0[2])
        check("the 3 ranks read disjoint windows", not overlap, f"{len(overlap)} shared tokens")

        # --- resume at the SAME micro-batch: must reproduce `after` exactly ----------------
        res = fresh()
        for l in res:
            l.resume_from_checkpoint_position(state, drawn_tokens=tokens_at_save)
        got = draw(res, 8)
        check("resume reproduces the next 8 steps EXACTLY on every rank", got == after,
              f"first divergence step {next((i for i,(x,y) in enumerate(zip(got,after)) if x!=y), None)}")
        check("resumed drawn_tokens continues (no double count / no reset)",
              res[0].drawn_tokens() == ref[0].drawn_tokens(),
              f"{res[0].drawn_tokens()} vs {ref[0].drawn_tokens()}")

        # --- a resume that never restored would re-read the corpus from the top -----------
        naive = fresh()
        check("control: NOT resuming really does restart the corpus (test is meaningful)",
              draw(naive, 1)[0] != after[0])

        # --- resume onto a different card => different micro-batch -------------------------
        for b_new in (2, 8):
            alt = [mod.WeightedMultiLoader(srcs, b_new, T, rank=r, world_size=WORLD, seed=1337)
                   for r in range(WORLD)]
            for l in alt:
                l.resume_from_checkpoint_position(state, drawn_tokens=tokens_at_save)
            check(f"micro-batch {B}->{b_new}: token progress preserved, not rescaled",
                  alt[0].drawn_base == tokens_at_save, f"{alt[0].drawn_base} vs {tokens_at_save}")
            s0 = draw(alt, 1)[0]
            ov = set(s0[0]) & set(s0[1]) | set(s0[0]) & set(s0[2]) | set(s0[1]) & set(s0[2])
            check(f"micro-batch {B}->{b_new}: ranks still disjoint after resume", not ov,
                  f"{len(ov)} shared tokens")
            check(f"micro-batch {B}->{b_new}: does not re-read from the corpus start",
                  s0[0] != draw(fresh(b_new), 1)[0][0])

        # --- an old checkpoint with no drawn_tokens must fall back, not zero out -----------
        legacy = dict(state)
        legacy.pop("drawn_tokens", None)
        lg = fresh()
        for l in lg:
            l.resume_from_checkpoint_position(legacy, drawn_tokens=None)
        check("legacy state without drawn_tokens still restores nonzero progress",
              lg[0].drawn_base > 0, str(lg[0].drawn_base))

        # --- a garbage/missing state must not silently look like a successful resume -------
        bad = fresh()
        for l in bad:
            l.resume_from_checkpoint_position(0, drawn_tokens=None)   # flat-loader int position
        check("incompatible resume state starts fresh instead of crashing",
              bad[0].drawn_base == 0 and bad[0].draws == 0)
    finally:
        shutil.rmtree(d, ignore_errors=True)

print("\n" + "=" * 64)
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL PASS — every slice resumes exactly where the last one stopped, on all 3 ranks,")
print("           and survives the micro-batch change a different card forces.")
