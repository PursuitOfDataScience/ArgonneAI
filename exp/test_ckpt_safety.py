"""Prove the checkpoint integrity contract, for pretrain.py AND continue_pretrain.py, in both trees.

The contract the owner asked for (2026-08-14): only the latest checkpoint is kept, a corrupt model
can never be committed, and once a checkpoint is committed the previous one is removed.

That makes the gates load-bearing in a way nothing else is: if they are wrong, latest-only
retention deletes the last good checkpoint and the run is unrecoverable. These tests exercise the
real save_checkpoint from each module -- no mocks of the code under test -- on CPU with a tiny
model, so they run in seconds and need no GPU.

    python exp/test_ckpt_safety.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn

TREES = ["/home/youzhi/ArgonneAI-4.5", "/home/youzhi/ArgonneAI"]
MODULES = ["pretrain", "continue_pretrain"]

failures = []


# pretrain.py runs `parser.parse_args()` and setup_distributed() at MODULE level, so importing it
# parses OUR argv. Feed it a minimal valid command line; with no RANK in env setup_distributed()
# returns (0,0,1) and nothing touches a GPU, so this stays a CPU-only test.
FAKE_ARGV = [
    "pretrain.py",
    "--tokenizer_path", "/dev/null",
    "--data_path", "/dev/null",
    "--checkpoint_dir", "/dev/null",
    "--batch_size", "1",
    "--block_size", "8",
    "--total_batch_size", "8",
]


def load(tree, modname):
    """Import <tree>/<modname>.py under a unique name without executing main()."""
    path = os.path.join(tree, modname + ".py")
    spec = importlib.util.spec_from_file_location(f"{modname}_{abs(hash(tree)) % 10**8}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    saved = sys.argv
    sys.argv = list(FAKE_ARGV)
    try:
        spec.loader.exec_module(mod)      # safe: the __main__ guard is at the end of the file
    finally:
        sys.argv = saved
    return mod


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 4)

    def forward(self, x):
        return self.b(self.a(x))


def save(mod, model, opt, step, d):
    """Call the module's real save_checkpoint, adapting to its signature."""
    import inspect
    sig = inspect.signature(mod.save_checkpoint).parameters
    args = [model, opt, None, step, step * 1000, 0.5, 0, d]
    if "dataset_epoch" in sig:                       # continue_pretrain takes 5 more
        args += [0, 0, 0, 1_000_000, "/dev/null"]
    return mod.save_checkpoint(*args)


def steps_on_disk(d):
    return sorted(f for f in os.listdir(d) if f.startswith("checkpoint_step_") and f.endswith(".pt"))


def check(label, cond, detail=""):
    print(f"    {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(f"{label} {detail}")


for tree in TREES:
    for modname in MODULES:
        print(f"\n=== {tree.split('/')[-1]} / {modname}.py ===")
        mod = load(tree, modname)
        d = tempfile.mkdtemp(prefix="ckptsafe_")
        try:
            model, opt = Tiny(), None
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            opt.step()   # give the optimizer real state to serialize

            # 1. a healthy save commits, and leaves exactly one checkpoint
            p1 = save(mod, model, opt, 100, d)
            check("healthy save commits", p1 is not None and os.path.exists(p1 or ""))
            check("no .tmp left behind", not any(f.endswith(".tmp") for f in os.listdir(d)),
                  str(os.listdir(d)))
            check("checkpoint_last.pt points at it",
                  os.path.realpath(os.path.join(d, "checkpoint_last.pt")) == os.path.realpath(p1 or ""))

            # 2. a second healthy save prunes the first -- latest-only retention
            p2 = save(mod, model, opt, 200, d)
            on_disk = steps_on_disk(d)
            check("second save commits", p2 is not None)
            check("previous checkpoint pruned (latest-only)", on_disk == ["checkpoint_step_200.pt"],
                  str(on_disk))

            # 3. THE CRITICAL ONE: a NaN weight must be refused, and must NOT prune the good one
            with torch.no_grad():
                model.a.weight[0, 0] = float("nan")
            p3 = save(mod, model, opt, 300, d)
            on_disk = steps_on_disk(d)
            check("NaN model REFUSED", p3 is None)
            check("good checkpoint SURVIVED the refusal", on_disk == ["checkpoint_step_200.pt"],
                  str(on_disk))
            check("refusal wrote no partial file", not any(f.endswith(".tmp") for f in os.listdir(d)),
                  str(os.listdir(d)))
            check("checkpoint_last.pt still points at the good one",
                  os.path.realpath(os.path.join(d, "checkpoint_last.pt")).endswith("checkpoint_step_200.pt"))

            # 4. an Inf weight is refused the same way
            with torch.no_grad():
                model.a.weight[0, 0] = 0.0
                model.b.bias[1] = float("inf")
            check("Inf model REFUSED", save(mod, model, opt, 400, d) is None)
            check("good checkpoint still there", steps_on_disk(d) == ["checkpoint_step_200.pt"])

            # 5. a truncated write must be caught by gate 2 and must not prune
            with torch.no_grad():
                model.b.bias[1] = 0.0
            real_save = torch.save

            def truncating_save(obj, path, *a, **k):
                real_save(obj, path, *a, **k)
                with open(path, "r+b") as f:      # lop off the tail: valid header, missing data
                    f.truncate(max(1, os.path.getsize(path) // 3))

            mod.torch.save = truncating_save
            try:
                p5 = save(mod, model, opt, 500, d)
            finally:
                mod.torch.save = real_save
            on_disk = steps_on_disk(d)
            check("truncated write REFUSED", p5 is None)
            check("truncated .tmp discarded", not any(f.endswith(".tmp") for f in os.listdir(d)),
                  str(os.listdir(d)))
            check("good checkpoint survived truncation", on_disk == ["checkpoint_step_200.pt"],
                  str(on_disk))

            # 6. after all those refusals a healthy save must still work and prune
            p6 = save(mod, model, opt, 600, d)
            check("recovers: healthy save after refusals", p6 is not None)
            check("and prunes to latest-only", steps_on_disk(d) == ["checkpoint_step_600.pt"],
                  str(steps_on_disk(d)))
            ck = torch.load(p6, map_location="cpu", weights_only=False)
            check("committed checkpoint is loadable and correct",
                  ck["global_step"] == 600 and all(torch.isfinite(v).all() for v in ck["model_state_dict"].values()))

            # 7. abandoned .tmp writes must be collected, or a preempted slice leaks ~25 GB each.
            #    prune_old_checkpoints deliberately ignores .tmp, so this is the only collector.
            stale = os.path.join(d, "checkpoint_step_555.pt.tmp")
            fresh = os.path.join(d, "checkpoint_step_777.pt.tmp")
            for f in (stale, fresh):
                with open(f, "wb") as fh:
                    fh.write(b"partial")
            os.utime(stale, (0, 0))                      # ancient => definitely abandoned
            mod.cleanup_stale_tmp_checkpoints(d)
            check("abandoned .tmp removed", not os.path.exists(stale))
            check("recent .tmp left alone (could be an active write)", os.path.exists(fresh))
            check("cleanup did not touch the real checkpoint", steps_on_disk(d) == ["checkpoint_step_600.pt"],
                  str(steps_on_disk(d)))
            check("cleanup does not disturb checkpoint_last.pt",
                  os.path.realpath(os.path.join(d, "checkpoint_last.pt")).endswith("checkpoint_step_600.pt"))
            os.remove(fresh)
            # a .tmp must never be mistaken for a resumable checkpoint
            with open(os.path.join(d, "checkpoint_step_999.pt.tmp"), "wb") as fh:
                fh.write(b"partial")
            check("resume ignores .tmp and picks the committed checkpoint",
                  os.path.realpath(mod.get_latest_checkpoint_path(d)).endswith("checkpoint_step_600.pt"),
                  str(mod.get_latest_checkpoint_path(d)))
        finally:
            shutil.rmtree(d, ignore_errors=True)

print("\n" + "=" * 64)
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL PASS — corrupt weights and torn writes are refused, and a refusal never")
print("           destroys the last good checkpoint, in all 4 trainer files.")
