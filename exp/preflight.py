"""Preflight for run_full_training.sh — run BEFORE every launch.

Why this exists: on 2026-08-14 I edited the worker to add runtime card detection and the replaced
text range silently swallowed `BLOCK_SIZE=1024`. `bash -n` passed (an undefined variable is legal
bash), the job queued for 1.5 hours, won GPUs, computed
TOTAL_BATCH_SIZE = 16 * 3 * 0 * 11 = 0, and died in under a second — wasting the slot and one of
five failure-retries, each of which blacklists the node it failed on.

Syntax checking cannot catch that. This asserts the RESOLVED values instead, without executing the
script (executing it would need SLURM, GPUs, module load and the marker files).

    python exp/preflight.py            # exits nonzero and explains if anything is wrong
"""
import os
import re
import subprocess
import sys

WORKER = "run_full_training.sh"
TRAINER = "pretrain.py"
WANT_EFFECTIVE = 540_672

fails, notes = [], []


def grab(src, pattern, label, cast=int):
    m = re.search(pattern, src, re.M)
    if not m:
        fails.append(f"{label}: NOT FOUND in {WORKER} — this is the BLOCK_SIZE failure mode")
        return None
    try:
        return cast(m.group(1))
    except ValueError:
        fails.append(f"{label}: found but unparseable ({m.group(1)!r})")
        return None


src = open(WORKER).read()

ngpus = grab(src, r"^NGPUS=(\d+)", "NGPUS")
block = grab(src, r"^BLOCK_SIZE=(\d+)", "BLOCK_SIZE")
chunk = grab(src, r"^PRETRAIN_CHUNK=\$\{A45_CHUNK:-(\d+)\}", "PRETRAIN_CHUNK")
stride = grab(src, r"^PRETRAIN_CKPT_STRIDE=\$\{A45_CKPT_STRIDE:-(\d+)\}", "PRETRAIN_CKPT_STRIDE")
stride94 = grab(src, r"PRETRAIN_CKPT_STRIDE=\$\{A45_CKPT_STRIDE_94G:-(\d+)\}", "stride on >=90GB card")

# every card branch must land on the SAME effective batch, or the LR recipe silently changes
# depending on which node a slice happens to win.
branches = re.findall(r'_CARD_SEEN="([^"]+)";\s*_MB=(\d+);\s*_GA=(\d+)', src)
if not branches:
    fails.append("card-detection branches (_MB/_GA): NOT FOUND")
elif ngpus and block:
    for name, mb, ga in branches:
        eff = int(mb) * ngpus * block * int(ga)
        tag = "OK " if eff == WANT_EFFECTIVE else "BAD"
        notes.append(f"  {tag} {name:<16} micro {mb:>2} x accum {ga:>2} x {ngpus} x {block} = {eff:,}")
        if eff != WANT_EFFECTIVE:
            fails.append(f"{name}: effective batch {eff:,} != {WANT_EFFECTIVE:,}")

# The token budget sets BOTH the WSD schedule length and the stop point, and it used to come
# only from an exported shell variable -- so a relaunch from a fresh login silently became a
# 2.9x shorter run. Assert the resolved default, that every configured shard exists with a valid
# llm.c header, and that the implied repetition stays inside the ~4x free-repeat bound.
WANT_TOKENS = 110_000_000_000
budget = grab(src, r"^A45_TRAIN_TOKENS=\$\{A45_TRAIN_TOKENS:-(\d+)\}", "A45_TRAIN_TOKENS")
if budget is not None and budget != WANT_TOKENS:
    fails.append(f"A45_TRAIN_TOKENS resolves to {budget:,}, want {WANT_TOKENS:,} "
                 f"(0 would silently shorten the run to the corpus size)")

_shards = subprocess.run(
    ["bash", "-c", f'eval "$(sed -n \'247,254p\' {WORKER})"; echo "$PRETRAIN_SOURCES"'],
    capture_output=True, text=True).stdout.strip()
_total = 0
for _spec in [s for s in _shards.split(",") if s]:
    _p, _w = _spec.rsplit(":", 1)
    if not os.path.exists(_p):
        fails.append(f"training shard MISSING: {_p}")
        continue
    with open(_p, "rb") as _f:
        _magic = int.from_bytes(_f.read(4), "little")
    if _magic != 20240801:
        fails.append(f"training shard has bad llm.c magic {_magic}: {_p}")
    _total += (os.path.getsize(_p) - 1024) // 4
if _total:
    _rep = (budget or 0) / _total
    notes.append(f"  {'OK ' if _rep <= 4.0 else 'BAD'} corpus {_total:,} tok, budget {budget:,} "
                 f"=> {_rep:.2f} passes, {int((budget or 0)/WANT_EFFECTIVE):,} steps")
    if _rep > 4.0:
        fails.append(f"budget implies {_rep:.2f} passes over the corpus, beyond the ~4x free-repeat bound")

if chunk is not None and chunk != 0:
    fails.append(f"PRETRAIN_CHUNK={chunk}, want 0 — chunk=0 is the +22.9% systems win")
if stride is not None and stride != 2:
    fails.append(f"PRETRAIN_CKPT_STRIDE={stride}, want 2 — the measured optimum (+6.1%)")
if stride94 is not None and stride94 != 2:
    fails.append(f"94GB-card stride={stride94}, want 2 — a4.0's 16 was tuned for its 32-layer arch")

# the trainer must be building the 4.5 arch, and every top-level def must precede __main__
# (the NameError that killed the first production slice).
tsrc = open(TRAINER).read()
if not re.search(r"^A45 = True", tsrc, re.M):
    fails.append("pretrain.py has A45 = False — would train the 1.04B a4.0 arch into the 4.5 ckpt dir")
import ast
tree = ast.parse(tsrc)
guards = [n.lineno for n in tree.body
          if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
          and getattr(n.test.left, "id", "") == "__name__"]
if guards:
    late = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.lineno > guards[0]]
    if late:
        fails.append(f"defs after the __main__ guard, will NameError at runtime: {late}")

for sh in (WORKER, "weekend.sh", "night.sh"):
    if subprocess.run(["bash", "-n", sh]).returncode != 0:
        fails.append(f"{sh}: bash -n failed")

# night.sh builds its GPU submission inside a --wrap string array, so a correctly-quoted
# night.sh can still emit a BROKEN body. weekend.sh does not use that construct, which is why
# it passed while night.sh was broken for hours. Dry-run both and require a clean exit.
for sh in ("weekend.sh", "night.sh"):
    r = subprocess.run(["./" + sh, "--dry-run"], capture_output=True, text=True)
    if r.returncode != 0 or "command not found" in (r.stdout + r.stderr):
        fails.append(f"{sh} --dry-run failed: {(r.stderr or r.stdout).strip().splitlines()[-1:] }")

# The checkpoint integrity contract is the one thing whose failure is UNRECOVERABLE: latest-only
# retention means a bad commit deletes the last good checkpoint. Execute the gates, don't just
# read them. ~10s on CPU, and it covers continue_pretrain.py too (the phase A/B trainer, which
# had NO gates at all until 2026-08-14).
#
# test_loader_resume.py guards the other silent-corruption path: ~576 chained slices all
# resume, and a resume that restarts a cursor or overlaps two ranks trains on a fraction of
# the corpus with no crash and no loss spike to give it away.
for _test in ("exp/test_ckpt_safety.py", "exp/test_loader_resume.py", "test_a45_arch.py"):
    _t = subprocess.run([sys.executable, _test], capture_output=True, text=True)
    if _t.returncode != 0:
        tail = (_t.stdout + _t.stderr).strip().splitlines()[-6:]
        fails.append(f"{_test} FAILED:\n      " + "\n      ".join(tail))

print("preflight: run_full_training.sh + pretrain.py")
print("\n".join(notes))
print(f"  chunk={chunk} stride={stride} stride94={stride94} A45=True guard-ordering=ok")
if fails:
    print("\nFAIL:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("\nPASS — safe to launch")
