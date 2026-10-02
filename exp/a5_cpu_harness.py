"""Run a trainer's REAL main() on CPU at a toy width. Test rig for exp/test_a5_trainer.py.

Nothing under test is mocked: the optimizer build, the KD loss, both DDP micro-step branches, the
ZeRO shard-merge save and the resume all execute exactly as in production. The shims below replace
only what a GPU-less node lacks, and each one is placement or telemetry, never math:
  * Tensor.pin_memory needs a CUDA driver (host-memory placement only);
  * setup_distributed() calls torch.cuda.set_device(local_rank);
  * the [HBM] log line divides by get_device_properties().total_memory;
  * DDP insists device_ids=None for a CPU module, and DEVICE is forced to "cpu" on every rank.

    python exp/a5_cpu_harness.py <trainer.py> <trainer args...>                      # 1 process
    torchrun --nproc_per_node=2 exp/a5_cpu_harness.py <trainer.py> <trainer args...>  # 2 ranks, gloo

Two trainer layouts are supported:
  * pretrain.py parses argv at MODULE level: it is imported, A5_TOY_ARCH='{"HIDDEN_SIZE": 64, ...}'
    overrides its architecture constants (read inside main()), and main() is called.
  * continue_pretrain.py parses argv, sets DEVICE and inits NCCL inside its __main__ block: it is run
    as __main__ with three exact, count-checked substitutions (gloo, DEVICE="cpu", DDP device_ids=None).
    Its architecture comes from the checkpoint it resumes (by design), so A5_TOY_ARCH does not apply;
    on CPU its hardcoded autocast("cuda") disables itself, so it trains in fp32 here.
"""
import importlib.util
import json
import os
import sys

import torch


class _Props:
    total_memory = 1 << 30
    name = "cpu"


torch.Tensor.pin_memory = lambda self, *a, **k: self
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.get_device_properties = lambda *a, **k: _Props()
torch.cuda.max_memory_reserved = lambda *a, **k: 0

trainer = os.path.abspath(sys.argv[1])
sys.argv = [trainer] + sys.argv[2:]
src = open(trainer).read()

if "\nargs = parser.parse_args()" in src:                     # pretrain.py layout
    spec = importlib.util.spec_from_file_location("trainer_under_test", trainer)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)      # parses argv and inits the process group (gloo on CPU)
    for key, value in json.loads(os.environ.get("A5_TOY_ARCH", "{}")).items():
        if not hasattr(mod, key):
            raise SystemExit(f"harness: {trainer} has no module constant {key}")
        setattr(mod, key, value)
    mod.DEVICE = "cpu"
    _DDP = mod.DDP
    mod.DDP = lambda module, device_ids=None, **kw: _DDP(module, device_ids=None, **kw)
    mod.main()
else:                                                         # continue_pretrain.py layout
    for old, new in (('dist.init_process_group("nccl")', 'dist.init_process_group("gloo")'),
                     ('DEVICE = f"cuda:{LOCAL_RANK}"', 'DEVICE = "cpu"'),
                     ('model = DDP(model, device_ids=[LOCAL_RANK], gradient_as_bucket_view=True)',
                      'model = DDP(model, device_ids=None, gradient_as_bucket_view=True)')):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"harness: expected exactly one {old!r} in {trainer}, found {n}")
        src = src.replace(old, new)
    exec(compile(src, trainer, "exec"), {"__name__": "__main__", "__file__": trainer})
