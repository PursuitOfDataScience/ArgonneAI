"""Run pretrain.py's real main() on the GPU at another model shape, for a5 sizing.

pretrain.py parses argv at module level and reads its architecture constants inside main(), so a shape
is set by importing the module, overriding the constants from A5_ARCH (JSON), and calling main(). Nothing
else is touched: same optimizer build, KD loss, doc_mask ids, compile and logging as production.

    A5_ARCH='{"HIDDEN_SIZE": 3072, "NUM_LAYERS": 24, "NUM_HEADS": 12, "NUM_KV_HEADS": 2, "INTERMEDIATE_SIZE": 8448}' \
        python exp/a5_gpu_sizing.py pretrain.py <pretrain args...>
"""
import importlib.util
import json
import os
import sys

trainer = os.path.abspath(sys.argv[1])
sys.argv = [trainer] + sys.argv[2:]
spec = importlib.util.spec_from_file_location("trainer_under_test", trainer)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
for key, value in json.loads(os.environ.get("A5_ARCH", "{}")).items():
    if not hasattr(mod, key):
        raise SystemExit(f"sizing: {trainer} has no module constant {key}")
    setattr(mod, key, value)
print(f"[sizing] arch override: {os.environ.get('A5_ARCH', '{}')}", flush=True)
mod.main()
