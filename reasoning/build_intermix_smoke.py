"""
Build a small INTERMIX doc-manifest for a smoke test (§13 plan):
  broad general (FineWeb, the pretrain corpus) + math (FineMath), 50:50 by DOC
  count, in the doc-bin format midtraining.py's DocManifestDataLoader consumes.

Why: §12/§13 showed pure-diet midtraining forgets. The proposed fix is to inject
math WITH general replay from the healthy pretrain distribution. This builds the
mixed corpus so we can midtrain argonne-3.0-base on it and re-probe both axes.

Mechanics:
- FineMath is already doc-bin (packed ~13568-tok docs). We reference N of its
  shards directly (absolute paths).
- FineWeb is a flat uint32 token stream (the pretrain .bin). We carve a slice into
  fixed-length "docs" (DOC_LEN tokens) and write one doc-bin shard + .lengths.npy.
- The DocManifestDataLoader takes ONE T-token window per doc per epoch, so the
  effective token mix is set by DOC COUNT -> we match FineWeb doc count to the
  chosen FineMath shards' total docs for ~50:50.
- Merged manifest uses absolute bin/lengths paths (tokenized_dir="/" so
  os.path.join is a no-op), letting it draw from two different tokenized_dirs.
"""

import json
import os
import numpy as np

VOCAB = 151669
DOC_LEN = 2050          # > block_size(2048)+1 so every FineWeb doc yields a T+1 window
NUM_FINEMATH_SHARDS = 6 # ~6 * per-shard docs of packed math

FINEWEB_BIN = "/project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21-binary/train.bin"
FINEMATH_MANIFEST = "/project/rcc/youzhi/data/finemath/finemath-4plus_qwen3_docbin_manifest.json"
OUT_DIR = "/project/rcc/youzhi/data/intermix_smoke"
FW_SHARD = os.path.join(OUT_DIR, "fineweb_slice.bin")
FW_LENGTHS = os.path.join(OUT_DIR, "fineweb_slice.lengths.npy")
OUT_MANIFEST = os.path.join(OUT_DIR, "intermix_smoke_manifest.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ---- FineMath: pick N shards, resolve absolute paths + doc counts ----
with open(FINEMATH_MANIFEST) as f:
    fm = json.load(f)
fm_dir = fm["tokenized_dir"]
fm_files = [x for x in fm["files"] if int(x["docs_kept"]) > 0][:NUM_FINEMATH_SHARDS]
fm_entries = []
fm_docs_total = 0
for x in fm_files:
    docs = int(x["docs_kept"])
    fm_docs_total += docs
    fm_entries.append({
        "bin_path": os.path.join(fm_dir, x["bin_path"]),
        "lengths_path": os.path.join(fm_dir, x["lengths_path"]),
        "docs_kept": docs,
        "source_relpath": "finemath/" + x["source_relpath"],
    })
print(f"FineMath: {len(fm_entries)} shards, {fm_docs_total:,} docs (packed ~13568-tok each)")

# ---- FineWeb: carve a slice into DOC_LEN docs, matching FineMath doc count ----
fw_docs = fm_docs_total                       # 50:50 by doc count
fw_tokens_needed = fw_docs * DOC_LEN
# The pretrain .bin has a 1024-byte header (256 int32), then uint32 tokens
# (see pretrain.py: np.memmap(..., dtype=np.uint32, offset=256*4)).
HEADER_BYTES = 256 * 4
mm = np.memmap(FINEWEB_BIN, dtype=np.uint32, mode="r", offset=HEADER_BYTES)
assert len(mm) >= fw_tokens_needed, f"FineWeb bin too small: have {len(mm):,}, need {fw_tokens_needed:,}"
head = np.asarray(mm[:1000], dtype=np.int64)
assert head.min() >= 0 and head.max() < VOCAB, f"FineWeb dtype/vocab check failed: max token {head.max()} (expected <{VOCAB}); is it uint32?"
print(f"FineWeb: carving {fw_docs:,} docs x {DOC_LEN} = {fw_tokens_needed:,} tokens from {FINEWEB_BIN}")

slice_tokens = np.asarray(mm[:fw_tokens_needed], dtype=np.uint32)
slice_tokens.tofile(FW_SHARD)                 # flat uint32, docs are contiguous DOC_LEN blocks
np.save(FW_LENGTHS, np.full(fw_docs, DOC_LEN, dtype=np.int64))
del mm, slice_tokens
print(f"  wrote {FW_SHARD} + {FW_LENGTHS}")

# ---- Merged manifest ----
fw_entry = {
    "bin_path": FW_SHARD,
    "lengths_path": FW_LENGTHS,
    "docs_kept": fw_docs,
    "source_relpath": "fineweb_slice",
}
total_tokens = fw_tokens_needed + sum(  # rough (finemath raw token totals not needed by loader)
    0 for _ in fm_entries
)
manifest = {
    "tokenized_dir": "/",                 # entries are absolute; join is a no-op
    "qwen_tokens_kept": int(fw_tokens_needed),  # reporting only; loader uses docs*T
    "files": [fw_entry] + fm_entries,
}
with open(OUT_MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nMerged manifest -> {OUT_MANIFEST}")
print(f"  files: 1 FineWeb ({fw_docs:,} docs) + {len(fm_entries)} FineMath ({fm_docs_total:,} docs)")
print(f"  doc balance: {fw_docs:,} general : {fm_docs_total:,} math  (~50:50 by doc => ~50:50 tokens trained)")
print("DONE")
