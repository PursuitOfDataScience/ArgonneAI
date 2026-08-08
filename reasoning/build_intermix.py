"""
Build the PRODUCTION intermix doc-manifest for the single-phase midtraining fix
(§13): broad general (FineWeb, the pretrain corpus) + math (FineMath), mixed by
DOC count, in the doc-bin format midtraining.py's DocManifestDataLoader consumes.

This is the production-scale sibling of build_intermix_smoke.py. Differences:
- DOC_LEN >= block_size+1 (default 13570 for the 13568 production block) so every
  FineWeb doc yields a full T+1 window at the production sequence length.
- Uses ALL FineMath shards by default (env NUM_FINEMATH_SHARDS=0 => all).
- GENERAL_RATIO controls the general:math token split by DOC count (default 1.0 =
  50:50; e.g. 1.5 = 60:40 general-heavy, to lean harder against forgetting).

The DocManifestDataLoader takes ONE T-token window per doc per epoch, so the
effective token mix equals the DOC-count ratio. midtraining.sh must run this
manifest with DOC_SHUFFLE=1 (weekend.sh/night.sh set DOC_SHUFFLE_OVERRIDE=1) so
the two sources interleave instead of training sequentially.

Env knobs: DOC_LEN, NUM_FINEMATH_SHARDS, GENERAL_RATIO, OUT_DIR.
"""

import json
import os
import numpy as np

VOCAB = 151669
HEADER_BYTES = 256 * 4  # pretrain .bin header (see pretrain.py: offset=256*4)

DOC_LEN = int(os.environ.get("DOC_LEN", "13570"))            # > production block_size(13568)+1
NUM_FINEMATH_SHARDS = int(os.environ.get("NUM_FINEMATH_SHARDS", "0"))  # 0 => all
GENERAL_RATIO = float(os.environ.get("GENERAL_RATIO", "1.0"))  # general_docs / math_docs
OUT_DIR = os.environ.get("OUT_DIR", "/project/rcc/youzhi/data/intermix")

FINEWEB_BIN = "/project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21-binary/train.bin"
FINEMATH_MANIFEST = "/project/rcc/youzhi/data/finemath/finemath-4plus_qwen3_docbin_manifest.json"
FW_SHARD = os.path.join(OUT_DIR, "fineweb_slice.bin")
FW_LENGTHS = os.path.join(OUT_DIR, "fineweb_slice.lengths.npy")
OUT_MANIFEST = os.path.join(OUT_DIR, "intermix_manifest.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ---- FineMath: resolve shards -> absolute paths + doc counts ----
with open(FINEMATH_MANIFEST) as f:
    fm = json.load(f)
fm_dir = fm["tokenized_dir"]
fm_files = [x for x in fm["files"] if int(x["docs_kept"]) > 0]
if NUM_FINEMATH_SHARDS > 0:
    fm_files = fm_files[:NUM_FINEMATH_SHARDS]
fm_entries, fm_docs_total = [], 0
for x in fm_files:
    docs = int(x["docs_kept"])
    fm_docs_total += docs
    fm_entries.append({
        "bin_path": os.path.join(fm_dir, x["bin_path"]),
        "lengths_path": os.path.join(fm_dir, x["lengths_path"]),
        "docs_kept": docs,
        "source_relpath": "finemath/" + x["source_relpath"],
    })
print(f"FineMath: {len(fm_entries)} shards, {fm_docs_total:,} docs")

# ---- FineWeb: carve GENERAL_RATIO * fm_docs docs of DOC_LEN tokens ----
fw_docs = int(round(fm_docs_total * GENERAL_RATIO))
fw_tokens_needed = fw_docs * DOC_LEN
mm = np.memmap(FINEWEB_BIN, dtype=np.uint32, mode="r", offset=HEADER_BYTES)
assert len(mm) >= fw_tokens_needed, f"FineWeb too small: have {len(mm):,}, need {fw_tokens_needed:,}"
head = np.asarray(mm[:1000], dtype=np.int64)
assert 0 <= head.min() and head.max() < VOCAB, f"FineWeb dtype/vocab check failed (max {head.max()})"
print(f"FineWeb: carving {fw_docs:,} docs x {DOC_LEN} = {fw_tokens_needed:,} tokens ({fw_tokens_needed*4/1e9:.1f} GB)")

# Stream the slice to disk in chunks (avoid a 28GB in-RAM copy).
CHUNK = 200_000_000
with open(FW_SHARD, "wb") as out:
    written = 0
    while written < fw_tokens_needed:
        n = min(CHUNK, fw_tokens_needed - written)
        np.asarray(mm[written:written + n], dtype=np.uint32).tofile(out)
        written += n
        print(f"  ... {written:,}/{fw_tokens_needed:,} tokens", flush=True)
np.save(FW_LENGTHS, np.full(fw_docs, DOC_LEN, dtype=np.int64))
del mm
print(f"  wrote {FW_SHARD} + {FW_LENGTHS}")

# ---- Merged manifest (absolute paths; tokenized_dir='/' makes join a no-op) ----
fw_entry = {"bin_path": FW_SHARD, "lengths_path": FW_LENGTHS, "docs_kept": fw_docs, "source_relpath": "fineweb_slice"}
manifest = {"tokenized_dir": "/", "qwen_tokens_kept": int(fw_tokens_needed), "files": [fw_entry] + fm_entries}
with open(OUT_MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nMerged manifest -> {OUT_MANIFEST}")
print(f"  {fw_docs:,} general docs : {fm_docs_total:,} math docs  "
      f"(~{fw_docs/(fw_docs+fm_docs_total)*100:.0f}:{fm_docs_total/(fw_docs+fm_docs_total)*100:.0f} by doc => by tokens trained)")
print("DONE")
