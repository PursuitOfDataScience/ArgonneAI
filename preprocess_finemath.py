#!/usr/bin/env python3
"""Tokenize FineMath into the SAME doc-aware docbin format midtraining.py consumes.

The midtraining loader (`DocManifestDataLoader`) slices ONE contiguous T+1 window
from within a SINGLE document, so every stored "document" must be >= block_size+1
(13569) tokens or it crashes. FineMath docs are short web pages, so we PACK many
source docs (EOS-separated) into >=16k-token super-documents — mirroring how the
16k-32k longmino pool is laid out.

Per source parquet -> one shard trio, matching the pool:
  <name>.bin          concatenated uint32 token ids of all super-docs
  <name>.lengths.npy  uint32 length of each super-doc
  <name>.meta.json    shard stats
Plus a top-level manifest with the keys DocManifestDataLoader reads
(tokenized_dir, qwen_tokens_kept, files[].{bin_path,lengths_path,docs_kept,source_relpath}).

EOS policy matches the pool: append tokenizer.eos_token_id after each source doc.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Globals per worker process (set in _init_worker).
_TOK = None
_EOS = None


def _init_worker(tokenizer_path):
    global _TOK, _EOS
    _TOK = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _EOS = _TOK.eos_token_id
    if _EOS is None:
        raise RuntimeError("tokenizer has no eos_token_id")


def _process_parquet(args):
    """Tokenize+pack one parquet file -> one (bin, lengths, meta) trio."""
    src_path, out_bin, out_lengths, out_meta, text_col, min_superdoc, row_batch = args
    rel = os.path.basename(src_path)

    # Resume: skip if the trio already exists and meta says done.
    if os.path.exists(out_bin) and os.path.exists(out_lengths) and os.path.exists(out_meta):
        try:
            m = json.load(open(out_meta))
            if m.get("status") == "done":
                return {"source_relpath": rel, "bin_path": os.path.basename(out_bin),
                        "lengths_path": os.path.basename(out_lengths),
                        "docs_kept": int(m["docs_kept"]),
                        "qwen_tokens_kept": int(m["qwen_tokens_kept"]),
                        "docs_seen": int(m.get("docs_seen", 0)), "status": "skipped_existing"}
        except Exception:
            pass

    pf = pq.ParquetFile(src_path)
    superdoc_lengths = []          # length of each emitted super-doc
    buf = []                       # current super-doc token buffer
    docs_seen = 0
    total_tokens = 0

    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    tmp_bin = out_bin + ".tmp"
    fbin = open(tmp_bin, "wb")

    def flush_superdoc():
        nonlocal buf, total_tokens
        if len(buf) >= min_superdoc:
            arr = np.asarray(buf, dtype=np.uint32)
            fbin.write(arr.tobytes())
            superdoc_lengths.append(len(buf))
            total_tokens += len(buf)
            buf = []
            return True
        return False

    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=[text_col])
        texts = table.column(text_col).to_pylist()
        # Batch-encode for speed.
        for i in range(0, len(texts), row_batch):
            chunk = [t for t in texts[i:i + row_batch] if t]
            if not chunk:
                continue
            encs = _TOK(chunk, add_special_tokens=False)["input_ids"]
            for ids in encs:
                docs_seen += 1
                buf.extend(ids)
                buf.append(_EOS)
                if len(buf) >= min_superdoc:
                    flush_superdoc()
    # Drop the trailing remainder if it's below the window requirement.
    fbin.close()
    os.replace(tmp_bin, out_bin)

    lengths_arr = np.asarray(superdoc_lengths, dtype=np.uint32)
    np.save(out_lengths, lengths_arr)

    meta = {
        "source_relpath": rel,
        "docs_seen": docs_seen,
        "docs_kept": int(len(superdoc_lengths)),          # = number of super-docs
        "qwen_tokens_kept": int(total_tokens),
        "min_superdoc_tokens": int(min_superdoc),
        "min_kept_length": int(lengths_arr.min()) if len(lengths_arr) else 0,
        "max_kept_length": int(lengths_arr.max()) if len(lengths_arr) else 0,
        "mean_kept_length": float(lengths_arr.mean()) if len(lengths_arr) else 0.0,
        "status": "done",
    }
    json.dump(meta, open(out_meta, "w"), indent=2)
    return {"source_relpath": rel, "bin_path": os.path.basename(out_bin),
            "lengths_path": os.path.basename(out_lengths),
            "docs_kept": int(len(superdoc_lengths)),
            "qwen_tokens_kept": int(total_tokens),
            "docs_seen": docs_seen, "status": "done"}


def main(a):
    src_dir = a.data_dir
    out_dir = a.output_dir
    os.makedirs(out_dir, exist_ok=True)
    parquets = sorted(f for f in os.listdir(src_dir) if f.endswith(".parquet"))
    if not parquets:
        raise SystemExit(f"no parquet files in {src_dir}")
    print(f"[finemath] {len(parquets)} parquet files; tokenizer={a.tokenizer_path}; "
          f"min_superdoc={a.min_superdoc}; workers={a.workers}", flush=True)

    tasks = []
    for p in parquets:
        stem = os.path.splitext(p)[0]
        tasks.append((
            os.path.join(src_dir, p),
            os.path.join(out_dir, stem + ".bin"),
            os.path.join(out_dir, stem + ".lengths.npy"),
            os.path.join(out_dir, stem + ".meta.json"),
            a.text_column, a.min_superdoc, a.row_batch,
        ))

    results = []
    with ProcessPoolExecutor(max_workers=a.workers,
                             initializer=_init_worker,
                             initargs=(a.tokenizer_path,)) as ex:
        futs = {ex.submit(_process_parquet, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[done] {r['source_relpath']}: {r['docs_kept']} super-docs, "
                  f"{r['qwen_tokens_kept']:,} tokens ({r['status']})", flush=True)

    results.sort(key=lambda r: r["source_relpath"])
    total_docs = sum(r["docs_kept"] for r in results)
    total_tokens = sum(r["qwen_tokens_kept"] for r in results)
    total_seen = sum(r["docs_seen"] for r in results)

    manifest = {
        "phase": "finemath_midtrain",
        "source_repo": "HuggingFaceTB/finemath",
        "source_config": os.path.basename(src_dir.rstrip("/")),
        "tokenizer_path": a.tokenizer_path,
        "tokenized_dir": out_dir,
        "selection_rule": f"pack EOS-separated source docs into super-docs >= {a.min_superdoc} tokens",
        "docs_seen": total_seen,
        "docs_kept": total_docs,
        "qwen_tokens_kept": int(total_tokens),
        "qwen_tokens_kept_billions": total_tokens / 1e9,
        "storage_format": {
            "bin_dtype": "uint32",
            "lengths_dtype": "uint32",
            "eos_policy": "append tokenizer.eos_token_id after each source doc",
            "packing": f"document-aware super-docs >= {a.min_superdoc} tokens; one parquet -> one bin/lengths/meta trio",
        },
        "files": results,
    }
    manifest_path = os.path.join(os.path.dirname(out_dir.rstrip("/")),
                                 os.path.basename(out_dir.rstrip("/")) + "_manifest.json")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"\n[finemath] TOTAL: {total_docs:,} super-docs, {total_tokens:,} tokens "
          f"({total_tokens/1e9:.2f}B)\n[finemath] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--data_dir", required=True, help="dir of FineMath parquet files")
    ap.add_argument("--output_dir", required=True, help="docbin output dir")
    ap.add_argument("--text_column", default="text")
    ap.add_argument("--min_superdoc", type=int, default=16384,
                    help="min tokens per packed super-doc (must exceed block_size; 13568 -> use >=13569)")
    ap.add_argument("--row_batch", type=int, default=256, help="rows per tokenizer batch call")
    ap.add_argument("--workers", type=int, default=16)
    main(ap.parse_args())
