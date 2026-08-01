#!/usr/bin/env python3
"""build_pretrain_mix.py -- Argonne-3.5 base pretraining corpus: FineWeb-dominant ~85/15.

Interleaves FineWeb CC-MAIN-2025-21 (flat llm.c .bin, ~55.21B tok) with FineMath-4plus
(docbin: 64 HEADERLESS uint32 shards, ~10.10B tok) into ONE flat llm.c .bin
(magic 20240801, uint32) that pretrain.py's DataLoader reads densely.

The natural token ratio is 55.21 : 10.10 = 84.5 / 15.5, i.e. already ~85/15, so BOTH
sources are consumed IN FULL and simply globally interleaved at CHUNK granularity with a
deterministic proportional schedule -- the two sources are mixed throughout (never trained
sequentially, which would re-cause the catastrophic forgetting of §13) while staying
FineWeb-dominant. A block_size (1024) training window straddles a chunk boundary only rarely
(<=1024/chunk) and harmlessly.

Output: <out_bin> (+ a copy of the tokenizer alongside it, if --tokenizer_src is given).
Header format matches preprocess_data.py: 256 int32, [0]=MAGIC, token count at byte offset 8.
"""
import os
import sys
import time
import json
import argparse
import shutil
import numpy as np

MAGIC = 20240801
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4  # 1024


def fineweb_stream(path):
    """FineWeb flat llm.c .bin: 1024-byte header then uint32 tokens."""
    with open(path, "rb") as f:
        magic = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    if magic != MAGIC:
        raise ValueError(f"FineWeb bin {path}: bad magic {magic} (expected {MAGIC})")
    return np.memmap(path, dtype=np.uint32, mode="r", offset=HEADER_BYTES)


def finemath_shards(manifest_path):
    """FineMath docbin shards are HEADERLESS uint32 token streams (offset 0)."""
    m = json.load(open(manifest_path))
    tid = m["tokenized_dir"]
    paths = []
    for it in m["files"]:
        if int(it["docs_kept"]) <= 0:
            continue
        p = os.path.join(tid, it["bin_path"])
        if not os.path.exists(p):
            raise FileNotFoundError(f"FineMath shard missing: {p}")
        paths.append(p)
    mms = [np.memmap(p, dtype=np.uint32, mode="r") for p in paths]
    return mms, [len(x) for x in mms], m.get("tokenizer_path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fineweb_bin", default="/project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21-binary/train.bin")
    ap.add_argument("--finemath_manifest", default="/project/rcc/youzhi/data/finemath/finemath-4plus_qwen3_docbin_manifest.json")
    ap.add_argument("--out_bin", default="/project/rcc/youzhi/data/pretrain_mix_fw_fm_85_15/train.bin")
    ap.add_argument("--chunk", type=int, default=262144, help="interleave granularity in tokens (default 256K)")
    ap.add_argument("--max_fineweb_tokens", type=int, default=0, help="0 = use all fineweb")
    ap.add_argument("--max_finemath_tokens", type=int, default=0, help="0 = use all finemath")
    ap.add_argument("--tokenizer_src", default="/project/rcc/youzhi/data/fineweb/data/CC-MAIN-2025-21-binary/tokenizer", help="tokenizer dir to copy next to out_bin (\"\" to skip)")
    ap.add_argument("--smoke_total", type=int, default=0, help="if >0, write only ~this many total tokens (scaled ~85/15) for validation")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_bin), exist_ok=True)

    fw = fineweb_stream(args.fineweb_bin)
    fw_total = len(fw) if not args.max_fineweb_tokens else min(len(fw), args.max_fineweb_tokens)

    fm_mms, fm_lens, fm_tok_path = finemath_shards(args.finemath_manifest)
    fm_total_all = int(sum(fm_lens))
    fm_total = fm_total_all if not args.max_finemath_tokens else min(fm_total_all, args.max_finemath_tokens)

    if args.smoke_total:
        scale = args.smoke_total / (fw_total + fm_total)
        fw_total = max(args.chunk, int(fw_total * scale))
        fm_total = max(args.chunk, int(fm_total * scale))

    total = fw_total + fm_total
    p_fw = fw_total / total
    C = args.chunk
    print(f"[mix] fineweb={fw_total:,}  finemath={fm_total:,}  total={total:,} ({total/1e9:.3f}B)  "
          f"fineweb_frac={p_fw:.4f} finemath_frac={1-p_fw:.4f}  chunk={C}", flush=True)
    print(f"[mix] fineweb bin={args.fineweb_bin}\n[mix] finemath manifest tokenizer={fm_tok_path}\n[mix] out={args.out_bin}", flush=True)

    fw_pos = 0
    fm_idx = 0
    fm_off = 0
    fm_emitted = 0

    def next_fw(n):
        nonlocal fw_pos
        n = min(n, fw_total - fw_pos)
        if n <= 0:
            return None
        buf = np.asarray(fw[fw_pos:fw_pos + n])
        fw_pos += n
        return buf

    def next_fm(n):
        nonlocal fm_idx, fm_off, fm_emitted
        n = min(n, fm_total - fm_emitted)
        if n <= 0:
            return None
        parts = []
        need = n
        while need > 0 and fm_idx < len(fm_mms):
            avail = fm_lens[fm_idx] - fm_off
            take = min(avail, need)
            if take > 0:
                parts.append(np.asarray(fm_mms[fm_idx][fm_off:fm_off + take]))
                fm_off += take
                need -= take
            if fm_off >= fm_lens[fm_idx]:
                fm_idx += 1
                fm_off = 0
        buf = parts[0] if len(parts) == 1 else np.concatenate(parts)
        fm_emitted += len(buf)
        return buf

    t0 = time.time()
    written = 0
    fw_written = 0
    fm_written = 0
    acc = 0.0
    next_report = 0
    with open(args.out_bin, "wb") as f:
        header = np.zeros(HEADER_INTS, dtype=np.int32)
        header[0] = MAGIC
        f.write(header.tobytes())
        while fw_pos < fw_total or fm_emitted < fm_total:
            fw_left = fw_pos < fw_total
            fm_left = fm_emitted < fm_total
            if fw_left and fm_left:
                acc += p_fw
                if acc >= 1.0:
                    use_fw = True
                    acc -= 1.0
                else:
                    use_fw = False
            else:
                use_fw = fw_left
            buf = next_fw(C) if use_fw else next_fm(C)
            if buf is None or len(buf) == 0:
                continue
            f.write(buf.astype(np.uint32, copy=False).tobytes())
            n = len(buf)
            written += n
            if use_fw:
                fw_written += n
            else:
                fm_written += n
            if written >= next_report:
                el = time.time() - t0
                rate = written / el if el > 0 else 0
                print(f"[mix] {written/1e9:6.2f}B / {total/1e9:.2f}B ({100*written/total:5.1f}%)  "
                      f"fw={fw_written/1e9:.2f}B fm={fm_written/1e9:.2f}B  {rate/1e6:.1f}M tok/s", flush=True)
                next_report += 5_000_000_000  # every 5B tokens

    # backfill the token count into the header (byte offset 8: low32, high32)
    with open(args.out_bin, "r+b") as f:
        f.seek(8)
        f.write(np.array([written & 0xFFFFFFFF], dtype=np.uint32).tobytes())
        f.write(np.array([written >> 32], dtype=np.uint32).tobytes())

    if args.tokenizer_src and os.path.isdir(args.tokenizer_src):
        dst = os.path.join(os.path.dirname(args.out_bin), "tokenizer")
        if os.path.abspath(dst) != os.path.abspath(args.tokenizer_src):
            shutil.copytree(args.tokenizer_src, dst, dirs_exist_ok=True)

    el = time.time() - t0
    gb = os.path.getsize(args.out_bin) / (1024**3)
    print(f"\n[mix] DONE: wrote {written:,} tokens ({written/1e9:.3f}B) -> {args.out_bin} ({gb:.2f} GB)")
    print(f"[mix] actual fineweb={fw_written:,} ({100*fw_written/max(1,written):.2f}%)  "
          f"finemath={fm_written:,} ({100*fm_written/max(1,written):.2f}%)")
    print(f"[mix] time={el/60:.1f} min  rate={written/max(1e-9,el)/1e6:.1f}M tok/s")


if __name__ == "__main__":
    main()
