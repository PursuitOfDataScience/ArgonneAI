#!/usr/bin/env python
"""Build the phase-C long-context corpus: long arXiv documents + replay, interleaved.

NOTHING IS RE-TOKENIZED. The owner tokenized proof-pile-2 arXiv on 2026-07-29 into
`proof_pile2_arxiv_qwen3_docbin` (28.6B tok / 100 shards / 1.54M docs); this script only SELECTS
from it and copies bytes. The other two sources are already-flat bins from earlier phases. The whole
job is a memmap copy and runs in a few minutes.

WHY A MIX AND NOT JUST ARXIV. Phase B trained on `reasoning_midtrain_flat.bin`, which is itself a
10-source blend (github_code, openmath, mixture_of_thoughts, fineweb_edu, instruct_chat, ...). A
pure-arXiv phase C would change the DATA COMPOSITION at the same time as the context length, and
this project has already lost a model that way: the §12 incident, where a math-heavy stage broke
general knowledge badly enough that SFT+DPO could not recover it. For a4 specifically the base-gate
probe found GENERAL to be the binding axis (12-13/15, flat over 11.6B tokens), so a stage with no
general data pushes on exactly the weakest axis. Hence 50% long arXiv + 50% replay.

WHY min_doc_tokens == the training block. The long portion draws only documents at least as long as
the context window, so a window generally sits INSIDE one document rather than spanning a boundary
(there is no document-boundary attention masking in the flat stream). Supply is not a constraint:
10.25B tokens live in docs >= 32,768, and the long portion needs ~3B.

WHY A COARSE INTERLEAVE CHUNK. Sources must be mixed THROUGHOUT the run, never concatenated
sequentially, or the tail of training sees one distribution and re-introduces forgetting. But the
chunk must also be much larger than the context window, otherwise a source switch would chop long
arXiv documents in half. 8Mi tokens is ~256x the window and still gives ~700 alternations over 6B.

Output format is byte-identical to `build_reasoning_corpus.py flatten` and to what phase B consumed:
magic 20240801, 256*int32 header, 64-bit token count at hdr[2] (low) / hdr[3] (high), then uint32.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

MAGIC, HDR = 20240801, 256


class FlatSource:
    """An existing flat .bin (256*int32 header, then uint32 tokens)."""

    def __init__(self, name, path, weight):
        self.name, self.weight = name, weight
        self.mm = np.memmap(path, dtype=np.uint32, mode="r", offset=HDR * 4)
        self.n = len(self.mm)
        self.pos = 0

    def avail(self):
        return self.n - self.pos

    def take(self, k):
        k = min(k, self.avail())
        if k <= 0:
            return None
        buf = np.asarray(self.mm[self.pos:self.pos + k])
        self.pos += k
        return buf


class DocbinLongSource:
    """Documents >= min_doc_tokens from a docbin dir (<stem>.bin + <stem>.lengths.npy)."""

    def __init__(self, name, src_dir, min_doc_tokens, weight, glob_pat="*.bin"):
        self.name, self.weight = name, weight
        self.docs = []  # (shard_path, start_tok, length)
        total = 0
        for b in sorted(glob.glob(os.path.join(src_dir, glob_pat))):
            lp = b[:-4] + ".lengths.npy"
            if not os.path.exists(lp):
                continue
            L = np.load(lp).astype(np.int64)
            off = np.concatenate([[0], np.cumsum(L)[:-1]])
            keep = np.nonzero(L >= min_doc_tokens)[0]
            for i in keep:
                self.docs.append((b, int(off[i]), int(L[i])))
            total += int(L[keep].sum())
        self.n = total
        self.di = 0
        self.doff = 0
        self._cache_path, self._cache_mm = None, None
        print(f"[{name}] {len(self.docs):,} docs >= {min_doc_tokens:,} tok -> {total/1e9:.2f}B available")

    def _mm(self, path):
        if path != self._cache_path:
            self._cache_mm = np.memmap(path, dtype=np.uint32, mode="r")
            self._cache_path = path
        return self._cache_mm

    def avail(self):
        return self.n - self.emitted() if False else None  # not needed; take() reports exhaustion

    def take(self, k):
        out, need = [], k
        while need > 0 and self.di < len(self.docs):
            path, start, length = self.docs[self.di]
            rem = length - self.doff
            t = min(rem, need)
            mm = self._mm(path)
            out.append(np.asarray(mm[start + self.doff:start + self.doff + t]))
            self.doff += t
            need -= t
            if self.doff >= length:
                self.di += 1
                self.doff = 0
        if not out:
            return None
        return out[0] if len(out) == 1 else np.concatenate(out)


def write_bin(path, sources, target, chunk):
    """Weighted-fair chunk interleave: each round, serve the source furthest behind its quota."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wsum = sum(s.weight for s in sources)
    quota = {s.name: target * s.weight / wsum for s in sources}
    got = {s.name: 0 for s in sources}
    written, t0, nxt = 0, time.time(), 0
    with open(path, "wb") as f:
        f.write(np.zeros(HDR, dtype=np.int32).tobytes())
        while written < target:
            live = [s for s in sources if got[s.name] < quota[s.name]]
            if not live:
                break
            s = min(live, key=lambda s: got[s.name] / quota[s.name])
            buf = s.take(min(chunk, int(quota[s.name] - got[s.name]), target - written))
            if buf is None or len(buf) == 0:
                quota[s.name] = got[s.name]  # exhausted -> stop scheduling it
                continue
            f.write(buf.astype(np.uint32, copy=False).tobytes())
            written += len(buf)
            got[s.name] += len(buf)
            if written >= nxt:
                print(f"  {written/1e9:5.2f}B / {target/1e9:.2f}B  "
                      f"{written/max(1e-9, time.time()-t0)/1e6:5.1f}M tok/s", flush=True)
                nxt += 1_000_000_000
    with open(path, "r+b") as f:
        f.seek(0); f.write(np.array([MAGIC], dtype=np.int32).tobytes())
        f.seek(8)
        f.write(np.array([written & 0xFFFFFFFF], dtype=np.uint32).tobytes())
        f.write(np.array([written >> 32], dtype=np.uint32).tobytes())
    return written, got


def verify(path, expect):
    hdr = np.fromfile(path, dtype=np.uint32, count=4)
    mm = np.memmap(path, dtype=np.uint32, mode="r", offset=HDR * 4)
    count = int(hdr[2]) | (int(hdr[3]) << 32)
    mx = int(np.asarray(mm[:20_000_000]).max())
    ok = hdr[0] == MAGIC and count == len(mm) == expect and mx < 151669
    print(f"[verify] {os.path.basename(path)}: magic={hdr[0]} count={count:,} memmap={len(mm):,} "
          f"max_id={mx} -> {'OK' if ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv_dir", default="/project/rcc/youzhi/data/proof_pile2_arxiv_qwen3_docbin/data")
    ap.add_argument("--arxiv_glob", default="arXiv_0[0-8]*.bin",
                    help="TRAINING shards only. Shards arXiv_09* are RESERVED as the held-out set "
                         "for the position-bucketed NLL gate: that eval draws its windows from this "
                         "same docbin, so training on every shard would silently contaminate the one "
                         "measurement that says whether the extension worked.")
    ap.add_argument("--replay_reasoning", default="/project/rcc/youzhi/data/reasoning_anneal/reasoning_midtrain_flat.bin")
    ap.add_argument("--replay_edu", default="/project/rcc/youzhi/data/argonne4_pretrain/edu_flat.bin")
    ap.add_argument("--out_bin", required=True)
    ap.add_argument("--val_bin", default="")
    ap.add_argument("--block", type=int, default=32768, help="training window; sets the long-doc filter")
    ap.add_argument("--target_tokens", type=float, default=6.0e9)
    ap.add_argument("--val_tokens", type=float, default=40e6)
    ap.add_argument("--w_long", type=int, default=50)
    ap.add_argument("--w_reasoning", type=int, default=25)
    ap.add_argument("--w_edu", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=8 * 1024 * 1024)
    args = ap.parse_args()

    def build(sources, path, target, tag):
        print(f"\n=== {tag}: {target/1e9:.3f}B tokens -> {path} ===")
        n, got = write_bin(path, sources, int(target), args.chunk)
        gb = os.path.getsize(path) / 1024 ** 3
        print(f"[done] {n:,} tokens ({n/1e9:.3f}B, {gb:.2f} GB)")
        for k, v in got.items():
            print(f"    {k:<22} {v/1e9:6.3f}B  ({100*v/max(1,n):4.1f}%)")
        return n

    def mk():
        return [
            DocbinLongSource("arxiv_long", args.arxiv_dir, args.block, args.w_long, args.arxiv_glob),
            FlatSource("replay_reasoning", args.replay_reasoning, args.w_reasoning),
            FlatSource("replay_edu", args.replay_edu, args.w_edu),
        ]

    # Validation is carved FIRST so it is disjoint from training: each source's cursor advances past
    # the val slice before the main build starts on that same source object.
    srcs = mk()
    if args.val_bin:
        nv = build(srcs, args.val_bin, args.val_tokens, "HELD-OUT VAL (carved first, disjoint)")
        verify(args.val_bin, nv)
    nm = build(srcs, args.out_bin, args.target_tokens, "MAIN")
    ok = verify(args.out_bin, nm)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
