#!/usr/bin/env python
"""Filter a doc-aware docbin to its LONG documents only, preserving the exact same on-disk
contract (`*.bin` raw uint32 tokens + `*.lengths.npy` + `*.meta.json`), so that
`build_reasoning_corpus.py flatten` can consume the result unchanged.

WHY THIS EXISTS (INFRA M+314). build_reasoning_corpus.py's `min_doc_tokens` is a TOKENIZE-time
filter: `cmd_tokenize` applies it while rendering raw text, and `cmd_flatten` never reads it --
flatten globs `$RC_OUT_ROOT/<name>/*.bin` and memmaps the token streams densely, with no per-doc
awareness at all. So for a source that is ALREADY tokenized (the 28.6B-token proof-pile-2 arxiv
docbin) setting `min_doc_tokens=13568` in the registry does NOTHING, and phase B would extend
context over a stream whose windows mostly straddle document boundaries. That is the failure the
phase-B data gate warns about in as many words: "would be 'extending context' over padding and
would look like it worked".

WHY 2x THE BLOCK SIZE IS THE DEFAULT, measured not assumed (exact, from the lengths arrays alone;
"single-doc windows" = P(a uniformly chosen block-sized window lies inside ONE document)):
    ALL docs           1,542,673 docs  28.633B tok   41.6%   median 13,668
    docs >= 13,568       777,690 docs  22.459B tok   53.0%   median 22,417
    docs >= 27,136       274,362 docs  12.860B tok   71.1%   median 37,624
    docs >= 54,272        56,171 docs   4.925B tok   84.5%   median 70,646
Filtering at 1x the block size barely helps (41.6 -> 53.0%) because a doc of exactly one window
still straddles as soon as the stream offset is not aligned to it. 2x buys 71.1% and still leaves
12.86B tokens, ~2x phase B's ~6.0B budget. 4x would be purer but is UNDER budget, so it is not a
candidate. `--min_tokens` is a parameter, not a constant, because that trade is the arguable part.
"""
import argparse, glob, json, os, sys
import numpy as np

CHUNK = 1 << 24   # 16M tokens = 64 MB per write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="dir holding *.bin + *.lengths.npy")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_tokens", type=int, default=27136, help="keep docs with >= this many tokens")
    p.add_argument("--tier", default="longctx")
    p.add_argument("--source", default="longctx_arxiv")
    p.add_argument("--budget_tokens", type=int, default=0, help="0 = record the full kept total")
    a = p.parse_args()

    bins = sorted(b for b in glob.glob(os.path.join(a.in_dir, "*.bin")) if not b.endswith(".tmp"))
    if not bins:
        sys.exit(f"no *.bin under {a.in_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    files, tot_kept, tot_docs, tot_seen = [], 0, 0, 0

    for i, b in enumerate(bins):
        lp = b[:-4] + ".lengths.npy"
        L = np.load(lp).astype(np.int64)
        sz = os.path.getsize(b)
        # VALIDATE THE PARSE, every shard: these docbins are RAW uint32 with NO header (flatten
        # memmaps them with no offset). If that is ever false the offsets below are garbage and the
        # output would be silently misaligned tokens, which no downstream check would catch.
        if sz // 4 != L.sum() or sz % 4:
            sys.exit(f"{os.path.basename(b)}: {sz} bytes != 4*{L.sum()} tokens -- layout changed, refusing")
        ends = np.cumsum(L)
        starts = ends - L
        keep = np.flatnonzero(L >= a.min_tokens)
        tot_seen += len(L)
        name = f"shard_{i:05d}"
        ob = os.path.join(a.out_dir, name + ".bin")
        if len(keep) == 0:
            print(f"[{name}] {os.path.basename(b)}: 0 of {len(L):,} docs kept -- skipped", flush=True)
            continue
        src = np.memmap(b, dtype=np.uint32, mode="r")
        wrote = 0
        with open(ob + ".tmp", "wb") as fh:
            for d in keep:
                s, e = int(starts[d]), int(ends[d])
                while s < e:                       # bounded write so RSS stays flat on a 10M-token doc
                    t = min(e - s, CHUNK)
                    fh.write(src[s:s + t].tobytes())
                    s += t
                    wrote += t
        del src
        os.replace(ob + ".tmp", ob)                # atomic: a partial shard must never be globbable
        kl = L[keep]
        np.save(os.path.join(a.out_dir, name + ".lengths.npy"), kl.astype(np.uint32))
        assert os.path.getsize(ob) == 4 * int(kl.sum()) == 4 * wrote, "written bytes != kept tokens"
        meta = dict(source_relpath=os.path.basename(b), status="ok",
                    bin_path=ob, lengths_path=os.path.join(a.out_dir, name + ".lengths.npy"),
                    docs_seen=int(len(L)), docs_kept=int(len(kl)),
                    qwen_tokens_kept=int(kl.sum()), min_doc_tokens=a.min_tokens,
                    filtered_from=b)
        json.dump(meta, open(os.path.join(a.out_dir, name + ".meta.json"), "w"), indent=1)
        files.append(meta)
        tot_kept += int(kl.sum()); tot_docs += int(len(kl))
        print(f"[{name}] {os.path.basename(b)}: {len(kl):,}/{len(L):,} docs, "
              f"{kl.sum()/1e9:.4f}B tok (cum {tot_kept/1e9:.3f}B)", flush=True)

    json.dump(dict(source=a.source, tier=a.tier,
                   budget_tokens=a.budget_tokens or tot_kept, scale=1.0,
                   min_doc_tokens=a.min_tokens, docs_seen=tot_seen, docs_kept=tot_docs,
                   qwen_tokens_kept=tot_kept, files=files),
              open(os.path.join(a.out_dir, "_source_manifest.json"), "w"), indent=1)
    print(f"\n[done] {tot_docs:,}/{tot_seen:,} docs, {tot_kept/1e9:.3f}B tokens -> {a.out_dir}")


if __name__ == "__main__":
    main()
