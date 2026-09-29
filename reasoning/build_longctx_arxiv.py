#!/usr/bin/env python3
"""build_longctx_arxiv.py -- tokenize proof-pile-2 arXiv into the Qwen3 docbin contract.

WHY A SEPARATE SCRIPT: build_reasoning_corpus.py is driven by a hardcoded SOURCES table, so adding
this source would mean editing that tracked pipeline file. Owner directive 2026-07-29 was to get the
DATA ready but NOT wire it into the existing pretraining yet, so this writes to its own output dir and
touches no launcher, no config, and no existing corpus.

OUTPUT (identical contract to preprocess_finemath.py / build_reasoning_corpus.py, so
DocManifestDataLoader and `build_reasoning_corpus.py flatten` can consume it later unchanged):
  <out>/data/<stem>.bin           uint32 token ids, one EOS appended per document
  <out>/data/<stem>.lengths.npy   uint32 length of each document (INCLUDING its EOS)
  <out>/data/<stem>.meta.json     per-shard stats
  <out>_manifest.json             combined manifest (tokenized_dir + per-file bin/lengths paths)

DELIBERATE CHOICE -- NO LENGTH FILTER. The longmino pool pre-filtered to >=13,568 tokens at tokenize
time, which permanently discarded its short tail and hard-capped the bucket at 32k. Here every
document is kept and its exact length recorded in .lengths.npy, so ANY later length filter or budget
is a cheap selection over the lengths array rather than a re-tokenization. Measured input: 28.47B
Qwen3 tokens, 1.55M docs, median 14,620 / p90 36,770 / max 146,752.

Resumable: a shard whose .bin + .lengths.npy + .meta.json all exist is skipped, so a timed-out job
just needs resubmitting.
"""
import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SRC_DIR = "/project/rcc/youzhi/data/EleutherAI_proof-pile-2/arxiv/train"
OUT_DIR = "/project/rcc/youzhi/data/proof_pile2_arxiv_qwen3_docbin"
TOKENIZER = "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base"
EOS_ID = 151643          # <|endoftext|>, same as every other Argonne corpus
BATCH = 8                # docs per tokenizer call. Small ON PURPOSE: individual arXiv docs reach
                         # 327k tokens, and a Python int list costs ~28 B/token, so a big batch is
                         # what blew up the first attempt. 8 keeps a worker's live set well bounded.

_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    return _TOK


def do_shard(relpath):
    """Tokenize ONE .jsonl.zst shard -> (.bin, .lengths.npy, .meta.json). Returns the meta dict."""
    import zstandard
    src = os.path.join(SRC_DIR, relpath)
    stem = relpath[:-len(".jsonl.zst")] if relpath.endswith(".jsonl.zst") else relpath
    out_sub = os.path.join(OUT_DIR, "data")
    os.makedirs(out_sub, exist_ok=True)
    bin_p = os.path.join(out_sub, stem + ".bin")
    len_p = os.path.join(out_sub, stem + ".lengths.npy")
    met_p = os.path.join(out_sub, stem + ".meta.json")
    if os.path.exists(bin_p) and os.path.exists(len_p) and os.path.exists(met_p):
        try:
            m = json.load(open(met_p)); m["status"] = "skipped_existing"; return m
        except Exception:
            pass  # corrupt meta -> redo

    tok = _tok()
    t0 = time.time()
    lengths = []
    docs_seen = bad = 0
    batch = []

    # STREAM tokens straight to disk. The first version accumulated the whole shard in a list and
    # then np.concatenate'd it -- ~2 GB peak per worker on a 0.25B-token shard, x16 workers =
    # OUT_OF_MEMORY at 22.3 GiB (job 52782682). Writing per batch keeps a worker's live set to one
    # batch, which matters here because individual arXiv docs reach 327k tokens.
    fout = open(bin_p + ".tmp", "wb")

    def flush(batch):
        if not batch:
            return
        for ids in tok(batch, add_special_tokens=False).input_ids:
            ids.append(EOS_ID)
            np.asarray(ids, dtype=np.uint32).tofile(fout)
            lengths.append(len(ids))

    try:
        with open(src, "rb") as fh:
            rd = zstandard.ZstdDecompressor().stream_reader(fh)
            for line in io.TextIOWrapper(rd, encoding="utf-8"):
                try:
                    d = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                docs_seen += 1
                t = d.get("text") or d.get("content") or ""
                if not t:
                    continue
                batch.append(t)
                if len(batch) >= BATCH:
                    flush(batch); batch = []
        flush(batch)
    finally:
        fout.close()

    if not lengths:
        os.remove(bin_p + ".tmp")
        meta = dict(source_relpath=relpath, status="empty", docs_seen=docs_seen,
                    bad_json_lines=bad, docs_kept=0, qwen_tokens_kept=0)
        json.dump(meta, open(met_p, "w"), indent=1)
        return meta

    L = np.asarray(lengths, dtype=np.uint32)
    # rename last, so a killed job leaves no half-written shard the resume logic would trust
    np.save(len_p + ".tmp.npy", L)
    os.replace(bin_p + ".tmp", bin_p)
    os.replace(len_p + ".tmp.npy", len_p)

    meta = dict(
        source_relpath=relpath, status="ok",
        bin_path=os.path.relpath(bin_p, OUT_DIR), lengths_path=os.path.relpath(len_p, OUT_DIR),
        source_bytes=os.path.getsize(src), docs_seen=docs_seen, docs_kept=int(L.size),
        bad_json_lines=bad, min_tokens_filter=0,
        qwen_tokens_kept=int(L.sum()), qwen_tokens_seen=int(L.sum()),
        min_kept_length=int(L.min()), max_kept_length=int(L.max()),
        mean_kept_length=float(L.mean()),
        docs_ge_13568=int((L >= 13568).sum()), tokens_ge_13568=int(L[L >= 13568].sum()),
        docs_ge_24576=int((L >= 24576).sum()), tokens_ge_24576=int(L[L >= 24576].sum()),
        docs_ge_32768=int((L >= 32768).sum()), tokens_ge_32768=int(L[L >= 32768].sum()),
        seconds=round(time.time() - t0, 1),
    )
    json.dump(meta, open(met_p, "w"), indent=1)
    return meta


def cmd_tokenize(args):
    shards = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".jsonl.zst"))
    if args.limit:
        shards = shards[:args.limit]
    os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
    print("tokenizing %d shards -> %s  (%d workers)" % (len(shards), OUT_DIR, args.workers), flush=True)
    done = 0; tot_tok = 0; tot_doc = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_shard, s): s for s in shards}
        for fu in as_completed(futs):
            s = futs[fu]
            try:
                m = fu.result()
            except Exception as e:
                print("  FAILED %-24s %s" % (s, str(e)[:110]), flush=True); continue
            done += 1
            tot_tok += m.get("qwen_tokens_kept", 0); tot_doc += m.get("docs_kept", 0)
            print("  [%3d/%3d] %-24s %7d docs %10d tok  %s (%.0fs)  running total %.2fB"
                  % (done, len(shards), s, m.get("docs_kept", 0), m.get("qwen_tokens_kept", 0),
                     m.get("status"), m.get("seconds", 0), tot_tok / 1e9), flush=True)
    print("\ntokenize done in %.1f min: %.2fB tokens, %.2fM docs" %
          ((time.time() - t0) / 60, tot_tok / 1e9, tot_doc / 1e6), flush=True)


def cmd_manifest(args):
    """Assemble the combined manifest from the per-shard meta.json files."""
    sub = os.path.join(OUT_DIR, "data")
    metas = []
    for f in sorted(os.listdir(sub)):
        if f.endswith(".meta.json"):
            try:
                m = json.load(open(os.path.join(sub, f)))
                if m.get("status") in ("ok", "skipped_existing") and m.get("docs_kept"):
                    metas.append(m)
            except Exception as e:
                print("  bad meta %s: %s" % (f, e))
    if not metas:
        print("no shard metas found -- run tokenize first"); return
    tot_tok = sum(m["qwen_tokens_kept"] for m in metas)
    tot_doc = sum(m["docs_kept"] for m in metas)
    man = dict(
        source_repo="EleutherAI/proof-pile-2", source_bucket="arxiv/train",
        source_raw_dir=SRC_DIR, tokenized_dir=OUT_DIR, tokenizer_path=TOKENIZER,
        storage_format="per-shard uint32 .bin + uint32 .lengths.npy + .meta.json",
        selection_rule="NO length filter -- every document kept, exact length in .lengths.npy so any "
                       "later length/budget selection is a cheap slice instead of a re-tokenization",
        eos_policy="one EOS (151643) appended per document, included in its length",
        purpose="long-context corpus; NOT wired into any training stage (owner directive 2026-07-29)",
        raw_file_count=len(metas),
        docs_kept=tot_doc, qwen_tokens_kept=tot_tok,
        qwen_tokens_kept_billions=round(tot_tok / 1e9, 4),
        mean_kept_length=round(tot_tok / max(1, tot_doc), 2),
        docs_ge_13568=sum(m.get("docs_ge_13568", 0) for m in metas),
        tokens_ge_13568=sum(m.get("tokens_ge_13568", 0) for m in metas),
        docs_ge_24576=sum(m.get("docs_ge_24576", 0) for m in metas),
        tokens_ge_24576=sum(m.get("tokens_ge_24576", 0) for m in metas),
        docs_ge_32768=sum(m.get("docs_ge_32768", 0) for m in metas),
        tokens_ge_32768=sum(m.get("tokens_ge_32768", 0) for m in metas),
        files=[dict(bin_path=os.path.join(OUT_DIR, m["bin_path"]),
                    lengths_path=os.path.join(OUT_DIR, m["lengths_path"]),
                    source_relpath=m["source_relpath"], docs_kept=m["docs_kept"],
                    qwen_tokens_kept=m["qwen_tokens_kept"]) for m in metas],
    )
    out = OUT_DIR + "_manifest.json"
    json.dump(man, open(out, "w"), indent=1)
    print("wrote %s" % out)
    print("  %d shards, %.2fM docs, %.3fB tokens (mean %.0f tok/doc)"
          % (len(metas), tot_doc / 1e6, tot_tok / 1e9, man["mean_kept_length"]))
    for b in (13568, 24576, 32768):
        print("  >=%-6d %8.2f%% of tokens (%.3fB) in %.2fM docs"
              % (b, 100 * man["tokens_ge_%d" % b] / tot_tok, man["tokens_ge_%d" % b] / 1e9,
                 man["docs_ge_%d" % b] / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tokenize"); p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("manifest")
    a = ap.parse_args()
    (cmd_tokenize if a.cmd == "tokenize" else cmd_manifest)(a)
