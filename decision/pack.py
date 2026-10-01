"""Tokenize decision records (built/*.jsonl) into memory-mapped arrays the trainer and evaluator read directly.

    python decision/pack.py --built built --out packed

Per split, a directory with:
    tokens.npy   int32, every example's input_ids back to back
    index.npy    int64 (N, 10): start, length, ans_pos, opt_start, n_opt, label, type, source, soft_start, truncated
    opt_pos.npy  int32, option read positions (relative to the example start), back to back
    soft.npy     float32, soft targets back to back (soft_start = -1 when an example has none)
    meta.json    source names, type names, example ids, per-source counts, token totals
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision.schema import Encoder, question_from_record, MAX_LEN  # noqa: E402

TYPES = ["choice", "score", "noul"]
BASE = os.environ.get("DECISION_BASE", "PursuitOfDataScience/argonne-4.5-base-ctx13568")
_enc = None


def _init(tok_dir, max_len):
    global _enc
    from transformers import PreTrainedTokenizerFast
    from decision.modeling import local_dir
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(local_dir(tok_dir), "tokenizer.json"))
    _enc = Encoder(tok, max_len)


def _encode(line):
    r = json.loads(line)
    try:
        e = _enc.encode(r["state"], question_from_record(r))
    except ValueError as ex:
        return None, f"{r['id']}: {ex}"
    return (r["id"], r["source"], TYPES.index(r["type"]), r["label"], r.get("soft"), e), None


def pack_split(src_path, out_dir, tok_dir, max_len, workers):
    t0 = time.time()
    with open(src_path) as f:
        lines = f.readlines()
    os.makedirs(out_dir, exist_ok=True)
    sources, ids, errors = {}, [], []
    tok_chunks, opt_chunks, soft_chunks, index = [], [], [], []
    start = opt_start = soft_start = 0
    with Pool(workers, initializer=_init, initargs=(tok_dir, max_len)) as pool:
        for res, err in pool.imap(_encode, lines, chunksize=64):
            if err:
                errors.append(err)
                continue
            rid, src, typ, label, soft, e = res
            sid = sources.setdefault(src, len(sources))
            ids.append(rid)
            n = len(e["input_ids"])
            k = len(e["opt_pos"])
            if soft is not None and len(soft) != k:
                errors.append(f"{rid}: soft has {len(soft)} entries for {k} options")
                ids.pop()
                continue
            tok_chunks.append(np.asarray(e["input_ids"], dtype=np.int32))
            opt_chunks.append(np.asarray(e["opt_pos"], dtype=np.int32))
            ss = -1
            if soft is not None:
                soft_chunks.append(np.asarray(soft, dtype=np.float32))
                ss = soft_start
                soft_start += k
            index.append((start, n, e["ans_pos"], opt_start, k, label, typ, sid, ss, int(e["truncated"])))
            start += n
            opt_start += k
    np.save(os.path.join(out_dir, "tokens.npy"), np.concatenate(tok_chunks) if tok_chunks else np.zeros(0, np.int32))
    np.save(os.path.join(out_dir, "opt_pos.npy"), np.concatenate(opt_chunks) if opt_chunks else np.zeros(0, np.int32))
    np.save(os.path.join(out_dir, "soft.npy"), np.concatenate(soft_chunks) if soft_chunks else np.zeros(0, np.float32))
    idx = np.asarray(index, dtype=np.int64).reshape(-1, 10)
    np.save(os.path.join(out_dir, "index.npy"), idx)
    names = sorted(sources, key=sources.get)
    per_src = {names[s]: int((idx[:, 7] == s).sum()) for s in range(len(names))}
    meta = {"sources": names, "types": TYPES, "ids": ids, "n": len(ids), "tokens": int(start),
            "per_source": per_src, "truncated": int(idx[:, 9].sum()) if len(idx) else 0,
            "max_len": max_len, "len_p50": int(np.median(idx[:, 1])) if len(idx) else 0,
            "len_max": int(idx[:, 1].max()) if len(idx) else 0, "errors": errors[:50], "n_errors": len(errors)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[pack] {os.path.basename(src_path)}: {len(ids)} examples, {start:,} tokens, median len {meta['len_p50']}, "
          f"max {meta['len_max']}, truncated {meta['truncated']}, errors {len(errors)} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--built", default="built")
    ap.add_argument("--out", default="packed")
    ap.add_argument("--tokenizer", default=BASE)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--splits", default="train,val,test,ood")
    args = ap.parse_args()
    for split in args.splits.split(","):
        p = os.path.join(args.built, f"{split}.jsonl")
        if os.path.exists(p):
            pack_split(p, os.path.join(args.out, split), args.tokenizer, args.max_len, args.workers)


if __name__ == "__main__":
    main()
