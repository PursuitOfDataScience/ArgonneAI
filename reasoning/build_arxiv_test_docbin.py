#!/usr/bin/env python3
"""Tokenize proof-pile-2's arXiv TEST split into the Qwen3 docbin contract (one .bin + .lengths.npy),
as HELD-OUT long documents for position-bucketed NLL (reasoning/exp_longctx_learning.py --docbin).

Why the test split: the a4.5 context extension (phase B) consumed the whole long-document pool of
proof-pile-2's arXiv TRAIN split (build_longctx_arxiv.py's OUT_DIR), so no
document there is held out for argonne-4.5-base-ctx13568. The test split was never tokenized for any
training run. Same tokenizer and EOS policy as build_longctx_arxiv.py (one EOS appended per document).

  python reasoning/build_arxiv_test_docbin.py [--workers 8]
"""
import argparse, glob, io, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

SRC = "/project/rcc/youzhi/data/EleutherAI_proof-pile-2/arxiv/test"
OUT = "/project/rcc/youzhi/data/proof_pile2_arxiv_test_qwen3_docbin"
TOKENIZER = "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base"
EOS = 151643

_tok = None


def _texts(path):
    import zstandard
    with open(path, "rb") as fh:
        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]


def _tokenize_file(path):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    docs = []
    for t in _texts(path):
        ids = _tok(t, add_special_tokens=False)["input_ids"]
        ids.append(EOS)
        docs.append(np.asarray(ids, dtype=np.uint32))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(SRC, "*.jsonl.zst")))
    assert len(files) == 100, f"expected 100 test shards, found {len(files)}"
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        per_file = list(ex.map(_tokenize_file, files))
    docs = [d for f in per_file for d in f]
    lengths = np.asarray([len(d) for d in docs], dtype=np.uint32)
    stem = os.path.join(OUT, "data", "arXiv_test")
    np.concatenate(docs).astype(np.uint32).tofile(stem + ".bin.tmp")
    np.save(stem + ".lengths.tmp.npy", lengths)
    os.replace(stem + ".bin.tmp", stem + ".bin")
    os.replace(stem + ".lengths.tmp.npy", stem + ".lengths.npy")
    assert os.path.getsize(stem + ".bin") == 4 * int(lengths.sum()), "bin size != 4 x sum(lengths)"
    meta = {"source": "EleutherAI/proof-pile-2 arxiv/test (100 files)", "tokenizer": TOKENIZER, "eos": EOS,
            "docs": int(len(lengths)), "tokens": int(lengths.sum()),
            "docs_ge_24577": int((lengths >= 24577).sum()), "docs_ge_13569": int((lengths >= 13569).sum()),
            "median": int(np.median(lengths)), "p90": int(np.percentile(lengths, 90)), "max": int(lengths.max())}
    json.dump(meta, open(stem + ".meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
