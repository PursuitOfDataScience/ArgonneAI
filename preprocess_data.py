"""
Preprocess parquet data into llm.c binary format.
Parallel version: uses multiprocessing across parquet files and batch tokenization.
Writes tokens incrementally to avoid memory blowup.
"""
import os
import sys
import glob
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from tqdm import tqdm

MAGIC = 20240801
BATCH_SIZE = 5000  # docs per tokenizer batch
FLUSH_SIZE = 1_000_000  # flush to disk every 1M tokens


def process_parquet(task):
    """Process a single parquet file into a temporary binary shard."""
    file_idx, filepath, total_files, tokenizer_path, output_dir, text_column = task
    filename = os.path.basename(filepath)
    shard_path = os.path.join(output_dir, f"shard_{file_idx:05d}.bin")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id

    pf = pq.ParquetFile(filepath)
    num_row_groups = pf.metadata.num_row_groups
    buf = []
    doc_count = 0
    tok_count = 0
    start = time.time()

    with open(shard_path, "wb") as f:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[text_column]):
            texts = batch.column(text_column).to_pylist()
            texts = [t for t in texts if t is not None and len(t) > 0]
            if not texts:
                continue

            encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
            for ids in encoded["input_ids"]:
                buf.extend(ids)
                buf.append(eos_token_id)

            doc_count += len(texts)

            if len(buf) >= FLUSH_SIZE:
                arr = np.array(buf, dtype=np.uint32)
                f.write(arr.tobytes())
                tok_count += len(buf)
                buf = []

        if buf:
            arr = np.array(buf, dtype=np.uint32)
            f.write(arr.tobytes())
            tok_count += len(buf)
            buf = []

    elapsed = time.time() - start
    rate = tok_count / elapsed if elapsed > 0 else 0
    return file_idx, filename, shard_path, tok_count, doc_count, rate, elapsed


def main(args):
    print("=" * 60)
    print("Preprocessing Parquet Data to Binary Format (Parallel)")
    print("=" * 60)

    output_dir = args.output_dir
    final_file = os.path.join(output_dir, "train.bin")
    os.makedirs(output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files")
    if len(parquet_files) == 0:
        print(f"ERROR: No parquet files in {args.data_dir}")
        sys.exit(1)

    num_workers = args.workers if args.workers > 0 else min(cpu_count(), len(parquet_files), 32)
    print(f"Using {num_workers} workers")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Text column: {args.text_column}")
    print(f"Output: {final_file}")
    print(f"Flush size: {FLUSH_SIZE:,} tokens")
    print("=" * 60)

    tasks = [
        (i, f, len(parquet_files), args.tokenizer_path, output_dir, args.text_column)
        for i, f in enumerate(parquet_files)
    ]

    start_time = time.time()

    results = []
    with Pool(num_workers) as pool:
        with tqdm(total=len(tasks), desc="Tokenizing", unit="file") as pbar:
            for result in pool.imap_unordered(process_parquet, tasks):
                results.append(result)
                file_idx, filename, _, tok_count, doc_count, rate, elapsed = result
                tqdm.write(
                    f"[{file_idx+1}/{len(parquet_files)}] {filename} | "
                    f"{doc_count:,} docs | {tok_count:,} tokens | "
                    f"{rate:,.0f} tok/s | {elapsed:.1f}s"
                )
                pbar.update(1)

    tokenize_time = time.time() - start_time
    print(f"\nTokenization done in {tokenize_time/60:.1f} min")

    results.sort(key=lambda x: x[0])

    print("Concatenating shards...")
    total_tokens = 0
    total_docs = 0

    with open(final_file, "wb") as f:
        header = np.zeros(256, dtype=np.int32)
        header[0] = MAGIC
        f.write(header.tobytes())

        for _, _, shard_path, tok_count, doc_count, _, _ in tqdm(
            results, desc="Concatenating", unit="shard"
        ):
            total_tokens += tok_count
            total_docs += doc_count
            with open(shard_path, "rb") as sf:
                while True:
                    chunk = sf.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.remove(shard_path)

    with open(final_file, "r+b") as f:
        f.seek(8)
        f.write(np.array([total_tokens & 0xFFFFFFFF], dtype=np.int32).tobytes())
        f.write(np.array([total_tokens >> 32], dtype=np.int32).tobytes())

    print("Saving tokenizer alongside data...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    elapsed = time.time() - start_time
    file_size = os.path.getsize(final_file) / (1024**3)
    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"Total: {total_docs:,} documents, {total_tokens:,} tokens")
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Rate: {total_tokens/elapsed:,.0f} tok/s")
    print(f"File: {final_file} ({file_size:.2f} GB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with parquet files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text", help="Column name containing text")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (0 = auto)")
    args = parser.parse_args()
    main(args)