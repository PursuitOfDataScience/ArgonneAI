"""
Preprocess parquet data into llm.c binary format.
Parallel version: uses multiprocessing across parquet files and batch tokenization.

python3 preprocess_data.py \
  --tokenizer_path /path/to/tokenizer \
  --data_dir /path/to/fineweb-parquets \
  --output_dir /path/to/output \
  --workers 32
"""
import os
import glob
import time
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
import pyarrow.parquet as pq

MAGIC = 20240801
BATCH_SIZE = 5000  # docs per tokenizer batch

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True, help="Directory with parquet files")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--text_column", type=str, default="text", help="Column name containing text")
parser.add_argument("--workers", type=int, default=0, help="Number of workers (0 = auto)")
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
FINAL_FILE = os.path.join(OUTPUT_DIR, "train.bin")


def process_parquet(task):
    """Process a single parquet file into a temporary binary shard."""
    file_idx, filepath, total_files, tokenizer_path, output_dir, text_column = task
    filename = os.path.basename(filepath)
    shard_path = os.path.join(output_dir, f"shard_{file_idx:05d}.bin")

    # Each worker loads its own tokenizer (not picklable across processes)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id

    pf = pq.ParquetFile(filepath)
    all_tokens = []
    doc_count = 0
    start = time.time()

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[text_column]):
        texts = batch.column(text_column).to_pylist()
        texts = [t for t in texts if t is not None and len(t) > 0]
        if not texts:
            continue

        # Batch tokenize — much faster than one-by-one
        encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        for ids in encoded["input_ids"]:
            all_tokens.extend(ids)
            all_tokens.append(eos_token_id)

        doc_count += len(texts)

    # Write shard
    tokens_array = np.array(all_tokens, dtype=np.uint32)
    tokens_array.tofile(shard_path)

    elapsed = time.time() - start
    tok_count = len(all_tokens)
    rate = tok_count / elapsed if elapsed > 0 else 0
    print(f"[{file_idx+1}/{total_files}] {filename} | {doc_count:,} docs | {tok_count:,} tokens | {rate:,.0f} tok/s | {elapsed:.1f}s")

    return shard_path, tok_count, doc_count


def main():
    print("=" * 60)
    print("Preprocessing Parquet Data to Binary Format (Parallel)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files")

    num_workers = args.workers if args.workers > 0 else min(cpu_count(), len(parquet_files), 32)
    print(f"Using {num_workers} workers")

    # Build task list
    tasks = [
        (i, f, len(parquet_files), args.tokenizer_path, OUTPUT_DIR, args.text_column)
        for i, f in enumerate(parquet_files)
    ]

    start_time = time.time()

    # Process in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_parquet, tasks)

    # Concatenate shards into final binary
    print("\nConcatenating shards...")
    total_tokens = 0
    total_docs = 0

    with open(FINAL_FILE, "wb") as f:
        # Write header placeholder
        header = np.zeros(256, dtype=np.int32)
        header[0] = MAGIC
        f.write(header.tobytes())

        # Append each shard in order
        for shard_path, tok_count, doc_count in results:
            total_tokens += tok_count
            total_docs += doc_count
            with open(shard_path, "rb") as sf:
                while True:
                    chunk = sf.read(64 * 1024 * 1024)  # 64MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
            os.remove(shard_path)  # clean up shard

    # Update header with final token count
    with open(FINAL_FILE, "r+b") as f:
        f.seek(8)
        f.write(np.array([total_tokens & 0xFFFFFFFF], dtype=np.int32).tobytes())
        f.write(np.array([total_tokens >> 32], dtype=np.int32).tobytes())

    # Save tokenizer alongside data
    print("Saving tokenizer alongside data...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))

    elapsed = time.time() - start_time
    file_size = os.path.getsize(FINAL_FILE) / (1024**3)
    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"Total: {total_docs:,} documents, {total_tokens:,} tokens")
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Rate: {total_tokens/elapsed:,.0f} tok/s")
    print(f"File: {FINAL_FILE} ({file_size:.2f} GB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()