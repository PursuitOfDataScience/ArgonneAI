"""
build_a4_data.py -- build the argonne4.0 PRETRAIN corpus (the validated 50/30/20 recipe's
per-source flat bins) from headerless docbin shards.

Converts docbin shards (concatenated uint32 Qwen3 token ids) into llm.c flat .bin files
(256*int32 header, magic 20240801, then uint32 tokens) that pretrain.py's DataLoader /
WeightedMultiLoader read directly. The launcher (run_full_training.sh) points --train_sources at
edu.bin:50, math.bin:30, code.bin:20; the weighted sampler realizes the ratio, so these bins do
NOT need to be pre-mixed -- just the largest clean per-source bins you can build.

Unlike the campaign's proxy build (capped at 1-1.6B/source), this uses ALL available shards by
default (edu ~2B, math ~9.5B, code ~7.5B ~= 19B combined). Streaming/chunked so host RAM stays tiny;
idempotent (skips a target already >= the requested size).

  !! DATA GAP (2026-07-20): edu is only ~2B tokenized but carries the highest recipe weight (50%),
  so a large run repeats edu heavily. Tokenize MORE FineWeb-Edu (upstream is ~1.3T) into
  EDU_DOCBIN_DIR and re-run to close it. math (~9.5B ~= all FineMath-4plus) and code (~7.5B) are
  better; FineMath-3plus (~34B) can extend math. See ARGONNE4.0.md.
"""
import os, glob, json, argparse, numpy as np

MAGIC = 20240801
CHUNK = 64 << 20  # 64M tokens per write

# docbin sources (override via env if they move)
EDU_DIR  = os.environ.get("EDU_DOCBIN_DIR",  "/project/rcc/youzhi/data/reasoning_anneal/fineweb_edu_a4")
MATH_DIR = os.environ.get("MATH_DOCBIN_DIR", "/project/rcc/youzhi/data/finemath/finemath-4plus_qwen3_docbin")
CODE_DIR = os.environ.get("CODE_DOCBIN_DIR", "/project/rcc/youzhi/data/reasoning_anneal/github_code")


def shards(d, pat):
    return sorted(glob.glob(os.path.join(d, pat)))


def write_flat(out_path, shard_paths, max_tokens=None, skip_head_tokens=0):
    """Concat docbin shards -> one llm.c flat bin. skip_head_tokens drops the first N tokens of the
    FIRST shard so a train arm and its val slice (carved from the same shard) do not overlap."""
    want = max_tokens if max_tokens else float("inf")
    if os.path.exists(out_path):
        have = (os.path.getsize(out_path) - 1024) // 4
        if have >= min(want, 1) and (max_tokens is None or have >= max_tokens * 0.98):
            print(f"  SKIP {os.path.basename(out_path)} (have {have:,} tok)")
            return have
    written = 0
    with open(out_path, "wb") as f:
        hdr = np.zeros(256, dtype=np.int32); hdr[0] = MAGIC; hdr[1] = 1
        f.write(hdr.tobytes())  # placeholder count; rewritten at end
        for i, sp in enumerate(shard_paths):
            if written >= want:
                break
            toks = np.memmap(sp, dtype=np.uint32, mode="r")
            if i == 0 and skip_head_tokens:
                toks = toks[skip_head_tokens:]
            for c0 in range(0, len(toks), CHUNK):
                if written >= want:
                    break
                chunk = np.asarray(toks[c0:c0 + CHUNK])
                if written + len(chunk) > want:
                    chunk = chunk[: int(want - written)]
                chunk.astype(np.uint32).tofile(f)
                written += len(chunk)
        f.seek(0)
        hdr[2] = written if written < 2**31 else -1
        f.write(hdr.tobytes())
    # >2^31 tokens overflow the int32 header count field, so we store -1 (above). The pretrain.py
    # loader IGNORES the header count and derives length from the memmap file size, so a single
    # large bin is fine -- this matches the existing 20.8B fineweb-binary-qwen3/train.bin. No cap.
    ovf = " [>2^31: hdr count=-1, loader uses file length]" if written >= 2**31 else ""
    print(f"  WROTE {os.path.basename(out_path)}: {written:,} tok{ovf} from {len(shard_paths)} shards")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/rcc/youzhi/data/argonne4_pretrain",
                    help="output dir for the flat per-source bins + val bins")
    ap.add_argument("--edu_cap", type=int, default=0, help="max edu tokens (0 = all shards)")
    ap.add_argument("--math_cap", type=int, default=0, help="max math tokens (0 = all shards)")
    ap.add_argument("--code_cap", type=int, default=0, help="max code tokens (0 = all shards)")
    ap.add_argument("--val_tokens", type=int, default=3_000_000, help="held-out val tokens/domain")
    ap.add_argument("--skip_val", action="store_true", help="do not (re)build val bins")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    edu, math, code = shards(EDU_DIR, "*.bin"), shards(MATH_DIR, "*.bin"), shards(CODE_DIR, "*.bin")
    print(f"docbin shards: edu={len(edu)} math={len(math)} code={len(code)}")
    assert edu and math and code, "missing docbin shards; set EDU/MATH/CODE_DOCBIN_DIR"

    # held-out per-domain val bins from the LAST shard (train arms exclude that shard).
    if not args.skip_val:
        write_flat(f"{args.out}/val_edu.bin",  [edu[-1]],  max_tokens=args.val_tokens)
        write_flat(f"{args.out}/val_math.bin", [math[-1]], max_tokens=args.val_tokens)
        write_flat(f"{args.out}/val_code.bin", [code[-1]], max_tokens=args.val_tokens)

    # training per-source bins (all-but-val shards; capped only if requested)
    ntok = {}
    ntok["edu"]  = write_flat(f"{args.out}/edu_flat.bin",  edu[:-1],  max_tokens=(args.edu_cap or None))
    ntok["math"] = write_flat(f"{args.out}/finemath_flat.bin", math[:-1], max_tokens=(args.math_cap or None))
    ntok["code"] = write_flat(f"{args.out}/code_flat.bin", code[:-1], max_tokens=(args.code_cap or None))

    combined = sum(ntok.values())
    manifest = {
        "recipe": "50% edu / 30% math / 20% code (validated 2026-07-20)",
        "train_arms": {k: f"{args.out}/{ 'finemath' if k=='math' else k }_flat.bin" for k in ntok},
        "tokens": ntok, "combined_tokens": combined,
        "val": {d: f"{args.out}/val_{d}.bin" for d in ("edu", "math", "code")},
        "sources": {"edu": EDU_DIR, "math": MATH_DIR, "code": CODE_DIR},
    }
    with open(f"{args.out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\ncombined train tokens: {combined:,}  (edu {ntok['edu']:,} / math {ntok['math']:,} / code {ntok['code']:,})")
    print(f"manifest -> {args.out}/manifest.json")
    edu_frac_ceiling = ntok["edu"] / (0.50)   # at 50% weight, edu-limited budget before >1 epoch of edu
    print(f"\nNOTE: at the 50% edu weight, edu ({ntok['edu']:,} tok) is the binding source: a run of "
          f"~{edu_frac_ceiling/1e9:.0f}B tokens already repeats edu ~once. For a bigger run either "
          f"tokenize more FineWeb-Edu or accept edu multi-epoch (<=~4x ~= free, Muennighoff).")
    print(f"Set the launcher to these bins:  A4_EDU={args.out}/edu_flat.bin "
          f"A4_MATH={args.out}/finemath_flat.bin A4_CODE={args.out}/code_flat.bin A4_TRAIN_TOKENS=<budget>")


if __name__ == "__main__":
    main()
