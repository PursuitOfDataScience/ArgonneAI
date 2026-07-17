#!/usr/bin/env python3
"""build_reasoning_corpus.py -- tokenize+binarize the Argonne-3.5 CODE/MATH/REASONING/TOOL anneal corpus.

WHY (see base-for-reasoning.md): the 3.0/3.5 reasoning ceiling is a base-data limit, not a
downstream one. The current base pretrain is FineWeb+FineMath 85/15 with ZERO code and no
synthetic reasoning/tool traces. This builds the missing tiers -- code, math-reasoning,
long-CoT reasoning, and <tool_call>/<tool_response> tool-use -- plus a fineweb-edu REPLAY
anchor (the §13 anti-forgetting lesson), into the SAME doc-aware docbin format that
pretrain.py / midtraining.py's DocManifestDataLoader already consumes.

OUTPUT (per preprocess_finemath.py's contract): for every processed raw shard, a trio
  <src>/<stem>.bin           uint32 token ids of EOS-separated super-docs (>= --min_superdoc)
  <src>/<stem>.lengths.npy   uint32 length of each super-doc
  <src>/<stem>.meta.json     shard stats (actual tokens kept, docs seen/kept/dropped)
Plus a combined top-level manifest with the keys DocManifestDataLoader reads
(tokenized_dir="/" + absolute bin_path/lengths_path per file, qwen_tokens_kept, files[]).
Consume with doc_shuffle=1 so ALL tiers interleave globally (never sequential -> no §13 forgetting).

The ONE tokenizer for the whole corpus is Qwen3-0.6B-Base (same as finemath): <think>(151667)/
</think>(151668)/<tool_call>(151657)/<tool_response>(151665) all map to SINGLE in-vocab ids, so
reasoning/tool structure bakes into the base at ~1 token each (base-for-reasoning.md §4).

QUALITY GATES applied per source (all configurable in SOURCES):
  * length: drop rendered docs < min_doc_tokens or > max_doc_tokens (the long-tail R1
    degenerate-loop filter; also keeps <think> open+close inside a trainable window).
  * think-closure: if a doc has <think> it must also have </think> (drop truncated CoT).
  * per-problem dedup: cap N solutions per identical problem within a shard (comp-prog has
    only ~35k unique questions over ~3.9M rows; math has many solutions/problem).
  * DECONTAMINATION: drop any doc sharing a 13-gram with the CLEAN eval gate sets
    (svamp/asdiv/mawps/gsm-plus + MATH-test + HumanEval), so the anneal cannot poison the
    very metric base-for-reasoning.md §2.4 relies on. GSM8K is already contaminated upstream
    (project memory) and is intentionally NOT a training source here.

USAGE
  # 1) tokenize one source (submit each as its own CPU SLURM job for parallelism)
  python build_reasoning_corpus.py tokenize --source github_code --workers 16 --scale 1.0
  # 2) after all sources done, build the combined manifest (trims each tier to its budget)
  python build_reasoning_corpus.py finalize --scale 1.0
  # helpers
  python build_reasoning_corpus.py list                 # show sources + budgets at a scale
  python build_reasoning_corpus.py inspect --source X    # dump 2 rendered samples (no writes)
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

TOKENIZER = "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base"
OUT_ROOT = os.environ.get("RC_OUT_ROOT", "/project/rcc/youzhi/data/reasoning_anneal")
MANIFEST_PATH = os.path.join(OUT_ROOT, "reasoning_anneal_manifest.json")
MIN_SUPERDOC = 16384          # >= any block we train at (midtrain 13568, pretrain 1024)
NGRAM_N = 13                  # decontamination shingle size (Llama/GPT-style)

THINK_OPEN, THINK_CLOSE = 151667, 151668

# Clean eval gate sets to decontaminate AGAINST (never train on the metric).
# (path, candidate text fields -- first present wins)
DECONTAM_EVALS = [
    ("/project/rcc/youzhi/data/svamp_clean", ["question", "Body", "problem", "text"]),
    ("/project/rcc/youzhi/data/asdiv_clean", ["question", "body", "problem", "text"]),
    ("/project/rcc/youzhi/data/mawps_clean", ["question", "problem", "text"]),
    ("/project/rcc/youzhi/data/gsmplus_test", ["question", "problem", "text"]),
    ("/project/rcc/youzhi/data/nlile_hendrycks-MATH-benchmark", ["problem", "question", "text"]),
    ("/project/rcc/youzhi/data/humaneval_ds", ["prompt", "text", "problem"]),
]

# ---------------------------------------------------------------------------
# Source registry. budget_tokens are for the DEFAULT ~30B anneal (--scale 1.0);
# --scale multiplies every budget. n_shards = how many raw files to sample
# (spread evenly for diversity); each selected shard is capped at budget/n_shards.
# ---------------------------------------------------------------------------
def _g(*p):
    return sorted(glob.glob(os.path.join(*p)))


SOURCES = {
    # ---- CODE (the highest-value missing ingredient) --------------------------------
    "github_code": dict(
        tier="code", fmt="parquet", renderer="code",
        files=lambda: _g("/project/rcc/youzhi/data/nick007x_github-code-2025/above-2-stars", "train_*.parquet"),
        columns=["repo_id", "file_path", "content"],
        renderer_args=dict(header=True),
        budget_tokens=8_000_000_000, n_shards=16,
        min_doc_tokens=32, max_doc_tokens=32768,
        require_think_closure=False, decontam=True,
        dedup_key=None,
    ),
    "comp_prog": dict(
        tier="code", fmt="jsonl", renderer="compprog",
        files=lambda: _g("/project/rcc/youzhi/data/datasets--nvidia--Nemotron-Competitive-Programming-v1/snapshots", "*", "data", "*.jsonl"),
        renderer_args=dict(strip_prefix_regex=r"^You are a helpful( and harmless)? assistant\..*?```\s*\n\n"),
        budget_tokens=4_000_000_000, n_shards=6,
        min_doc_tokens=64, max_doc_tokens=32768,
        require_think_closure=True, decontam=True,
        drop_splits=["test", "valid", "validation"],
        dedup_key="__user0__", dedup_cap=3,
    ),
    # ---- MATH (complements the FineMath already in the base) ------------------------
    "openmath": dict(
        tier="math", fmt="jsonl", renderer="math_ps",
        files=lambda: [p for p in _g("/project/rcc/youzhi/data/nvidia_OpenMathReasoning_curated", "*", "shards", "shard_*.jsonl")
                       if "before_qwen_cpu_cleanup" not in p and "/genselect/" not in p],
        renderer_args=dict(append_answer=True),
        budget_tokens=8_000_000_000, n_shards=24,
        min_doc_tokens=64, max_doc_tokens=32768,
        require_think_closure=True, decontam=True,
        dedup_key="problem", dedup_cap=4,
    ),
    # ---- REASONING (long-CoT, breaks the ceiling) -----------------------------------
    "am_r1": dict(
        tier="reasoning", fmt="jsonl", renderer="chat",
        files=lambda: _g("/project/rcc/youzhi/data/a-m-team_AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_curated/shards", "shard_*.jsonl"),
        renderer_args=dict(messages_key="messages"),
        budget_tokens=3_000_000_000, n_shards=24,
        min_doc_tokens=64, max_doc_tokens=24576,
        require_think_closure=True, decontam=True,
        dedup_key="__user0__", dedup_cap=2,
    ),
    "mixture_of_thoughts": dict(
        tier="reasoning", fmt="arrow", renderer="chat",
        files=lambda: _g("/project/rcc/youzhi/data/open-r1_Mixture-of-Thoughts/train", "data-*.arrow"),
        renderer_args=dict(messages_key="messages"),
        budget_tokens=2_000_000_000, n_shards=15,
        min_doc_tokens=64, max_doc_tokens=24576,
        require_think_closure=True, decontam=True,
        dedup_key="__user0__", dedup_cap=2,
    ),
    "thinking_05m": dict(
        tier="reasoning", fmt="arrow", renderer="chat",
        files=lambda: _g("/project/rcc/youzhi/data/PursuitOfDataScience_0.5M-thinking/train", "data-*.arrow"),
        renderer_args=dict(messages_key="messages"),  # verified/overridden after profiling
        budget_tokens=1_000_000_000, n_shards=25,
        min_doc_tokens=64, max_doc_tokens=24576,
        require_think_closure=True, decontam=True,
        dedup_key="__user0__", dedup_cap=2,
    ),
    # ---- TOOL USE (bake <tool_call>/<tool_response> into the base -- md §4) ----------
    "agentic": dict(
        tier="tool", fmt="jsonl", renderer="chat",
        files=lambda: _g("/project/rcc/youzhi/data/datasets--nvidia--Nemotron-SFT-Agentic-v2/snapshots", "*", "data", "*.jsonl"),
        renderer_args=dict(messages_key="messages", emit_tools=True, keep_system=True),
        budget_tokens=1_000_000_000, n_shards=3,
        min_doc_tokens=64, max_doc_tokens=24576,
        require_think_closure=False, decontam=True,
        dedup_key=None,
    ),
    # ---- GENERAL REPLAY anchor (anti-forgetting; NOT a source of new capability) ----
    "fineweb_edu": dict(
        tier="general", fmt="arrow", renderer="text",
        files=lambda: _g("/project/rcc/youzhi/data/fineweb-edu", "fineweb-edu-train-*-of-00218.arrow"),
        columns=None, renderer_args=dict(text_col="text"),
        budget_tokens=3_000_000_000, n_shards=24,
        min_doc_tokens=32, max_doc_tokens=32768,
        require_think_closure=False, decontam=False,  # general web; keep it fast
        dedup_key=None,
    ),
    # ---- Defined but OFF by default (medium quality, chatty; tangential to anneal) ---
    "instruct_chat": dict(
        tier="instruction", fmt="jsonl", renderer="chat", enabled=False,
        files=lambda: _g("/project/rcc/youzhi/data/datasets--nvidia--Nemotron-SFT-Instruction-Following-Chat-v2/snapshots", "*", "data", "*.jsonl"),
        renderer_args=dict(messages_key="messages"),
        budget_tokens=2_000_000_000, n_shards=2,
        min_doc_tokens=32, max_doc_tokens=16384,
        require_think_closure=False, decontam=True,
        dedup_key=None,
    ),
}


# ---------------------------------------------------------------------------
# Per-worker globals
# ---------------------------------------------------------------------------
_TOK = None
_EOS = None
_DECONTAM = None  # frozenset of 13-gram strings


def _norm_words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def _shingles(s, n=NGRAM_N):
    # Generator (not a list) so _contaminated short-circuits on the first hit and never
    # materializes ~1.3M shingles for a huge doc.
    w = _norm_words(s)
    if len(w) < n:
        if w:
            yield " ".join(w)
        return
    for i in range(len(w) - n + 1):
        yield " ".join(w[i:i + n])


def _init_worker(tokenizer_path, decontam_grams):
    global _TOK, _EOS, _DECONTAM
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _EOS = _TOK.eos_token_id
    if _EOS is None:
        raise RuntimeError("tokenizer has no eos_token_id")
    _DECONTAM = decontam_grams


def _contaminated(text):
    if not _DECONTAM:
        return False
    for g in _shingles(text):
        if g in _DECONTAM:
            return True
    return False


# ---------------------------------------------------------------------------
# Renderers: record dict -> pretraining string (or None to skip)
# ---------------------------------------------------------------------------
def render_text(rec, a):
    t = rec.get(a.get("text_col", "text"))
    return t if isinstance(t, str) and t.strip() else None


def render_code(rec, a):
    c = rec.get("content")
    if not isinstance(c, str) or not c.strip():
        return None
    if a.get("header"):
        return f"# {rec.get('repo_id', '')}/{rec.get('file_path', '')}\n{c}"
    return c


def render_math_ps(rec, a):
    p = (rec.get("problem") or "").strip()
    s = (rec.get("generated_solution") or "").strip()
    if not s:
        return None
    txt = f"Problem:\n{p}\n\nSolution:\n{s}" if p else s
    ans = rec.get("expected_answer")
    if a.get("append_answer") and ans and "\\boxed" not in s:
        txt += f"\n\nAnswer: {str(ans).strip()}"
    return txt


def _tools_preamble(tools):
    """Render a `tools` list into Qwen-style <tools>...</tools> so the call<->schema link is learnable."""
    body = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return ("# Tools\nYou may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{body}\n</tools>")


def _seg_for_msg(m):
    """One chat turn -> a ChatML block, preserving <think>/<tool_call>/<tool_response>.
    Returns None for an empty turn. reasoning_content (a SEPARATE R1 field, e.g. agentic /
    comp-prog) is wrapped in <think>...</think>; inline <think> in content is passed through."""
    role = m.get("role", "") or ""
    content = m.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    if role in ("tool", "tool_response", "observation", "function"):
        if not content.strip():
            return None
        return f"<|im_start|>tool\n<tool_response>\n{content}\n</tool_response><|im_end|>"
    pieces = []
    rc = m.get("reasoning_content")
    if rc and str(rc).strip():
        pieces.append(f"<think>\n{str(rc).strip()}\n</think>")
    if content.strip():
        pieces.append(content)
    tc = m.get("tool_calls")
    if tc:
        pieces.append("\n".join(
            f"<tool_call>\n{json.dumps(c.get('function', c), ensure_ascii=False)}\n</tool_call>"
            for c in tc))
    if not pieces:
        return None
    return f"<|im_start|>{role}\n{chr(10).join(pieces)}<|im_end|>"


def render_chat(rec, a):
    msgs = rec.get(a.get("messages_key", "messages")) or rec.get("conversations")
    if not msgs:
        return None
    # normalize {from,value} (ShareGPT) -> {role,content}
    norm = []
    for m in msgs:
        if "from" in m and "role" not in m:
            m = {"role": {"human": "user", "gpt": "assistant"}.get(m["from"], m["from"]),
                 "content": m.get("value", "")}
        norm.append(dict(m))
    # tool-use: fold the available `tools` schema into the system turn so the base learns
    # which <tool_call> maps to which signature (base-for-reasoning.md §4).
    if a.get("emit_tools") and rec.get("tools"):
        tp = _tools_preamble(rec["tools"])
        si = next((i for i, m in enumerate(norm) if m.get("role") == "system"), None)
        if si is not None:
            norm[si]["content"] = ((norm[si].get("content") or "") + "\n\n" + tp).strip()
        else:
            norm.insert(0, {"role": "system", "content": tp})
    segs = []
    for m in norm:
        if m.get("role") == "system" and not (a.get("keep_system") and (m.get("content") or "").strip()):
            continue
        s = _seg_for_msg(m)
        if s:
            segs.append(s)
    return "\n".join(segs) if segs else None


def render_compprog(rec, a):
    msgs = rec.get("messages") or []
    if len(msgs) < 2:
        return None
    u = (msgs[0].get("content") or "").strip()
    rgx = a.get("strip_prefix_regex")
    if rgx:
        u = re.sub(rgx, "", u, count=1, flags=re.DOTALL).strip()
    if u == "-":
        u = ""
    asst = msgs[-1]
    rc = (asst.get("reasoning_content") or "").strip()
    body = (asst.get("content") or "").strip()
    if rc:
        body = f"<think>\n{rc}\n</think>\n\n{body}"
    parts = ([u] if u else []) + ([body] if body else [])
    out = "\n\n".join(parts).strip()
    return out or None


RENDERERS = {
    "text": render_text, "code": render_code, "math_ps": render_math_ps,
    "chat": render_chat, "compprog": render_compprog,
}


def _dedup_hash(rec, key):
    if key == "__user0__":
        msgs = rec.get("messages") or rec.get("conversations") or []
        src = (msgs[0].get("content") or msgs[0].get("value") or "") if msgs else ""
    else:
        src = rec.get(key) or ""
    return hashlib.blake2b(str(src).strip().encode("utf-8", "ignore"), digest_size=8).digest()


def _row_ok(rec, cfg):
    ds = cfg.get("drop_splits")
    if ds and str(rec.get("split", "")).lower() in ds:
        return False
    return True


# ---------------------------------------------------------------------------
# Record iteration (format-agnostic)
# ---------------------------------------------------------------------------
def iter_records(path, fmt, columns=None):
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    elif fmt == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        # Stream in bounded row batches (NOT whole row groups) so 16 workers x a large
        # code row group don't OOM -- keeps peak host RAM at the tokenizer+decontam floor.
        for batch in pf.iter_batches(batch_size=256, columns=columns):  # 256 (not 1024): bounds the
            for row in batch.to_pylist():
                yield row
    elif fmt == "arrow":
        try:
            from datasets import Dataset
            for row in Dataset.from_file(path):
                yield row
        except Exception:
            import pyarrow as pa
            with pa.ipc.open_stream(path) as r:   # HF arrow STREAM files (e.g. 0.5M-thinking)
                for batch in r:
                    for row in batch.to_pylist():
                        yield row
    else:
        raise ValueError(f"unknown fmt {fmt}")


# ---------------------------------------------------------------------------
# Worker: tokenize + pack ONE raw shard into a docbin trio, capped at cap_tokens.
# ---------------------------------------------------------------------------
def _process_shard(task):
    src_path, fmt, columns, renderer_key, renderer_args, cfg, out_bin, out_lengths, out_meta, cap_tokens = task
    rel = os.path.basename(src_path)

    if os.path.exists(out_bin) and os.path.exists(out_lengths) and os.path.exists(out_meta):
        try:
            m = json.load(open(out_meta))
            if m.get("status") == "done":
                m["status"] = "skipped_existing"
                return m
        except Exception:
            pass

    render = RENDERERS[renderer_key]
    min_tok = cfg.get("min_doc_tokens", 32)
    max_tok = cfg.get("max_doc_tokens", 1 << 30)
    need_close = cfg.get("require_think_closure", False)
    dedup_key = cfg.get("dedup_key")
    dedup_cap = cfg.get("dedup_cap", 1)
    do_decontam = cfg.get("decontam", False)

    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    tmp_bin = out_bin + ".tmp"
    fbin = open(tmp_bin, "wb")

    superdoc_lengths = []
    buf = []
    total_tokens = 0
    docs_seen = docs_kept = 0
    d_short = d_long = d_unclosed = d_contam = d_dup = d_empty = 0
    seen = {}

    def flush():
        nonlocal buf, total_tokens
        if len(buf) >= MIN_SUPERDOC:
            arr = np.asarray(buf, dtype=np.uint32)
            fbin.write(arr.tobytes())
            superdoc_lengths.append(len(buf))
            total_tokens += len(buf)
            buf = []
            return True
        return False

    for rec in iter_records(src_path, fmt, columns):
        docs_seen += 1
        if not _row_ok(rec, cfg):
            continue
        if dedup_key:
            h = _dedup_hash(rec, dedup_key)
            c = seen.get(h, 0)
            if c >= dedup_cap:
                d_dup += 1
                continue
            seen[h] = c + 1
        text = render(rec, renderer_args)
        if not text or not text.strip():
            d_empty += 1
            continue
        # Early length guard: a text far over the token cap will be dropped by max_tok anyway;
        # skip it BEFORE decontam-shingling + tokenize so a few multi-MB code files can't OOM
        # the worker. max_tok*8 chars => well over max_tok tokens (~4 chars/tok) -> no valid drop.
        if len(text) > max_tok * 8:
            d_long += 1
            continue
        if do_decontam and _contaminated(text):
            d_contam += 1
            continue
        ids = _TOK(text, add_special_tokens=False)["input_ids"]
        n = len(ids)
        if n < min_tok:
            d_short += 1
            continue
        if n > max_tok:
            d_long += 1
            continue
        if need_close and (THINK_OPEN in ids) and (THINK_CLOSE not in ids):
            d_unclosed += 1
            continue
        docs_kept += 1
        buf.extend(ids)
        buf.append(_EOS)
        if len(buf) >= MIN_SUPERDOC:
            flush()
            if total_tokens >= cap_tokens:
                break

    fbin.close()
    os.replace(tmp_bin, out_bin)
    lengths_arr = np.asarray(superdoc_lengths, dtype=np.uint32)
    np.save(out_lengths, lengths_arr)

    meta = {
        "source_relpath": rel,
        "bin_path": os.path.abspath(out_bin),
        "lengths_path": os.path.abspath(out_lengths),
        "docs_kept": int(len(superdoc_lengths)),          # number of super-docs (loader unit)
        "source_docs_kept": int(docs_kept),               # number of raw docs packed
        "docs_seen": int(docs_seen),
        "qwen_tokens_kept": int(total_tokens),
        "cap_tokens": int(cap_tokens),
        "dropped": dict(short=d_short, too_long=d_long, unclosed_think=d_unclosed,
                        contaminated=d_contam, dedup=d_dup, empty=d_empty),
        "min_superdoc_tokens": MIN_SUPERDOC,
        "mean_superdoc_len": float(lengths_arr.mean()) if len(lengths_arr) else 0.0,
        "status": "done",
    }
    json.dump(meta, open(out_meta, "w"), indent=2)
    return meta


# ---------------------------------------------------------------------------
# Decontam set builder (main process)
# ---------------------------------------------------------------------------
def build_decontam():
    from datasets import load_from_disk, Dataset
    grams = set()
    n_items = 0
    for root, fields in DECONTAM_EVALS:
        rows = []
        try:
            try:
                dd = load_from_disk(root)
                splits = dd.keys() if hasattr(dd, "keys") else [None]
                for sp in splits:
                    d = dd[sp] if sp is not None else dd
                    rows.extend(list(d))
            except Exception:
                for af in _g(root, "**", "*.arrow") or _g(root, "*.arrow"):
                    rows.extend(list(Dataset.from_file(af)))
        except Exception as e:
            print(f"[decontam] WARN could not load {root}: {e}", flush=True)
            continue
        fld = None
        if rows:
            for cand in fields:
                if cand in rows[0]:
                    fld = cand
                    break
        added = 0
        for r in rows:
            t = r.get(fld) if fld else None
            if isinstance(t, str) and t.strip():
                for g in _shingles(t):
                    grams.add(g)
                added += 1
        n_items += added
        print(f"[decontam] {os.path.basename(root)}: field={fld} items={added}", flush=True)
    print(f"[decontam] {n_items} eval problems -> {len(grams):,} unique {NGRAM_N}-grams", flush=True)
    return frozenset(grams)


# ---------------------------------------------------------------------------
# Shard selection
# ---------------------------------------------------------------------------
def select_files(all_files, n_shards):
    if n_shards >= len(all_files) or n_shards <= 0:
        return list(all_files)
    if n_shards == 1:
        return [all_files[0]]
    idx = sorted({round(i * (len(all_files) - 1) / (n_shards - 1)) for i in range(n_shards)})
    return [all_files[i] for i in idx]


def source_outdir(name):
    return os.path.join(OUT_ROOT, name)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_list(args):
    print(f"{'source':22} {'tier':12} {'enabled':8} {'budget@scale':>14} {'n_shards':>8}  files")
    for name, cfg in SOURCES.items():
        if not cfg.get("enabled", True) and not args.all:
            continue
        files = cfg["files"]()
        bud = int(cfg["budget_tokens"] * args.scale)
        print(f"{name:22} {cfg['tier']:12} {str(cfg.get('enabled', True)):8} "
              f"{bud/1e9:12.2f}B {cfg['n_shards']:8d}  {len(files)} raw files")


def cmd_inspect(args):
    cfg = SOURCES[args.source]
    files = cfg["files"]()
    if not files:
        raise SystemExit(f"no files for {args.source}")
    render = RENDERERS[cfg["renderer"]]
    print(f"[inspect] {args.source}: {len(files)} files; first={files[0]}")
    shown = 0
    for rec in iter_records(files[0], cfg["fmt"], cfg.get("columns")):
        t = render(rec, cfg.get("renderer_args", {}))
        if t:
            print("=" * 80)
            print(t[:1600])
            shown += 1
            if shown >= args.n:
                break


def cmd_tokenize(args):
    cfg = SOURCES[args.source]
    all_files = cfg["files"]()
    if not all_files:
        raise SystemExit(f"no raw files for source {args.source}")
    sel = select_files(all_files, cfg["n_shards"])
    budget = int(cfg["budget_tokens"] * args.scale)
    cap_per_shard = math.ceil(budget / max(1, len(sel)))
    outdir = source_outdir(args.source)
    os.makedirs(outdir, exist_ok=True)
    print(f"[tokenize] {args.source} tier={cfg['tier']} fmt={cfg['fmt']} renderer={cfg['renderer']}", flush=True)
    print(f"[tokenize] {len(all_files)} raw -> {len(sel)} selected; budget={budget/1e9:.2f}B "
          f"cap/shard={cap_per_shard/1e9:.3f}B; out={outdir}", flush=True)

    decontam = build_decontam() if (cfg.get("decontam") and not args.no_decontam) else frozenset()

    # Only the picklable filter fields cross into the worker (NOT the `files` lambda).
    wcfg = {k: cfg.get(k) for k in ("min_doc_tokens", "max_doc_tokens", "require_think_closure",
                                    "dedup_key", "dedup_cap", "decontam", "drop_splits")}
    tasks = []
    for p in sel:
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(os.path.basename(p))[0])
        tasks.append((p, cfg["fmt"], cfg.get("columns"), cfg["renderer"], cfg.get("renderer_args", {}),
                      wcfg, os.path.join(outdir, stem + ".bin"), os.path.join(outdir, stem + ".lengths.npy"),
                      os.path.join(outdir, stem + ".meta.json"), cap_per_shard))

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(TOKENIZER, decontam)) as ex:
        futs = {ex.submit(_process_shard, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[done] {r['source_relpath']}: {r['docs_kept']} super-docs "
                  f"{r['qwen_tokens_kept']:,} tok ({r['status']}) dropped={r['dropped']}", flush=True)

    tot = sum(r["qwen_tokens_kept"] for r in results)
    json.dump({"source": args.source, "tier": cfg["tier"], "budget_tokens": budget,
               "scale": args.scale, "files": results, "qwen_tokens_kept": int(tot)},
              open(os.path.join(outdir, "_source_manifest.json"), "w"), indent=2)
    print(f"[tokenize] {args.source} TOTAL {tot/1e9:.3f}B tokens in {(time.time()-t0)/60:.1f} min", flush=True)


def cmd_finalize(args):
    """Merge per-source shards into ONE manifest, trimming each tier to its (scaled) budget."""
    files_out = []
    total = 0
    by_tier = {}
    for name, cfg in SOURCES.items():
        if not cfg.get("enabled", True) and name not in (args.include or []):
            continue
        sm = os.path.join(source_outdir(name), "_source_manifest.json")
        if not os.path.exists(sm):
            print(f"[finalize] SKIP {name}: not tokenized yet ({sm} missing)")
            continue
        m = json.load(open(sm))
        budget = int(cfg["budget_tokens"] * args.scale)
        shards = sorted([f for f in m["files"] if f["docs_kept"] > 0],
                        key=lambda f: f["source_relpath"])
        used = 0
        for f in shards:
            if used >= budget:
                break
            files_out.append({"bin_path": f["bin_path"], "lengths_path": f["lengths_path"],
                              "docs_kept": int(f["docs_kept"]), "source_relpath": f"{name}/{f['source_relpath']}",
                              "tier": cfg["tier"]})
            used += f["qwen_tokens_kept"]
        by_tier[cfg["tier"]] = by_tier.get(cfg["tier"], 0) + used
        total += used
        print(f"[finalize] {name:20} tier={cfg['tier']:11} used {used/1e9:6.3f}B "
              f"({len([x for x in files_out if x['source_relpath'].startswith(name+'/')])} shards)")

    manifest = {
        "phase": "reasoning_anneal",
        "tokenizer_path": TOKENIZER,
        "tokenized_dir": "/",                       # bin_path/lengths_path are absolute
        "min_superdoc_tokens": MIN_SUPERDOC,
        "scale": args.scale,
        "qwen_tokens_kept": int(total),
        "qwen_tokens_kept_billions": total / 1e9,
        "tokens_by_tier_billions": {k: v / 1e9 for k, v in sorted(by_tier.items())},
        "files": files_out,
    }
    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump(manifest, open(MANIFEST_PATH, "w"), indent=2)
    print(f"\n[finalize] {total/1e9:.3f}B tokens, {len(files_out)} shards")
    print(f"[finalize] by tier (B): " + ", ".join(f"{k}={v/1e9:.2f}" for k, v in sorted(by_tier.items())))
    print(f"[finalize] manifest -> {MANIFEST_PATH}")


def cmd_flatten(args):
    """Interleave the per-source docbin .bin token streams into ONE dense flat llm.c .bin
    (magic 20240801, uint32) for the PRETRAINING-phase continue_pretrain.py (which reads a flat
    .bin densely). Chunk-granular weighted-fair interleave -> all sources mixed THROUGHOUT (never
    sequential -> no §13 forgetting), budget-proportional. Same shape as build_pretrain_mix.py."""
    MAGIC, HDR = 20240801, 256
    out_bin = args.out_bin or os.path.join(OUT_ROOT, "reasoning_anneal_flat.bin")
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    srcs = []
    for name, cfg in SOURCES.items():
        if not cfg.get("enabled", True) and name not in (args.include or []):
            continue
        bins = sorted(b for b in glob.glob(os.path.join(source_outdir(name), "*.bin"))
                      if not b.endswith(".tmp") and os.path.getsize(b) >= 4)  # skip 0-token (empty) shards
        if not bins:
            print(f"[flatten] SKIP {name}: no non-empty .bin (not tokenized yet)")
            continue
        mms = [np.memmap(b, dtype=np.uint32, mode="r") for b in bins]
        avail = int(sum(len(m) for m in mms))
        budget = int(cfg["budget_tokens"] * args.scale)
        use = min(avail, budget) if budget > 0 else avail
        frac = max(0.0, min(1.0, args.holdout_frac))
        split = int(round((1.0 - frac) * use))            # main=[0,split)  holdout=[split,use)  (DISJOINT)
        if args.part == "main":
            start, end = 0, split
        elif args.part == "holdout":
            start, end = split, use
        else:
            start, end = 0, use
        srcs.append(dict(name=name, tier=cfg["tier"], mms=mms, use=(end - start),
                         emitted=0, idx=0, off=0, start=start))
        print(f"[flatten] {name:22} tier={cfg['tier']:11} avail={avail/1e9:6.3f}B "
              f"part={args.part} window=[{start/1e9:.3f}B,{end/1e9:.3f}B) emit={(end - start)/1e9:6.3f}B ({len(bins)} bins)")

    # Pre-skip each source to the start of its window (the holdout part starts mid-stream). This is
    # what makes main/holdout share NO source tokens -> genuinely DISJOINT, same-composite slices.
    def _skip(s, n):
        while n > 0 and s["idx"] < len(s["mms"]):
            m = s["mms"][s["idx"]]
            t = min(len(m) - s["off"], n)
            s["off"] += t; n -= t
            if s["off"] >= len(m):
                s["idx"] += 1; s["off"] = 0
    for s in srcs:
        if s.get("start", 0) > 0:
            _skip(s, s["start"])

    total = sum(s["use"] for s in srcs)
    if total <= 0:
        raise SystemExit("[flatten] no tokens to flatten")

    def take(s, n):
        n = min(n, s["use"] - s["emitted"])
        if n <= 0:
            return None
        parts, need = [], n
        while need > 0 and s["idx"] < len(s["mms"]):
            m = s["mms"][s["idx"]]
            avail_m = len(m) - s["off"]
            t = min(avail_m, need)
            if t > 0:
                parts.append(np.asarray(m[s["off"]:s["off"] + t]))
                s["off"] += t
                need -= t
            if s["off"] >= len(m):
                s["idx"] += 1
                s["off"] = 0
        if not parts:
            return None
        buf = parts[0] if len(parts) == 1 else np.concatenate(parts)
        s["emitted"] += len(buf)
        return buf

    C = args.chunk
    written = 0
    per = {s["name"]: 0 for s in srcs}
    t0 = time.time()
    next_report = 0
    with open(out_bin, "wb") as f:
        hdr = np.zeros(HDR, dtype=np.int32)
        hdr[0] = MAGIC
        f.write(hdr.tobytes())
        while True:
            live = [s for s in srcs if s["emitted"] < s["use"]]
            if not live:
                break
            s = min(live, key=lambda s: s["emitted"] / s["use"])   # most-behind source
            buf = take(s, C)
            if buf is None or len(buf) == 0:
                s["use"] = s["emitted"]                            # exhausted early -> stop scheduling
                continue
            f.write(buf.astype(np.uint32, copy=False).tobytes())
            written += len(buf)
            per[s["name"]] += len(buf)
            if written >= next_report:
                el = time.time() - t0
                print(f"[flatten] {written/1e9:6.2f}B / {total/1e9:.2f}B  {written/max(1e-9, el)/1e6:.1f}M tok/s", flush=True)
                next_report += 5_000_000_000
    with open(out_bin, "r+b") as f:
        f.seek(8)
        f.write(np.array([written & 0xFFFFFFFF], dtype=np.uint32).tobytes())
        f.write(np.array([written >> 32], dtype=np.uint32).tobytes())
    gb = os.path.getsize(out_bin) / (1024 ** 3)
    print(f"\n[flatten] DONE: {written:,} tokens ({written/1e9:.3f}B) -> {out_bin} ({gb:.2f} GB), chunk={C}")
    for s in srcs:
        print(f"  {s['name']:22} {per[s['name']]/1e9:6.3f}B ({100*per[s['name']]/max(1, written):4.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--scale", type=float, default=1.0); p.add_argument("--all", action="store_true")
    p = sub.add_parser("inspect"); p.add_argument("--source", required=True); p.add_argument("--n", type=int, default=2)
    p = sub.add_parser("tokenize")
    p.add_argument("--source", required=True); p.add_argument("--workers", type=int, default=16)
    p.add_argument("--scale", type=float, default=1.0); p.add_argument("--no_decontam", action="store_true")
    p = sub.add_parser("finalize")
    p.add_argument("--scale", type=float, default=1.0); p.add_argument("--include", nargs="*", default=[])
    p = sub.add_parser("flatten")   # -> dense flat llm.c .bin(s); --holdout_frac/--part carve DISJOINT slices
    p.add_argument("--scale", type=float, default=1.0); p.add_argument("--chunk", type=int, default=262144)
    p.add_argument("--out_bin", default=""); p.add_argument("--include", nargs="*", default=[])
    p.add_argument("--holdout_frac", type=float, default=0.0,
                   help="Reserve this fraction of EACH source's used tokens for a disjoint held-out slice "
                        "(e.g. 0.25 -> main=first 75%%, holdout=last 25%% of every source; same composite).")
    p.add_argument("--part", choices=["all", "main", "holdout"], default="all",
                   help="all=whole corpus (default); main=first (1-holdout_frac) of each source; "
                        "holdout=last holdout_frac. main+holdout are DISJOINT + same-composite.")
    args = ap.parse_args()
    {"list": cmd_list, "inspect": cmd_inspect, "tokenize": cmd_tokenize,
     "finalize": cmd_finalize, "flatten": cmd_flatten}[args.cmd](args)


if __name__ == "__main__":
    main()
