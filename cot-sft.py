#!/usr/bin/env python3
"""
SFT training script for custom HuggingFace CausalLM models.

Key features:
- Dynamically imports --model_def to register custom HF model types.
- Loads model/tokenizer from local or remote --model_path.
- Extends context by updating rope_theta + max_position_embeddings and
  rebuilding rotary_emb at EVERY layer from --model_def.
- Loads chat data from either:
  - datasets load_from_disk() directory, or
  - JSONL file with `messages` column.
- Data is expected to already contain <think>...</think> blocks in
  assistant turns. Generation-prompt rendering keeps
  enable_thinking=True so templates do not inject an empty
  <think>...</think> scaffold that breaks supervision alignment.
- Masks loss so only assistant tokens contribute to training.
- Manual input/label shift in ShiftedLossTrainer because ArgonneModel's
  forward() computes loss as cross_entropy(logits, labels) with NO
  internal shift — the caller must align inputs and targets.
- Auto-detects the end-of-turn token from the chat template and sets
  it as eos_token so generation stops correctly.
- Supports DDP/multi-GPU when launched with torchrun.
"""

import argparse
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"


QUALITY_QUESTIONS = [
    "Explain what a black hole is in a way a 10-year-old would understand.",
    "Does charge curve spacetime? Explain why or why not.",
    "Why were elements heavier than lithium not produced in large amounts during Big Bang nucleosynthesis?",
    "What is the typical orientation of polarizing filters in sunglasses, and why?",
    "If the pH of a solution is raised from 7.00 to 12.0, how does OH- concentration change?",
    "Write a short poem about the ocean at night.",
]
MAX_NEW_TOKENS_QUALITY = 1024
QUALITY_DO_SAMPLE = False
QUALITY_TEMPERATURE = 0.7
QUALITY_TOP_P = 0.9
QUALITY_TOP_K = 40
QUALITY_SEED_THINK = False
QUALITY_FORCE_MCQ_POSTPROCESS = False
QUALITY_FORCE_NON_MCQ_POSTPROCESS = True
QUALITY_LOG_RAW = False
QUALITY_NO_REPEAT_NGRAM = 10
QUALITY_REPETITION_PENALTY = 1.0
NO_TEXT_GENERATION = True
QUALITY_TOKENS_AFTER_ANSWER = 96
QUALITY_LOOP_NGRAM = 12
QUALITY_LOOP_REPEATS = 4
QUALITY_MAX_THINK_WORDS = 1024
QUALITY_THINK_PROMPT = (
    "Reason inside <think>...</think> in plain language, then use that reasoning to produce the final answer without repeating the reasoning verbatim."
)
QUALITY_SYSTEM_PROMPT = (
    "You are a careful scientific reasoning assistant. "
    "Use the reasoning to reach the conclusion. Answer directly and specifically in plain natural language. Do not write code or meta commentary."
)
GENERIC_REASONING_FALLBACK_PATTERNS = (
    "apply the key principle and state the conclusion directly.",
    "apply the key principle",
    "state the conclusion directly",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# EOS detection from chat template
# ---------------------------------------------------------------------------

def detect_eos_from_template(tokenizer) -> int:
    """Find the end-of-turn token the chat template places after assistant
    content by rendering a minimal conversation and walking backwards past
    any trailing whitespace tokens."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]

    for i in range(len(ids) - 1, -1, -1):
        token_str = tokenizer.decode([ids[i]]).strip()
        if token_str:
            return ids[i]
    return tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Model definition loader
# ---------------------------------------------------------------------------

def import_model_definition(model_def_path: str):
    if not os.path.isfile(model_def_path):
        raise FileNotFoundError(f"--model_def not found: {model_def_path}")
    module_name = f"custom_model_def_{abs(hash(os.path.abspath(model_def_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, model_def_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model definition from: {model_def_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "RotaryEmbedding"):
        raise AttributeError(
            f"RotaryEmbedding class not found in model definition: {model_def_path}"
        )
    if hasattr(module, "ArgonneModel"):
        missing = list(getattr(module.ArgonneModel, "_keys_to_ignore_on_load_missing", []) or [])
        if r"lm_head\.weight" not in missing:
            missing.append(r"lm_head\.weight")
        module.ArgonneModel._keys_to_ignore_on_load_missing = missing
    return module


def load_hf_state_dict(model_dir: str):
    from safetensors.torch import load_file

    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single_path):
        print(f"Loading weights: {single_path}", flush=True)
        state_dict = load_file(single_path)
        print(f"Loaded {len(state_dict)} tensors from safetensors.", flush=True)
        return state_dict

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json found in: {model_dir}"
        )

    print(f"Loading sharded weights index: {index_path}", flush=True)
    with open(index_path) as f:
        index_obj = json.load(f)
    weight_map = index_obj.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"No weight_map found in safetensors index: {index_path}")

    shard_names = list(dict.fromkeys(weight_map.values()))
    state_dict = {}
    for shard_name in shard_names:
        shard_path = os.path.join(model_dir, shard_name)
        print(f"Loading shard: {shard_path}", flush=True)
        shard_state = load_file(shard_path)
        overlap = state_dict.keys() & shard_state.keys()
        if overlap:
            sample = ", ".join(sorted(list(overlap))[:4])
            raise ValueError(f"Duplicate tensor keys across shards: {sample}")
        state_dict.update(shard_state)
    print(
        f"Loaded {len(state_dict)} tensors from {len(shard_names)} safetensors shards.",
        flush=True,
    )
    return state_dict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(data_path: str) -> Dataset:
    if os.path.isdir(data_path):
        ds = load_from_disk(data_path)
        if isinstance(ds, DatasetDict) and "train" in ds:
            ds = ds["train"]
    elif os.path.isfile(data_path) and data_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=data_path, split="train")
    else:
        raise ValueError(
            f"--data_path must be a load_from_disk directory or a .jsonl file, got: {data_path}"
        )

    if "messages" not in ds.column_names:
        raise ValueError(f"'messages' column not found in dataset columns: {ds.column_names}")
    return ds


def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize message dicts, dropping any with invalid roles or empty content."""
    cleaned: List[Dict[str, str]] = []
    for m in messages or []:
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def extract_input_ids(tokenized: Any) -> List[int]:
    """Normalize apply_chat_template outputs into a plain List[int]."""
    ids: Any
    if isinstance(tokenized, dict):
        ids = tokenized.get("input_ids")
    elif hasattr(tokenized, "input_ids"):
        ids = getattr(tokenized, "input_ids")
    else:
        ids = tokenized

    if ids is None:
        raise ValueError("apply_chat_template did not return input_ids")
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


# ---------------------------------------------------------------------------
# Assistant-only loss masking
# ---------------------------------------------------------------------------

def assistant_has_reasoning_format(messages: List[Dict[str, str]]) -> bool:
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            text = msg["content"]
            if "<think>" not in text or "</think>" not in text:
                return False
            after = text.split("</think>", 1)[1].strip()
            return bool(after)
    return False


THINK_SPAN_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
ANSWER_MARKERS = ("**Answer:**", "**Answer**", "Answer:")
TRAINING_META_PATTERNS = (
    "the user asks",
    "the user asked",
    "the user is asking",
    "the user is",
    "the user likely",
    "the user might",
    "the system",
    "system says",
    "the question",
    "this question",
    "we need to answer",
    "we need to respond",
    "we are asked",
    "let's",
    "let me think",
    "problem:",
    "thus answer",
    "thus final",
    "final answer",
    "multiple choice question",
    "multiple-choice",
    "options are",
    "the options",
    "option a",
    "option b",
    "option c",
    "option d",
    "provide complete code",
    "final output",
    "make sure to",
    "system instruction",
    "short answer",
    "short explanation",
    "concise answer",
    "concise explanation",
    "provide answer",
    "the instruction",
    "instruction:",
    "helpful assistant",
    "assistant should",
    "the prompt",
    "provide a short reasoning",
    "provide a brief reasoning",
    "provide concise reasoning",
    "provide reasoning",
    "short reasoning",
    "brief reasoning",
    "the reasoning should be",
    "the answer should be",
    "the answer should be a short, direct statement",
    "the reasoning should be short",
    "the reasoning should be short and to the point",
    "then output the answer",
    "output the answer",
    "the problem is a simple factual question",
    "the problem is straightforward",
    "the problem is simple",
    "simple factual question",
    "answer directly and specifically",
    "short and to the point",
    "provide code",
    "didn't ask for code",
    "did not ask for code",
    "didn t ask for code",
    "didn't request code",
    "did not request code",
    "be concise",
    "will also include reasoning",
    "no meta commentary",
    "no code",
    "code snippet",
    "<a/b/c/d>",
    "a/b/c/d",
) + GENERIC_REASONING_FALLBACK_PATTERNS
TRAINING_ANSWER_META_PATTERNS = (
    "<a/b/c/d>",
    "a/b/c/d",
    "code snippet",
    "def ",
    "print(",
    "if __name__",
    "```",
    "provide a short reasoning",
    "provide a brief reasoning",
    "provide concise reasoning",
    "provide reasoning",
    "short reasoning",
    "brief reasoning",
    "the reasoning should be",
    "the answer should be",
    "then output the answer",
    "output the answer",
    "the problem is a simple factual question",
    "the problem is straightforward",
    "the problem is simple",
    "simple factual question",
    "answer directly and specifically",
) + GENERIC_REASONING_FALLBACK_PATTERNS
TRAINING_MIN_THINK_WORDS = 12
TRAINING_MAX_THINK_WORDS = 512
TRAINING_MAX_ANSWER_WORDS = 96
TRAINING_MAX_ANSWER_OVERLAP = 0.85
TRAINING_REPEAT_NGRAM = 6
TRAINING_REPEAT_MAX_COUNT = 4
TRAINING_SELF_REF_META_RE = re.compile(
    r"\b(?:i|we)\s+(?:need|must|should|have to)\s+(?:answer|respond|output|provide|give|write|produce|state|explain)\b",
    flags=re.IGNORECASE,
)


def has_excessive_word_repetition(words: List[str]) -> bool:
    if len(words) < TRAINING_REPEAT_NGRAM * TRAINING_REPEAT_MAX_COUNT:
        return False

    unique_ratio = len(set(words)) / float(len(words))
    if unique_ratio < 0.32:
        return True

    ngrams = [
        tuple(words[i : i + TRAINING_REPEAT_NGRAM])
        for i in range(0, len(words) - TRAINING_REPEAT_NGRAM + 1)
    ]
    if not ngrams:
        return False
    return Counter(ngrams).most_common(1)[0][1] >= TRAINING_REPEAT_MAX_COUNT


def word_overlap_ratio(left: List[str], right: List[str]) -> float:
    left_set = {
        re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", w.lower())
        for w in left
        if w
    }
    right_set = {
        re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", w.lower())
        for w in right
        if w
    }
    left_set.discard("")
    right_set.discard("")
    if not right_set:
        return 0.0
    return len(left_set & right_set) / float(len(right_set))


def clean_training_think_span(think_text: str) -> str:
    normalized = re.sub(r"(?i)<\s*a\s*/\s*b\s*/\s*c\s*/\s*d\s*>", " ", think_text)
    normalized = re.sub(r"(?i)\b(a\s*/\s*b\s*/\s*c\s*/\s*d)\b", " ", normalized)
    normalized = re.sub(
        r"(?is)\bA\s*:.*?\bB\s*:.*?\bC\s*:.*?\bD\s*:.*?(?=(?:\n|$))",
        " ",
        normalized,
    )
    sentence_candidates = re.split(r"(?<=[.!?])\s+|\n+", normalized)

    kept_sentences: List[str] = []
    seen_sentences: set[str] = set()
    for sent in sentence_candidates:
        s = re.sub(r"\s+", " ", sent).strip()
        if not s or s.startswith("```"):
            continue
        if "```" in s:
            continue
        lower = s.lower()
        if any(pat in lower for pat in TRAINING_META_PATTERNS):
            continue
        if re.search(r"\b(?:the\s+)?user\b", lower):
            continue
        if re.search(r"\bsystem\b", lower):
            continue
        if re.search(r"\b(?:instruction|assistant|prompt|final line)\b", lower):
            continue
        if re.search(r"\bthe answer is\b", lower):
            continue
        if re.search(r"\banswer\b", lower):
            continue
        if re.search(r"\bcode\b", lower):
            continue
        if TRAINING_SELF_REF_META_RE.search(lower):
            continue
        if re.search(r"(?:\boption\s*[A-D]\b|\b[A-D]\s*:)", s, flags=re.IGNORECASE):
            continue
        if lower.startswith(("answer:", "**answer", "thus answer", "final answer")):
            continue
        if len(s.split()) < 4:
            continue
        if s in seen_sentences:
            continue
        seen_sentences.add(s)
        kept_sentences.append(s)

    cleaned = re.sub(r"\s+", " ", " ".join(kept_sentences)).strip()
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) < TRAINING_MIN_THINK_WORDS:
        return ""
    if len(words) > TRAINING_MAX_THINK_WORDS:
        words = words[:TRAINING_MAX_THINK_WORDS]
    if has_excessive_word_repetition(words):
        return ""
    return " ".join(words)


def parse_abcd_options(text: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for m in re.finditer(r"([A-D]):\s*(.*?)(?=\s+[A-D]:|$)", text, flags=re.IGNORECASE | re.DOTALL):
        option_text = re.sub(r"\s+", " ", m.group(2)).strip()
        if option_text:
            options[m.group(1).upper()] = option_text
    return options


def extract_first_answer_text(text: str) -> Optional[str]:
    candidate = extract_answer_block(text)
    if candidate is None:
        return None
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in candidate.splitlines()]
    for line in lines:
        if not line:
            continue
        line = re.sub(r"^[*\-#\s:]+", "", line).strip()
        line = line.strip("* ").strip()
        if not line:
            continue
        lower = line.lower()
        if any(pat in lower for pat in TRAINING_ANSWER_META_PATTERNS):
            return None
        if "<think" in lower or "</think" in lower:
            return None
        if lower.startswith(("the user", "user asks", "system instruction", "you are")):
            return None
        return line
    return None


def extract_answer_block(text: str) -> Optional[str]:
    if "</think>" not in text:
        return None
    candidate = text.split("</think>", 1)[1].strip()
    if not candidate:
        return None

    for marker in ANSWER_MARKERS:
        if marker in candidate:
            candidate = candidate.split(marker, 1)[1].strip()
            break

    lines: List[str] = []
    for raw_line in candidate.splitlines():
        line = re.sub(r"^[*\-#\s:]+", "", raw_line).strip()
        line = line.strip("* ").strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)

    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()

    answer = "\n".join(lines).strip()
    return answer or None


def normalize_compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_first_answer_letter(text: str) -> Optional[str]:
    answer_line = extract_first_answer_text(text)
    if not answer_line:
        return None
    lead = re.match(r"^([ABCD])(?:\b|[\):.\-])", answer_line, flags=re.IGNORECASE)
    if lead:
        return lead.group(1).upper()
    opt = re.search(r"\b(?:option|choice)\s*([ABCD])\b", answer_line, flags=re.IGNORECASE)
    if opt:
        return opt.group(1).upper()
    return None


def maybe_expand_mcq_letter_answer(answer_line: str, user_text: str) -> str:
    stripped = answer_line.strip()
    with_text = re.match(r"^([ABCD])\s*[\)\].:\-–]\s*(.+)$", stripped, flags=re.IGNORECASE)
    if with_text and with_text.group(2).strip():
        return with_text.group(2).strip()

    m = re.match(r"^([ABCD])(?:\b|[\):.\-].*)?$", stripped, flags=re.IGNORECASE)
    if not m:
        return stripped
    letter = m.group(1).upper()
    options = parse_abcd_options(user_text)
    option_text = options.get(letter, "").strip()
    if option_text:
        return option_text
    return ""


def canonicalize_reasoning_turn(text: str, user_text: str = "") -> Optional[str]:
    span = THINK_SPAN_RE.search(text)
    if span is None:
        return None
    think_clean = clean_training_think_span(span.group(1))
    if not think_clean:
        return None

    answer_block = extract_answer_block(text)
    if answer_block is None:
        return None

    answer_block = answer_block.strip()
    answer_lines = [ln for ln in answer_block.splitlines() if ln.strip()]
    if len(answer_lines) == 1:
        answer_block = maybe_expand_mcq_letter_answer(answer_lines[0], user_text)
    else:
        answer_block = "\n".join(answer_lines).strip()

    if not answer_block:
        return None
    if re.fullmatch(r"[ABCD]", answer_block, flags=re.IGNORECASE):
        return None
    answer_words = answer_block.split()
    if len(answer_words) > TRAINING_MAX_ANSWER_WORDS:
        return None
    lower_answer = answer_block.lower()
    if any(
        tok in lower_answer
        for tok in (
            "user",
            "system",
            "instruction",
            "helpful assistant",
            "code",
            "provide a short reasoning",
            "provide a brief reasoning",
            "provide concise reasoning",
            "provide reasoning",
            "short reasoning",
            "brief reasoning",
            "the reasoning should be",
            "the answer should be",
            "then output the answer",
            "output the answer",
            "the problem is a simple factual question",
            "the problem is straightforward",
            "the problem is simple",
            "simple factual question",
            "answer directly and specifically",
            "short and to the point",
        )
    ):
        return None
    if any(tok in lower_answer for tok in GENERIC_REASONING_FALLBACK_PATTERNS):
        return None
    if word_overlap_ratio(answer_words, think_clean.split()) > TRAINING_MAX_ANSWER_OVERLAP:
        return None
    if normalize_compact_text(answer_block) == normalize_compact_text(think_clean):
        return None

    return f"<think>\n{think_clean}\n</think>\n\n**Answer:** {answer_block}"


def maybe_truncate_think_span(text: str, tokenizer, max_think_tokens: int) -> str:
    if max_think_tokens <= 0:
        return text
    match = THINK_SPAN_RE.search(text)
    if not match:
        return text

    think_content = match.group(1).strip()
    think_ids = tokenizer.encode(think_content, add_special_tokens=False)
    if len(think_ids) <= max_think_tokens:
        return text

    truncated_ids = think_ids[:max_think_tokens]
    truncated_think = tokenizer.decode(truncated_ids, skip_special_tokens=False).strip()
    start, end = match.span(1)
    return text[:start] + "\n" + truncated_think + "\n" + text[end:]


def find_last_assistant_turn(messages: List[Dict[str, str]]) -> Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant" and messages[i]["content"].strip():
            return i
    return None


def build_masked_example(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
    max_think_tokens: int = 0,
    preserve_raw_reasoning: bool = False,
) -> Optional[Dict[str, List[int]]]:
    """Build one supervised example with loss only on the final assistant turn."""
    if not messages:
        return None
    target_idx = find_last_assistant_turn(messages)
    if target_idx is None or target_idx <= 0:
        return None

    last_user_idx = None
    for i in range(target_idx - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None
    target_turn = dict(messages[target_idx])
    if not assistant_has_reasoning_format([target_turn]):
        return None
    if preserve_raw_reasoning:
        target_turn["content"] = target_turn["content"].strip()
    else:
        canonical_target = canonicalize_reasoning_turn(
            target_turn["content"],
            user_text=messages[last_user_idx]["content"],
        )
        if canonical_target is None:
            return None
        target_turn["content"] = canonical_target
    target_turn["content"] = maybe_truncate_think_span(
        target_turn["content"],
        tokenizer=tokenizer,
        max_think_tokens=max_think_tokens,
    )

    # Keep the full conversation history up to the target assistant turn so
    # multi-turn corpora (for example Ultrachat) preserve their context.
    context = messages[:target_idx]

    prefix_ids = extract_input_ids(
        tokenizer.apply_chat_template(
            context,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )
    full_ids = extract_input_ids(
        tokenizer.apply_chat_template(
            context + [target_turn],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    )

    if len(prefix_ids) >= len(full_ids):
        return None
    if len(full_ids) > max_seq_len:
        target_ids = full_ids[len(prefix_ids):]
        if len(target_ids) >= max_seq_len:
            return None
        keep_prefix = max_seq_len - len(target_ids)
        prefix_ids = prefix_ids[-keep_prefix:] if keep_prefix > 0 else []
        full_ids = prefix_ids + target_ids

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    if all(v == -100 for v in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LazySFTDataset(TorchDataset):
    def __init__(
        self,
        ds: Dataset,
        tokenizer,
        max_seq_len: int,
        max_think_tokens: int = 0,
        max_reasoning_rows: int = 0,
        preserve_raw_reasoning: bool = False,
    ):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_think_tokens = max_think_tokens
        self.preserve_raw_reasoning = preserve_raw_reasoning
        self.rng = random.Random(20260330)

        candidate_indices = list(range(len(ds)))
        if "num_tokens" in ds.column_names:
            # Skip pathological ultra-long chats so dataset filtering stays tractable.
            safety_max_tokens = max_seq_len * 8
            candidate_indices = [
                i
                for i, n in enumerate(ds["num_tokens"])
                if isinstance(n, (int, float)) and int(n) <= safety_max_tokens
            ]
        rng = random.Random(20260330)
        rng.shuffle(candidate_indices)

        if max_reasoning_rows > 0 and len(candidate_indices) > max_reasoning_rows:
            candidate_indices = candidate_indices[:max_reasoning_rows]

        # Keep the full reasoning pool and validate rows lazily during batching.
        # This avoids a full dataset pre-scan at startup for very large corpora.
        self.indices = candidate_indices
        if not self.indices:
            raise RuntimeError("No usable examples with messages found.")

    def __len__(self) -> int:
        return len(self.indices)

    def _build(self, raw_idx: int) -> Optional[Dict[str, List[int]]]:
        example = self.ds[raw_idx]
        msgs = example.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return None
        conv = clean_messages(msgs)
        if not conv:
            return None
        if not assistant_has_reasoning_format(conv):
            return None
        return build_masked_example(
            conv,
            self.tokenizer,
            self.max_seq_len,
            max_think_tokens=self.max_think_tokens,
            preserve_raw_reasoning=self.preserve_raw_reasoning,
        )

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        built = self._build(self.indices[idx % len(self.indices)])
        if built is not None:
            return built
        for _ in range(64):
            built = self._build(self.indices[self.rng.randrange(len(self.indices))])
            if built is not None:
                return built
        raise RuntimeError("Failed to build a valid tokenized example.")


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            labels = f["labels"]
            pad = max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad)
            batch_attention_mask.append(mask + [0] * pad)
            batch_labels.append(labels + [-100] * pad)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Quality / eval helpers
# ---------------------------------------------------------------------------

def chat_to_input_ids(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    return [int(x) for x in encoded]


def build_quality_prompt(question: str) -> str:
    # Keep the probe close to training-time inputs: the model sees the raw
    # question, then continues in the learned <think>...</think> format.
    return question.strip()


def build_quality_messages(question: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": build_quality_prompt(question)}]


def has_repeated_ngram_tail(token_ids: List[int], n: int, repeats: int) -> bool:
    if n <= 0 or repeats <= 1:
        return False
    if len(token_ids) < n * repeats:
        return False
    tail = token_ids[-n:]
    for i in range(2, repeats + 1):
        if token_ids[-i * n : -(i - 1) * n] != tail:
            return False
    return True


def get_no_repeat_ngram_banned_tokens(token_ids: List[int], n: int) -> List[int]:
    if n <= 1 or len(token_ids) + 1 < n:
        return []
    prefix = tuple(token_ids[-(n - 1):]) if n > 1 else tuple()
    banned: List[int] = []
    for i in range(0, len(token_ids) - n + 1):
        if tuple(token_ids[i : i + n - 1]) == prefix:
            banned.append(token_ids[i + n - 1])
    if not banned:
        return []
    return sorted(set(banned))


def apply_repetition_penalty(logits: torch.Tensor, token_ids: List[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not token_ids:
        return logits

    unique_token_ids = sorted({int(tok) for tok in token_ids})
    if not unique_token_ids:
        return logits

    token_index = torch.tensor(unique_token_ids, device=logits.device, dtype=torch.long)
    token_scores = logits.index_select(-1, token_index)
    adjusted_scores = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
    logits = logits.clone()
    logits.index_copy_(-1, token_index, adjusted_scores)
    return logits


@torch.no_grad()
def generate_quality_sequence(
    model_to_eval,
    tokenizer,
    input_ids: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    eos_id = tokenizer.eos_token_id
    generated = input_ids
    start_len = input_ids.shape[1]
    answer_start_len: Optional[int] = None

    while generated.shape[1] < max_length:
        chunk = generated[:, -model_to_eval.config.max_position_embeddings :]
        outputs = model_to_eval(chunk)
        logits = outputs.logits[:, -1, :]
        gen_ids = generated[0].tolist()
        if QUALITY_NO_REPEAT_NGRAM > 1:
            banned_tokens = get_no_repeat_ngram_banned_tokens(gen_ids, QUALITY_NO_REPEAT_NGRAM)
            if banned_tokens and len(banned_tokens) < logits.size(-1):
                logits[:, banned_tokens] = float("-inf")
        logits = apply_repetition_penalty(logits, gen_ids, QUALITY_REPETITION_PENALTY)

        if QUALITY_DO_SAMPLE:
            temperature = max(QUALITY_TEMPERATURE, 1e-5)
            logits = logits / temperature

            if QUALITY_TOP_K > 0:
                top_values, _ = torch.topk(logits, min(QUALITY_TOP_K, logits.size(-1)))
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
            if QUALITY_TOP_P is not None and 0.0 < QUALITY_TOP_P < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > QUALITY_TOP_P
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)
        next_id = int(next_token.item())
        if eos_id is not None and next_id == eos_id:
            break

        gen_len = generated.shape[1] - start_len
        if gen_len < 16 or gen_len % 16 != 0:
            continue

        gen_ids = generated[0, start_len:].tolist()
        if has_repeated_ngram_tail(gen_ids, n=QUALITY_LOOP_NGRAM, repeats=QUALITY_LOOP_REPEATS):
            break

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        answer_pos = text.find("**Answer:")
        if answer_pos == -1:
            answer_pos = text.find("**Answer**")
        if answer_pos != -1:
            if answer_start_len is None:
                answer_start_len = gen_len
            elif gen_len - answer_start_len >= QUALITY_TOKENS_AFTER_ANSWER:
                break

    return generated


def is_mcq_question(question: str) -> bool:
    q = question.upper()
    return all(f"{letter}:" in q for letter in ("A", "B", "C", "D"))


def parse_mcq_options(question: str) -> Dict[str, str]:
    return parse_abcd_options(question)


def is_yes_no_question(question: str) -> bool:
    q = question.strip().lower()
    if "yes or no" in q or "yes/no" in q:
        return True
    if any(
        phrase in q
        for phrase in (
            "why or why not",
            "explain why",
            "explain how",
            "explain whether",
            "what is the reason",
            "why does",
            "why do",
            "why is",
            "how does",
            "how do",
            "what happens",
        )
    ):
        return False
    first = q.split(" ", 1)[0] if q else ""
    if first in {"is", "are", "was", "were", "does", "do", "has", "have", "had", "can", "could", "should", "will", "would", "did", "may", "might", "shall", "must"}:
        return True
    return False


def extract_yes_no_from_answer_line(text: str) -> Optional[str]:
    answer_line = extract_first_answer_text(text)
    if not answer_line:
        return None
    lower = answer_line.lower()
    yes_pos = lower.find("yes")
    no_pos = lower.find("no")
    if yes_pos == -1 and no_pos == -1:
        return None
    if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos):
        return "Yes."
    if no_pos != -1:
        return "No."
    return None


def clean_think_text(raw_text: str) -> str:
    text = raw_text.strip()
    if "<think>" in text:
        text = text.split("<think>", 1)[1]
    if "</think>" in text:
        text = text.split("</think>", 1)[0]
    for marker in ("**Answer:**", "**Answer**", "Answer:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    boilerplate = (
        "the user asks:",
        "the user is",
        "the user question",
        "the user gave",
        "the user just asked",
        "the user wants a simple answer",
        "the user wants a concise answer",
        "the user wants a short poem",
        "the user wants answer",
        "the system says",
        "system says",
        "system instruction",
        "instruction:",
        "multiple choice",
        "we'll output the answer",
        "we will output the answer",
        "we'll answer with the answer",
        "we'll keep it concise",
        "we'll keep it natural",
        "we'll keep it short",
        "we'll provide a concise answer",
        "we'll provide a short reasoning",
        "provide a simple python script",
        "we could also include",
        "we might also include",
        "that would be a complete code",
        "that would be complete code",
        "we'll produce a short code",
        "we'll produce a small program",
        "we'll produce code",
        "we'll output the code",
        "short snippet",
        "snippet that prints",
        "the code is simple",
        "if __name__ == \"__main__\"",
        "print the answer",
        "provide the answer",
        "provide a concise answer",
        "provide a short poem",
        "provide a brief explanation",
        "provide concise answer",
        "provide concise reasoning",
        "answer is",
        "the answer is",
        "not asking for code",
        "didn't ask for code",
        "did not ask for code",
        "provide code",
        "provide a concise answer",
        "short answer",
        "short explanation",
        "we need to answer",
        "thus final",
        "provide reasoning",
        "provide a short reasoning",
        "provide a brief reasoning",
        "provide concise reasoning",
        "short reasoning",
        "brief reasoning",
        "the reasoning should be",
        "the answer should be",
        "then output the answer",
        "output the answer",
        "the problem is a simple factual question",
        "the problem is straightforward",
        "the problem is simple",
        "simple factual question",
        "answer directly and specifically",
        "short and to the point",
        "question:",
    ) + GENERIC_REASONING_FALLBACK_PATTERNS
    # Keep only substantive sentences and drop meta boilerplate.
    sentence_candidates = re.split(r"(?<=[.!?])\\s+|\\n+", text)
    kept: List[str] = []
    seen_sentences: set[str] = set()
    for sent in sentence_candidates:
        s = sent.strip()
        if not s:
            continue
        lower = s.lower()
        normalized = re.sub(r"\s+", " ", lower)
        if normalized in seen_sentences:
            continue
        seen_sentences.add(normalized)
        if any(marker in lower for marker in boilerplate):
            continue
        if lower.count("p_{\\text{") >= 2:
            continue
        if s.startswith("**Answer") or s.startswith("Answer:"):
            continue
        if len(s.split()) < 4:
            continue
        kept.append(s)
        if len(" ".join(kept).split()) >= QUALITY_MAX_THINK_WORDS:
            break
    cleaned = " ".join(kept).strip()
    if cleaned:
        words = cleaned.split()
        if len(words) > QUALITY_MAX_THINK_WORDS:
            cleaned = " ".join(words[:QUALITY_MAX_THINK_WORDS]).strip()
        return cleaned
    return ""


def clean_open_ended_answer_text(raw_text: str, think_text: str = "") -> str:
    text = raw_text.strip()
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    for marker in ANSWER_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    text = text.strip("* `\"'").strip()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    lower = text.lower()
    if any(pat in lower for pat in GENERIC_REASONING_FALLBACK_PATTERNS):
        return ""
    if think_text and normalize_compact_text(text) == normalize_compact_text(think_text):
        return ""
    return text


@torch.no_grad()
def select_mcq_answer_letter(
    model_to_eval,
    tokenizer,
    question: str,
    think_text: str,
    device: torch.device,
) -> str:
    prompt_ids = chat_to_input_ids(
        tokenizer,
        build_quality_messages(question),
        add_generation_prompt=True,
    )
    answer_prefix = f"<think>\n{think_text}\n</think>\n\n**Answer:**"
    prefix_ids = prompt_ids + tokenizer.encode(answer_prefix, add_special_tokens=False)
    max_pos = model_to_eval.config.max_position_embeddings

    def score_suffix(suffix_ids: List[int]) -> float:
        if not suffix_ids:
            return float("-inf")
        ids = list(prefix_ids)
        total = 0.0
        for tok in suffix_ids:
            chunk = torch.tensor([ids[-max_pos:]], dtype=torch.long, device=device)
            logits = model_to_eval(chunk).logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            total += float(log_probs[0, tok].item())
            ids.append(tok)
        return total / max(1, len(suffix_ids))

    best_letter = "A"
    best_score = float("-inf")
    options = parse_mcq_options(question)
    for letter in ("A", "B", "C", "D"):
        variants = []
        option_text = options.get(letter, "").strip()
        text_variants = [f" {letter}", letter, f"**{letter}**"]
        if option_text:
            text_variants.extend(
                [
                    f" {letter}: {option_text}",
                    f" {letter} ({option_text})",
                ]
            )
        for v in text_variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if ids:
                variants.append(ids)
        if not variants:
            continue
        score = max(score_suffix(ids) for ids in variants)
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


@torch.no_grad()
def select_yes_no_answer(
    model_to_eval,
    tokenizer,
    question: str,
    think_text: str,
    device: torch.device,
) -> str:
    prompt_ids = chat_to_input_ids(
        tokenizer,
        build_quality_messages(question),
        add_generation_prompt=True,
    )
    answer_prefix = f"<think>\n{think_text}\n</think>\n\n**Answer:**"
    prefix_ids = prompt_ids + tokenizer.encode(answer_prefix, add_special_tokens=False)
    max_pos = model_to_eval.config.max_position_embeddings

    def score_variant(variant: str) -> float:
        suffix_ids = tokenizer.encode(variant, add_special_tokens=False)
        if not suffix_ids:
            return float("-inf")
        ids = list(prefix_ids)
        total = 0.0
        for tok in suffix_ids:
            chunk = torch.tensor([ids[-max_pos:]], dtype=torch.long, device=device)
            logits = model_to_eval(chunk).logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            total += float(log_probs[0, tok].item())
            ids.append(tok)
        return total / max(1, len(suffix_ids))

    yes_score = max(score_variant(" Yes"), score_variant("Yes"))
    no_score = max(score_variant(" No"), score_variant("No"))
    return "Yes." if yes_score >= no_score else "No."


@torch.no_grad()
def generate_open_ended_answer(
    model_to_eval,
    tokenizer,
    question: str,
    think_text: str,
    device: torch.device,
) -> str:
    think_text = think_text.strip()
    if not think_text:
        return ""

    prompt_ids = chat_to_input_ids(
        tokenizer,
        build_quality_messages(question),
        add_generation_prompt=True,
    )
    answer_prefix = f"<think>\n{think_text}\n</think>\n\n**Answer:**"
    prefix_ids = prompt_ids + tokenizer.encode(answer_prefix, add_special_tokens=False)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    max_length = min(
        model_to_eval.config.max_position_embeddings,
        input_ids.shape[1] + MAX_NEW_TOKENS_QUALITY,
    )
    out = generate_quality_sequence(
        model_to_eval=model_to_eval,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_length=max_length,
    )
    gen_ids = out[0, input_ids.shape[1] :].tolist()
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(eos_id)]
    raw_answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return clean_open_ended_answer_text(raw_answer, think_text=think_text)


@torch.no_grad()
def answer_questions(model, tokenizer, questions: List[str], tag: str, step: int) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    model_to_eval = model.module if hasattr(model, "module") else model
    was_training = model_to_eval.training
    model_to_eval.eval()

    device = next(model_to_eval.parameters()).device
    old_use_cache = getattr(model_to_eval.config, "use_cache", None)
    if old_use_cache is not None:
        model_to_eval.config.use_cache = True
    print("\n" + "=" * 90)
    print(f"[QUALITY] {tag} | step={step}")
    print("=" * 90)

    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    
    total_think_close = 0
    total_nonempty_think = 0
    total_answer_format = 0
    total_answer_after_close = 0

    for i, q in enumerate(questions, start=1):
        answer_text = ""
        prompt_ids = chat_to_input_ids(
            tokenizer,
            build_quality_messages(q),
            add_generation_prompt=True,
        )
        if QUALITY_SEED_THINK:
            # Seed generation with <think> to match CoT-style continuation.
            prompt_ids = prompt_ids + think_ids
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        max_length = min(
            model_to_eval.config.max_position_embeddings,
            input_ids.shape[1] + MAX_NEW_TOKENS_QUALITY,
        )
        gen_kwargs = {
            "input_ids": input_ids,
            "max_length": max_length,
            "do_sample": QUALITY_DO_SAMPLE,
        }
        if QUALITY_DO_SAMPLE:
            gen_kwargs["temperature"] = QUALITY_TEMPERATURE
            gen_kwargs["top_p"] = QUALITY_TOP_P
            if QUALITY_TOP_K > 0:
                gen_kwargs["top_k"] = QUALITY_TOP_K
        out = generate_quality_sequence(
            model_to_eval=model_to_eval,
            tokenizer=tokenizer,
            input_ids=gen_kwargs["input_ids"],
            max_length=gen_kwargs["max_length"],
        )
        gen_ids = out[0, input_ids.shape[1] :].tolist()
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if is_mcq_question(q) and (QUALITY_FORCE_MCQ_POSTPROCESS or QUALITY_FORCE_NON_MCQ_POSTPROCESS):
            think_text = clean_think_text(raw_text)
            answer_letter = extract_first_answer_letter(raw_text)
            if answer_letter is None:
                answer_letter = select_mcq_answer_letter(
                    model_to_eval=model_to_eval,
                    tokenizer=tokenizer,
                    question=q,
                    think_text=think_text,
                    device=device,
                )
            answer_text = answer_letter
            text = f"<think>\n{think_text}\n</think>\n\n**Answer:** {answer_letter}"
        elif QUALITY_FORCE_NON_MCQ_POSTPROCESS:
            think_text = clean_think_text(raw_text)
            if is_yes_no_question(q):
                answer_text = extract_yes_no_from_answer_line(raw_text)
                if answer_text is None:
                    answer_text = select_yes_no_answer(
                        model_to_eval=model_to_eval,
                        tokenizer=tokenizer,
                        question=q,
                        think_text=think_text,
                        device=device,
                    )
            else:
                # Open-ended eval is raw end-to-end now: never regenerate a fresh
                # answer from the extracted reasoning span.
                answer_text = clean_open_ended_answer_text(
                    extract_answer_block(raw_text) or "",
                    think_text=think_text,
                )
            text = f"<think>\n{think_text}\n</think>\n\n**Answer:** {answer_text}"
        else:
            text = raw_text
            answer_text = extract_first_answer_text(raw_text) or ""
        print(f"\nQ{i}: {q}")
        if QUALITY_LOG_RAW:
            print(f"RAW_A{i}: {raw_text}")
        print(f"A{i}: {text}")
        
        has_think_close = "</think>" in text
        is_objective = is_mcq_question(q) or is_yes_no_question(q)
        has_answer_format = has_think_close and bool(answer_text.strip())

        if has_think_close:
            total_think_close += 1
            think_content = text.split("</think>", 1)[0].strip()
            if think_content:
                total_nonempty_think += 1

        if has_answer_format:
            total_answer_format += 1
            if has_think_close:
                if is_objective:
                    close_pos = text.find("</think>")
                    answer_pos = min(
                        p for p in (text.find("**Answer:"), text.find("**Answer**")) if p != -1
                    )
                    if answer_pos > close_pos:
                        total_answer_after_close += 1
                else:
                    total_answer_after_close += 1

    total_questions = len(questions)
    print(
        "\n[FORMAT CHECK] "
        f"</think> tags: {total_think_close}/{total_questions} | "
        f"non-empty think span: {total_nonempty_think}/{total_questions} | "
        f"**Answer:** format: {total_answer_format}/{total_questions} | "
        f"answer after </think>: {total_answer_after_close}/{total_questions}"
    )
    print(
        "[FORMAT_METRIC] "
        f"tag={tag} step={step} total={total_questions} "
        f"think_close={total_think_close} nonempty_think={total_nonempty_think} "
        f"answer_format={total_answer_format} answer_after_close={total_answer_after_close}"
    )
    print("=" * 90 + "\n")
    print("\n" + "=" * 90 + "\n")

    if old_use_cache is not None:
        model_to_eval.config.use_cache = old_use_cache
    if was_training:
        model_to_eval.train()


class QualityCallback(TrainerCallback):
    def __init__(self, tokenizer, every_steps: int = 200) -> None:
        self.tokenizer = tokenizer
        self.every_steps = every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if NO_TEXT_GENERATION:
            return control
        if state.global_step > 0 and state.global_step % self.every_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                answer_questions(model, self.tokenizer, QUALITY_QUESTIONS, "DURING_SFT", state.global_step)
        return control


class StopAfterCheckpointSaveCallback(TrainerCallback):
    def __init__(self, enabled: bool = False, slice_steps: int = 0) -> None:
        self.enabled = enabled
        self.slice_steps = max(0, int(slice_steps))
        self.start_step: Optional[int] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_step = int(state.global_step)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled or self.slice_steps <= 0:
            return control
        if self.start_step is None:
            self.start_step = max(0, int(state.global_step) - 1)
        target_step = self.start_step + self.slice_steps
        if int(state.global_step) >= target_step:
            print(
                f"Slice step target reached at step {state.global_step}; saving checkpoint.",
                flush=True,
            )
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps > 0 and state.global_step >= max_steps:
            return control
        print(
            f"Checkpoint saved at step {state.global_step}; stopping this training slice.",
            flush=True,
        )
        control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# Shifted-loss Trainer
# ---------------------------------------------------------------------------
# ArgonneModel.forward() does NOT shift internally — it computes:
#   loss = cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
# So the caller must provide:
#   x = input_ids[:, :-1]   (input tokens 0..N-2)
#   y = labels[:, 1:]        (target tokens 1..N-1)
# This way logits[i] (predicted from token i) is trained against token i+1.

class ShiftedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        x = input_ids[:, :-1].contiguous()
        y = labels[:, 1:].contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1].contiguous()

        outputs = model(
            input_ids=x,
            attention_mask=attention_mask,
            labels=y,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# RoPE replacement helper
# ---------------------------------------------------------------------------

def replace_rotary_embeddings(model, RotaryEmbedding, rope_theta: float, max_seq_len: int) -> None:
    """Replace the rotary embedding on the model *and* inside every
    transformer block's attention layer so that all layers use the new
    rope_theta and max_position_embeddings."""
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Top-level (used by ArgonneModel — shared RoPE called in forward()).
    if hasattr(model, "rotary_emb"):
        model.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
        )

    # Per-block replacement (safety net in case any layer holds its own).
    blocks = None
    for attr in ("blocks", "layers", "decoder", "h"):
        if hasattr(model, attr):
            blocks = getattr(model, attr)
            break
    if blocks is None and hasattr(model, "model"):
        for attr in ("blocks", "layers", "decoder", "h"):
            if hasattr(model.model, attr):
                blocks = getattr(model.model, attr)
                break

    if blocks is not None:
        for blk in blocks:
            for sub in [blk] + [getattr(blk, a, None) for a in ("attn", "self_attn", "attention")]:
                if sub is not None and hasattr(sub, "rotary_emb"):
                    sub.rotary_emb = RotaryEmbedding(
                        head_dim,
                        max_position_embeddings=max_seq_len,
                        base=rope_theta,
                    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT for custom HuggingFace model")
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_def", required=True, help="Path to custom model .py")
    p.add_argument("--data_path", required=True, help="JSONL or load_from_disk path")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer_path", default=None, help="Path to tokenizer (defaults to model_path)")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument(
        "--model_context_length",
        type=int,
        default=0,
        help=(
            "RoPE/config context length for the saved model. "
            "Defaults to max(max_seq_length, loaded config max_position_embeddings)."
        ),
    )
    p.add_argument(
        "--preserve_raw_reasoning",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use the original assistant <think>...</think> turn as-is instead of canonicalizing it",
    )
    p.add_argument(
        "--max_think_tokens",
        type=int,
        default=128,
        help="If > 0, truncate tokens inside <think>...</think> to this length for training",
    )
    p.add_argument(
        "--max_reasoning_rows",
        type=int,
        default=0,
        help="If > 0, cap the number of reasoning rows kept after cheap filtering",
    )
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_epochs", type=float, default=1)
    p.add_argument("--max_steps", type=int, default=-1, help="If > 0, override num_epochs with fixed step count")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--quality_steps", type=int, default=100)
    p.add_argument("--max_new_tokens_quality", type=int, default=1024)
    p.add_argument("--quality_do_sample", type=int, default=0, choices=[0, 1])
    p.add_argument("--quality_temperature", type=float, default=0.7)
    p.add_argument("--quality_top_p", type=float, default=0.9)
    p.add_argument("--quality_top_k", type=int, default=40)
    p.add_argument("--quality_seed_think", type=int, default=0, choices=[0, 1])
    p.add_argument(
        "--quality_no_repeat_ngram",
        type=int,
        default=10,
        help="Apply no-repeat ngram decoding during quality generations when > 1",
    )
    p.add_argument(
        "--quality_repetition_penalty",
        type=float,
        default=1.0,
        help="Apply repetition penalty during quality generations when > 1.0",
    )
    p.add_argument("--quality_force_mcq_postprocess", type=int, default=0, choices=[0, 1])
    p.add_argument("--quality_force_non_mcq_postprocess", type=int, default=1, choices=[0, 1])
    p.add_argument("--quality_log_raw", type=int, default=0, choices=[0, 1])
    p.add_argument(
        "--no_text_generation",
        type=int,
        default=1,
        choices=[0, 1],
        help="Skip all sample-question text generation probes when set to 1",
    )
    p.add_argument(
        "--quality_questions_file",
        type=str,
        default="/home/youzhi/ArgonneAI/cot_experiments/root_cause/quality_questions_mcq.txt",
        help="Text file with one quality-eval question per line",
    )
    p.add_argument("--run_before_quality", type=int, default=0, choices=[0, 1])
    p.add_argument("--run_after_quality", type=int, default=1, choices=[0, 1])
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=46)
    p.add_argument("--save_strategy", type=str, default="no", choices=["steps", "no"])
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--skip_final_save", action="store_true", help="Skip final save_pretrained output")
    p.add_argument("--resume_from_checkpoint", default=None, help="HF checkpoint path to resume from")
    p.add_argument(
        "--exit_after_checkpoint_save",
        action="store_true",
        help="Stop the training process after the next checkpoint save so the job wrapper can resubmit",
    )
    p.add_argument(
        "--slice_steps",
        type=int,
        default=0,
        help="If > 0 with --exit_after_checkpoint_save, force a checkpoint after this many resumed training steps",
    )
    return p.parse_args()


def get_local_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def load_quality_questions_from_file(path: str) -> List[str]:
    questions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q:
                questions.append(q)
    if not questions:
        raise ValueError(f"No non-empty questions found in --quality_questions_file: {path}")
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    global MAX_NEW_TOKENS_QUALITY, QUALITY_DO_SAMPLE, QUALITY_TEMPERATURE, QUALITY_TOP_P, QUALITY_TOP_K
    global QUALITY_SEED_THINK, QUALITY_FORCE_MCQ_POSTPROCESS, QUALITY_FORCE_NON_MCQ_POSTPROCESS
    global QUALITY_LOG_RAW, NO_TEXT_GENERATION
    global QUALITY_NO_REPEAT_NGRAM, QUALITY_REPETITION_PENALTY
    MAX_NEW_TOKENS_QUALITY = args.max_new_tokens_quality
    QUALITY_DO_SAMPLE = args.quality_do_sample == 1
    QUALITY_TEMPERATURE = args.quality_temperature
    QUALITY_TOP_P = args.quality_top_p
    QUALITY_TOP_K = max(0, args.quality_top_k)
    QUALITY_SEED_THINK = args.quality_seed_think == 1
    QUALITY_FORCE_MCQ_POSTPROCESS = args.quality_force_mcq_postprocess == 1
    QUALITY_FORCE_NON_MCQ_POSTPROCESS = args.quality_force_non_mcq_postprocess == 1
    QUALITY_LOG_RAW = args.quality_log_raw == 1
    NO_TEXT_GENERATION = args.no_text_generation == 1
    QUALITY_NO_REPEAT_NGRAM = max(0, args.quality_no_repeat_ngram)
    QUALITY_REPETITION_PENALTY = max(1.0, args.quality_repetition_penalty)

    global QUALITY_QUESTIONS
    if not NO_TEXT_GENERATION and args.quality_questions_file:
        QUALITY_QUESTIONS = load_quality_questions_from_file(args.quality_questions_file)
    if args.preserve_raw_reasoning == 1 and args.max_think_tokens > 0:
        print(
            f"WARNING: preserve_raw_reasoning=1 but max_think_tokens={args.max_think_tokens}; "
            "long raw <think> spans will still be truncated.",
            flush=True,
        )
    if (
        not NO_TEXT_GENERATION
        and args.quality_force_non_mcq_postprocess == 1
        and args.quality_force_mcq_postprocess == 0
        and "mcq" in os.path.basename(args.quality_questions_file).lower()
    ):
        print(
            "WARNING: MCQ quality questions are being routed through non-MCQ postprocessing. "
            "This can make the output log misleading.",
            flush=True,
        )

    device = get_local_device()

    model_module = import_model_definition(args.model_def)
    RotaryEmbedding = getattr(model_module, "RotaryEmbedding")

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"

    # Auto-detect the end-of-turn token from the chat template and set it
    # as eos so that generation stops at the right place.
    detected_eos_id = detect_eos_from_template(tokenizer)
    if detected_eos_id != tokenizer.eos_token_id:
        old_eos = repr(tokenizer.eos_token)
        tokenizer.eos_token_id = detected_eos_id
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(detected_eos_id)
        print(
            f"EOS updated: {old_eos} -> {repr(tokenizer.eos_token)} "
            f"(id={detected_eos_id}) [detected from chat template]"
        )
    else:
        print(f"EOS token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")

    # Manual model construction + weight loading to ensure lm_head is
    # properly tied to embed_tokens. AutoModelForCausalLM.from_pretrained
    # fails to re-establish the tie for this custom model.

    ArgonneConfig = getattr(model_module, "ArgonneConfig")
    ArgonneModel = getattr(model_module, "ArgonneModel")

    if os.path.isdir(args.model_path):
        config_path = os.path.join(args.model_path, "config.json")
        print(f"Loading config: {config_path}", flush=True)
        with open(config_path) as f:
            config_dict = json.load(f)
        state_dict = load_hf_state_dict(args.model_path)
    elif os.path.isfile(args.model_path) and args.model_path.endswith(".pt"):
        print(f"Loading training checkpoint: {args.model_path}", flush=True)
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"'model_state_dict' not found in checkpoint: {args.model_path}")
        config_dict = {}
        state_dict = checkpoint["model_state_dict"]
        print(f"Loaded {len(state_dict)} tensors from checkpoint state dict.", flush=True)
        from continue_pretrain import HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS

        config_dict.update(
            {
                "hidden_size": HIDDEN_SIZE,
                "num_hidden_layers": NUM_LAYERS,
                "num_attention_heads": NUM_HEADS,
                "num_key_value_heads": NUM_KV_HEADS,
            }
        )
    else:
        raise ValueError(
            f"--model_path must be a HuggingFace model directory or a .pt training checkpoint, got: {args.model_path}"
        )

    config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}
    vocab_size = None
    for key in ("embed_tokens.weight", "lm_head.weight"):
        if key in state_dict:
            vocab_size = int(state_dict[key].shape[0])
            break
    if vocab_size is not None:
        if len(tokenizer) != vocab_size:
            print(
                f"Tokenizer size {len(tokenizer)} differs from checkpoint vocab {vocab_size}; "
                "using checkpoint vocab for the model.",
                flush=True,
            )
        config_dict["vocab_size"] = vocab_size

    config = ArgonneConfig(**config_dict)
    model = ArgonneModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(
        f"State dict applied. missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}",
        flush=True,
    )
    model.tie_weights()
    print("Weights tied (embed_tokens <-> lm_head).", flush=True)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters ({args.precision})")
    print(f"Tied: embed_tokens == lm_head -> {model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}")

    # ---- Extend RoPE / context ----
    loaded_context_length = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    model_context_length = (
        args.model_context_length
        if args.model_context_length > 0
        else max(args.max_seq_length, loaded_context_length)
    )
    model.config.rope_theta = args.rope_theta
    model.config.max_position_embeddings = model_context_length
    model.config.block_size = model_context_length
    replace_rotary_embeddings(model, RotaryEmbedding, args.rope_theta, model_context_length)
    print(
        "Context setup: "
        f"training max_seq_length={args.max_seq_length}, "
        f"model max_position_embeddings={model_context_length}, "
        f"loaded max_position_embeddings={loaded_context_length}, "
        f"rope_theta={args.rope_theta}",
        flush=True,
    )

    # ---- Flash attention + gradient checkpointing ----
    model.config.use_flash_attention = True
    if hasattr(model, "blocks"):
        for blk in model.blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "use_flash_attention"):
                blk.attn.use_flash_attention = True
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    # ---- Data ----
    print(f"Loading dataset from: {args.data_path}", flush=True)
    ds = load_training_data(args.data_path)
    print(f"Loaded dataset with {len(ds):,} rows from {args.data_path}")

    train_dataset = LazySFTDataset(
        ds,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        max_think_tokens=args.max_think_tokens,
        max_reasoning_rows=args.max_reasoning_rows,
        preserve_raw_reasoning=args.preserve_raw_reasoning == 1,
    )
    print(f"Reasoning rows after cheap filters: {len(train_dataset):,}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_rank0 = torch.distributed.get_rank() == 0
    else:
        is_rank0 = True
    if is_rank0 and args.run_before_quality == 1 and not NO_TEXT_GENERATION:
        answer_questions(model, tokenizer, QUALITY_QUESTIONS, "BEFORE_SFT", 0)

    os.makedirs(args.output_dir, exist_ok=True)

    bf16 = args.precision == "bf16"
    fp16 = args.precision == "fp16"
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        bf16=bf16,
        fp16=fp16,
        save_strategy=args.save_strategy,
        save_steps=max(1, args.save_steps),
        save_total_limit=args.save_total_limit if args.save_strategy != "no" else None,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )

    # ShiftedLossTrainer because ArgonneModel.forward() does NOT shift
    # labels internally — it computes cross_entropy(logits, labels) directly.
    trainer = ShiftedLossTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
        callbacks=[
            QualityCallback(tokenizer=tokenizer, every_steps=args.quality_steps),
            StopAfterCheckpointSaveCallback(
                enabled=args.exit_after_checkpoint_save,
                slice_steps=args.slice_steps,
            ),
        ],
    )

    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    if is_rank0 and args.run_after_quality == 1 and not NO_TEXT_GENERATION:
        answer_questions(model, tokenizer, QUALITY_QUESTIONS, "AFTER_SFT", int(trainer.state.global_step))
    trainer_max_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
    stopped_for_next_slice = (
        args.exit_after_checkpoint_save
        and trainer_max_steps > 0
        and int(trainer.state.global_step) < trainer_max_steps
    )
    if stopped_for_next_slice:
        print("Skipping final model/tokenizer save for incomplete checkpoint slice.")
    elif args.skip_final_save:
        print("Skipping final model/tokenizer save (--skip_final_save).")
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        trainer.save_state()
        print(f"Saved final model/tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
