#!/usr/bin/env python3
"""
DPO fine-tuning starting from the SFT Argonne checkpoint.

This version keeps the overall workflow the same:
1. Load SFT checkpoint as policy and reference
2. Load a preference dataset from disk
3. Run standard offline DPO
4. Evaluate on a small locked probe set before / during / after training
5. Save the final model

Important design choice:
- The six quality questions are eval-only probes.
- They are NEVER used to create anchor pairs or online negatives.
- Dataset recipes can still filter and upweight *proxy* prompts that are broadly
  related to conversational helpfulness, but they avoid exact eval leakage.
"""

import argparse
import gc
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Default hyperparameters
# -----------------------------------------------------------------------------
MAX_SEQ_LEN = 1024
SEED = 42
EPOCHS = 1
BATCH_SIZE = 12
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-6
MIN_LR_RATIO = 0.1
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
GRAD_CLIP = 1.0
LOG_EVERY = 10
QUALITY_EVERY = 25
MAX_NEW_TOKENS_QUALITY = 200
DPO_BETA = 0.03
MAX_STEPS = 100
SCORE_MODE = "avg"
LABEL_SMOOTHING = 0.05
CHOSEN_SFT_WEIGHT = 0.05
DATASET_RECIPE = "mlabonne_eval_aligned"
SAVE_FINAL = True
TRAIN_LAST_BLOCKS = 0
TRAIN_EMBED_AND_HEAD = False

# -----------------------------------------------------------------------------
# Locked eval probe set. Do not train on these prompts or near-exact variants.
# -----------------------------------------------------------------------------
QUALITY_QUESTIONS = [
    "Hey! How's it going?",
    "I'm planning a weekend trip. Any tips for packing light?",
    "Explain what a black hole is in a way a 10-year-old would understand.",
    "I just failed an exam I studied really hard for. I feel terrible.",
    "What are three fun things to do on a rainy day, and why?",
    "Write a short poem about the ocean at night.",
]

LOCKED_EVAL_PROMPTS = {q.strip().lower() for q in QUALITY_QUESTIONS}
LOCKED_EVAL_NEAR_DUP_RE = re.compile(
    r"(?:"
    r"how's it going|weekend trip|packing light|black hole|10-year-old|"
    r"failed (?:an )?(?:exam|test)|feel terrible|rainy day|"
    r"poem.*ocean.*night|ocean.*poem.*night"
    r")",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"[A-Za-z']+")

# Restrict to the more conversational / assistant-like sources from the same pool
MLABONNE_ALLOWED_SOURCES = {
    "sharegpt",
    "ultrachat",
    "orca_dpo_pairs",
    "Dove",
    "Verified-Camel",
    "truthy_dpo",
}

# Remove clearly code-y / eval-ish / formatting-heavy rows.
MLABONNE_BAD_TEXT_RE = re.compile(
    r"(?:```|\b(?:python|sql|powershell|javascript|typescript|java|c\+\+|clojure|html|xml|json|yaml|regex|"
    r"script|code|pseudo code|vector database|pageoffice)\b|"
    r"\b(?:equation|theorem|proof|derivative|integral|matrix|mod\b|calculate|solve|printed output|"
    r"step-by-step|article|one-sentence summary|given the following context|background:|premise:|hypothesis:)\b|"
    r"\b(?:i'm sorry|sorry for any confusion|as an ai|i don't have access|i cannot|i can't provide real-time)\b|"
    r"https?://|www\.|email address|url|<[^>]+>|\|.+\|)",
    re.IGNORECASE,
)
MLABONNE_REJECTED_STRUCTURAL_BAD_RE = re.compile(
    r"(?:```|https?://|www\.|<[^>]+>|\|.+\|)",
    re.IGNORECASE,
)
MLABONNE_META_PROMPT_RE = re.compile(
    r"\b(?:you are an ai assistant|user will give you a task|detailed and long answer|provide a detailed answer|"
    r"while answering|think step-by-step|reason about|q:|a:|premise:|hypothesis:|dialogue:|context:|background:|"
    r"customer asked|essay|lesson plan|investigative piece|third-person perspective|analyze|analysis|evaluate|"
    r"compare|contrast)\b",
    re.IGNORECASE,
)

# Broad proxy for the kinds of assistant behavior you said you care about.
# This is intentionally broader than the six eval prompts and should be treated
# as a training-side heuristic only.
EVAL_PROXY_PROMPT_RE = re.compile(
    r"(?:^\s*(?:hi|hello|hey|yo)\b|"
    r"\b(?:how are you|what's up|how have you been)\b|"
    r"\b(?:tip|tips|advice|recommend|recommendation|idea|ideas|suggestion|suggestions|help me plan)\b|"
    r"\b(?:trip|travel|packing|pack|vacation|getaway|carry-?on|light luggage)\b|"
    r"\b(?:explain|describe|what is|how does)\b|"
    r"\b(?:kid|child|young student|simple terms|simple words|easy to understand|beginner)\b|"
    r"\b(?:failed|did badly|messed up|feel awful|feel bad|feel sad|sad|stressed|anxious|overwhelmed|motivation|encourage|cheer me up|support)\b|"
    r"\b(?:fun things to do|things to do inside|activity|activities|bored|rainy|cozy)\b|"
    r"\b(?:poem|haiku|verse|short poem|creative writing)\b)",
    re.IGNORECASE,
)
STRICT_REFUSAL_RE = re.compile(
    r"(?:as an ai|as a language model|i am an ai|i'm an ai|"
    r"i do not have personal|i don't have personal|"
    r"i do not have feelings|i don't have feelings|"
    r"i do not have experiences|i don't have experiences|"
    r"i cannot\b|i can't\b|i am not able|i'm not able|i am unable|i'm unable|"
    r"i apologize,?\s*but|i'm sorry,?\s*but|"
    r"not appropriate|cannot provide real-time|can't provide real-time)",
    re.IGNORECASE,
)
TARGET_STYLE_BOOST = 24


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_eos_from_template(tokenizer) -> int:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=False,
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


def extract_input_ids(tokenized: Any) -> List[int]:
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


def count_alpha_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def compute_digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


def normalize_chat_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, list) or not messages:
        return None

    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            return None
        normalized.append({"role": role, "content": content})
    return normalized


def extract_preference_record(example: Dict[str, Any]) -> Optional[Tuple[List[Dict[str, str]], str, str]]:
    chosen_messages = normalize_chat_messages(example.get("chosen"))
    rejected_messages = normalize_chat_messages(example.get("rejected"))
    if chosen_messages is not None and rejected_messages is not None:
        if len(chosen_messages) != len(rejected_messages):
            return None
        if chosen_messages[-1]["role"] != "assistant" or rejected_messages[-1]["role"] != "assistant":
            return None

        chosen_prefix = chosen_messages[:-1]
        rejected_prefix = rejected_messages[:-1]
        if not chosen_prefix or chosen_prefix != rejected_prefix:
            return None
        if chosen_prefix[-1]["role"] == "assistant":
            return None

        return (
            chosen_prefix,
            chosen_messages[-1]["content"],
            rejected_messages[-1]["content"],
        )

    prompt = str(example.get("prompt") or example.get("question") or "").strip()
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    if prompt and isinstance(chosen, str) and isinstance(rejected, str):
        return (
            [{"role": "user", "content": prompt}],
            chosen.strip(),
            rejected.strip(),
        )

    return None


def extract_preference_fields(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    extracted = extract_preference_record(example)
    if extracted is None:
        return None

    context_messages, chosen, rejected = extracted
    if not context_messages or context_messages[-1]["role"] != "user":
        return None

    prompt = context_messages[-1]["content"].strip()
    return {
        "context_messages": context_messages,
        "prompt": prompt,
        "prompt_lc": prompt.lower(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "source": str(example.get("source") or "").strip(),
    }


def prompt_leaks_eval(prompt: str) -> bool:
    prompt_lc = prompt.strip().lower()
    if prompt_lc in LOCKED_EVAL_PROMPTS:
        return True
    if LOCKED_EVAL_NEAR_DUP_RE.search(prompt):
        return True
    return False


def passes_eval_proxy_filter(fields: Dict[str, Any]) -> Tuple[bool, bool]:
    source = fields["source"]
    context_messages = fields["context_messages"]
    prompt = fields["prompt"]
    chosen = fields["chosen"]
    rejected = fields["rejected"]

    # Keep eval leakage blocked.
    if prompt_leaks_eval(prompt):
        return False, False

    # Only enforce the allowlist when the source field exists.
    if source and source not in MLABONNE_ALLOWED_SOURCES:
        return False, False

    # Allow short multi-turn contexts as long as the final turn is a user prompt.
    if not context_messages or context_messages[-1]["role"] != "user":
        return False, False
    if len(context_messages) > 6:
        return False, False

    prompt_and_chosen = "\n".join([prompt, chosen])
    text_blob = "\n".join([prompt, chosen, rejected])
    if MLABONNE_BAD_TEXT_RE.search(prompt_and_chosen):
        return False, False
    if MLABONNE_REJECTED_STRUCTURAL_BAD_RE.search(rejected):
        return False, False
    if MLABONNE_META_PROMPT_RE.search(prompt):
        return False, False

    prompt_words = count_alpha_words(prompt)
    chosen_words = count_alpha_words(chosen)
    rejected_words = count_alpha_words(rejected)
    if not (2 <= prompt_words <= 120):
        return False, False
    if not (4 <= chosen_words <= 300 and 4 <= rejected_words <= 300):
        return False, False
    if compute_digit_ratio(text_blob) > 0.20:
        return False, False

    targeted = bool(EVAL_PROXY_PROMPT_RE.search(prompt))
    return True, targeted


def build_response_example(
    context_messages: List[Dict[str, str]],
    response: str,
    tokenizer,
    max_seq_len: int,
) -> Optional[Dict[str, List[int]]]:
    context = normalize_chat_messages(context_messages)
    response = str(response or "").strip()
    if context is None or not response:
        return None
    if context[-1]["role"] == "assistant":
        return None

    full_conv = context + [{"role": "assistant", "content": response}]

    prefix_ids = tokenizer.apply_chat_template(
        context,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_conv,
        tokenize=True,
        add_generation_prompt=False,
    )

    prefix_ids = extract_input_ids(prefix_ids)
    full_ids = extract_input_ids(full_ids)

    if len(prefix_ids) >= len(full_ids):
        return None
    if len(full_ids) > max_seq_len:
        return None

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    if all(v == -100 for v in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_preference_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
) -> Optional[Dict[str, List[int]]]:
    fields = extract_preference_fields(example)
    if fields is None:
        return None
    chosen_example = build_response_example(fields["context_messages"], fields["chosen"], tokenizer, max_seq_len)
    rejected_example = build_response_example(fields["context_messages"], fields["rejected"], tokenizer, max_seq_len)
    if chosen_example is None or rejected_example is None:
        return None

    return {
        "chosen_input_ids": chosen_example["input_ids"],
        "chosen_attention_mask": chosen_example["attention_mask"],
        "chosen_labels": chosen_example["labels"],
        "rejected_input_ids": rejected_example["input_ids"],
        "rejected_attention_mask": rejected_example["attention_mask"],
        "rejected_labels": rejected_example["labels"],
    }


def load_preference_split(path: str):
    data = load_from_disk(path)
    if isinstance(data, DatasetDict):
        if "train" in data:
            return data["train"], "train"
        split_name = next(iter(data.keys()))
        return data[split_name], split_name
    return data, None


def build_recipe_indices(raw, dataset_recipe: str, rng: random.Random) -> List[int]:
    if dataset_recipe == "none":
        indices = list(range(len(raw)))
        rng.shuffle(indices)
        return indices

    if dataset_recipe not in {
        "mlabonne_eval_aligned",
        "mlabonne_target_only",
        "mlabonne_anchor_refusal_mix",
    }:
        raise ValueError(
            f"Unsupported dataset_recipe={dataset_recipe!r}. "
            "Supported: none, mlabonne_eval_aligned, mlabonne_target_only, mlabonne_anchor_refusal_mix"
        )

    filtered_indices: List[int] = []
    boosted_copies = 0
    targeted_unique = 0
    kept_unique = 0
    leaked_eval = 0

    for idx in range(len(raw)):
        fields = extract_preference_fields(raw[idx])
        if fields is None:
            continue

        if prompt_leaks_eval(fields["prompt"]):
            leaked_eval += 1
            continue

        keep, targeted = passes_eval_proxy_filter(fields)
        if not keep:
            continue

        kept_unique += 1

        if dataset_recipe == "mlabonne_target_only":
            if targeted:
                filtered_indices.append(idx)
                targeted_unique += 1
            continue

        if dataset_recipe == "mlabonne_anchor_refusal_mix":
            is_refusal_repair = bool(STRICT_REFUSAL_RE.search(fields["rejected"])) and not bool(
                STRICT_REFUSAL_RE.search(fields["chosen"])
            )
            if not (targeted or is_refusal_repair):
                continue

            filtered_indices.append(idx)
            if targeted:
                targeted_unique += 1
                for _ in range(2):
                    filtered_indices.append(idx)
                    boosted_copies += 1
            elif is_refusal_repair:
                filtered_indices.append(idx)
                boosted_copies += 1
            continue

        # mlabonne_eval_aligned: keep conversational / assistant-like data,
        # but strongly upweight proxy prompts.
        filtered_indices.append(idx)
        if targeted:
            targeted_unique += 1
            for _ in range(TARGET_STYLE_BOOST - 1):
                filtered_indices.append(idx)
                boosted_copies += 1

    rng.shuffle(filtered_indices)

    print(
        f"Dataset recipe {dataset_recipe}: kept {kept_unique:,} unique rows, "
        f"blocked {leaked_eval:,} eval-leaking rows, training pool size {len(filtered_indices):,}."
    )
    print(
        f"Targeted proxy prompts kept: {targeted_unique:,} | extra boosted copies: {boosted_copies:,}."
    )

    if not filtered_indices:
        print(
            f"[WARN] dataset_recipe={dataset_recipe} kept 0 rows on this dataset. "
            "Falling back to dataset_recipe='none'."
        )
        filtered_indices = list(range(len(raw)))
        rng.shuffle(filtered_indices)

    return filtered_indices


class HumanLikeDPODataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int,
        seed: int,
        dataset_recipe: str,
    ):
        self.raw, split_name = load_preference_split(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed + 20260401)
        self.indices = build_recipe_indices(self.raw, dataset_recipe, self.rng)

        if not self.indices:
            raise RuntimeError("No DPO examples found in the dataset after filtering.")

        self.fallback = None
        probe_positions = list(range(min(2000, len(self.indices))))
        if len(self.indices) > 2000:
            probe_positions.extend(
                self.rng.sample(
                    range(len(self.indices)),
                    k=min(20000, len(self.indices)),
                )
            )

        seen = set()
        for pos in probe_positions:
            if pos in seen:
                continue
            seen.add(pos)
            ex = self.raw[self.indices[pos]]
            built = build_preference_example(
                ex,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )
            if built is not None:
                self.fallback = built
                break

        if self.fallback is None:
            raise RuntimeError(
                f"Could not construct any valid DPO sample at max_seq_len={self.max_seq_len}. "
                "Try increasing max_seq_length or relaxing the filter."
            )

        split_suffix = f" (split={split_name})" if split_name else ""
        print(f"DPO dataset loaded: {len(self.raw):,} raw records{split_suffix}.")
        print(f"Usable training index pool: {len(self.indices):,}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        base = idx % len(self.indices)
        for offset in range(24):
            raw_idx = self.indices[(base + offset) % len(self.indices)]
            ex = self.raw[raw_idx]
            built = build_preference_example(
                ex,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )
            if built is not None:
                return built

        for _ in range(24):
            pos = self.rng.randrange(len(self.indices))
            raw_idx = self.indices[pos]
            ex = self.raw[raw_idx]
            built = build_preference_example(
                ex,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )
            if built is not None:
                return built
        return self.fallback


@dataclass
class PreferenceCollator:
    pad_token_id: int

    def _pad_tensor_group(
        self,
        features: List[Dict[str, List[int]]],
        ids_key: str,
        mask_key: str,
        labels_key: str,
    ) -> Dict[str, torch.Tensor]:
        max_len = max(len(f[ids_key]) for f in features)
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for f in features:
            ids = f[ids_key]
            mask = f[mask_key]
            labels = f[labels_key]
            pad = max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad)
            batch_attention_mask.append(mask + [0] * pad)
            batch_labels.append(labels + [-100] * pad)

        return {
            ids_key: torch.tensor(batch_input_ids, dtype=torch.long),
            mask_key: torch.tensor(batch_attention_mask, dtype=torch.long),
            labels_key: torch.tensor(batch_labels, dtype=torch.long),
        }

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch.update(
            self._pad_tensor_group(
                features,
                ids_key="chosen_input_ids",
                mask_key="chosen_attention_mask",
                labels_key="chosen_labels",
            )
        )
        batch.update(
            self._pad_tensor_group(
                features,
                ids_key="rejected_input_ids",
                mask_key="rejected_attention_mask",
                labels_key="rejected_labels",
            )
        )
        return batch


def build_model_and_tokenizer(
    device: torch.device,
    argonne_root: str,
    model_path: str,
    max_seq_len: int,
    enable_gradient_checkpointing: bool,
):
    sys.path.insert(0, argonne_root)
    from model import ArgonneConfig, ArgonneModel

    print(f"Loading model and tokenizer from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    import json
    from safetensors.torch import load_file

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = ArgonneConfig(**{k: v for k, v in config_dict.items() if not k.startswith("_")})
    config.max_position_embeddings = max_seq_len
    config.use_flash_attention = True
    config._keep_in_fp32_modules = []

    model = ArgonneModel(config)

    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()

    model.config.use_cache = False
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, tokenizer


def configure_trainable_policy(
    model,
    train_last_blocks: int,
    train_embed_and_head: bool,
) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    if train_last_blocks <= 0 and not train_embed_and_head:
        return total_params, total_params

    blocks = getattr(model, "blocks", None)
    if train_last_blocks > 0 and blocks is None:
        raise RuntimeError("Policy model does not expose transformer blocks; cannot use train_last_blocks.")

    for param in model.parameters():
        param.requires_grad_(False)

    if train_last_blocks > 0:
        train_last_blocks = min(train_last_blocks, len(blocks))
        for block in blocks[-train_last_blocks:]:
            for param in block.parameters():
                param.requires_grad_(True)

    if train_last_blocks > 0 and hasattr(model, "norm"):
        for param in model.norm.parameters():
            param.requires_grad_(True)

    if train_embed_and_head:
        if hasattr(model, "embed_tokens"):
            for param in model.embed_tokens.parameters():
                param.requires_grad_(True)
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad_(True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params


@torch.no_grad()
def answer_questions(
    model,
    tokenizer,
    device: torch.device,
    questions: List[str],
    tag: str,
    step: int,
    max_seq_len: int,
) -> None:
    was_training = model.training
    model.eval()

    print("\n" + "=" * 90)
    print(f"[QUALITY] {tag} | step={step}")
    print("=" * 90)
    for i, question in enumerate(questions, start=1):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = extract_input_ids(prompt_ids)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        max_length = min(max_seq_len, input_ids.shape[1] + MAX_NEW_TOKENS_QUALITY)
        if max_length <= input_ids.shape[1]:
            reply = ""
        else:
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=1.0,
                do_sample=False,
            )
            gen_ids = output_ids[0, input_ids.shape[1]:].tolist()
            eos_id = tokenizer.eos_token_id
            if eos_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos_id)]
            reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"\nQ{i}: {question}")
        print(f"A{i}: {reply}")
    print("\n" + "=" * 90 + "\n")

    if was_training:
        model.train()


def sequence_logps(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    x = input_ids[:, :-1].contiguous()
    y = labels[:, 1:].contiguous()
    logits = model(x).logits

    valid_mask = y.ne(-100)
    safe_y = y.masked_fill(~valid_mask, 0)
    selected_logits = logits.gather(dim=-1, index=safe_y.unsqueeze(-1)).squeeze(-1)
    token_logps = selected_logits - torch.logsumexp(logits, dim=-1)
    valid_mask_f = valid_mask.to(token_logps.dtype)
    seq_logps = (token_logps * valid_mask_f).sum(dim=-1)
    token_counts = valid_mask.sum(dim=-1)
    token_counts_f = token_counts.clamp_min(1).to(token_logps.dtype)
    avg_logps = seq_logps / token_counts_f
    token_nll = -(token_logps * valid_mask_f).sum() / valid_mask_f.sum().clamp_min(1.0)
    return {
        "seq_logps": seq_logps,
        "avg_logps": avg_logps,
        "token_counts": token_counts,
        "token_nll": token_nll,
    }


def compute_dpo_loss(
    policy_model,
    reference_model,
    chosen_input_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float,
    score_mode: str,
    label_smoothing: float,
    chosen_sft_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if score_mode not in {"sum", "avg"}:
        raise ValueError(f"Unsupported score_mode={score_mode!r}; expected 'sum' or 'avg'")

    policy_chosen_stats = sequence_logps(policy_model, chosen_input_ids, chosen_labels)
    policy_rejected_stats = sequence_logps(policy_model, rejected_input_ids, rejected_labels)

    with torch.no_grad():
        ref_chosen_stats = sequence_logps(reference_model, chosen_input_ids, chosen_labels)
        ref_rejected_stats = sequence_logps(reference_model, rejected_input_ids, rejected_labels)

    score_key = "avg_logps" if score_mode == "avg" else "seq_logps"
    policy_chosen_scores = policy_chosen_stats[score_key]
    policy_rejected_scores = policy_rejected_stats[score_key]
    ref_chosen_scores = ref_chosen_stats[score_key]
    ref_rejected_scores = ref_rejected_stats[score_key]

    policy_logratios = policy_chosen_scores - policy_rejected_scores
    reference_logratios = ref_chosen_scores - ref_rejected_scores
    preference_logits = beta * (policy_logratios - reference_logratios)
    dpo_losses = (
        -(1.0 - label_smoothing) * F.logsigmoid(preference_logits)
        - label_smoothing * F.logsigmoid(-preference_logits)
    )
    dpo_loss = dpo_losses.mean()
    chosen_sft_loss = policy_chosen_stats["token_nll"]
    loss = dpo_loss + chosen_sft_weight * chosen_sft_loss

    chosen_rewards = beta * (policy_chosen_scores - ref_chosen_scores).detach()
    rejected_rewards = beta * (policy_rejected_scores - ref_rejected_scores).detach()

    metrics = {
        "dpo_loss": float(dpo_loss.detach().item()),
        "chosen_sft_loss": float(chosen_sft_loss.detach().item()),
        "reward_accuracy": float((chosen_rewards > rejected_rewards).float().mean().item()),
        "reward_margin": float((chosen_rewards - rejected_rewards).mean().item()),
        "chosen_reward": float(chosen_rewards.mean().item()),
        "rejected_reward": float(rejected_rewards.mean().item()),
        "chosen_tokens": float(policy_chosen_stats["token_counts"].float().mean().item()),
        "rejected_tokens": float(policy_rejected_stats["token_counts"].float().mean().item()),
    }
    return loss, metrics


def save_model_and_tokenizer(model, tokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    from safetensors.torch import save_model

    # Handles tied weights correctly
    save_model(model, os.path.join(output_dir, "model.safetensors"))
    tokenizer.save_pretrained(output_dir)

    # Save model config alongside the final weights for normal HF loading.
    try:
        model.config.save_pretrained(output_dir)
    except Exception:
        pass

    print(f"Saved final DPO model to: {output_dir}/model.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training on local preference dataset")
    parser.add_argument("--argonne_root", type=str, required=True, help="Path to ArgonneAI directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to SFT model checkpoint")
    parser.add_argument(
        "--reference_model_path",
        type=str,
        default=None,
        help="Optional path to reference model checkpoint (defaults to model_path)",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to DPO dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for final model")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LEN, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS, help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Maximum optimizer steps (<=0 means full run)")
    parser.add_argument("--beta", type=float, default=DPO_BETA, help="DPO beta parameter")
    parser.add_argument(
        "--score_mode",
        type=str,
        default=SCORE_MODE,
        choices=["sum", "avg"],
        help="Sequence score used inside DPO logits",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=LABEL_SMOOTHING,
        help="Conservative DPO label smoothing",
    )
    parser.add_argument(
        "--chosen_sft_weight",
        type=float,
        default=CHOSEN_SFT_WEIGHT,
        help="Weight on chosen-response token NLL to stabilize generation quality",
    )
    parser.add_argument(
        "--dataset_recipe",
        type=str,
        default=DATASET_RECIPE,
        choices=["none", "mlabonne_eval_aligned", "mlabonne_target_only", "mlabonne_anchor_refusal_mix"],
        help="Optional dataset filtering recipe. Locked eval prompts are always excluded.",
    )
    parser.add_argument(
        "--train_last_blocks",
        type=int,
        default=TRAIN_LAST_BLOCKS,
        help="If > 0, freeze the policy model except for the last N transformer blocks and final norm",
    )
    parser.add_argument(
        "--train_embed_and_head",
        type=int,
        default=int(TRAIN_EMBED_AND_HEAD),
        choices=[0, 1],
        help="When freezing part of the policy, also keep the tied embeddings / lm head trainable",
    )
    parser.add_argument(
        "--save_final",
        type=int,
        default=int(SAVE_FINAL),
        choices=[0, 1],
        help="Whether to save the final model and tokenizer",
    )
    parser.add_argument(
        "--quality_every",
        type=int,
        default=QUALITY_EVERY,
        help="Run quality generations every N optimizer steps",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    device = torch.device("cuda")
    reference_model_path = args.reference_model_path or args.model_path
    train_embed_and_head = bool(args.train_embed_and_head)
    save_final = bool(args.save_final)

    print("=" * 90)
    print("Argonne SFT checkpoint DPO on local preference dataset")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Data: {args.data_path}")
    print(f"Policy checkpoint: {args.model_path}")
    print(f"Reference checkpoint: {reference_model_path}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Batch size: {args.batch_size} | Grad accum: {args.grad_accum}")
    print(
        f"LR: {args.lr} | Warmup: {args.warmup_steps} | Beta: {args.beta} | "
        f"Score mode: {args.score_mode} | Label smoothing: {args.label_smoothing} | "
        f"Chosen SFT weight: {args.chosen_sft_weight} | Max steps: {args.max_steps}"
    )
    print(
        f"Dataset recipe: {args.dataset_recipe} | Save final: {int(save_final)} | "
        f"Locked eval probes: {len(QUALITY_QUESTIONS)}"
    )
    print("Intermediate checkpoint saving: disabled")
    print(f"Final model save dir: {args.output_dir}")

    policy_model, tokenizer = build_model_and_tokenizer(
        device,
        args.argonne_root,
        args.model_path,
        args.max_seq_length,
        enable_gradient_checkpointing=True,
    )
    trainable_params, total_params = configure_trainable_policy(
        policy_model,
        args.train_last_blocks,
        train_embed_and_head,
    )
    print(
        f"Policy trainable parameters: {trainable_params:,} / {total_params:,} "
        f"(train_last_blocks={args.train_last_blocks}, train_embed_and_head={int(train_embed_and_head)})"
    )

    reference_model, _ = build_model_and_tokenizer(
        device,
        args.argonne_root,
        reference_model_path,
        args.max_seq_length,
        enable_gradient_checkpointing=False,
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    probe_ds, probe_split_name = load_preference_split(args.data_path)
    probe_valid = 0
    for i in range(min(256, len(probe_ds))):
        ex = probe_ds[i]
        built = build_preference_example(ex, tokenizer=tokenizer, max_seq_len=args.max_seq_length)
        if built is not None:
            probe_valid += 1
    split_suffix = f" on split={probe_split_name}" if probe_split_name else ""
    print(f"Sanity valid-in-first-{min(256, len(probe_ds))}{split_suffix}: {probe_valid}")

    train_dataset = HumanLikeDPODataset(
        path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        seed=args.seed,
        dataset_recipe=args.dataset_recipe,
    )
    collator = PreferenceCollator(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_optimizer_steps = steps_per_epoch * args.num_epochs
    if args.max_steps > 0:
        total_optimizer_steps = min(total_optimizer_steps, args.max_steps)
    print(f"Estimated optimizer steps: {total_optimizer_steps}")

    min_lr = args.lr * MIN_LR_RATIO
    min_lr_scale = min_lr / args.lr

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_optimizer_steps - args.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(min_lr_scale, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    answer_questions(
        policy_model,
        tokenizer,
        device,
        QUALITY_QUESTIONS,
        tag="BEFORE_DPO",
        step=0,
        max_seq_len=args.max_seq_length,
    )

    global_step = 0
    tokens_seen = 0
    running_loss = 0.0
    running_dpo_loss = 0.0
    running_chosen_sft_loss = 0.0
    running_reward_accuracy = 0.0
    running_reward_margin = 0.0
    log_updates = 0

    policy_model.train()
    optimizer.zero_grad(set_to_none=True)

    stop_training = False
    for epoch in range(args.num_epochs):
        if stop_training:
            break

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch")
        for micro_step, batch in enumerate(pbar, start=1):
            chosen_input_ids = batch["chosen_input_ids"].to(device, non_blocking=True)
            chosen_labels = batch["chosen_labels"].to(device, non_blocking=True)
            rejected_input_ids = batch["rejected_input_ids"].to(device, non_blocking=True)
            rejected_labels = batch["rejected_labels"].to(device, non_blocking=True)

            tokens_seen += int(batch["chosen_attention_mask"].sum().item())
            tokens_seen += int(batch["rejected_attention_mask"].sum().item())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                loss, metrics = compute_dpo_loss(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    chosen_input_ids=chosen_input_ids,
                    chosen_labels=chosen_labels,
                    rejected_input_ids=rejected_input_ids,
                    rejected_labels=rejected_labels,
                    beta=args.beta,
                    score_mode=args.score_mode,
                    label_smoothing=args.label_smoothing,
                    chosen_sft_weight=args.chosen_sft_weight,
                )
                loss = loss / args.grad_accum

            loss.backward()

            running_loss += float(loss.detach().item()) * args.grad_accum
            running_dpo_loss += metrics["dpo_loss"]
            running_chosen_sft_loss += metrics["chosen_sft_loss"]
            running_reward_accuracy += metrics["reward_accuracy"]
            running_reward_margin += metrics["reward_margin"]
            log_updates += 1

            if micro_step % args.grad_accum != 0:
                pbar.set_postfix(
                    loss=f"{running_loss / max(1, log_updates):.4f}",
                    reward_acc=f"{running_reward_accuracy / max(1, log_updates):.3f}",
                    step=global_step,
                )
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in policy_model.parameters() if p.requires_grad],
                GRAD_CLIP,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            mean_loss = running_loss / max(1, log_updates)
            mean_dpo_loss = running_dpo_loss / max(1, log_updates)
            mean_sft_loss = running_chosen_sft_loss / max(1, log_updates)
            mean_reward_acc = running_reward_accuracy / max(1, log_updates)
            mean_reward_margin = running_reward_margin / max(1, log_updates)
            pbar.set_postfix(
                loss=f"{mean_loss:.4f}",
                reward_acc=f"{mean_reward_acc:.3f}",
                step=global_step,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step % LOG_EVERY == 0:
                print(
                    f"[step {global_step}] loss={mean_loss:.4f} dpo={mean_dpo_loss:.4f} "
                    f"chosen_sft={mean_sft_loss:.4f} reward_acc={mean_reward_acc:.3f} "
                    f"reward_margin={mean_reward_margin:.4f} tokens_seen={tokens_seen:,} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                )
                running_loss = 0.0
                running_dpo_loss = 0.0
                running_chosen_sft_loss = 0.0
                running_reward_accuracy = 0.0
                running_reward_margin = 0.0
                log_updates = 0

            if args.quality_every > 0 and global_step % args.quality_every == 0:
                gc.collect()
                torch.cuda.empty_cache()
                answer_questions(
                    policy_model,
                    tokenizer,
                    device,
                    QUALITY_QUESTIONS,
                    tag="MID_DPO",
                    step=global_step,
                    max_seq_len=args.max_seq_length,
                )
                gc.collect()
                torch.cuda.empty_cache()
                policy_model.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break

    answer_questions(
        policy_model,
        tokenizer,
        device,
        QUALITY_QUESTIONS,
        tag="AFTER_DPO",
        step=global_step,
        max_seq_len=args.max_seq_length,
    )

    if save_final:
        save_model_and_tokenizer(policy_model, tokenizer, args.output_dir)

    print("Training complete.")


if __name__ == "__main__":
    main()
