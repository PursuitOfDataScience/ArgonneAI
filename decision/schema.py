"""Question types, the prompt layout, and tokenization for the Argonne decision model.

The model answers in ONE forward pass and never generates text. A question is laid out after the shared
`state` (the context, up to the base model's 13,568 tokens) as:

    {state}

    Question: {instructions}          <- "Rate:" for Score, "Statement:" for Noul
    Options:
    (1) {option 1}
    (2) {option 2}
    ...
    Answer:

The decision head reads the hidden state at the final token of "Answer:" and scores it against the hidden
state at the newline that closes each option line, so an option's score comes from the model's own reading of
that option in context. That is what lets one pass handle up to 255 options, and why the answer can never fall
outside the schema: there is nothing to generate, only a softmax over the options that were given.

Pieces are tokenized separately and concatenated, because the tokenizer merges across boundaries (":\\n" is one
token): encoding piece by piece fixes exactly where each option ends, identically in training and serving.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

MAX_OPTIONS = 255          # Jev's Choice limit
MAX_LEN = 13568            # argonne-4.5-base-ctx13568's trained context
MAX_OPTION_TOKENS = 96     # one option line is clipped to this many tokens
NL = 198                   # "\n" in the Qwen3 tokenizer
NLNL = 271                 # "\n\n"
HEADERS = {"choice": "Question", "score": "Rate", "noul": "Statement"}
NOUL_OPTIONS = ["true", "false"]   # Noul is a two-option question; noul = P(option 1)


@dataclass
class Choice:
    """Pick one of `criteria` (2 to 255 strings)."""
    instructions: str
    criteria: Sequence[str]
    type: str = field(default="choice", init=False)


@dataclass
class Score:
    """Rate on a user-defined scale. `legend` maps a numeric level to its description, e.g.
    {0: "Calm", 1: "Frustrated but civil", 2: "Very angry"}. Returns the expected level."""
    instructions: str
    legend: Dict[float, str]
    type: str = field(default="score", init=False)


@dataclass
class Noul:
    """Probability that the statement in `instructions` is true of the state."""
    instructions: str
    type: str = field(default="noul", init=False)


def question_options(q) -> List[str]:
    """(option labels, option texts) as they appear in the prompt, for any question type."""
    if q.type == "choice":
        return list(q.criteria)
    if q.type == "score":
        return [q.legend[k] for k in sorted(q.legend, key=float)]
    return list(NOUL_OPTIONS)


def option_tags(q) -> List[str]:
    """The tag printed before each option: 1..N for Choice and Noul, the level value for Score."""
    if q.type == "score":
        return [_fmt_level(k) for k in sorted(q.legend, key=float)]
    return [str(i + 1) for i in range(len(question_options(q)))]


def _fmt_level(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


class Encoder:
    """Turns (state, question) into token ids plus the positions the decision head reads."""

    def __init__(self, tokenizer, max_len: int = MAX_LEN):
        self.tok = tokenizer
        self.max_len = max_len
        self._cache = {}

    def enc(self, text: str) -> List[int]:
        if not text:
            return []
        if len(text) < 64:                       # labels and headers repeat millions of times
            hit = self._cache.get(text)
            if hit is None:
                hit = self._cache[text] = self.tok.encode(text, add_special_tokens=False)
            return list(hit)
        return self.tok.encode(text, add_special_tokens=False)

    def question_block(self, q):
        """Token ids of the question block and the offsets (within the block) of each option's closing NL
        and of the answer token."""
        opts = question_options(q)
        if not 2 <= len(opts) <= MAX_OPTIONS:
            raise ValueError(f"a {q.type} needs 2..{MAX_OPTIONS} options, got {len(opts)}")
        ids = self.enc(f"{HEADERS[q.type]}: {q.instructions.strip()}") + [NL] + self.enc("Options:") + [NL]
        opt_pos = []
        for tag, text in zip(option_tags(q), opts):
            line = self.enc(f"({tag}) {str(text).strip()}")[:MAX_OPTION_TOKENS]
            ids += line + [NL]
            opt_pos.append(len(ids) - 1)
        ids += self.enc("Answer:")
        return ids, opt_pos, len(ids) - 1

    def encode(self, state: str, q):
        """Full sequence for one question. The state is clipped to fit MAX_LEN, keeping its head and tail
        (a document's opening and its most recent text both tend to carry the evidence)."""
        block, opt_pos, ans = self.question_block(q)
        state_ids = self.enc(state.strip()) if state and state.strip() else []
        budget = self.max_len - len(block) - (1 if state_ids else 0)
        if budget < 0:
            raise ValueError(f"question block alone is {len(block)} tokens, over the {self.max_len} limit")
        truncated = len(state_ids) > budget
        if truncated:
            marker = self.enc(" [...] ")
            keep = max(budget - len(marker), 0)
            head = keep // 2
            state_ids = state_ids[:head] + marker + state_ids[len(state_ids) - (keep - head):] if keep > 0 else []
            state_ids = state_ids[:budget]
        prefix = state_ids + ([NLNL] if state_ids else [])
        off = len(prefix)
        return {"input_ids": prefix + block, "opt_pos": [p + off for p in opt_pos], "ans_pos": ans + off,
                "n_state": len(state_ids), "truncated": truncated}


def question_from_record(r):
    """Build a question object from a unified data record (see build_data.py)."""
    t = r["type"]
    if t == "choice":
        return Choice(r["instructions"], r["options"])
    if t == "score":
        return Score(r["instructions"], {lv: txt for lv, txt in zip(r["levels"], r["options"])})
    if t == "noul":
        return Noul(r["instructions"])
    raise ValueError(f"unknown question type {t!r}")
