#!/usr/bin/env python3
"""Audit the LIVE Hub config of every published Argonne model against the release invariants.

WHY THIS EXISTS. thinking_training.md §37 found four silent release-config defects and recorded the
lesson as "ALWAYS diff the staged config against the LIVE Hub config". §42 then found a fifth, on
models that had already been published for weeks -- because the lesson was a habit, not a check, and
habits do not run themselves. This is the check.

It reads configs from the Hub (not from disk, not from the staging dir) because the Hub copy is the
one users get, and it is the only copy that can silently drift from what was staged.

THE INVARIANTS, and why each one is a defect rather than a preference:

  auto_map present         without it `from_pretrained(trust_remote_code=True)` cannot find
                           ArgonneModel and a standalone load FAILS OUTRIGHT.
  window disabled          `interleaved_local_attention` / `local_attention_window` must be
                           false/null. model.py applies the window ONLY on the flash-attn-2 path
                           (model.py ~line 376, gated on `_flash_attn_available`); the SDPA
                           fallback ignores it. Every Argonne pretrain ran on flash-attn-4, which
                           does not expose `flash_attn.flash_attn_interface`, so the window has
                           never been active and these weights only ever saw FULL attention.
                           Publishing the flags hands a 256-token window on odd layers to any user
                           who happens to have flash-attn-2 installed -- silently, and at whatever
                           context the card advertises.
  loss_chunk_size == 0     nonzero makes forward() return `logits=None` whenever
                           `self.training and labels is not None`. Inert at inference, a footgun on
                           a base model whose purpose is fine-tuning.
  dtype/torch_dtype        both should name the published weight dtype.
  ctx sanity               block_size == max_position_embeddings. ArgonneConfig.__init__ maps a
                           `block_size` kwarg ONTO max_position_embeddings and it WINS over an
                           explicit value, so a stale block_size silently caps the context.
  eos_token_id             chat/instruct/think models need 151645 (<|im_end|>) or .generate() never
                           stops at end of turn. Bases legitimately use 151643 (<|endoftext|>) or
                           null, so this is only enforced for chat-shaped repos.

Exit status is 1 if any repo fails, so this can gate a release.

Usage:
  python reasoning/hub_config_audit.py                  # audit the known Argonne family
  python reasoning/hub_config_audit.py --author PursuitOfDataScience   # discover argonne* repos
  python reasoning/hub_config_audit.py --repos a b c
"""
import argparse
import json
import sys

CHAT_EOS = 151645          # <|im_end|>
BASE_EOS_OK = (151643, None)

# Repos whose eos_token_id must be the turn terminator, not the document terminator.
CHAT_SHAPED = ("instruct", "think", "chat")


def is_chat_shaped(repo_id):
    low = repo_id.lower()
    return any(k in low for k in CHAT_SHAPED)


def audit(repo_id, cfg):
    """Return a list of problem strings; empty means the config is release-clean."""
    problems = []
    if not cfg.get("auto_map"):
        problems.append("auto_map MISSING -> standalone trust_remote_code load fails")
    if cfg.get("interleaved_local_attention") or cfg.get("local_attention_window") is not None:
        problems.append(
            f"sliding window ADVERTISED (interleaved={cfg.get('interleaved_local_attention')}, "
            f"window={cfg.get('local_attention_window')}) -> a flash-attn-2 user gets attention "
            f"the weights never saw"
        )
    if cfg.get("loss_chunk_size"):
        problems.append(
            f"loss_chunk_size={cfg.get('loss_chunk_size')} -> forward() returns logits=None when "
            f"fine-tuning with labels"
        )
    bs, mpe = cfg.get("block_size"), cfg.get("max_position_embeddings")
    if bs is not None and mpe is not None and bs != mpe:
        problems.append(f"block_size {bs} != max_position_embeddings {mpe} -> context silently capped")
    # `dtype` is a NEWER transformers key than `torch_dtype`; a legacy config carrying only
    # torch_dtype is fine and flagging it just trains the reader to ignore this report. Only a real
    # disagreement (both present, different) or neither present is a defect.
    d, td = cfg.get("dtype"), cfg.get("torch_dtype")
    if d is not None and td is not None and d != td:
        problems.append(f"dtype {d!r} != torch_dtype {td!r} -> which one wins is version-dependent")
    elif d is None and td is None:
        problems.append("neither dtype nor torch_dtype set -> loads in fp32 by default")
    eos = cfg.get("eos_token_id")
    if is_chat_shaped(repo_id):
        if eos != CHAT_EOS:
            problems.append(f"eos_token_id {eos} != {CHAT_EOS} (<|im_end|>) on a chat-shaped repo "
                            f"-> .generate() never stops at end of turn")
    elif eos not in BASE_EOS_OK:
        problems.append(f"eos_token_id {eos} unexpected for a base (want 151643 or null)")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default=None,
                    help="discover this author's argonne* models instead of using the built-in list")
    ap.add_argument("--repos", nargs="*", default=None)
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    if a.repos:
        repos = a.repos
    elif a.author:
        repos = sorted(m.id for m in api.list_models(author=a.author)
                       if "argonne" in m.id.split("/")[-1].lower())
    else:
        repos = sorted(m.id for m in api.list_models(author="PursuitOfDataScience")
                       if "argonne" in m.id.split("/")[-1].lower())

    failed, skipped, clean = {}, [], []
    for r in repos:
        try:
            cfg = json.load(open(hf_hub_download(r, "config.json")))
        except Exception as exc:
            skipped.append((r, f"{type(exc).__name__}"))
            continue
        # Only the custom architecture has these invariants; a Qwen/Llama derivative does not.
        if cfg.get("model_type") not in ("argonne", "argonne2"):
            skipped.append((r, f"model_type={cfg.get('model_type')} (not an Argonne arch)"))
            continue
        p = audit(r, cfg)
        (failed.setdefault(r, p) if p else clean.append(r))

    print(f"=== CLEAN ({len(clean)}) ===")
    for r in clean:
        print(f"  ok    {r}")
    print(f"\n=== NEEDS FIXING ({len(failed)}) ===")
    for r, ps in failed.items():
        print(f"  FAIL  {r}")
        for p in ps:
            print(f"          - {p}")
    if skipped:
        print(f"\n=== SKIPPED ({len(skipped)}) ===")
        for r, why in skipped:
            print(f"  skip  {r}  ({why})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
