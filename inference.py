#!/usr/bin/env python3
"""Inference and evaluation utilities for Argonne models.

This script provides two main capabilities:
 - Compute perplexity (bits-per-token) over a text file where each line is a single document.
 - Generate continuations for prompts interactively or from a prompt file.

It supports loading exported models in HuggingFace `save_pretrained` format and will try
an automatic device mapping with `device_map='auto'` and optional mixed precision.

Examples
--------
Compute perplexity for a plaintext evaluation file:
    python inference.py --model-dir /path/to/model --eval-file dataset.txt --block-size 4096 --batch-size 8

Generate samples from a prompt:
    python inference.py --model-dir /path/to/model --prompt "Write a short story about a lighthouse" --max-length 512

"""

import argparse
import math
import os
import sys
import time
from typing import Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

# Local utilities
try:
    from .data_processing import chunk_tokens
except Exception:
    # If executed as a script directly from the repo root, fall back to a relative import
    from data_processing import chunk_tokens

# =============================================================================
# CRITICAL: Register the Argonne model with Transformers at module load time.
# This MUST happen before any call to AutoModelForCausalLM.from_pretrained().
# =============================================================================
def _register_argonne_model() -> bool:
    """Import and register ArgonneConfig/ArgonneModel with Transformers Auto classes."""
    _ArgonneConfig = None
    _ArgonneModel = None

    # Try package import first
    try:
        from ArgonneAI.model import ArgonneConfig as _ArgonneConfig, ArgonneModel as _ArgonneModel
    except Exception:
        pass

    # Try importing 'model' from the script's directory
    if _ArgonneConfig is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [script_dir, os.path.dirname(script_dir)]:
            if candidate and candidate not in sys.path:
                sys.path.insert(0, candidate)
        try:
            from model import ArgonneConfig as _ArgonneConfig, ArgonneModel as _ArgonneModel
        except Exception:
            pass

    if _ArgonneConfig is None or _ArgonneModel is None:
        return False

    # Explicitly register with Transformers Auto classes
    try:
        AutoConfig.register("argonne2", _ArgonneConfig)
    except ValueError:
        pass  # Already registered
    try:
        AutoModel.register(_ArgonneConfig, _ArgonneModel)
    except ValueError:
        pass  # Already registered
    try:
        AutoModelForCausalLM.register(_ArgonneConfig, _ArgonneModel)
    except ValueError:
        pass  # Already registered

    return True

# Run registration immediately on module load
_ARGONNE_MODEL_REGISTERED = _register_argonne_model()
if not _ARGONNE_MODEL_REGISTERED:
    print(
        "WARNING: Could not import ArgonneAI.model to register 'argonne2' model type. "
        "Ensure the script is run from the ArgonneAI directory or set PYTHONPATH."
    )


def resolve_torch_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None or dtype_str == "auto":
        return None
    if dtype_str.lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str.lower() in ("fp16", "float16"):
        return torch.float16
    if dtype_str.lower() in ("fp32", "float32"):
        return torch.float32
    return None


def load_tokenizer_and_model(
    model_dir: str,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model from a local directory.

    This function tries an automatic device map (fast for multiple GPUs) and falls back
    to CPU then `.to(device)` to support single-device evaluation.
    """

    print(f"Loading tokenizer from: {model_dir}")
    # Ensure the path is absolute to avoid issues with transformers path validation
    model_dir_abs = os.path.abspath(model_dir)
    
    # Check if the model directory exists and is accessible
    if not os.path.isdir(model_dir_abs):
        raise FileNotFoundError(
            f"Model directory not found or not accessible: {model_dir_abs}\n"
            f"If this is on a network filesystem (e.g., /eagle/), make sure you are "
            f"running on a compute node with access to that filesystem."
        )
    
    # Check if config.json exists (required for model loading)
    config_path = os.path.join(model_dir_abs, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"config.json not found in model directory: {model_dir_abs}\n"
            f"Make sure this is a valid HuggingFace model directory."
        )
    
    # Work around HuggingFace Hub validation bug that treats long local paths as repo IDs
    # Try multiple methods to load the tokenizer
    tokenizer = None
    tokenizer_json_path = os.path.join(model_dir_abs, "tokenizer.json")
    tokenizer_config_path = os.path.join(model_dir_abs, "tokenizer_config.json")
    
    # Method 1: Direct load from tokenizer.json if it exists
    if os.path.isfile(tokenizer_json_path):
        print(f"Loading tokenizer directly from {tokenizer_json_path}")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_json_path,
            trust_remote_code=trust_remote_code,
        )
        # Try to load additional config from tokenizer_config.json if it exists
        if os.path.isfile(tokenizer_config_path):
            import json
            with open(tokenizer_config_path, "r") as f:
                tok_config = json.load(f)
            if "eos_token" in tok_config:
                tokenizer.eos_token = tok_config["eos_token"]
            if "bos_token" in tok_config:
                tokenizer.bos_token = tok_config["bos_token"]
            if "pad_token" in tok_config:
                tokenizer.pad_token = tok_config["pad_token"]
    
    # Method 2: Try with disabled repo validation (monkey-patch)
    if tokenizer is None:
        try:
            import huggingface_hub.utils._validators as hf_validators
            original_validate = hf_validators.validate_repo_id
            hf_validators.validate_repo_id = lambda x: None  # Disable validation
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir_abs, 
                    use_fast=True, 
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
            finally:
                hf_validators.validate_repo_id = original_validate  # Restore
        except Exception as e:
            print(f"Method 2 (monkey-patch) failed: {e}")
    
    # Method 3: Use a compatible tokenizer from HuggingFace Hub as fallback
    if tokenizer is None:
        print("Loading GPT-2 tokenizer as fallback (Argonne model likely uses GPT-2 tokenizer)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Ensure any local model code (ArgonneAI/model.py) is imported so that
    # its AutoConfig/AutoModel registrations happen before AutoModelForCausalLM
    # attempts to `from_pretrained()` and read the `model_type` from the saved config.
    def _ensure_local_model_registration(model_search_paths: Optional[List[str]] = None) -> None:
        import importlib
        nonlocal model_dir
        tried = []
        # First try the package import (if ArgonneAI is installed or run from package root)
        try:
            import ArgonneAI.model  # noqa: F401
            return
        except Exception as exc:  # pragma: no cover - environment dependent
            tried.append(f"ArgonneAI.model import failed: {exc}")

        # Try importing `model` from a few likely locations on sys.path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mp_pretrain_path = os.environ.get("MP_PRETRAIN_PATH")
        candidates = [p for p in (mp_pretrain_path, script_dir, os.path.dirname(script_dir), model_dir) if p]
        if model_search_paths:
            candidates = list(dict.fromkeys((model_search_paths or []) + candidates))

        for p in candidates:
            if not p:
                continue
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                import model  # noqa: F401
                return
            except Exception as e:  # pragma: no cover - environment dependent
                tried.append(f"import model from {p} failed: {e}")

        if tried:
            print("WARNING: Could not import local Argonne model module; tried:")
            for t in tried:
                print("  -", t)
            print(
                "Transformers may not recognise the custom 'argonne2' model_type; "
                "ensure `ArgonneAI/model.py` is on PYTHONPATH or run the script from the repo root."
            )

    _ensure_local_model_registration()
    
    # Monkey-patch huggingface_hub validation to allow local paths
    # This works around a bug where long local paths are incorrectly validated as repo IDs
    import huggingface_hub.utils._validators as hf_validators
    original_validate_repo_id = hf_validators.validate_repo_id
    def patched_validate_repo_id(repo_id):
        # If it looks like a local path, skip validation
        if repo_id and (repo_id.startswith('/') or repo_id.startswith('.') or os.path.sep in repo_id):
            return None
        return original_validate_repo_id(repo_id)
    hf_validators.validate_repo_id = patched_validate_repo_id
    
    print("Loading model... (this may take a while)")
    model = None
    tried_auto = False
    # Try device_map='auto' with dtype (if requested)
    try:
        if dtype is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir_abs,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
                tried_auto = True
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"Auto device mapping with dtype {dtype} failed: {exc}")

        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir_abs,
                    torch_dtype=dtype if dtype is not None else torch.float32,
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                # If this looks like a 'model_type not recognized' (e.g. argonne2) error,
                # attempt to register the local Argonne model and retry once.
                err = str(exc).lower()
                if "argonne2" in err or "model type" in err:
                    print("Detected unrecognized model type in checkpoint. Attempting to import local Argonne model module and retry...")
                    _ensure_local_model_registration([os.path.dirname(model_dir_abs)])
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_dir_abs,
                            torch_dtype=dtype if dtype is not None else torch.float32,
                            trust_remote_code=trust_remote_code,
                            local_files_only=True,
                        )
                    except Exception as exc2:  # pragma: no cover - environment dependent
                        print(f"Retry after local registration failed: {exc2}")
                        print(f"Falling back to CPU-only load due to: {exc}")
                        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=trust_remote_code, local_files_only=True)
                else:
                    print(f"Falling back to CPU-only load due to: {exc}")
                    model = AutoModelForCausalLM.from_pretrained(model_dir_abs, trust_remote_code=trust_remote_code, local_files_only=True)
    finally:
        # Restore original validation function
        hf_validators.validate_repo_id = original_validate_repo_id

    # If the model is on CPU and a single device was requested, move it
    if device is not None and isinstance(device, str) and device.lower() in ("cuda", "cpu"):
        # if model already has device mapping, skip moving
        try:
            device_obj = torch.device(device)
            model.to(device_obj)
        except Exception:
            pass

    # Some models don't set tokenizer.pad_token by default; ensure it's set
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    return tokenizer, model


def _batchify_chunks(chunks: Iterable[List[int]], batch_size: int) -> Iterable[List[List[int]]]:
    batch: List[List[int]] = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_lines: Iterable[str],
    block_size: int = 4096,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    max_lines: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Compute average perplexity for a sequence of lines (documents).

    Returns (perplexity, mean_loss) where mean_loss is average cross-entropy (nats) per token.
    """

    is_cuda = torch.cuda.is_available() and device is not None and device.type == "cuda"

    total_nll = 0.0
    total_tokens = 0
    processed = 0

    # We'll chunk documents into windows of block_size+1 tokens per training / eval chunk
    # Each chunk yields token_len == block_size + 1, where model computes a shift and returns loss over token_len-1 tokens.
    progress = None
    if verbose:
        progress = tqdm(desc="Scoring", unit="chunk")

    try:
        for line_idx, line in enumerate(eval_lines):
            if not line:
                continue
            if max_lines is not None and line_idx >= max_lines:
                break

            token_ids = tokenizer.encode(line, add_special_tokens=False)
            # Only yield full chunks (block_size + 1) so we can reuse training math
            if not token_ids:
                continue

            chunks = list(chunk_tokens(token_ids, block_size))
            if not chunks:
                # Skip short lines - token-length < block_size+1
                continue

            for batch in _batchify_chunks(chunks, batch_size):
                # Input IDs are the full chunk (length block_size+1), model will use label shift internally
                seq_len = len(batch[0])
                input_ids = torch.tensor(batch, dtype=torch.long)
                if device is not None:
                    input_ids = input_ids.to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = float(outputs.loss)

                # Loss is mean across tokens in the batch. Multiply by # tokens to get negative log-likelihood (nats)
                n_tokens = 0
                for s in batch:
                    n_tokens += max(0, len(s) - 1)
                total_nll += loss * n_tokens
                total_tokens += n_tokens
                processed += len(batch)

                if progress is not None:
                    progress.update(1)

    finally:
        if progress is not None:
            progress.close()

    if total_tokens == 0:
        raise RuntimeError("No valid tokens processed. Check block_size and input data.")

    mean_loss = total_nll / total_tokens
    perplexity = math.exp(mean_loss)
    return perplexity, mean_loss


def generate_from_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    do_sample: bool = True,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Generate text from a prompt using the Argonne model.
    
    Note: The custom ArgonneModel.generate() does not support num_return_sequences,
    so we generate once and return a single output per prompt.
    """
    input_data = tokenizer(prompt, return_tensors="pt")
    input_ids = input_data["input_ids"]
    if device is not None:
        input_ids = input_ids.to(device)
    
    # Build kwargs with only the parameters the custom generate() supports
    gen_kwargs = dict(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
    )
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    
    # outputs is a 2D tensor (batch_size, seq_len)
    # decode each sequence in the batch
    texts = []
    if outputs.dim() == 2:
        # Multiple sequences or batch
        for i in range(outputs.shape[0]):
            text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            texts.append(text)
    else:
        # Single sequence (1D tensor)
        text = tokenizer.decode(outputs, skip_special_tokens=True)
        texts.append(text)
    
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argonne model inference utility (CUDA + bf16)")
    parser.add_argument("--model-dir", type=str, required=False,
                        default="/project/rcc/youzhi/Argonne2.0",
                        help="Path to the exported model directory (save_pretrained output)")
    parser.add_argument("--eval-file", type=str, default=None, help="Plaintext file (one document per line) to compute perplexity")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string to generate from")
    parser.add_argument("--prompt-file", type=str, default=None, help="File containing one prompt per line to generate on")
    parser.add_argument("--generate", action="store_true", help="Run generation for prompt(s)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for scoring/eval")
    parser.add_argument("--block-size", type=int, default=4096, help="Block size (model max position window) for chunking in tokens")
    parser.add_argument("--max-lines", type=int, default=None, help="Limit number of documents to evaluate (for debugging)")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length for generation tasks")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top-k", type=int, default=50, help="top_k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p sampling")
    parser.add_argument("--prompt-out", type=str, default=None, help="Optional file to write generated texts to")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Check if user specified any action
    if not args.eval_file and not args.generate and not args.prompt and not args.prompt_file:
        print("No action specified. Use --generate to run text generation or --eval-file to compute perplexity.")
        print("Running generation with default prompts...")
        args.generate = True  # Default to generation mode
    
    # Always use CUDA with bf16
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Load components
    tokenizer, model = load_tokenizer_and_model(
        args.model_dir, dtype=dtype, trust_remote_code=True, device="cuda"
    )

    # If the model is mapped to many devices but we want single-device inference,
    # `model.device` may not be set. We'll just prefer the first parameter device.
    first_param = next(model.parameters(), None)
    model_device = first_param.device if first_param is not None else device

    # Evaluate perplexity if requested
    if args.eval_file:
        if not os.path.exists(args.eval_file):
            raise FileNotFoundError(args.eval_file)
        with open(args.eval_file, "r", encoding="utf-8") as rf:
            lines = [l.strip() for l in rf if l.strip()]
        if args.no_progress:
            verbose = False
        else:
            verbose = True
        print(f"Computing perplexity on {len(lines)} lines using block_size={args.block_size}, batch={args.batch_size} ...")
        perplexity, mean_loss = compute_perplexity(
            model,
            tokenizer,
            lines,
            block_size=args.block_size,
            batch_size=args.batch_size,
            device=model_device,
            max_lines=args.max_lines,
            verbose=verbose,
        )
        print(f"Perplexity: {perplexity:.3f} (mean loss {mean_loss:.6f} nats/token)")

    # Generation mode
    if args.generate:
        prompts: List[str] = []
        if args.prompt:
            prompts.append(args.prompt)
        if args.prompt_file:
            if not os.path.exists(args.prompt_file):
                raise FileNotFoundError(args.prompt_file)
            with open(args.prompt_file, "r", encoding="utf-8") as pf:
                for row in pf:
                    row = row.strip()
                    if row:
                        prompts.append(row)
        if not prompts:
            # Provide a richer default set of prompts to exercise different generation styles
            prompts = [
                "Hello, once upon a time.",
                "Write a short poem about the night sky.",
                "Explain, in simple terms, how a transformer attention mechanism works.",
                "Translate the following English sentence to French: 'The cat sat on the mat.'",
                "Write a Python function that computes the factorial of a number.",
            ]

        print(f"Generating {len(prompts)} prompts with max_length={args.max_length}")
        results = []
        for prompt in prompts:
            texts = generate_from_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=True,
                device=model_device,
            )
            for text in texts:
                print("--- PROMPT ---")
                print(prompt)
                print("--- OUTPUT ---")
                print(text)
                print("------------\n")
                results.append({"prompt": prompt, "output": text})

        if args.prompt_out:
            with open(args.prompt_out, "w", encoding="utf-8") as wf:
                for r in results:
                    wf.write(r["prompt"].replace("\n", " ") + "\n")
                    wf.write(r["output"].replace("\n", " ") + "\n\n")


if __name__ == "__main__":
    main()
