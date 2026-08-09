# Local project rules (Argonne) — not committed

## Inference: ALWAYS use a real serving engine (vLLM / SGLang). No exceptions.

For **any** job that generates tokens or evaluates the model — sampling, self-consistency,
best-of-N, STaR/rejection-sampling generation, GRPO rollouts, lm-eval, ANY benchmark — the
**first thing to reach for is vLLM** (or SGLang). Do **not** default to the naive HF path
(`AutoModelForCausalLM.generate` / a per-token Python decode loop / lm-eval's HF `bs=1`
backend). That path is ~40 tok/s and ~9% HBM on this model; vLLM is ~10-50× faster and fills
the card. Using the slow path when the fast one exists is the mistake to never repeat.

**The custom `argonne2` arch is already ported to vLLM and validated** (token-for-token exact
vs `model.py`): `reasoning/vllm_argonne.py`. It is the source of truth — reuse it.

### How to run vLLM for this arch (the gotchas, baked in)
- `import vllm_argonne; vllm_argonne.register()` **before** constructing `LLM(...)` — it registers
  `argonne2` with vLLM + `AutoConfig`, and applies the transformers-5.x tokenizer shim.
- Env: `export VLLM_ENABLE_V1_MULTIPROCESSING=0` and `export PYTHONPATH=<repo>/reasoning:<repo>`
  so `register_model` reaches the engine process (custom-model gotcha).
- `gpu_memory_utilization=0.90` (the HBM target — see below), `dtype="bfloat16"`, `trust_remote_code=True`.
- config.json needs `auto_map` for `from_pretrained(trust_remote_code=True)` to work standalone.

### Reusable vLLM tooling (prefer these; don't rewrite the HF path)
- `reasoning/vllm_grade.py` — fast greedy GSM8K/MATH pass@1 grader.
- `reasoning/vllm_bon.py` — sampling (K/problem) + verifier best-of-N (self-consistency for free).
- `reasoning/run_lmeval_vllm.py` — **lm-eval via the vLLM backend**. Use this for benchmarks, NOT
  `reasoning/run_lmeval.py` (that's the slow HF `bs=1` path; keep only as a last-resort fallback).

### The ONE inherent exception
lm-eval's **HF** backend (`ArgonneHFLM` in `run_lmeval.py`) is forced to `bs=1` (no padding on this
arch) → ~9% HBM and slow. That's expected, not a bug — but it's exactly why you should use the
**vLLM backend** instead. Only fall back to HF `bs=1` if a task genuinely can't run under vLLM.

## GPU / HBM
- **Target 90% HBM** on every job (vLLM: `gpu_memory_utilization=0.90`; training: size batch/seq).
- **Use 1× H100** by default (`--constraint=H100 --gres=gpu:1`). Not H200, not multi-GPU, unless
  asked. Don't hog: cancel superseded/slow jobs and prefer the fast (vLLM) path.

## SLURM
- Every `sbatch` must carry `--exclude=midway3-0423,midway3-[0298,0377-0378,0603-0606]`
  (other groups' nodes — do not run there). 0602 is usable.
- All `.sh` are git-ignored (`*/*` rule); `.py`/`.md` need `git add -f` to commit. Never commit `.sh`.
- Submit from the repo root so `#SBATCH --output=report/...` resolves correctly.

## Untracked `.sh` do NOT travel with a branch/worktree — never hardcode a worktree path in them
Lesson (2026-07-17): `weekend.sh`/`night.sh`/`run_full_training.sh` broke with
`ERROR: run script not found: /home/youzhi/ArgonneAI-3.5/run_full_training.sh` after the
`argonne3.5` branch moved out of its throwaway worktree `/home/youzhi/ArgonneAI-3.5` into the
main clone `/home/youzhi/ArgonneAI`. Why it broke, and why nothing "carried it over":
- A **branch switch / `git checkout` only restores TRACKED files.** Untracked & git-ignored files
  are never touched by a checkout and are NOT associated with any branch or commit.
- **Every `.sh` here is git-ignored by our own policy**, so the launcher scripts are untracked —
  they live only as on-disk files in whatever directory they were created in. When the branch
  consolidated into the main clone, the `.py`/`.md` came via the branch but the `.sh` had to be
  **manually copied**, and they arrived still carrying the hardcoded `REPO_ROOT="/home/youzhi/ArgonneAI-3.5"`
  pointing at the (now-deleted) worktree. Git can't catch this — it doesn't track these files.
  (`git worktree remove` even refuses when untracked files are present unless forced — a hint they
  don't move with the worktree.)
- **Rule:** never hardcode a worktree-specific absolute path in an untracked launcher. For a
  script run directly, derive the root from its own location:
  `REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"`. For a script
  sbatch'd (SLURM spools a COPY, so `$0`/`BASH_SOURCE` don't point at the repo), use an
  env-overridable absolute default (`REPO_ROOT="${REPO_ROOT:-/home/youzhi/ArgonneAI}"`) and thread
  `REPO_ROOT` through every `--export=ALL,...` list (launcher → first slice → self-resubmits) so a
  moved repo works end-to-end. All three scripts were fixed this way on 2026-07-17.

## Every published HF model MUST be linked from the GitHub README (rule added 2026-08-02)
When a model goes up on Hugging Face, it is not "shipped" until the repo README on `main` links
it. The README's model table is the index people actually find things through; a model that
exists only on HF is effectively undiscoverable from the code that produced it.
- Add a row to the family table at the top of `README.md` (params / context / training tokens /
  HF link) AND a section for the model, in the same shape as the existing Argonne 3.0/2.5/… entries.
- Link BOTH directions: the HF card cites the GitHub source files, the GitHub README cites the
  HF model. A reader landing on either one can reach the other.
- Applies to every release. Currently required: **argonne-3.5-base** and **Argonne-3.5-think**.
