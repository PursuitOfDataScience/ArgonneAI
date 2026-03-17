#!/usr/bin/env python3
import argparse
import os
import re
import statistics
import subprocess
import textwrap
import time
from copy import deepcopy
from pathlib import Path


WORKDIR = Path("/home/youzhi/ArgonneAI/att_res")
REPO_ROOT = WORKDIR.parent
REPORT_DIR = WORKDIR / "report"
RESULTS_TSV = WORKDIR / "results.tsv"
COAUTHOR = "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
NGPUS = 2

RESULTS_HEADER = "\t".join(
    [
        "exp_id",
        "batch_size",
        "block_size",
        "grad_accum",
        "total_batch_tokens",
        "precision",
        "torch_compile",
        "gradient_checkpointing",
        "flash_attention",
        "extra_settings",
        "steps_completed",
        "s_per_step",
        "tokens_per_sec",
        "last_loss",
        "status",
        "notes",
    ]
)

BASELINE_ROW = "\t".join(
    [
        "0",
        "1",
        "1024",
        "4",
        "8192",
        "bf16",
        "1",
        "1",
        "1",
        "none",
        "9104",
        "1.01",
        "8100",
        "4.0048",
        "CANCELLED",
        "baseline - cancelled by slurm after 2.6h",
    ]
)


def run(cmd, cwd=WORKDIR, check=True):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {cmd}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def ensure_results_file():
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(f"{RESULTS_HEADER}\n{BASELINE_ROW}\n")


def total_tokens(cfg):
    return cfg["batch_size"] * NGPUS * cfg["block_size"] * cfg["grad_accum"]


def cfg_signature(cfg):
    keys = [
        "batch_size",
        "block_size",
        "grad_accum",
        "precision",
        "torch_compile",
        "torch_compile_mode",
        "gradient_checkpointing",
        "flash_attention",
        "use_attn_res",
        "attn_res_block_size",
        "cpus_per_task",
        "mem_gb",
        "alloc_conf",
        "nccl_p2p_disable",
    ]
    return tuple(cfg[k] for k in keys)


def make_base_cfg():
    return {
        "batch_size": 1,
        "block_size": 1024,
        "grad_accum": 4,
        "precision": "bf16",
        "torch_compile": 1,
        "torch_compile_mode": "default",
        "gradient_checkpointing": 1,
        "flash_attention": 1,
        "use_attn_res": 1,
        "attn_res_block_size": 4,
        "cpus_per_task": 4,
        "mem_gb": 32,
        "alloc_conf": "expandable_segments:True",
        "nccl_p2p_disable": "",
        "label": "baseline-like",
        "notes": "baseline-like",
    }


def build_stage1():
    cfgs = []
    baseline_sig = cfg_signature(make_base_cfg())
    blocks_to_bs = {
        1024: [1, 2, 4],
        512: [1, 2, 4, 8],
        256: [4, 8, 16],
    }
    for block_size, batches in blocks_to_bs.items():
        for use_attn_res in [1, 0]:
            for batch_size in batches:
                cfg = make_base_cfg()
                cfg["block_size"] = block_size
                cfg["batch_size"] = batch_size
                cfg["grad_accum"] = max(1, 8192 // (NGPUS * block_size * batch_size))
                cfg["use_attn_res"] = use_attn_res
                cfg["label"] = (
                    f"s1-b{block_size}-bs{batch_size}-ga{cfg['grad_accum']}"
                    f"-attnres{use_attn_res}-compile1-default"
                )
                cfg["notes"] = "stage1 micro-batch/sequence/attnres sweep at ~8192 tokens"
                if cfg_signature(cfg) != baseline_sig:
                    cfgs.append(cfg)
    return cfgs


def top_configs(records, n=5):
    usable = [
        r
        for r in records
        if r["tokens_per_sec"] is not None and r["status"] in {"OK", "TIMEOUT"}
    ]
    usable.sort(key=lambda r: r["tokens_per_sec"], reverse=True)
    out = []
    seen = set()
    for rec in usable:
        sig = cfg_signature(rec["config"])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(rec["config"])
        if len(out) >= n:
            break
    return out


def build_stage2(records, planned_sigs):
    cfgs = []
    seeds = top_configs(records, n=5)
    if not seeds:
        seeds = [make_base_cfg()]
    for i, seed in enumerate(seeds, start=1):
        variants = []
        v = deepcopy(seed)
        v["torch_compile"] = 0
        v["torch_compile_mode"] = "default"
        v["gradient_checkpointing"] = 0
        v["label"] = f"s2-top{i}-compile0-gc0"
        v["notes"] = "stage2 compile-off control"
        variants.append(v)

        v = deepcopy(seed)
        v["torch_compile"] = 0
        v["torch_compile_mode"] = "default"
        v["gradient_checkpointing"] = 1
        v["label"] = f"s2-top{i}-compile0-gc1"
        v["notes"] = "stage2 true gradient checkpointing tradeoff"
        variants.append(v)

        v = deepcopy(seed)
        v["torch_compile"] = 1
        v["torch_compile_mode"] = "reduce-overhead"
        v["gradient_checkpointing"] = seed["gradient_checkpointing"]
        v["label"] = f"s2-top{i}-compile1-reduce-overhead"
        v["notes"] = "stage2 compile mode test: reduce-overhead"
        variants.append(v)

        v = deepcopy(seed)
        v["torch_compile"] = 1
        v["torch_compile_mode"] = "max-autotune"
        v["gradient_checkpointing"] = seed["gradient_checkpointing"]
        v["label"] = f"s2-top{i}-compile1-max-autotune"
        v["notes"] = "stage2 compile mode test: max-autotune"
        variants.append(v)

        for cfg in variants:
            sig = cfg_signature(cfg)
            if sig in planned_sigs:
                continue
            planned_sigs.add(sig)
            cfgs.append(cfg)
    return cfgs


def best_config(records):
    usable = [
        r
        for r in records
        if r["tokens_per_sec"] is not None and r["status"] in {"OK", "TIMEOUT"}
    ]
    if not usable:
        return make_base_cfg()
    usable.sort(key=lambda r: r["tokens_per_sec"], reverse=True)
    return deepcopy(usable[0]["config"])


def build_stage3(records, planned_sigs):
    cfgs = []
    seed = best_config(records)
    scales = [2, 3, 4, 6, 8]
    for scale in scales:
        v = deepcopy(seed)
        v["grad_accum"] = max(1, seed["grad_accum"] * scale)
        v["label"] = f"s3-best-totalx{scale}"
        v["notes"] = "stage3 total batch scaling via grad accumulation"
        sig = cfg_signature(v)
        if sig not in planned_sigs:
            planned_sigs.add(sig)
            cfgs.append(v)

    for cpus in [8, 16, 24]:
        v = deepcopy(seed)
        v["cpus_per_task"] = cpus
        v["label"] = f"s3-best-cpus{cpus}"
        v["notes"] = "stage3 dataload/host overhead via cpus-per-task"
        sig = cfg_signature(v)
        if sig not in planned_sigs:
            planned_sigs.add(sig)
            cfgs.append(v)

    allocs = [
        "expandable_segments:True,max_split_size_mb:256",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
        "expandable_segments:True,max_split_size_mb:64",
    ]
    for alloc in allocs:
        v = deepcopy(seed)
        v["alloc_conf"] = alloc
        v["label"] = f"s3-best-alloc-{alloc.replace(':', '').replace(',', '-')}"
        v["notes"] = "stage3 CUDA allocator tuning"
        sig = cfg_signature(v)
        if sig not in planned_sigs:
            planned_sigs.add(sig)
            cfgs.append(v)
    return cfgs


def build_fallback(records, exp_id, planned_sigs):
    seed = best_config(records)
    v = deepcopy(seed)
    blocks = [1024, 768, 512, 384, 256]
    batches = [1, 2, 4, 8, 16]
    compile_modes = ["default", "reduce-overhead", "max-autotune"]
    v["block_size"] = blocks[exp_id % len(blocks)]
    v["batch_size"] = batches[exp_id % len(batches)]
    base_ga = max(1, 8192 // (NGPUS * v["block_size"] * v["batch_size"]))
    v["grad_accum"] = max(1, base_ga + (exp_id % 3))
    v["use_attn_res"] = exp_id % 2
    v["flash_attention"] = 1 if (exp_id % 4) != 0 else 0
    v["torch_compile"] = 1 if (exp_id % 3) != 0 else 0
    v["gradient_checkpointing"] = 1 if v["torch_compile"] == 0 else seed["gradient_checkpointing"]
    v["torch_compile_mode"] = compile_modes[exp_id % len(compile_modes)]
    v["cpus_per_task"] = [4, 8, 16, 24][exp_id % 4]
    v["alloc_conf"] = [
        "expandable_segments:True",
        "expandable_segments:True,max_split_size_mb:256",
        "expandable_segments:True,max_split_size_mb:128",
        "expandable_segments:True,max_split_size_mb:64",
    ][exp_id % 4]
    v["label"] = f"s4-fallback-{exp_id}"
    v["notes"] = "fallback mutation around current best"
    sig = cfg_signature(v)
    if sig in planned_sigs:
        return None
    planned_sigs.add(sig)
    return v


def write_run_script(exp_id, cfg):
    report_out = f"report/{exp_id}-train.out"
    report_err = f"report/{exp_id}-train.err"
    content = textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name=attres-{exp_id}
        #SBATCH --account=rcc-staff
        #SBATCH --partition=test
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={cfg['cpus_per_task']}
        #SBATCH --mem={cfg['mem_gb']}G
        #SBATCH --gres=gpu:2
        #SBATCH --constraint=a100
        #SBATCH --time=00:20:00
        #SBATCH --output={report_out}
        #SBATCH --error={report_err}

        module load python/miniforge-25.3.0
        unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
        source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

        export PYTHONUNBUFFERED=1
        export PYTORCH_CUDA_ALLOC_CONF="{cfg['alloc_conf']}"
        export NCCL_ASYNC_ERROR_HANDLING=1
        """
    )
    if cfg["nccl_p2p_disable"]:
        content += f'export NCCL_P2P_DISABLE={cfg["nccl_p2p_disable"]}\n'
    content += textwrap.dedent(
        f"""\

        cd /home/youzhi/ArgonneAI/att_res

        TOKENIZER=/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base
        DATA=/project/rcc/youzhi/fineweb-binary-qwen3/train.bin
        CKPT_DIR=/project/rcc/youzhi/llm.c/test
        NGPUS=2

        BATCH_SIZE={cfg['batch_size']}
        BLOCK_SIZE={cfg['block_size']}
        GRAD_ACCUM={cfg['grad_accum']}
        TOTAL_BATCH_SIZE=$((BATCH_SIZE * NGPUS * BLOCK_SIZE * GRAD_ACCUM))

        torchrun --nproc_per_node=$NGPUS train_llm_c.py \\
          --tokenizer_path $TOKENIZER \\
          --data_path $DATA \\
          --checkpoint_dir $CKPT_DIR \\
          --lr 1e-4 \\
          --batch_size $BATCH_SIZE \\
          --total_batch_size $TOTAL_BATCH_SIZE \\
          --block_size $BLOCK_SIZE \\
          --precision {cfg['precision']} \\
          --flash_attention {cfg['flash_attention']} \\
          --weight_decay 0.1 \\
          --adam_beta1 0.9 \\
          --adam_beta2 0.999 \\
          --grad_clip 1.0 \\
          --warmup_steps 1000 \\
          --min_lr_ratio 0.1 \\
          --checkpoint_interval 999999 \\
          --max_epochs 1 \\
          --torch_compile {cfg['torch_compile']} \\
          --torch_compile_mode {cfg['torch_compile_mode']} \\
          --gradient_checkpointing {cfg['gradient_checkpointing']} \\
          --use_attn_res {cfg['use_attn_res']} \\
          --attn_res_block_size {cfg['attn_res_block_size']} \\
          --reset_schedule 1
        """
    )
    path = WORKDIR / f"run_{exp_id}.sh"
    path.write_text(content)
    path.chmod(0o755)
    return path


def wait_for_job(job_id):
    while True:
        q = run(f'squeue -j "{job_id}" -h', cwd=WORKDIR, check=False)
        if not q.stdout.strip():
            return
        time.sleep(15)


def get_state(job_id):
    for _ in range(30):
        s = run(
            f'sacct -j "{job_id}" --format=JobIDRaw,State --parsable2 --noheader',
            cwd=WORKDIR,
            check=False,
        )
        lines = [x.strip() for x in s.stdout.splitlines() if x.strip()]
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 2 and parts[0] == str(job_id):
                return parts[1].split()[0]
        time.sleep(2)
    return "UNKNOWN"


def parse_steps_and_loss(out_text, err_text):
    step_loss = re.findall(r"Step\s+(\d+)\s+\|\s+Loss:\s+([0-9]+(?:\.[0-9]+)?)", out_text)
    if step_loss:
        steps = int(step_loss[-1][0])
        loss = float(step_loss[-1][1])
        return steps, loss

    pbar_steps = [int(m.group(1)) for m in re.finditer(r"Training:[^\n]*\|\s*(\d+)/\d+", err_text)]
    if pbar_steps:
        return max(pbar_steps), None
    return 0, None


def parse_s_per_step(err_text):
    vals = []
    for m in re.finditer(r"([0-9]+(?:\.[0-9]+)?)s/step", err_text):
        v = float(m.group(1))
        if 0 < v < 1000:
            vals.append(v)
    for m in re.finditer(r"([0-9]+(?:\.[0-9]+)?)step/s", err_text):
        rate = float(m.group(1))
        if rate > 0:
            vals.append(1.0 / rate)
    if not vals:
        return None
    tail = vals[-400:]
    if len(tail) > 40:
        tail = tail[len(tail) // 10 :]
    return float(statistics.median(tail))


def fmt_num(v, digits=4):
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def append_result(
    exp_id,
    cfg,
    steps_completed,
    s_per_step,
    tokens_per_sec,
    last_loss,
    status,
    notes,
):
    extra = (
        f"compile_mode={cfg['torch_compile_mode']};"
        f"alloc={cfg['alloc_conf']};"
        f"cpus={cfg['cpus_per_task']};"
        f"mem={cfg['mem_gb']}G;"
        f"attn_res_block={cfg['attn_res_block_size']}"
    )
    row = "\t".join(
        [
            str(exp_id),
            str(cfg["batch_size"]),
            str(cfg["block_size"]),
            str(cfg["grad_accum"]),
            str(total_tokens(cfg)),
            cfg["precision"],
            str(cfg["torch_compile"]),
            str(cfg["gradient_checkpointing"]),
            str(cfg["flash_attention"]),
            extra,
            str(steps_completed),
            fmt_num(s_per_step, 4),
            fmt_num(tokens_per_sec, 2),
            fmt_num(last_loss, 4),
            status,
            notes.replace("\t", " "),
        ]
    )
    with RESULTS_TSV.open("a") as f:
        f.write(row + "\n")


def commit_experiment(exp_id, cfg, status, s_per_step):
    sps = "na" if s_per_step is None else f"{s_per_step:.4f}"
    msg = f"exp {exp_id}: {cfg['label']} ({status}, {sps} s/step)"
    add = run(
        f"git --no-pager add run_{exp_id}.sh report/{exp_id}-train.out report/{exp_id}-train.err results.tsv",
        cwd=WORKDIR,
        check=False,
    )
    if add.returncode != 0:
        return
    commit = run(
        f'git --no-pager commit -m "{msg}" -m "{COAUTHOR}"',
        cwd=WORKDIR,
        check=False,
    )
    if commit.returncode != 0:
        print(f"[exp {exp_id}] commit skipped: {commit.stderr.strip()}")


def run_one_experiment(exp_id, cfg):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_path = write_run_script(exp_id, cfg)
    submit = run(f"sbatch --parsable {run_path.name}", cwd=WORKDIR, check=True)
    job_id = submit.stdout.strip().split(";")[0]
    print(f"[exp {exp_id}] submitted job {job_id} :: {cfg['label']}", flush=True)

    wait_for_job(job_id)
    state = get_state(job_id)
    print(f"[exp {exp_id}] job {job_id} finished with state={state}", flush=True)

    tail_err = run(f"tail -50 report/{exp_id}-train.err", cwd=WORKDIR, check=False)
    tail_out = run(f"tail -80 report/{exp_id}-train.out", cwd=WORKDIR, check=False)
    if tail_err.stdout.strip():
        print(f"[exp {exp_id}] tail err:\n{tail_err.stdout}", flush=True)
    if tail_out.stdout.strip():
        print(f"[exp {exp_id}] tail out:\n{tail_out.stdout}", flush=True)

    out_file = WORKDIR / f"report/{exp_id}-train.out"
    err_file = WORKDIR / f"report/{exp_id}-train.err"
    out_text = out_file.read_text(errors="ignore") if out_file.exists() else ""
    err_text = err_file.read_text(errors="ignore") if err_file.exists() else ""
    both = f"{out_text}\n{err_text}"

    oom = bool(re.search(r"CUDA out of memory|out of memory|OUT_OF_MEMORY", both, flags=re.IGNORECASE))
    steps_completed, last_loss = parse_steps_and_loss(out_text, err_text)
    s_per_step = parse_s_per_step(err_text)
    tps = (total_tokens(cfg) / s_per_step) if s_per_step else None

    if oom:
        status = "OOM"
    elif state.startswith("TIMEOUT"):
        status = "OK" if (s_per_step is not None and steps_completed >= 20) else "TIMEOUT"
    elif state.startswith("COMPLETED"):
        status = "OK" if s_per_step is not None else "ERROR"
    else:
        status = "OK" if (s_per_step is not None and steps_completed >= 20) else "ERROR"

    notes = f"{cfg['notes']}; slurm_state={state}"
    if oom:
        notes += "; oom_detected"
    append_result(exp_id, cfg, steps_completed, s_per_step, tps, last_loss, status, notes)
    commit_experiment(exp_id, cfg, status, s_per_step)

    return {
        "exp_id": exp_id,
        "config": deepcopy(cfg),
        "job_id": job_id,
        "state": state,
        "steps_completed": steps_completed,
        "s_per_step": s_per_step,
        "tokens_per_sec": tps,
        "last_loss": last_loss,
        "status": status,
        "notes": notes,
    }


def git_init_if_needed():
    chk = run("git rev-parse --is-inside-work-tree", cwd=WORKDIR, check=False)
    if chk.returncode == 0 and chk.stdout.strip() == "true":
        return
    run("git init", cwd=WORKDIR, check=True)


def main():
    parser = argparse.ArgumentParser(description="Autonomous throughput sweeper")
    parser.add_argument("--max_experiments", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    git_init_if_needed()
    ensure_results_file()

    records = []
    planned_sigs = set()
    queue = []

    stage1 = build_stage1()
    for cfg in stage1:
        sig = cfg_signature(cfg)
        if sig in planned_sigs:
            continue
        planned_sigs.add(sig)
        queue.append(cfg)

    stage = 1
    exp_id = 1
    while exp_id <= args.max_experiments:
        if not queue:
            if stage == 1:
                stage = 2
                queue.extend(build_stage2(records, planned_sigs))
            elif stage == 2:
                stage = 3
                queue.extend(build_stage3(records, planned_sigs))

        if not queue:
            fallback = None
            for bump in range(0, 100):
                fallback = build_fallback(records, exp_id + bump, planned_sigs)
                if fallback is not None:
                    queue.append(fallback)
                    break
            if fallback is None:
                raise RuntimeError("Unable to generate a new fallback configuration")

        cfg = queue.pop(0)
        if args.dry_run:
            print(
                f"exp {exp_id:02d}: {cfg['label']} "
                f"(B={cfg['batch_size']}, T={cfg['block_size']}, GA={cfg['grad_accum']}, "
                f"total={total_tokens(cfg)}, tc={cfg['torch_compile']}/{cfg['torch_compile_mode']}, "
                f"gc={cfg['gradient_checkpointing']}, attn_res={cfg['use_attn_res']})"
            )
            exp_id += 1
            continue

        rec = run_one_experiment(exp_id, cfg)
        records.append(rec)
        exp_id += 1

    if args.dry_run:
        print(f"Planned {args.max_experiments} experiments.")
    else:
        best = best_config(records)
        print(
            f"Completed {len(records)} experiments. "
            f"Best label so far: {best['label']}, "
            f"B={best['batch_size']} T={best['block_size']} GA={best['grad_accum']}"
        )


if __name__ == "__main__":
    main()
