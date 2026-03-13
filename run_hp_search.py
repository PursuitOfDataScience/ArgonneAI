#!/usr/bin/env python3
"""
Hyperparameter search runner - runs experiments in parallel.
Phase 1: LR + Precision sweep (complete)
Phase 2: Batch size sweep
Phase 3: Sequence length sweep
"""

import os
import sys
import time
import subprocess
import threading
import numpy as np
from datetime import datetime

LOG_FILE = "/home/youzhi/ArgonneAI/report/experiments.tsv"
EXPERIMENT_DIR = "/home/youzhi/ArgonneAI"
MAX_CONCURRENT_JOBS = 2
RUN_ID_OFFSET = 24  # Phase 1 had 24 experiments

def log_experiment(run_id, params, val_loss, token_loss, tokens_per_sec, total_steps, notes=""):
    with open(LOG_FILE, 'a') as f:
        line = f"{run_id}\t{params.get('lr', '')}\t{params.get('batch_size', '')}\t{params.get('total_batch_size', '')}\t{params.get('block_size', '')}\t{params.get('warmup_steps', '')}\t{params.get('weight_decay', '')}\t{params.get('adam_beta1', '')}\t{params.get('adam_beta2', '')}\t{params.get('grad_clip', '')}\t{params.get('min_lr_ratio', '')}\t{params.get('precision', '')}\t{val_loss}\t{token_loss}\t{tokens_per_sec}\t{total_steps}\t{notes}\n"
        f.write(line)

active_jobs = {}
job_lock = threading.Lock()

def run_experiment(run_id, params, gpu_id=0, timeout=420):
    start_time = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    cmd = [
        sys.executable, "train_llm_c.py",
        "--lr", str(params.get('lr', 1e-4)),
        "--batch_size", str(params.get('batch_size', 2)),
        "--total_batch_size", str(params.get('total_batch_size', 65536)),
        "--block_size", str(params.get('block_size', 4096)),
        "--warmup_steps", str(params.get('warmup_steps', 0)),
        "--weight_decay", str(params.get('weight_decay', 0.1)),
        "--adam_beta1", str(params.get('adam_beta1', 0.9)),
        "--adam_beta2", str(params.get('adam_beta2', 0.999)),
        "--grad_clip", str(params.get('grad_clip', 1.0)),
        "--min_lr_ratio", str(params.get('min_lr_ratio', 0.1)),
        "--precision", str(params.get('precision', 'bf16')),
        "--num_steps", "999999",
        "--checkpoint_interval", "999999",
        "--run_id", str(run_id),
    ]

    output_file = f"{EXPERIMENT_DIR}/report/exp_{run_id}.out"
    val_loss = token_loss = tokens_per_sec = total_steps = notes = ""

    try:
        with open(output_file, 'w') as outf:
            proc = subprocess.Popen(cmd, cwd=EXPERIMENT_DIR, stdout=outf, stderr=subprocess.STDOUT, env=env)
            try:
                proc.wait(timeout=timeout + 120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        with open(output_file, 'r') as f:
            content = f.read()

        if "OutOfMemoryError" in content or "CUDA out of memory" in content:
            notes = "OOM"
            val_loss = "OOM"
        elif " nan" in content.lower():
            notes = "NaN"
            val_loss = "NaN"
        else:
            for line in content.split('\n'):
                if 'SUMMARY:' in line and 'val_loss=' in line:
                    parts = line.split()
                    for part in parts:
                        if 'val_loss=' in part:
                            val_loss = part.split('=')[1].rstrip(',')
                        if 'token_loss=' in part:
                            token_loss = part.split('=')[1].rstrip(',')
                        if 'tokens_per_sec=' in part:
                            tokens_per_sec = part.split('=')[1].rstrip(',')
                        if 'steps=' in part:
                            total_steps = part.split('=')[1]
            if not val_loss:
                notes = "No summary found"
    except Exception as e:
        notes = f"Exception: {str(e)[:50]}"

    log_experiment(run_id, params, val_loss, token_loss, tokens_per_sec, total_steps, notes)
    return {'run_id': run_id, 'params': params, 'val_loss': val_loss, 'notes': notes}

def run_job_in_thread(run_id, params, result_list, index, gpu_id=0):
    result = run_experiment(run_id, params, gpu_id=gpu_id)
    result_list[index] = result
    with job_lock:
        if run_id in active_jobs:
            del active_jobs[run_id]

def print_top5():
    results = []
    try:
        with open(LOG_FILE, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= 14:
                    loss = parts[12]
                    if loss and loss not in ['', 'OOM', 'NaN']:
                        try:
                            results.append((float(loss), line))
                        except:
                            pass
    except:
        pass

    if results:
        results.sort(key=lambda x: x[0])
        print("\n" + "=" * 60)
        print("Top 5 configs so far:")
        for loss, line in results[:5]:
            parts = line.strip().split('\t')
            print(f"  Run {parts[0]}: LR={parts[1]}, BS={parts[2]}, Prec={parts[11]}, Val Loss={loss}")

def run_phase_configs(configs, offset=0):
    """Run a list of configs with parallelism."""
    results = [None] * len(configs)
    threads = []
    next_idx = [0]
    gpu_counter = [0]

    def start_next():
        while next_idx[0] < len(configs):
            with job_lock:
                if len(active_jobs) < MAX_CONCURRENT_JOBS:
                    run_id = configs[next_idx[0]]['run_id']
                    params = configs[next_idx[0]]
                    active_jobs[run_id] = True
                    gpu_id = gpu_counter[0] % MAX_CONCURRENT_JOBS
                    gpu_counter[0] += 1
                    next_idx[0] += 1
                else:
                    return False
            t = threading.Thread(target=run_job_in_thread, args=(run_id, params, results, next_idx[0] - 1, gpu_id))
            t.start()
            threads.append(t)
            time.sleep(2)

    start_next()
    while next_idx[0] < len(configs) or any(t.is_alive() for t in threads):
        time.sleep(10)
        start_next()
        completed = sum(1 for r in results if r is not None)
        print(f"Progress: {completed}/{len(configs)} experiments completed", end='\r')

    print(f"\nCompleted {len(configs)} experiments")
    for r in results:
        if r:
            print(f"  Exp {r['run_id']}: {r['params']}, Val Loss: {r['val_loss']}, Notes: {r['notes']}")
    print_top5()
    return results

def run_phase2_batch_size():
    """Phase 2: Batch size sweep with best LR and precision."""
    print("\n" + "=" * 60)
    print("Phase 2: Batch Size Sweep (Experiments 25-30)")
    print("=" * 60)

    # Best params from Phase 1: LR=1e-4, bf16
    base_params = {
        'lr': 1e-4,
        'precision': 'bf16',
        'warmup_steps': 0,
        'weight_decay': 0.1,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'grad_clip': 1.0,
        'min_lr_ratio': 0.1,
    }

    configs = []
    # Test different batch sizes with scaled total batch size
    for bs, tbs in [(1, 65536), (2, 65536), (4, 65536), (8, 65536), (1, 131072), (2, 131072)]:
        params = {**base_params, 'batch_size': bs, 'total_batch_size': tbs, 'block_size': 4096, 'run_id': RUN_ID_OFFSET + len(configs) + 1}
        configs.append(params)

    return run_phase_configs(configs, RUN_ID_OFFSET)

def run_phase3_block_size():
    """Phase 3: Sequence length sweep."""
    print("\n" + "=" * 60)
    print("Phase 3: Block Size Sweep (Experiments 31-35)")
    print("=" * 60)

    # Best params: LR=1e-4, bf16, batch_size=2, total=65536
    base_params = {
        'lr': 1e-4,
        'precision': 'bf16',
        'batch_size': 2,
        'total_batch_size': 65536,
        'warmup_steps': 0,
        'weight_decay': 0.1,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'grad_clip': 1.0,
        'min_lr_ratio': 0.1,
    }

    configs = []
    for bs in [256, 512, 1024, 2048, 8192]:
        params = {**base_params, 'block_size': bs, 'run_id': 30 + len(configs) + 1}
        configs.append(params)

    return run_phase_configs(configs, 30)

def main():
    # Run Phase 2
    run_phase2_batch_size()

    # Run Phase 3
    run_phase3_block_size()

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
