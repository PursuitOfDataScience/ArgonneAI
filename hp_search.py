#!/usr/bin/env python3
"""
Hyperparameter search for LLM.c training.
Runs experiments and logs results to report/experiments.tsv
"""

import os
import sys
import time
import subprocess
import threading
import numpy as np
from datetime import datetime

# Experiment log file
LOG_FILE = "/home/youzhi/ArgonneAI/report/experiments.tsv"
EXPERIMENT_DIR = "/home/youzhi/ArgonneAI"

# Initialize log file with header if it doesn't exist
def init_log():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            header = "run_id\tlr\tmin_lr\tbatch_size\ttotal_batch_size\tblock_size\twarmup_steps\tweight_decay\tadam_beta1\tadam_beta2\tgrad_clip\tfinal_loss\ttokens_per_sec\ttotal_steps\ttotal_tokens\twall_time_sec\tnotes\n"
            f.write(header)

def log_experiment(run_id, params, final_loss, tokens_per_sec, total_steps, total_tokens, wall_time_sec, notes=""):
    """Log experiment results to TSV file."""
    with open(LOG_FILE, 'a') as f:
        line = f"{run_id}\t{params.get('lr', '')}\t{params.get('min_lr', '')}\t{params.get('batch_size', '')}\t{params.get('total_batch_size', '')}\t{params.get('block_size', '')}\t{params.get('warmup_steps', '')}\t{params.get('weight_decay', '')}\t{params.get('adam_beta1', '')}\t{params.get('adam_beta2', '')}\t{params.get('grad_clip', '')}\t{final_loss}\t{tokens_per_sec}\t{total_steps}\t{total_tokens}\t{wall_time_sec:.1f}\t{notes}\n"
        f.write(line)

def run_experiment(run_id, params, timeout=300):
    """Run a single experiment with given parameters."""
    start_time = time.time()

    # Build command line args
    cmd = [
        "python", "train_llm_c.py",
        "--lr", str(params.get('lr', 1e-4)),
        "--min_lr", str(params.get('min_lr', params.get('lr', 1e-4) * 0.1)),
        "--batch_size", str(params.get('batch_size', 2)),
        "--total_batch_size", str(params.get('total_batch_size', 65536)),
        "--block_size", str(params.get('block_size', 4096)),
        "--warmup_steps", str(params.get('warmup_steps', 0)),
        "--weight_decay", str(params.get('weight_decay', 0.1)),
        "--adam_beta1", str(params.get('adam_beta1', 0.9)),
        "--adam_beta2", str(params.get('adam_beta2', 0.999)),
        "--grad_clip", str(params.get('grad_clip', 1.0)),
        "--num_steps", "999999",  # Will be limited by timeout
        "--checkpoint_interval", "999999",  # No checkpoints during search
        "--run_id", str(run_id),
    ]

    # Create output file
    output_file = f"/home/youzhi/ArgonneAI/report/exp_{run_id}.out"

    try:
        # Run training
        result = subprocess.run(
            cmd,
            cwd=EXPERIMENT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # Extra time for startup/shutdown
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        wall_time = time.time() - start_time

        # Parse output for final loss and metrics
        final_loss = ""
        tokens_per_sec = ""
        total_steps = ""
        total_tokens = ""
        notes = ""

        # Check for OOM
        if "OutOfMemoryError" in result.stderr or "CUDA out of memory" in result.stderr:
            notes = "OOM"
            final_loss = "OOM"
        # Check for NaN
        elif "nan" in result.stdout.lower() or "nan" in result.stderr.lower():
            notes = "NaN"
            final_loss = "NaN"
        else:
            # Parse output
            lines = result.stdout.split('\n')
            for line in lines:
                if '| Loss:' in line and '| PPL:' in line:
                    parts = line.split('|')
                    for part in parts:
                        if 'Loss:' in part:
                            final_loss = part.split(':')[1].strip()
                        if 'Tokens:' in part:
                            total_tokens = part.split(':')[1].strip().replace(',', '')
                        if 'LR:' in part:
                            # Get last loss line as final
                            pass

            # Calculate tokens per second
            if total_tokens:
                try:
                    tokens_per_sec = float(total_tokens) / wall_time
                except:
                    pass

            # Count steps
            step_count = result.stdout.count('| Loss:')
            total_steps = step_count

        # If no loss found, check stderr for errors
        if not final_loss or final_loss == "":
            if result.returncode != 0:
                notes = f"Error: {result.stderr[:200]}"
            else:
                notes = "No output"

    except subprocess.TimeoutExpired:
        wall_time = time.time() - start_time
        # Try to parse partial results
        notes = "Timeout"
        # Still try to get last loss
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                for line in reversed(lines):
                    if '| Loss:' in line:
                        parts = line.split('|')
                        for part in parts:
                            if 'Loss:' in part:
                                final_loss = part.split(':')[1].strip()
                        break
        except:
            pass

    except Exception as e:
        wall_time = time.time() - start_time
        notes = f"Exception: {str(e)[:100]}"

    # Log results
    log_experiment(run_id, params, final_loss, tokens_per_sec, total_steps, total_tokens, wall_time, notes)

    return {
        'run_id': run_id,
        'params': params,
        'final_loss': final_loss,
        'tokens_per_sec': tokens_per_sec,
        'wall_time': wall_time,
        'notes': notes
    }


# Phase 1: Learning Rate Sweep (experiments 1-15)
def run_phase1():
    """Sweep learning rate broadly."""
    print("=" * 60)
    print("Phase 1: Learning Rate Sweep")
    print("=" * 60)

    # Broad sweep: 1e-5 to 3e-2 (log scale)
    lr_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    # Plus some intermediate values
    lr_values = sorted(set(lr_values))

    results = []
    for i, lr in enumerate(lr_values):
        run_id = i + 1
        params = {
            'lr': lr,
            'min_lr': lr * 0.1,
            'batch_size': 2,
            'total_batch_size': 65536,
            'block_size': 4096,
            'warmup_steps': 0,
            'weight_decay': 0.1,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'grad_clip': 1.0,
        }

        print(f"\nExperiment {run_id}: LR = {lr}")
        result = run_experiment(run_id, params, timeout=300)
        results.append(result)
        print(f"  Loss: {result['final_loss']}, Tokens/sec: {result['tokens_per_sec']}, Notes: {result['notes']}")

    # Print top 5
    print("\n" + "=" * 60)
    print("Top 5 configs so far:")
    valid_results = [r for r in results if r['final_loss'] not in ['OOM', 'NaN', '', None] and r['notes'] == '']
    if valid_results:
        valid_results.sort(key=lambda x: float(x['final_loss']))
        for r in valid_results[:5]:
            print(f"  Run {r['run_id']}: LR={r['params']['lr']}, Loss={r['final_loss']}")

    return results


if __name__ == "__main__":
    init_log()
    run_phase1()
