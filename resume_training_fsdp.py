import argparse

from training import DEFAULT_MAX_TRAINING_STEPS, train_fsdp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Argonne FSDP resume (use training.py for fresh pretraining)",
    )
    parser.add_argument("--data-glob", type=str, required=True, help="Glob pattern for dataset shards")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_MAX_TRAINING_STEPS)
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=str, default="fsdp_checkpoints")
    parser.add_argument("--resume-from", type=str, required=True)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fsdp(
        data_glob=args.data_glob,
        tokenizer_path=args.tokenizer_path,
        total_steps=args.total_steps,
        block_size=args.block_size,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        save_interval=args.save_interval,
        trust_remote_code=args.trust_remote_code,
        require_resume=True,
        entrypoint="resume_training_fsdp.py",
    )
