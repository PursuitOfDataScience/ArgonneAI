#!/usr/bin/env python3
"""Online RLVR via GRPO (Group Relative Policy Optimization) for Argonne 3.0.

Why this and not more STaR: rejection-sampling SFT (star_generate.py + SFT)
saturated — it sharpens FACT execution on the CoT path but never fixes
reasoning-CHAIN correctness (sum-loop, 2x+5, logic puzzles persist across
think_mix2/star/star2). GRPO puts reward on the FULL rollout: sample a group
of G traces per problem, reward = verified-correct \\boxed answer, and push the
policy toward the above-average traces and away from the below-average ones,
with a KL leash to a frozen reference so it doesn't collapse / forget.

Design notes for THIS model:
  - ArgonneModel.forward forces attention_mask=None (pure causal, no padding
    support). RIGHT-padding is therefore SAFE: a real token at position i only
    attends to <= i, so trailing pad tokens never affect a real token's logits.
    We right-pad each group to its max length and mask the loss to real tokens.
  - Sampling and log-prob must come from the SAME distribution for the policy
    gradient to be unbiased. We sample at temperature T with NO top_k/top_p
    truncation, and compute log-probs as log_softmax(logits / T). The frozen
    reference uses the same T.
  - One inner update per batch (mu=1), so the importance ratio is 1 at the
    point of the gradient and the clipped surrogate reduces to the group-
    baseline policy gradient -A * logp; we keep the KL (k3 estimator) leash.
  - KV-cache generation (model.py) makes rollout sampling ~10x faster, which
    is what makes online RL tractable here at all.
"""

import argparse, json, math, os, sys, time, datetime as _dt
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
# model.py lives at the repo ROOT (parent of reasoning/) and self-registers the
# `argonne2` arch at import time. Put BOTH reasoning/ and root on the path so the
# import resolves and AutoModel can load checkpoints that DON'T bundle model.py
# (e.g. soup_blend_a085, built by build_ckpt_soup.py — config has no auto_map).
for _p in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from model import ArgonneConfig, ArgonneModel  # noqa: F401
except ModuleNotFoundError:
    pass
# Reuse the verifier + problem loader + cached sampler from the STaR pipeline.
from star_generate import norm, extract_boxed, load_problems, batched_sample


def shaped_reward(text, gold, has_eos):
    """Graded, verifiable reward. Returns (reward, is_correct).

    Round-1 GRPO used a binary 0/1 reward; with group-normalized advantage that
    gives ZERO gradient whenever a whole group is all-correct or all-wrong (only
    ~3/8 groups carried signal). Grading the reward gives almost every group
    variance -> non-zero advantage, and the ordering
        correct > closed+wrong-answer > closed+no-answer > stopped-unclosed > looping
    directly pressures the policy to CLOSE </think> and stop the enumeration
    loops that ate the token budget, even before it can solve the problem.
    is_correct is tracked separately so logged accuracy stays honest (the policy
    can't game *that* — only the gold check counts).
    """
    closed = "</think>" in text
    pred = extract_boxed(text)
    if closed and pred is not None and pred == gold:
        return 1.0, True
    if closed and pred is not None:
        return 0.3, False          # right format, wrong answer
    if closed:
        return 0.15, False         # closed but no parseable \boxed
    if not has_eos:
        return -0.2, False         # never stopped -> truncated/looping
    return 0.0, False              # stopped without closing


def seq_token_logp(model, ids, temp):
    """Per-token log-prob of ids[:,1:] under the model. ids: [B, L] right-padded.
    Returns [B, L-1]. Caller masks to real continuation tokens.
    """
    logits = model(ids).logits[:, :-1, :].float() / temp
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/project/rcc/youzhi/models/instruct/think_star2_ckpts")
    ap.add_argument("--ref-path", default=None, help="frozen KL reference (default = model-path)")
    ap.add_argument("--out", default="/project/rcc/youzhi/models/instruct/think_grpo_ckpts")
    ap.add_argument("--source", choices=["gsm8k", "math", "both"], default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=0, help="0 = all")
    ap.add_argument("--steps", type=int, default=2000, help="upper bound; usually wall-limited via --max-hours")
    ap.add_argument("--prompts-per-step", type=int, default=12)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--target-hbm", type=float, default=0.0,
                    help="if >0, auto-pick the backward micro-batch to fill this fraction of the "
                         "DETECTED GPU HBM (mixed 80/94GiB pool). peak(micro) ~= 40 + 4.7*micro GiB.")
    ap.add_argument("--gen-group", type=int, default=0,
                    help="generation group size = rollouts/prompt. LARGE -> fills the otherwise-idle "
                         "HBM during rollout generation AND sharpens the group-relative advantage. "
                         "0 = use --group-size. Decoupled from the backward (see --bwd-micro).")
    ap.add_argument("--bwd-micro", type=int, default=0,
                    help="backward micro-batch: split each group's rollouts into chunks of this many "
                         "sequences per backward (the fp32 (n x seq x vocab) logits are the HBM "
                         "ceiling). 0 = whole group. Grads accumulate across chunks -> identical update.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=11.0, help="exit+save before wall limit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: needs GPU"); sys.exit(1)
    torch.manual_seed(args.seed)
    dev = "cuda"
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")

    print(f"Loading policy <- {args.model_path}", flush=True)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True).to(dev)
    print(f"Loading frozen ref <- {args.ref_path or args.model_path}", flush=True)
    ref = AutoModelForCausalLM.from_pretrained(
        args.ref_path or args.model_path, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True).to(dev)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr

    problems = load_problems(args.source, args.n_problems, seed=args.seed)
    print(f"GRPO: {len(problems)} problems, source={args.source} | "
          f"P={args.prompts_per_step} G={args.group_size} steps={args.steps} "
          f"lr={args.lr} beta={args.beta} temp={args.temperature} max_new={args.max_new_tokens}",
          flush=True)

    if args.target_hbm > 0:
        tot_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        # backward micro-batch fills the per-chunk fp32-logits peak: ~= 40 + 4.7*micro GiB
        # (40 = policy+grads+AdamW-fp32+frozen-ref; fit from the G=8/9/10 probes).
        micro_auto = max(6, min(14, round((args.target_hbm * tot_gib - 40.0) / 4.7)))
        if args.bwd_micro <= 0:
            args.bwd_micro = micro_auto
        # large generation group fills the idle HBM during rollouts + better advantage.
        if args.gen_group <= 0:
            args.gen_group = 128
        args.group_size = args.gen_group
        print(f"[auto-hbm] {tot_gib:.0f}GiB card, target {args.target_hbm:.0%} -> "
              f"gen_group={args.group_size} (rollouts/prompt), backward micro-batch={args.bwd_micro}",
              flush=True)
    else:
        if args.gen_group > 0:
            args.group_size = args.gen_group
    G = args.group_size
    MICRO = args.bwd_micro if args.bwd_micro > 0 else G
    order = torch.randperm(len(problems), generator=torch.Generator().manual_seed(args.seed)).tolist()
    ptr = 0
    t_start = time.time()
    run_reward, run_closed, run_correct, run_n = 0.0, 0.0, 0, 0

    for step in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        # ---- pull the next P problems (reshuffle on wrap) ----
        batch = []
        for _ in range(args.prompts_per_step):
            if ptr >= len(order):
                order = torch.randperm(len(problems)).tolist(); ptr = 0
            batch.append(problems[order[ptr]]); ptr += 1

        # ---- sample G rollouts per problem (cached, raw temp-T) ----
        groups = []  # each: dict(prompt_len, seqs[list of full id-lists], rewards, advs)
        policy.eval()
        for (q, gold, _tier) in batch:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=True,
                add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
            pids = (enc["input_ids"] if hasattr(enc, "keys") else enc)[0].tolist()
            # OOM-safe generation: a large group's prefill logits (Kg x prompt_len x vocab)
            # scale with prompt length; on a rare long prompt, halve the group and retry.
            Kg = G
            while True:
                try:
                    inp = torch.tensor([pids], device=dev).repeat(Kg, 1)
                    with torch.inference_mode():
                        gens = batched_sample(policy, inp, max_new_tokens=args.max_new_tokens,
                                              eos_id=eos_id, temperature=args.temperature,
                                              top_k=0, top_p=0)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if Kg <= 8:
                        raise
                    Kg = max(8, Kg // 2)
                    print(f"[gen-oom] long prompt -> retry Kg={Kg}", flush=True)
            seqs, rews = [], []
            for g in gens:
                has_eos = eos_id in g
                # strip trailing tokens after the first eos (batched_sample appends until done)
                if has_eos:
                    g = g[:g.index(eos_id) + 1]
                text = tok.decode(g, skip_special_tokens=True)
                r, correct = shaped_reward(text, gold, has_eos)
                rews.append(r)
                seqs.append(pids + g)
                run_correct += int(correct)
                run_closed += int("</think>" in text)
            groups.append({"plen": len(pids), "seqs": seqs, "rews": rews})
            run_reward += sum(rews)
            run_n += len(rews)

        # ---- advantages (group-normalized) ----
        for gr in groups:
            r = torch.tensor(gr["rews"])
            adv = (r - r.mean()) / (r.std() + 1e-4)
            gr["adv"] = adv.tolist()
            gr["has_signal"] = bool(r.std() > 0)

        # ---- gradient update (one inner epoch; backward per group, then step) ----
        policy.train()
        opt.zero_grad(set_to_none=True)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        n_back, kl_acc, loss_acc, tok_acc, n_oom = 0, 0.0, 0.0, 0, 0
        n_sig = max(1, sum(g["has_signal"] for g in groups))
        for gr in groups:
            if not gr["has_signal"]:
                continue
            plen = gr["plen"]
            gseqs, gadv = gr["seqs"], gr["adv"]
            Kg = len(gseqs)
            # Chunk the group's Kg rollouts into MICRO-sized backward passes. The fp32
            # (chunk x seq x vocab) logits in seq_token_logp are the HBM ceiling, so a big
            # generation group (fills HBM during rollouts) still backprops within memory.
            # loss = sum_chunk[ sum_seq(seq_loss) / (Kg * n_sig) ] == mean over the group's
            # rollouts, mean over signal groups -> identical to the un-chunked update; grads
            # accumulate across chunks. OOM-safe: a rare long chunk is skipped, not fatal.
            for cs in range(0, Kg, MICRO):
                cseqs = gseqs[cs:cs + MICRO]
                cadv = gadv[cs:cs + MICRO]
                m = len(cseqs)
                maxlen = max(len(s) for s in cseqs)
                try:
                    ids = torch.full((m, maxlen), eos_id, dtype=torch.long, device=dev)
                    genmask = torch.zeros((m, maxlen - 1), device=dev)
                    for i, s in enumerate(cseqs):
                        ids[i, :len(s)] = torch.tensor(s, device=dev)
                        # token-logp index j predicts full position j+1; gen tokens live at
                        # full positions [plen, len(s)) -> indices [plen-1, len(s)-1)
                        genmask[i, plen - 1:len(s) - 1] = 1.0
                    adv = torch.tensor(cadv, device=dev).unsqueeze(1)  # [m,1]
                    logp = seq_token_logp(policy, ids, args.temperature)        # grad
                    with torch.no_grad():
                        ref_logp = seq_token_logp(ref, ids, args.temperature).detach()
                    d = ref_logp - logp
                    kl = torch.exp(d) - d - 1.0
                    per_tok = -(adv * logp) + args.beta * kl
                    seq_tok = genmask.sum(1).clamp(min=1.0)
                    loss = ((per_tok * genmask).sum(1) / seq_tok).sum() / (Kg * n_sig)
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    n_oom += 1
                    logp = ref_logp = kl = per_tok = loss = ids = genmask = adv = None
                    torch.cuda.empty_cache()
                    continue
                n_back += 1
                kl_acc += float((kl * genmask).sum() / genmask.sum().clamp(min=1))
                loss_acc += float(loss); tok_acc += int(genmask.sum())

        gnorm = float("nan")
        if n_back > 0:
            gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip))
            opt.step()

        el = time.time() - t_start
        _tot = torch.cuda.get_device_properties(0).total_memory
        _hbm = torch.cuda.max_memory_allocated() / _tot
        print(f"[{step+1}/{args.steps}] acc={run_correct/max(1,run_n):.3f} "
              f"reward={run_reward/max(1,run_n):.3f} closed={run_closed/max(1,run_n):.2f} "
              f"sig_groups={n_sig}/{len(groups)} bwd_chunks={n_back} oom_skip={n_oom} "
              f"kl={kl_acc/max(1,n_back):.4f} loss={loss_acc:.4f} gnorm={gnorm:.2f} "
              f"lr={lr_at(step):.2e} | {el/(step+1):.1f}s/step "
              f"hbm={_hbm*100:.1f}% ({torch.cuda.max_memory_allocated()/1e9:.1f}/{_tot/1e9:.0f}GiB)",
              flush=True)
        # reset running stats each step for a per-step read
        run_reward = run_closed = 0.0; run_correct = run_n = 0

        # ---- checkpointing ----
        due = (step + 1) % args.save_steps == 0
        time_up = el > args.max_hours * 3600
        if due or time_up or step == args.steps - 1:
            policy.save_pretrained(args.out); tok.save_pretrained(args.out)
            print(f"saved -> {args.out} (step {step+1})", flush=True)
            if time_up:
                print("WALL LIMIT reached; exiting after save.", flush=True)
                break

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
