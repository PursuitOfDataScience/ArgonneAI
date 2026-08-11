#!/usr/bin/env python3
"""ON-POLICY DISTILLATION: per-token reverse KL to a teacher, on the STUDENT's own rollouts.

WHY THIS EXISTS, and why it is not a thirteenth imitation arm.
Every post-training lever run on argonne4-think so far is a LIKELIHOOD objective on a set of
sequences somebody else picked: stage-C CoT-SFT, the verify tier, RFT/STaR rounds 1-2,
distillation from 3.5-think, distillation from Llama-3.1-8B. Their combined effect is exactly
what a likelihood objective predicts -- pass@8 rose 62.4 -> 69.0 while greedy sat at ~43 and
`acc|ANSWERED` never left ~50% (3.5-think: 70%). Two structural reasons:

  1. **The fuel is capped by the student's own correctness.** `fail_taxonomy.py` on 93,912
     on-policy rollouts of the current best checkpoint: 44.1% of training problems are NEVER
     solved in 8 samples, and only 23.3% of rollouts are correct. RFT can train on the 23% and
     must throw the other 77% away. That is why round 2 saturated.
  2. **Imitating an off-policy teacher does not transfer.** Llama-3.1-8B solves 45.7% of the same
     problems (3x the student) and distilling its text was measured at 39.24 vs the 39.21
     baseline -- a null, with `acc|ANSWERED` DOWN to 46.1%. The student learned a style it cannot
     execute. Off-policy sequence imitation is the wrong channel.

Per-token reverse KL on the student's own traces fixes both. The traces come from the student, so
there is no distribution shift; the supervision is the teacher's full next-token distribution at
every state the student actually visits, so a WRONG trace is just as informative as a right one --
the 77% stops being waste. Reverse KL is mode-seeking, which is the property this line needs: the
failure is a diffuse argmax (greedy 43 under pass@8 69), and mode-seeking sharpens one mode
instead of spreading mass over eight. Discount zero: each token is graded on its own, as in
Thinking Machines' on-policy distillation recipe.

WHAT MAKES IT POSSIBLE HERE. argonne4 kept the Qwen3 tokenizer, so a Qwen3 teacher and the
student assign IDENTICAL ids to identical text -- verified, not assumed: tokenizer vocab (151,643
entries), merges, all 26 added tokens, and the `<think>`/`</think>`/`<|im_end|>` ids are equal
between `think_combo` and `Qwen3-4B-Thinking-2507`, and a real 329-token trace tokenises to the
same id sequence under both. So ONE token sequence can be fed to both models and the
distributions are directly comparable, position by position. (The teacher's 151,936-wide head is
sliced to the student's 151,669 and renormalised; those extra ids are reserved tokens the student
has no output for.)

The teacher is only ever run FORWARD, never sampled. vLLM arch support is therefore irrelevant --
a teacher vLLM 0.11.2 cannot serve is still usable here.

  python reasoning/opd_train.py \
      --student /project/rcc/youzhi/models/a4_think_final/think_combo \
      --model_def model.py \
      --teacher /project/rcc/youzhi/toxic-models/Qwen/Qwen3-4B-Thinking-2507 \
      --rollouts /project/rcc/youzhi/data/a4_dpo/a4_dpo_all.jsonl \
      --out /project/rcc/youzhi/models/a4_think_final/think_opd
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

RDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(RDIR)
for _p in (RDIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_cotsft():
    """cot-sft.py has a hyphen in its name, so it cannot be imported normally.

    Reuse its loaders rather than copying them: this model needs manual construction to re-tie
    lm_head (AutoModelForCausalLM silently fails to) and its rotary embedding rebuilt at every
    layer. Divergence between two copies of that sequence is exactly the class of bug that has
    cost this line whole runs.
    """
    path = os.path.join(RDIR, "cot-sft.py")
    spec = importlib.util.spec_from_file_location("cot_sft_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cot_sft_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

LABEL_ORDER = ["correct", "wrong", "unclosed", "no_answer"]


def build_rows(rollouts, tok, build_ids, max_seq_len, per_problem, labels_keep,
               eos_id, seed, hint_template="", solve_band=None):
    """One row per rollout: prompt ids + the trace the STUDENT actually generated.

    Stratified by label so a batch contains both states the student got right and states it got
    wrong -- the wrong ones are where the teacher has something to say. NOTHING is filtered for
    quality on purpose: unlike RFT, a bad trace is not waste here, it is the most informative
    state the teacher can be asked about.

    With `hint_template`, each row also carries a SECOND prompt for the teacher, containing
    privileged information the student does not get (the gold answer, and a verified reference
    solution when one of the rollouts found it). The completion is identical in both, so the two
    models are compared token-for-token at states the student visits while the teacher knows more.
    That is what turns a frozen copy of the student into a stronger teacher without a capacity gap
    -- the failure mode that made off-policy imitation of Llama-3.1-8B a null here.
    """
    by_q = defaultdict(list)
    with open(rollouts) as f:
        for line in f:
            r = json.loads(line)
            by_q[(r["pool"], r["question"])].append(r)

    rng = random.Random(seed)
    rows, stat = [], Counter()
    for (pool, q), rs in sorted(by_q.items()):
        if solve_band is not None:
            # ZONE-OF-PROXIMAL-DEVELOPMENT filter. 44.1% of these problems are never solved in K
            # samples and 67% of the math_train_hard ones are; on those a teacher holding the answer
            # produces a distribution the student has no route to, so the divergence is large and the
            # gradient is spent on something unlearnable. The mirror risk is real too -- a problem
            # solved 8/8 has nothing to teach. Off by default because it is a hypothesis, not a
            # finding: the arms that ran first deliberately included everything.
            nc = sum(1 for r in rs if r["label"] == "correct")
            if not (solve_band[0] <= nc <= solve_band[1]):
                stat["drop_outside_solve_band"] += 1
                continue
        buckets = defaultdict(list)
        for r in rs:
            if r["label"] in labels_keep:
                buckets[r["label"]].append(r)
        for v in buckets.values():
            rng.shuffle(v)
        # round-robin over labels so every problem contributes a MIX, not 3 copies of one mode
        picked, i = [], 0
        while len(picked) < per_problem:
            avail = [L for L in LABEL_ORDER if buckets.get(L)]
            if not avail:
                break
            L = avail[i % len(avail)]
            picked.append(buckets[L].pop())
            i += 1
        p_ids = build_ids(tok, q)
        t_ids = p_ids
        if hint_template:
            gold = str(rs[0].get("gold", ""))
            ref = ""
            good = [r for r in rs if r["label"] == "correct"]
            if good:
                # the shortest verified-correct trace: a reference DERIVATION, not just the answer.
                # 44.1% of problems are never solved in K samples, so many rows legitimately get
                # the answer alone -- which is the case RFT could never use at all.
                ref = min(good, key=lambda r: len(r["trace"]))["trace"]
                ref = ref.split("</think>")[0].replace("<think>", "").strip()
                ref = " A correct derivation is: " + tok.decode(
                    tok.encode(ref, add_special_tokens=False)[:256])
            hint = hint_template.format(gold=gold, solution=ref)
            t_ids = build_ids(tok, q + hint)
            stat["hint_with_solution" if ref else "hint_answer_only"] += 1
        for r in picked:
            tr = r["trace"]
            c_ids = tok.encode(tr, add_special_tokens=False)
            # A generation that ended on its own emitted <|im_end|>, which vLLM strips from the
            # text. An `unclosed` trace hit the token cap instead, so it has no terminator to
            # learn. Getting this backwards would train the model to stop mid-sentence.
            if r["label"] != "unclosed":
                c_ids = c_ids + [eos_id]
            if max(len(p_ids), len(t_ids)) + len(c_ids) > max_seq_len:
                stat["drop_too_long"] += 1
                continue
            if len(c_ids) < 8:
                stat["drop_too_short"] += 1
                continue
            rows.append({"ids": p_ids + c_ids, "n_prompt": len(p_ids),
                         "t_ids": t_ids + c_ids, "t_n_prompt": len(t_ids),
                         "n_comp": len(c_ids), "label": r["label"], "pool": pool})
            stat[f"keep_{r['label']}"] += 1
    rng.shuffle(rows)
    return rows, stat


def make_micro_batches(rows, max_batch_tokens, seed, window=256):
    """Group rows into micro-batches by PADDED TOKEN COUNT, not row count.

    Two reasons, both measured on this line rather than assumed:
      * MEMORY. The KD term materialises several [rows, T, 151669] fp32 tensors. With a row-count
        batch, peak HBM is set by the longest row that happens to land together -- an unlucky batch
        of four 1024-token traces OOMs 80% of the way through a run that had been fine for an hour.
        A token budget makes the peak a constant the probe can actually verify.
      * THROUGHPUT. These traces average 344 tokens with a p95 of 611, so a fixed row count either
        wastes the card on short batches or risks the long ones. Length-grouped batching was worth
        1.74x on a4's SFT for exactly this reason.
    Rows are shuffled, sorted inside windows (so a batch is length-homogeneous), packed, and then
    the resulting micro-batches are shuffled again so step order stays random.
    """
    rng = random.Random(seed)
    # the budget must bound the LONGER of the two packs: with a privileged hint the teacher's
    # sequence is up to ~285 tokens longer than the student's, and it is the teacher's forward that
    # would OOM first if the budget only counted the student.
    rowlen = lambda i: max(len(rows[i]["ids"]), len(rows[i].get("t_ids") or ()))
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    batches = []
    for w in range(0, len(idx), window):
        chunk = sorted(idx[w:w + window], key=rowlen)
        cur, cur_max = [], 0
        for i in chunk:
            L = rowlen(i)
            m = max(cur_max, L)
            if cur and (len(cur) + 1) * m > max_batch_tokens:
                batches.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(i)
                cur_max = m
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def _pack(batch, pad_id, id_key, plen_key, prefix_frac=1.0):
    T = max(len(b[id_key]) for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), T), dtype=torch.bool)     # True = a completion token
    for i, b in enumerate(batch):
        n = len(b[id_key])
        ids[i, :n] = torch.tensor(b[id_key], dtype=torch.long)
        p = b[plen_key]
        if prefix_frac < 1.0:
            # KD ON THE TRACE PREFIX ONLY. Two independent findings point here:
            #   * the failure is at the OPENING -- 79% of wrong traces differ from a correct derivation
            #     at equation index 0, median shared-equation prefix 0% (§41b);
            #   * the trace-lengthening is BODY IMITATION, not a closure-probability effect -- removing
            #     all gradient from the terminator logits changed trace length by one token (§41w).
            # So take the teacher's early decisions, where the information is, and stop before learning
            # its verbosity in the tail. The student's own tail is left entirely alone.
            n = p + max(8, int(round((n - p) * prefix_frac)))
        mask[i, p:min(n, len(b[id_key]))] = True
    return ids, mask


def collate(batch, pad_id, kd_prefix_frac=1.0):
    """Two packed batches over the SAME completions: the student's prompt and the teacher's.

    They are separate tensors because a privileged hint makes the teacher's prompt longer, so the
    completion sits at different absolute positions in each. Nothing needs to be reconciled -- each
    model runs its own forward pass and the loss only ever compares the two models' distributions
    for the same completion TOKEN, gathered by each sequence's own mask.
    """
    ids, mask = _pack(batch, pad_id, "ids", "n_prompt")
    t_ids, t_mask = _pack(batch, pad_id, "t_ids", "t_n_prompt")
    is_corr = torch.tensor([b["label"] == "correct" for b in batch], dtype=torch.bool)
    if kd_prefix_frac >= 1.0:
        return ids, mask, t_ids, t_mask, is_corr, mask, t_mask
    # separate, SHORTER masks for the KD term only; CE and the diagnostics keep the full completion.
    # Both sides are cut by the same fraction of the same completion, so the two gathers still line up
    # token-for-token -- asserted at the call site rather than trusted.
    _, kmask = _pack(batch, pad_id, "ids", "n_prompt", kd_prefix_frac)
    _, t_kmask = _pack(batch, pad_id, "t_ids", "t_n_prompt", kd_prefix_frac)
    return ids, mask, t_ids, t_mask, is_corr, kmask, t_kmask


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def gather_completion(logits, tgt_mask, V):
    """Flatten to [n_completion_tokens, V] the rows that PREDICT a completion token.

    Position t's logits predict token t+1, so the predicting positions are the completion mask
    shifted left by one. Gathering instead of masking in place is what lets the student and the
    teacher have different prompt lengths -- and it keeps padding out of the fp32 vocab math.
    """
    pred = tgt_mask[:, 1:]                       # [B, T-1] True where the NEXT token is completion
    return logits[:, :-1, :V][pred]              # [n, V], row-major = per-sequence order


def kd_loss(s_flat, t_flat, teacher_temp, div, keep_idx=None):
    """Per-token divergence between the student's and the teacher's next-token distributions.

    `revkl` = KL(student || teacher) is the default and is mode-SEEKING: it asks the student to put
    its mass where the teacher has mass, rather than to cover every mode the teacher has. That is
    the right direction here -- a 1.04B student spreading mass over eight modes is precisely this
    model's defect (greedy 43 under pass@8 69). `jsd` is symmetric and bounded, which is what the
    consensus-self-distillation literature uses when teacher and student are the same model
    conditioned differently and the two distributions are already close.
    """
    sf = s_flat
    tf = t_flat
    if keep_idx is not None:
        # ⚠️THE SURGICAL FIX FOR §41f. Dropping the `</think>` and eos COLUMNS from the divergence --
        # not just the positions whose target IS a terminator -- is what protects termination. The
        # damage in arm 1 came from the ~200 positions per trace where the target is ordinary text
        # and a long-CoT teacher assigns near-zero mass to closing: reverse KL pushes the student's
        # closing mass down at every one of them, and closure is the integral of that hazard. With
        # the columns removed, both distributions are renormalised over content tokens only, the
        # loss measures the SHAPE OF THE REASONING, and the terminator logits receive no gradient
        # from it at all -- so the student keeps its own trace-length prior while still learning
        # what to say next from a much stronger teacher.
        sf = sf.index_select(1, keep_idx)
        tf = tf.index_select(1, keep_idx)
    tl = tf.float()
    if teacher_temp != 1.0:
        tl = tl / teacher_temp
    log_ps = F.log_softmax(sf.float(), dim=-1)
    log_pt = F.log_softmax(tl, dim=-1)           # renormalised over the SHARED vocab
    if div == "fwdkl":
        pt = log_pt.exp()
        return (pt * (log_pt - log_ps)).sum(-1).mean()
    if div == "jsd":
        ps = log_ps.exp()
        pt = log_pt.exp()
        log_m = ((ps + pt) * 0.5).clamp_min(1e-12).log()
        return 0.5 * ((ps * (log_ps - log_m)).sum(-1) + (pt * (log_pt - log_m)).sum(-1)).mean()
    ps = log_ps.exp()
    return (ps * (log_ps - log_pt)).sum(-1).mean()


def ce_loss(s_flat, ids, tgt_mask, row_mask):
    """Plain next-token CE on selected rows -- the anchor term, off by default."""
    pred = tgt_mask[:, 1:]
    tgt = ids[:, 1:][pred]
    keep = row_mask.unsqueeze(1).expand_as(pred)[pred]
    if not bool(keep.any()):
        return s_flat.sum() * 0.0
    return F.cross_entropy(s_flat[keep].float(), tgt[keep])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--model_def", default="model.py")
    ap.add_argument("--tokenizer_path", default="")
    ap.add_argument("--teacher", default="",
                    help="a model dir, or 'self' to freeze a copy of the student as the teacher "
                         "(only meaningful together with --hint-template)")
    ap.add_argument("--hint-template", default="",
                    help="appended to the TEACHER's user turn only, e.g. "
                         "'\\n\\n(Reference: the correct answer is {gold}.)'. Fields: {gold}, "
                         "{solution}. Turns a frozen copy of the student into a better-informed "
                         "teacher with no capacity or style gap.")
    ap.add_argument("--div", default="revkl", choices=["revkl", "jsd", "fwdkl"])
    ap.add_argument("--solve-band", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                    help="keep only problems whose rollouts contain LO..HI correct ones, e.g. 1 7 "
                         "to drop both the never-solved and the already-mastered")
    ap.add_argument("--kd-prefix-frac", type=float, default=1.0,
                    help="apply the KD loss only to the first FRAC of each trace's completion tokens "
                         "(1.0 = all). Mechanism-matched to §41b (the failure is at the opening) and "
                         "§41w (the lengthening is body imitation, not a closure effect).")
    ap.add_argument("--exclude-terminators", type=int, default=0,
                    help="drop the </think> and eos COLUMNS from the divergence, so the teacher has "
                         "no influence on trace length. Required for a teacher whose trace-length "
                         "distribution differs from the student's -- see the closure-hazard note in "
                         "the training loop.")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-problem", type=int, default=3)
    ap.add_argument("--labels", nargs="*", default=["correct", "wrong", "unclosed", "no_answer"])
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--rope-theta", type=float, default=1000000.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-batch-tokens", type=int, default=8192,
                    help="padded tokens per micro-batch; sets peak HBM (see make_micro_batches)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.0,
                    help="CE on gold-verified rows only. 0 = pure on-policy distillation.")
    ap.add_argument("--teacher-temp", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--stats-out", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda"
    cot = _load_cotsft()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from clean_eval import build_ids

    tok_path = a.tokenizer_path or a.student
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    eos_id = cot.detect_eos_from_template(tok) or tok.eos_token_id
    print(f"[opd] tokenizer {tok_path}  len={len(tok)}  eos={eos_id}", flush=True)

    # ---- student ----------------------------------------------------------------------
    model_module = cot.import_model_definition(a.model_def)
    ArgonneConfig = getattr(model_module, "ArgonneConfig")
    ArgonneModel = getattr(model_module, "ArgonneModel")
    RotaryEmbedding = getattr(model_module, "RotaryEmbedding")

    def load_argonne(src_dir, train):
        """Manual construction, not from_pretrained: this arch needs lm_head re-tied to
        embed_tokens and its rotary embedding rebuilt at EVERY layer, and from_pretrained silently
        does neither. Shared by the student and by a 'self' teacher so they cannot diverge."""
        cfg_d = json.load(open(os.path.join(src_dir, "config.json")))
        cfg_d = {k: v for k, v in cfg_d.items() if not k.startswith("_")}
        sd = cot.load_hf_state_dict(src_dir)
        for key in ("embed_tokens.weight", "lm_head.weight"):
            if key in sd:
                cfg_d["vocab_size"] = int(sd[key].shape[0])
                break
        m = ArgonneModel(ArgonneConfig(**cfg_d))
        miss, unexp = m.load_state_dict(sd, strict=False)
        m.tie_weights()
        m.config.rope_theta = a.rope_theta
        m.config.max_position_embeddings = max(
            a.max_seq_len, int(cfg_d.get("max_position_embeddings", 0) or 0))
        m.config.block_size = m.config.max_position_embeddings
        cot.replace_rotary_embeddings(m, RotaryEmbedding, a.rope_theta,
                                     m.config.max_position_embeddings)
        m.config.use_flash_attention = True
        for blk in m.blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "use_flash_attention"):
                blk.attn.use_flash_attention = True
        m.config.use_cache = False
        m.config.loss_chunk_size = 0       # KD needs logits; the chunked-CE path returns None
        if train:
            m.gradient_checkpointing_enable()
        del sd
        return m, miss, unexp

    model, miss, unexp = load_argonne(a.student, True)
    V = int(model.config.vocab_size)
    print(f"[opd] student {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B  vocab={V}  "
          f"missing={len(miss)} unexpected={len(unexp)}  "
          f"tied={model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}",
          flush=True)
    model.to(dev)
    model.train()

    # ---- teacher: forward-only, bf16, frozen -------------------------------------------
    # A pure-CE pass (--kd-weight 0) needs NO teacher: §41ap's repair pass trains on the model's own
    # verified-correct traces to pull in the unclosed tail and the empty-think mode, and loading a 2.88B
    # teacher to multiply its output by zero would cost a forward pass per micro-batch for nothing.
    t0 = time.time()
    if a.kd_weight == 0.0:
        if a.ce_weight <= 0.0:
            raise SystemExit("--kd-weight 0 with --ce-weight 0 has no loss at all")
        teacher = None
        print(f"[opd] NO teacher: pure-CE pass (kd_weight=0, ce_weight={a.ce_weight}) on rows "
              f"labelled {sorted(a.labels)}", flush=True)
    self_teacher = (a.teacher == "self")
    if a.kd_weight == 0.0:
        pass
    elif self_teacher:
        if not a.hint_template:
            raise SystemExit("--teacher self without --hint-template is a no-op: the teacher would "
                             "be the student's own initial distribution and every KL term is 0")
        teacher, _, _ = load_argonne(a.student, False)
        teacher.to(torch.bfloat16)
        ttok = tok
    elif (os.path.isdir(a.teacher)
          and json.load(open(os.path.join(a.teacher, "config.json"))).get("model_type") == "argonne2"):
        # An Argonne-arch teacher (e.g. the released 3.5-think) must go through the same manual
        # construction as the student: AutoModelForCausalLM does not re-tie lm_head for this arch and
        # would load a teacher whose head is random -- a silently meaningless target distribution.
        teacher, tmiss, _ = load_argonne(a.teacher, False)
        teacher.to(torch.bfloat16)
        ttok = AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True)
        print(f"[opd] argonne-arch teacher, missing={len(tmiss)} "
              f"tied={teacher.embed_tokens.weight.data_ptr() == teacher.lm_head.weight.data_ptr()}",
              flush=True)
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            a.teacher, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True)
        ttok = AutoTokenizer.from_pretrained(a.teacher)
    if teacher is not None:
        teacher.config.use_cache = False
        teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    if teacher is not None:
        print(f"[opd] teacher {'self (frozen copy of the student)' if self_teacher else os.path.basename(a.teacher)}  "
              f"{sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B  "
              f"vocab={teacher.config.vocab_size}  div={a.div}  loaded in {time.time() - t0:.0f}s",
              flush=True)
    if a.hint_template:
        print(f"[opd] teacher hint template: {a.hint_template!r}", flush=True)

    # HARD GATE on the assumption the whole method rests on. If the two tokenizers ever disagree
    # on a single id, every KL term is computed against the teacher's distribution for a
    # DIFFERENT token and the run is silently meaningless.
    probe = ("Natalia sold clips to 48 friends in April, and then she sold half as many "
             "clips in May. How many clips did Natalia sell altogether?\n"
             "<think>\n48 / 2 = 24, so 48 + 24 = 72.\n</think>\n\nThe answer is $\\boxed{72}$.")
    ia = tok.encode(probe, add_special_tokens=False)
    ib = ia if teacher is None else ttok.encode(probe, add_special_tokens=False)
    if ia != ib:
        raise SystemExit(f"FATAL tokenizer mismatch: student {len(ia)} ids vs teacher {len(ib)} "
                         "-- per-token KD is only defined under identical tokenisation")
    for t in ("<think>", "</think>", "<|im_end|>"):
        if teacher is not None and tok.convert_tokens_to_ids(t) != ttok.convert_tokens_to_ids(t):
            raise SystemExit(f"FATAL special-token id mismatch on {t!r}")
    print(f"[opd] tokenizer identity verified on a {len(ia)}-token probe "
          f"(+ <think>/</think>/<|im_end|> ids)", flush=True)

    # ---- data ------------------------------------------------------------------------
    rows, dstat = build_rows(a.rollouts, tok, build_ids, a.max_seq_len, a.per_problem,
                             set(a.labels), eos_id, a.seed, a.hint_template,
                             tuple(a.solve_band) if a.solve_band else None)
    print(f"[opd] rows={len(rows):,}  " + "  ".join(f"{k}={v:,}" for k, v in sorted(dstat.items())),
          flush=True)
    if not rows:
        raise SystemExit("FATAL no training rows")
    mean_comp = sum(len(r["ids"]) - r["n_prompt"] for r in rows) / len(rows)
    print(f"[opd] mean completion tokens {mean_comp:.0f}  mean total {sum(len(r['ids']) for r in rows) / len(rows):.0f}",
          flush=True)

    mb = make_micro_batches(rows, a.max_batch_tokens, a.seed)
    seq_per_mb = sum(len(b) for b in mb) / len(mb)
    _rl = lambda i: max(len(rows[i]["ids"]), len(rows[i].get("t_ids") or ()))
    pad_frac = 1.0 - sum(sum(_rl(i) for i in b) for b in mb) / \
        sum(len(b) * max(_rl(i) for i in b) for b in mb)
    steps_per_epoch = len(mb) // a.grad_accum
    total_steps = a.max_steps if a.max_steps > 0 else steps_per_epoch * a.epochs
    print(f"[opd] micro-batches={len(mb):,}  {seq_per_mb:.1f} seq/micro  "
          f"{a.max_batch_tokens} tok budget  padding {pad_frac * 100:.1f}%  "
          f"accum={a.grad_accum}  eff {seq_per_mb * a.grad_accum:.0f} seq/step  "
          f"steps/epoch={steps_per_epoch}  total={total_steps}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / max(1, a.warmup)
        prog = (s - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    keep_idx = None
    if a.exclude_terminators:
        term = {tok.convert_tokens_to_ids("</think>"), int(eos_id)}
        keep_idx = torch.tensor([i for i in range(V) if i not in term],
                                dtype=torch.long, device=dev)
        print(f"[opd] terminator columns excluded from the divergence: {sorted(term)} "
              f"({V} -> {keep_idx.numel()} columns). The teacher cannot move trace length.",
              flush=True)

    pad_id = eos_id
    hist = []
    step = 0
    t_start = time.time()
    micro_i = 0
    # `n` counts every micro-step (loss terms); `diag_n` counts only the micro-steps the
    # diagnostics ran on. Dividing the diagnostics by `n` under grad_accum>1 under-reports them by
    # exactly that factor -- which is what made the 6-step probe read 38% agreement against the
    # real run's 77% on the same models.
    accum = {"kd": 0.0, "ce": 0.0, "n": 0, "diag_n": 0, "argmax_agree": 0.0, "close_ps": 0.0,
             "close_pt": 0.0, "close_n": 0, "haz_ps": 0.0, "haz_pt": 0.0}
    warned_hazard = False

    for ep in range(a.epochs):
        epoch_mb = mb if ep == 0 else make_micro_batches(rows, a.max_batch_tokens, a.seed + ep)
        for group in epoch_mb:
            batch = [rows[i] for i in group]
            ids, cmask, t_ids, t_cmask, is_corr, kmask, t_kmask = collate(
                batch, pad_id, a.kd_prefix_frac)
            ids, cmask, is_corr = ids.to(dev), cmask.to(dev), is_corr.to(dev)
            t_ids, t_cmask = t_ids.to(dev), t_cmask.to(dev)
            kmask, t_kmask = kmask.to(dev), t_kmask.to(dev)

            t_logits = None
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_logits = teacher(input_ids=t_ids).logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_out = model(input_ids=ids)
            s_logits = s_out.logits

            # Gathered, not masked in place: the teacher's prompt may be LONGER than the student's
            # (a privileged hint), so the same completion token sits at different absolute
            # positions. Both gathers walk their own sequence in order, so row k of each flat
            # tensor is the same completion token under both models.
            s_flat = gather_completion(s_logits, kmask, V)
            t_flat = None if t_logits is None else gather_completion(t_logits, t_kmask, V)
            if t_flat is not None and s_flat.shape[0] != t_flat.shape[0]:
                raise RuntimeError(f"completion-token count mismatch student {s_flat.shape[0]} vs "
                                   f"teacher {t_flat.shape[0]} -- the two prompts must share the "
                                   "completion exactly")
            L_kd = (torch.zeros((), device=dev) if t_flat is None
                    else kd_loss(s_flat, t_flat, a.teacher_temp, a.div, keep_idx))
            L = a.kd_weight * L_kd
            L_ce = torch.zeros((), device=dev)
            if a.ce_weight > 0:
                # CE keeps the FULL completion even when KD is prefix-only: the anchor's whole job is
                # to hold the model's own tail behaviour in place.
                L_ce = ce_loss(gather_completion(s_logits, cmask, V), ids, cmask, is_corr)
                L = L + a.ce_weight * L_ce
            (L / a.grad_accum).backward()

            # Diagnostics run on the FIRST micro-step of each accumulation group (micro_i is still
            # pre-increment here, so this fires exactly once per optimizer step): they need
            # another pass over a [n_tokens, 151669] tensor and paying that every micro-step would
            # slow the run for numbers only read in the log.
            if micro_i % a.grad_accum == 0 and t_flat is not None:
                with torch.no_grad():
                    n_tok = s_flat.shape[0]
                    if n_tok:
                        accum["diag_n"] += 1
                        sl = s_flat.detach()
                        accum["argmax_agree"] += float((sl.argmax(-1) == t_flat.argmax(-1)).sum()) / n_tok
                        # ⚠️THE CLOSURE HAZARD, and read this comment before changing it.
                        # The first version of this diagnostic reported p(`</think>`) only at
                        # positions where the student's trace ALREADY closed. That is a biased
                        # sample of exactly the states where teacher and student agree: it read a
                        # comfortable 0.72-0.88 against the student's ~1.00 all the way through a run
                        # that came out of the oven with a 96.95% UNCLOSED rate and greedy 1.75.
                        # What actually matters is the MARGINAL hazard -- the mean closing mass over
                        # EVERY completion position, because closure is what that hazard integrates
                        # to over a 200-token trace. A long-CoT teacher can hold reasonable mass on
                        # `</think>` at a finished derivation while holding essentially none of it
                        # mid-trace, and reverse KL then drives the student's per-position hazard to
                        # zero and it never terminates at all.
                        ci = tok.convert_tokens_to_ids("</think>")
                        ei = eos_id
                        s_lse, t_lse = sl.logsumexp(-1), t_flat.logsumexp(-1)
                        haz_s = ((sl[:, ci] - s_lse).exp() + (sl[:, ei] - s_lse).exp()).mean()
                        haz_t = ((t_flat[:, ci] - t_lse).exp() + (t_flat[:, ei] - t_lse).exp()).mean()
                        accum["haz_ps"] += float(haz_s)
                        accum["haz_pt"] += float(haz_t)
                        tgt_flat = ids[:, 1:][cmask[:, 1:]]
                        close_pos = tgt_flat == ci
                        if bool(close_pos.any()):
                            accum["close_ps"] += float((sl[:, ci] - s_lse)[close_pos].exp().mean())
                            accum["close_pt"] += float((t_flat[:, ci] - t_lse)[close_pos].exp().mean())
                            accum["close_n"] += 1
            accum["kd"] += float(L_kd)
            accum["ce"] += float(L_ce)
            accum["n"] += 1
            micro_i += 1

            if micro_i % a.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0 or step == 1:
                    n = max(1, accum["n"])
                    dn = max(1, accum["diag_n"])
                    cn = max(1, accum["close_n"])
                    print(f"[opd] step {step}/{total_steps}  revKL {accum['kd'] / n:.4f}  "
                          f"ce {accum['ce'] / n:.4f}  agree {accum['argmax_agree'] / dn * 100:.1f}%  "
                          f"haz s {accum['haz_ps'] / dn:.4f} t {accum['haz_pt'] / dn:.4f}  "
                          f"p(</think>|closed) s {accum['close_ps'] / cn:.2f} t "
                          f"{accum['close_pt'] / cn:.2f}  gnorm {float(gn):.2f}  "
                          f"lr {lr_at(step):.2e}  "
                          f"HBM {torch.cuda.max_memory_allocated() / 2**30:.1f}G  "
                          f"{(time.time() - t_start) / 60:.1f}min", flush=True)
                    hist.append({"step": step, "revKL": accum["kd"] / n, "ce": accum["ce"] / n,
                                 "agree": accum["argmax_agree"] / dn,
                                 "p_close_student": accum["close_ps"] / cn,
                                 "p_close_teacher": accum["close_pt"] / cn,
                                 "hazard_student": accum["haz_ps"] / dn,
                                 "hazard_teacher": accum["haz_pt"] / dn})
                    # THE kill signal. A teacher holding <1/5 of the student's per-position closing
                    # mass will drive termination to zero over a 200-token trace, whatever the
                    # conditional p(</think>|already closed) says. Loud, and early.
                    if (not warned_hazard and accum["haz_pt"] / dn < 0.2 * accum["haz_ps"] / dn):
                        warned_hazard = True
                        print(f"WARNING closure-hazard collapse risk: teacher "
                              f"{accum['haz_pt'] / dn:.5f} vs student {accum['haz_ps'] / dn:.5f} "
                              f"per position. This is how a run ends at 97% unclosed. Consider "
                              f"--ce-weight > 0 or a teacher with a matching trace-length "
                              f"distribution.", flush=True)
                    accum = {k: (0.0 if isinstance(v, float) else 0) for k, v in accum.items()}
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    print(f"[opd] done {step} steps in {(time.time() - t_start) / 60:.1f} min", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.to(torch.bfloat16)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    cpath = os.path.join(a.out, "config.json")
    c = json.load(open(cpath))
    c["eos_token_id"] = 151645       # the deployed stop token; 151643 never terminates a chat turn
    c["dtype"] = "bfloat16"
    c.pop("auto_map", None)
    json.dump(c, open(cpath, "w"), indent=2)
    # build_ids() renders the chat template, so a checkpoint without one silently evaluates on a
    # different prompt than it trained on. save_pretrained normally writes it; copy if it did not.
    for f in ("chat_template.jinja",):
        p = os.path.join(a.out, f)
        src = os.path.join(tok_path, f)
        if not os.path.exists(p) and os.path.exists(src):
            import shutil
            shutil.copy(src, p)
            print(f"[opd] copied {f} from {tok_path}", flush=True)
    # The soup builder and the vLLM port both index weights BY NAME. A save that renames or drops
    # a tensor loads as a fresh-init model and scores like noise, which reads as "the method
    # failed". Compare against the checkpoint we started from.
    from safetensors.torch import load_file
    src_keys = set(cot.load_hf_state_dict(a.student).keys())
    new_keys = set(load_file(os.path.join(a.out, "model.safetensors")).keys())
    if src_keys != new_keys:
        print(f"WARNING key-set drift: only-in-source={sorted(src_keys - new_keys)[:6]} "
              f"only-in-saved={sorted(new_keys - src_keys)[:6]}", flush=True)
    else:
        print(f"[opd] tensor key set identical to the source checkpoint ({len(new_keys)} tensors)",
              flush=True)
    print(f"[opd] saved -> {a.out}  eos={c['eos_token_id']}", flush=True)
    open(os.path.join(a.out, ".opd_complete"), "w").write(str(step))

    if a.stats_out:
        json.dump({"rows": len(rows), "data_stat": dstat, "steps": step,
                   "hist": hist, "args": vars(a)}, open(a.stats_out, "w"), indent=1)
        print(f"[opd] wrote {a.stats_out}", flush=True)


if __name__ == "__main__":
    main()
