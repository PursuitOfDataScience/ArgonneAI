#!/usr/bin/env python3
"""exp_longctx_learning.py -- IS long-context training actually effective learning?

Everything measured so far said only what FITS and how FAST it runs. Neither speaks to whether the
model actually LEARNS to use positions beyond what it was pretrained at. Both models were pretrained
at block 1024; phase B extends to 13,568 purely by RoPE theta=1e6 extrapolation, and that has never
been validated. This is the experiment that tests it.

THE A/B IS REAL -- both sides of the intervention exist as checkpoints:
  a35_pre   models/pretrain/checkpoint_step_308733.pt   base + reasoning anneal, trained ONLY at 1024
  a35_post  models/midtrain/checkpoint_step_311475.pt   the SAME model +1.34B tokens at block 13,568
plus, optionally,
  a4_anneal  the live a4 anneal checkpoint            1.04B params, trained ONLY at 1024 (no phase B)

METRIC: position-bucketed next-token NLL (nats) on held-out long documents, out to 24,576 -- i.e.
PAST phase B's own 13,568 training length, so extrapolation beyond training is measured too.

EVAL DATA: proof-pile-2 arXiv, tokenized. Genuinely held out for BOTH arms: 3.5 saw FineWeb+FineMath
(pretrain) then code/math/reasoning/tool (anneal). Neither includes arXiv.

FALSIFIABLE PREDICTIONS -- the outcomes are distinguishable, which is the point:
  H1  RoPE extrapolates unaided     -> a35_pre NLL keeps FALLING with position past 1024.
                                       Then phase B is largely unnecessary.
  H2  extrapolation breaks          -> a35_pre NLL goes FLAT or RISES past ~1024-4096.
                                       Then phase B is necessary, and H3 says whether it worked.
  H3  phase B is effective learning -> a35_post < a35_pre at long positions AND the gap GROWS with
                                       position. A uniform offset at all positions would instead mean
                                       phase B just did generic training, not context extension.
  H4  it generalizes past training  -> the post-vs-pre gap persists in the 13,568-24,576 buckets.
                                       If the gap collapses there, phase B taught only its own length.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F

BUCKETS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192),
           (8192, 13568), (13568, 20480), (20480, 24576)]
EVAL_LEN = 24576
# --eval_len extends both of the above (see main()); the 0-1024 control bucket is always kept
# because it is what makes the attribution airtight -- gains must appear ONLY at long positions.
EXTRA_EDGES = [32768, 40960, 49152, 65536]
DOCBIN = "/project/rcc/youzhi/data/proof_pile2_arxiv_qwen3_docbin/data"
TOKP = "/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base"


def load_ckpt(pt_path, repo, theta=1e6):
    """Build the model from `repo`'s constants and load a raw training .pt. Mirrors the extraction
    helper used by reasoning/a4_dose.sh so arch never has to be guessed."""
    sys.path.insert(0, repo)
    from model import ArgonneConfig, ArgonneModel
    from transformers import AutoTokenizer
    ns = {}
    for line in open(os.path.join(repo, "pretrain.py")):
        if re.match(r"^[A-Z][A-Z0-9_]*\s*=\s*(True|False|None|[-\d.eE]+)\s*(#.*)?$", line):
            exec(line, ns)
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    st = ck["model_state_dict"]
    for pfx in ("_orig_mod.", "module."):
        if any(k.startswith(pfx) for k in st):
            st = {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in st.items()}
    step = ck.get("global_step"); ck = None
    tok = AutoTokenizer.from_pretrained(TOKP, trust_remote_code=True)
    kw = dict(vocab_size=st["embed_tokens.weight"].shape[0], hidden_size=ns["HIDDEN_SIZE"],
              num_hidden_layers=ns["NUM_LAYERS"], num_attention_heads=ns["NUM_HEADS"],
              num_key_value_heads=ns["NUM_KV_HEADS"], max_position_embeddings=EVAL_LEN,
              rope_theta=theta, use_flash_attention=True, qk_norm=ns["ENABLE_QK_NORM"],
              v_norm=ns["ENABLE_V_NORM"], sandwich_norm=ns["ENABLE_SANDWICH_NORM"],
              z_loss_weight=ns["Z_LOSS_WEIGHT"],
              interleaved_local_attention=ns["ENABLE_INTERLEAVED_LOCAL_ATTENTION"],
              local_attention_window=(ns["LOCAL_ATTENTION_WINDOW"]
                                      if ns["ENABLE_INTERLEAVED_LOCAL_ATTENTION"] else None),
              logit_softcap=ns["LOGIT_SOFTCAP"], tie_word_embeddings=True)
    if "INTERMEDIATE_SIZE" in ns:
        kw["intermediate_size"] = ns["INTERMEDIATE_SIZE"]
    cfg = ArgonneConfig(**kw); cfg.block_size = EVAL_LEN; cfg._keep_in_fp32_modules = []
    m = ArgonneModel(cfg)
    miss, unexp = m.load_state_dict(st, strict=False)
    bad = [k for k in miss if "lm_head" not in k]
    assert not bad, "arch mismatch, missing: %s" % bad[:6]
    assert not unexp, "unexpected: %s" % list(unexp)[:6]
    return m.to(torch.bfloat16).to("cuda").eval(), step, len(tok)


def eval_windows(model, windows):
    """Per-bucket summed NLL + token count over each window. CE chunked in fp32 so the
    24,576 x 151,680 logit tensor never materialises in fp32."""
    tot = {b: 0.0 for b in BUCKETS}; cnt = {b: 0 for b in BUCKETS}
    with torch.no_grad():
        for w in windows:
            x = torch.from_numpy(w[:-1].astype(np.int64))[None].to("cuda")
            y = torch.from_numpy(w[1:].astype(np.int64))[None].to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).logits
            for (lo, hi) in BUCKETS:
                hi2 = min(hi, logits.shape[1])
                if hi2 <= lo:
                    continue
                s = 0.0
                for a in range(lo, hi2, 1024):
                    b = min(a + 1024, hi2)
                    s += F.cross_entropy(logits[0, a:b].float(), y[0, a:b],
                                         reduction="sum").item()
                tot[(lo, hi)] += s; cnt[(lo, hi)] += (hi2 - lo)
            del logits
            torch.cuda.empty_cache()
    return {b: (tot[b] / cnt[b] if cnt[b] else float("nan")) for b in BUCKETS}, cnt


def get_windows(n_docs, seed=0, glob_pat="*.bin"):
    """Held-out arXiv windows of exactly EVAL_LEN+1 tokens, taken from docs long enough to fill one."""
    bins = sorted(glob.glob(os.path.join(DOCBIN, glob_pat)))
    assert bins, "no tokenized arXiv shards matching %s" % glob_pat
    rng = np.random.default_rng(seed)
    out = []
    for bp in bins:
        L = np.load(bp[:-4] + ".lengths.npy")
        starts = np.concatenate([[0], np.cumsum(L.astype(np.int64))[:-1]])
        ok = np.flatnonzero(L >= EVAL_LEN + 1)
        if not len(ok):
            continue
        toks = np.memmap(bp, dtype=np.uint32, mode="r")
        for i in rng.permutation(ok):
            out.append(np.asarray(toks[starts[i]:starts[i] + EVAL_LEN + 1]))
            if len(out) >= n_docs:
                return out
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--arms", default="a35_pre,a35_post")
    ap.add_argument("--out", default="report/exp_longctx_learning.json")
    ap.add_argument("--eval_len", type=int, default=EVAL_LEN,
                    help="Window length; buckets auto-extend to cover it.")
    ap.add_argument("--docbin_glob", default="*.bin",
                    help="Restrict eval windows to these shards. REQUIRED once a model trains on "
                         "arXiv: phase C trains on proof-pile-2, so the default '*.bin' would draw "
                         "eval windows from TRAINED-ON documents. Phase C trains on arXiv_0[0-8]*; "
                         "pass 'arXiv_09*.bin' to stay on the reserved holdout shards.")
    a = ap.parse_args()

    ARMS = {
        "a35_pre":  ("/project/rcc/youzhi/models/pretrain/checkpoint_step_308733.pt", "/home/youzhi/ArgonneAI"),
        "a35_post": ("/project/rcc/youzhi/models/midtrain/checkpoint_step_311475.pt", "/home/youzhi/ArgonneAI"),
    }
    # a35_final: midtraining RAN TO COMPLETION 2026-08-01 (epoch 1 @step 321062, 6.02B tok @block 13568,
    # LR cooled 1e-4 -> 1e-5). a35_post above was only 22% in (step 311475, pre-cooldown), so re-measuring
    # on the final weights is what says whether the remaining 78% + the LR decay held the extension.
    mid = sorted(glob.glob("/project/rcc/youzhi/models/midtrain/checkpoint_step_*.pt"),
                 key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))
    if mid:
        ARMS["a35_final"] = (mid[-1], "/home/youzhi/ArgonneAI")
    # a4's live anneal ckpt (block-1024-only, no phase B) -- extrapolation baseline for a 1.04B model
    a4d = "/project/rcc/youzhi/models/argonne4_pretrain"
    a4 = sorted(glob.glob(os.path.join(a4d, "checkpoint_step_*.pt")),
                key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))
    if a4:
        ARMS["a4_anneal"] = (a4[-1], "/home/youzhi/ArgonneAI-4.0")

    if a.eval_len != EVAL_LEN:
        EVAL_LEN = a.eval_len
        edges = [e for e in [1024, 2048, 4096, 8192, 13568, 20480, 24576] + EXTRA_EDGES if e < EVAL_LEN]
        BUCKETS = list(zip([0] + edges, edges + [EVAL_LEN]))
    ARMS["a4_phaseb"] = ("/project/rcc/youzhi/models/argonne4_midtrain/checkpoint_step_109622.pt",
                         "/home/youzhi/ArgonneAI-4.0")
    c = sorted(glob.glob("/project/rcc/youzhi/models/argonne4_midtrain_c/checkpoint_step_*.pt"),
               key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))
    if c:
        ARMS["a4_phasec"] = (c[-1], "/home/youzhi/ArgonneAI-4.0")
    print("eval shards: %s   window %d" % (a.docbin_glob, EVAL_LEN))
    w = get_windows(a.docs, glob_pat=a.docbin_glob)
    print("eval: %d held-out arXiv windows of %d tokens (%.2fM tokens/arm)"
          % (len(w), EVAL_LEN, len(w) * EVAL_LEN / 1e6), flush=True)
    res = {}
    for name in a.arms.split(","):
        if name not in ARMS:
            print("  skip unknown arm %s" % name); continue
        pt, repo = ARMS[name]
        t0 = time.time()
        m, step, _ = load_ckpt(pt, repo)
        nll, cnt = eval_windows(m, w)
        res[name] = dict(ckpt=pt, step=step, nll={"%d-%d" % b: v for b, v in nll.items()},
                         tokens={"%d-%d" % b: c for b, c in cnt.items()})
        print("  %-10s step %-8s  " % (name, step)
              + "  ".join("%s:%.4f" % ("%dk" % (b[1] // 1024), nll[b]) for b in BUCKETS)
              + "   (%.0fs)" % (time.time() - t0), flush=True)
        del m; torch.cuda.empty_cache()

    print("\n%-14s " % "bucket" + "".join("%12s" % k for k in res) +
          ("%14s" % "post-pre" if {"a35_pre", "a35_post"} <= set(res) else ""))
    for b in BUCKETS:
        k = "%d-%d" % b
        row = "%-14s " % ("%d-%d" % b) + "".join("%12.4f" % res[m]["nll"][k] for m in res)
        if {"a35_pre", "a35_post"} <= set(res):
            d = res["a35_post"]["nll"][k] - res["a35_pre"]["nll"][k]
            row += "%+14.4f" % d
        print(row)
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nwrote %s" % a.out)
    print("READ: H1 pre keeps falling past 1k -> RoPE extrapolates unaided. H2 pre flat/rising -> it")
    print("      breaks. H3 post-pre negative AND growing with position -> phase B is real context")
    print("      extension. H4 gap persists in the 13.5k-24.5k buckets -> it generalizes past training.")
