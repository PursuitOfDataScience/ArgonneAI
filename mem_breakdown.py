#!/usr/bin/env python3
"""Where does the memory actually GO on an A100-40GB? Measure, don't extrapolate.

Both beagle3 cards OOM'd at every micro-batch, chunked or not, and the usage was nearly identical
at micro=2 and micro=4 -- so the blocker is not micro-batch-proportional. This attributes it stage
by stage, including the two things that are easy to forget:
  * AdamW's m/v are allocated LAZILY on the first .step(), not at construction.
  * resuming LOADS m/v from the checkpoint, so a resumed run peaks where a fresh one does not.
Then it re-measures with ZeroRedundancyOptimizer to show exactly what sharding buys.
"""
import os, sys, argparse, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/home/youzhi/ArgonneAI")
ap.add_argument("--micro", type=int, default=1)
ap.add_argument("--block", type=int, default=1024)
ap.add_argument("--vocab", type=int, default=151680)
ap.add_argument("--zero", type=int, default=0)
ap.add_argument("--ckpt", default="/project/rcc/youzhi/models/argonne45_pretrain/checkpoint_last.pt")
ap.add_argument("--load-opt", type=int, default=1, help="load optimizer state (a resume does)")
ap.add_argument("--fused", type=int, default=1, help="fused AdamW; suspect under ZeRO")
a = ap.parse_args()
sys.path.insert(0, a.repo)
from model import ArgonneConfig, ArgonneModel

rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
local = int(os.environ.get("LOCAL_RANK", 0))
if "RANK" in os.environ:
    dist.init_process_group("nccl")
torch.cuda.set_device(local)
MAIN = rank == 0
G = 2**30

def mark(tag):
    torch.cuda.synchronize()
    if MAIN:
        al = torch.cuda.memory_allocated()/G; pk = torch.cuda.max_memory_allocated()/G
        rs = torch.cuda.memory_reserved()/G
        free, tot = torch.cuda.mem_get_info()
        print(f"  {tag:<38s} alloc {al:6.2f}  peak {pk:6.2f}  reserved {rs:6.2f}  free {free/G:6.2f} / {tot/G:.2f} GiB", flush=True)

if MAIN:
    print(f"=== {torch.cuda.get_device_name(0)} | world={world} | micro={a.micro} | zero={a.zero} | load_opt={a.load_opt} ===")
mark("baseline (empty)")

cfg = ArgonneConfig(vocab_size=a.vocab, hidden_size=2560, num_hidden_layers=24,
    num_attention_heads=10, num_key_value_heads=2, intermediate_size=7040,
    max_position_embeddings=a.block, rope_theta=1e6, use_flash_attention=True,
    qk_norm=True, v_norm=True, sandwich_norm=True, z_loss_weight=0.0, mtp_horizon=1,
    mtp_loss_weight=0.0, interleaved_local_attention=False, local_attention_window=None,
    attn_pattern=None, sliding_window_size=2048, nope_global=False, attn_gate=False,
    mlp_type="swiglu", mtp_module_layers=0, doc_mask=False, logit_softcap=15.0,
    loss_chunk_size=0, tie_word_embeddings=True)
cfg._keep_in_fp32_modules = []
model = ArgonneModel(cfg).to("cuda")
model.gradient_checkpointing_enable(); model.checkpoint_stride = 2
mark("model on GPU (fp32 params)")

if world > 1:
    model = DDP(model, device_ids=[local], gradient_as_bucket_view=True)
    mark("after DDP wrap")

if a.zero:
    from torch.distributed.optim import ZeroRedundancyOptimizer
    kw = dict(lr=6e-4, betas=(0.9,0.95), weight_decay=0.1)
    if a.fused: kw['fused'] = True
    opt = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, **kw)
else:
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9,0.95), weight_decay=0.1, fused=True)
mark("optimizer constructed (m/v LAZY)")

if a.load_opt and os.path.exists(a.ckpt):
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    (model.module if world > 1 else model).load_state_dict(ck['model_state_dict'])
    mark("model_state_dict loaded")
    opt.load_state_dict(ck['optimizer_state_dict'])
    # The checkpoint is mapped to CPU, and ZeRO's load_state_dict does NOT relocate the per-param
    # `step` scalar the way plain Optimizer.load_state_dict does when the fused/capturable AdamW
    # kernel is in play. Result: m/v land on CUDA, `step` stays on CPU, and _fused_adamw_ rejects
    # the mismatch at the FIRST .step() -- long after the memory probe says everything fits.
    inner = getattr(opt, 'optim', opt)
    moved = 0
    for g in inner.param_groups:
        for prm in g['params']:
            st = inner.state.get(prm)
            if st and 'step' in st and hasattr(st['step'], 'device') and st['step'].device.type == 'cpu':
                st['step'] = st['step'].to(prm.device); moved += 1
    if MAIN and moved:
        print(f"  (relocated {moved} `step` scalars CPU -> CUDA)")
    mark("optimizer_state_dict loaded (m/v NOW real)")
    del ck

ids = torch.randint(0, a.vocab, (a.micro, a.block), device="cuda")
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, labels=ids)
        loss = out["loss"] if isinstance(out, dict) else out[0]
    mark("after forward")
    loss.backward()
    mark("after backward (grads live)")
    torch.nn.utils.clip_grad_norm_((model.module if world>1 else model).parameters(), 0.4)
    opt.step(); opt.zero_grad(set_to_none=True)
    mark("after optimizer.step()")
    if MAIN: print("  RESULT: this configuration FITS")
except torch.OutOfMemoryError as e:
    if MAIN: print(f"  RESULT: OOM -- {str(e)[:110]}")
except Exception as e:
    # NOT just OOM: ZeRO's step() can raise for structural reasons (fused kernels, param groups).
    # Catching only OOM previously hid the real cause behind a bare torchrun exitcode 1.
    if MAIN:
        import traceback
        print(f"  RESULT: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
if dist.is_initialized(): dist.destroy_process_group()
