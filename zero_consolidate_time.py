#!/usr/bin/env python3
"""How long does ZeroRedundancyOptimizer.consolidate_state_dict() actually take here?

Training under ZeRO works (17k tok/s on 4x A40) but the first checkpoint never appears and
"Saving checkpoint..." never prints -- and consolidate_if_sharded() runs immediately BEFORE that
print. So consolidation is the suspect. This times it in isolation, with progress heartbeats, to
tell SLOW-BUT-FINITE (raise the interval / accept it) from a TRUE HANG (needs a different save
strategy). PyTorch consolidates by gathering each rank's pickled local state_dict through the
process group, and this optimizer's state is 15.38 GiB, so pathological slowness is plausible.
"""
import os, sys, time, threading, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
sys.path.insert(0, "/home/youzhi/ArgonneAI")
from model import ArgonneConfig, ArgonneModel

CKPT = "/project/rcc/youzhi/models/argonne45_pretrain/checkpoint_last.pt"
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
dist.init_process_group("nccl"); torch.cuda.set_device(local)
MAIN = rank == 0
def log(m):
    if MAIN: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

cfg = ArgonneConfig(vocab_size=151680, hidden_size=2560, num_hidden_layers=24,
    num_attention_heads=10, num_key_value_heads=2, intermediate_size=7040,
    max_position_embeddings=1024, rope_theta=1e6, use_flash_attention=True, qk_norm=True,
    v_norm=True, sandwich_norm=True, z_loss_weight=0.0, mtp_horizon=1, mtp_loss_weight=0.0,
    interleaved_local_attention=False, local_attention_window=None, attn_pattern=None,
    sliding_window_size=2048, nope_global=False, attn_gate=False, mlp_type="swiglu",
    mtp_module_layers=0, doc_mask=False, logit_softcap=15.0, loss_chunk_size=0,
    tie_word_embeddings=True)
cfg._keep_in_fp32_modules = []
m = ArgonneModel(cfg).to("cuda"); m.gradient_checkpointing_enable(); m.checkpoint_stride = 2
m = DDP(m, device_ids=[local], gradient_as_bucket_view=True)
opt = ZeroRedundancyOptimizer(m.parameters(), optimizer_class=torch.optim.AdamW,
                              lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
log("built model + ZeRO optimizer")
ck = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
m.module.load_state_dict(ck['model_state_dict'])
opt.load_state_dict(ck['optimizer_state_dict'])
inner = getattr(opt, 'optim', opt)
for g in inner.param_groups:
    for p in g['params']:
        st = inner.state.get(p)
        if st and 'step' in st and torch.is_tensor(st['step']) and st['step'].device.type == 'cpu':
            st['step'] = st['step'].to(p.device)
del ck
log("loaded checkpoint state")

# one real step so the optimizer state is fully materialised, as it is at a real save point
ids = torch.randint(0, 151680, (2, 1024), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(input_ids=ids, labels=ids)
    loss = out["loss"] if isinstance(out, dict) else out[0]
loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
log("completed one optimizer step")

stop = threading.Event()
def beat():
    t0 = time.time()
    while not stop.wait(15):
        if MAIN: print(f"    ... consolidating, {time.time()-t0:.0f}s elapsed", flush=True)
threading.Thread(target=beat, daemon=True).start()

log("calling consolidate_state_dict(to=0) -- ALL ranks")
t0 = time.time()
opt.consolidate_state_dict(to=0)
dt = time.time() - t0
stop.set()
log(f"consolidate_state_dict RETURNED in {dt:.1f}s")
if MAIN:
    t1 = time.time(); sd = opt.state_dict()
    print(f"[{time.strftime('%H:%M:%S')}] state_dict() in {time.time()-t1:.1f}s, {len(sd.get('state',{}))} entries", flush=True)
    print(f"VERDICT: consolidation is SLOW-BUT-FINITE at {dt:.1f}s -- usable if << the 7h slice", flush=True)
dist.barrier(); dist.destroy_process_group()
