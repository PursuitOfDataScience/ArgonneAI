#!/usr/bin/env python3
"""Does a ZeRO-written checkpoint load into a PLAIN, unsharded AdamW?

The gate for using beagle3: a checkpoint that only reloads on the same GPU count is a trap, not a
migration path -- this run has already crossed three machines.

⚠️ DO NOT compare the optimizer-entry count against the number of model_state_dict TENSORS. This
model TIES embed_tokens.weight and lm_head.weight, so 339 state-dict keys correspond to 338 distinct
Parameters, and AdamW keys state by distinct parameter. An earlier version of this script asserted
339 == 338 and reported a correct merge as "looks like a SHARD".

Shapes are taken from each state entry's own exp_avg, so no model construction is needed and the
peak stays near the optimizer state itself (~15.4 GiB plus the dummy params).
"""
import sys, torch

path = sys.argv[1]
ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
osd, msd = ck['optimizer_state_dict'], ck['model_state_dict']
st, pg = osd.get('state', {}), osd.get('param_groups', [])
n = len(st)

print(f"file        : {path.split('/')[-1]}")
print(f"step/tokens : {ck['global_step']} / {ck['tokens_processed']:,}  loss {ck['loss']:.4f}")
print(f"tensors     : {len(msd)} state-dict keys -> {n} distinct params "
      f"(difference is the tied embedding/lm_head)")

keys_ok = sorted(st.keys()) == list(range(n))
pg_ok = bool(pg) and list(pg[0].get('params', [])) == list(range(n))
print(f"keys        : int-indexed 0..{n-1}? {keys_ok}")
print(f"param_groups: {len(pg)} group(s), params == range({n})? {pg_ok}")
if not (keys_ok and pg_ok):
    print("*** FAIL: not in plain-AdamW layout -- a shard-only save would look like this")
    raise SystemExit(1)

missing = [i for i in range(n) if 'exp_avg' not in st[i] or 'exp_avg_sq' not in st[i]]
if missing:
    print(f"*** FAIL: {len(missing)} entries lack exp_avg/exp_avg_sq (first: {missing[:5]})")
    raise SystemExit(1)

# Functional check: a plain AdamW over params shaped from the state itself must accept this dict.
params = [torch.nn.Parameter(torch.zeros(st[i]['exp_avg'].shape, dtype=torch.float32))
          for i in range(n)]
opt = torch.optim.AdamW(params, lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
try:
    opt.load_state_dict(osd)
except Exception as e:
    print(f"*** UNSHARDED LOAD FAILED: {type(e).__name__}: {str(e)[:300]}")
    raise SystemExit(1)

got = opt.state_dict()['state']
steps = {int(v['step']) for v in got.values() if 'step' in v}
print(f"UNSHARDED LOAD: OK -- {len(got)} entries restored")
print(f"AdamW step counter across entries: {sorted(steps)[:4]}{' ...' if len(steps) > 4 else ''}")
print("=> PORTABLE: resumable unsharded (Spark, H100) as well as sharded (beagle3)")
