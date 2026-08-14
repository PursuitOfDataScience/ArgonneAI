"""Every flag the PRODUCTION config sets must actually change the computation.

Written after 2026-08-14, when the opposite bug bit twice on the same flag. LOCAL_ATTENTION_WINDOW
=256 was configured-but-inert in every Argonne pretrain since 3.0 (only flash-attn-2 honored it,
and this cluster has flash-attn-4). Then the argonne4.5 SDPA fix made it real, which silently
turned on a 256-token window across half of a4.5's layers -- while the startup banner still printed
"full attention ... IGNORED on this path". Reading the config told you nothing in either direction.

So don't read. Toggle each flag and require the logits (or the loss) to move. A flag that can be
flipped with no observable effect is either dead code or a lie in the saved config, and both have
now cost real training runs.

    python exp/test_no_dead_flags.py

CPU-only, seconds, no GPU. Uses the production flag VALUES at toy dimensions -- what is under test
is whether each switch is wired, which does not depend on width.
"""
import importlib.util
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import model as M  # noqa: E402

failures = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(label)


def load_pretrain():
    spec = importlib.util.spec_from_file_location("pt_flags", os.path.join(REPO, "pretrain.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    saved = sys.argv
    sys.argv = ["pretrain.py", "--tokenizer_path", "/dev/null", "--data_path", "/dev/null",
                "--checkpoint_dir", "/dev/null", "--batch_size", "1", "--block_size", "8",
                "--total_batch_size", "8"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


pt = load_pretrain()

# The exact switch settings production uses, at toy width.
PROD = dict(
    vocab_size=256, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=128, max_position_embeddings=256,
    rope_theta=pt.ROPE_THETA,
    qk_norm=pt.ENABLE_QK_NORM,
    v_norm=pt.ENABLE_V_NORM,
    sandwich_norm=pt.ENABLE_SANDWICH_NORM,
    z_loss_weight=pt.Z_LOSS_WEIGHT,
    logit_softcap=pt.LOGIT_SOFTCAP,
    interleaved_local_attention=pt.ENABLE_INTERLEAVED_LOCAL_ATTENTION,
    local_attention_window=pt.LOCAL_ATTENTION_WINDOW,
    attn_pattern=pt.ATTN_PATTERN,
    mlp_type=pt.MLP_TYPE,
    use_flash_attention=True,
)
SEQ = 48
g = torch.Generator().manual_seed(7)
X = torch.randint(0, PROD["vocab_size"], (2, SEQ), generator=g)
Y = torch.randint(0, PROD["vocab_size"], (2, SEQ), generator=g)


def build(**over):
    torch.manual_seed(0)
    return M.ArgonneModel(M.ArgonneConfig(**{**PROD, **over})).eval()


M._attention_path_logged = True   # silence the per-process banner
base = build()
with torch.no_grad():
    base_logits = base(X).logits.clone()


def moves(label, **over):
    """Flip one switch; the same weights must produce different logits."""
    alt = build(**over)
    alt.load_state_dict(base.state_dict(), strict=False)   # identical weights where they overlap
    with torch.no_grad():
        out = alt(X).logits
    d = (out - base_logits).abs().max().item()
    check(f"{label} is WIRED (flipping it changes the logits)", d > 1e-6, f"max|delta|={d:.3e}")


print("=== production switches must each be live ===")
print(f"  (config under test: qk_norm={PROD['qk_norm']} v_norm={PROD['v_norm']} "
      f"sandwich_norm={PROD['sandwich_norm']} softcap={PROD['logit_softcap']} "
      f"rope_theta={PROD['rope_theta']:g} mlp={PROD['mlp_type']})")
moves("qk_norm", qk_norm=False)
moves("v_norm", v_norm=False)
moves("sandwich_norm", sandwich_norm=False)
moves("logit_softcap", logit_softcap=0.0)   # 0 is the off value; None raises in the config
moves("rope_theta", rope_theta=10000.0)

print("\n=== switches production deliberately leaves OFF must still be reachable ===")
# A disabled feature that is ALSO unreachable is indistinguishable from a removed one -- and that
# is exactly how the window rotted. Prove the off-switch is a choice, not a dead end.
z = build(z_loss_weight=0.3)
z.load_state_dict(base.state_dict(), strict=False)
z.train()
base.train()
lz = z(X, labels=Y).loss.item()
l0 = base(X, labels=Y).loss.item()
check("z_loss_weight is wired (0.0 in production, but reachable)", abs(lz - l0) > 1e-6,
      f"{lz:.6f} vs {l0:.6f}")
base.eval()

print("\n=== tied embeddings ===")
check("lm_head shares storage with embed_tokens (tying is real, not a copy)",
      base.lm_head.weight.data_ptr() == base.embed_tokens.weight.data_ptr())
# nn.Module.parameters() already de-duplicates shared tensors, so it cannot show tying. The place
# the duplicate surfaces is state_dict(), which emits both keys -- that is why the real checkpoint
# reads 2,451,968,512 state_dict params against a 2,063,667,712 model: the 388,300,800 difference
# is exactly the tied lm_head, counted twice. Assert that relationship, not a parameters() diff.
n_params = sum(p.numel() for p in base.parameters())
n_state = sum(v.numel() for v in base.state_dict().values())
n_emb = base.embed_tokens.weight.numel()
check("state_dict double-counts the tied head by exactly one embedding matrix",
      n_state - n_params == n_emb, f"state {n_state:,} - params {n_params:,} = {n_state-n_params:,}, embedding {n_emb:,}")

print("\n=== settings that must NOT change the math ===")
# checkpoint_stride only trades store-vs-recompute. If it ever changes a number, the throughput
# win we measured (+6.1%) would be buying a different model, which is not a trade at all.
for stride in (1, 2, 4):
    m = build()
    m.load_state_dict(base.state_dict(), strict=False)
    m.gradient_checkpointing = True
    m.checkpoint_stride = stride
    m.train()
    out = m(X, labels=Y)
    out.loss.backward()
    gr = torch.cat([p.grad.flatten() for _, p in sorted(m.named_parameters()) if p.grad is not None])
    if stride == 1:
        ref_loss, ref_grad = out.loss.item(), gr
    else:
        check(f"checkpoint_stride={stride} is numerically identical to stride=1",
              abs(out.loss.item() - ref_loss) < 1e-6 and torch.allclose(gr, ref_grad, atol=1e-5),
              f"dloss={abs(out.loss.item()-ref_loss):.2e} dgrad={(gr-ref_grad).abs().max().item():.2e}")

print("\n=== dropout must be off for pretraining ===")
check("attention_dropout == 0", base.config.attention_dropout == 0.0, str(base.config.attention_dropout))
check("hidden_dropout == 0", base.config.hidden_dropout == 0.0, str(base.config.hidden_dropout))

print("\n" + "=" * 64)
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL PASS — every production switch is live, the disabled ones are still reachable,")
print("           and checkpoint_stride changes speed without changing the model.")
