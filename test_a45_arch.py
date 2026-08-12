"""CPU smoke test for the argonne4.5 architecture additions.

Two jobs:
  1. PARITY -- with every new flag at its default, this model.py must be numerically identical to
     argonne4.0's. Every 4.0 checkpoint and launcher has to keep working untouched.
  2. EFFECT -- each new flag must actually do the thing it claims. The whole reason 4.5 exists is
     Finding A: `local_attention_window` has been configured-but-inert in production since 3.0
     because it was only wired into a flash-attn-2 path this cluster does not have. A flag that
     silently does nothing is the failure mode being tested for.

Run:  python test_a45_arch.py
"""

import subprocess
import sys

import torch

import model as new_model


def _load_baseline():
    """Import argonne4.0's model.py alongside this one for the parity check."""
    import importlib.util
    import tempfile
    import os

    import transformers

    src = subprocess.run(
        ["git", "show", "argonne4.0:model.py"],
        capture_output=True, text=True, check=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    ).stdout
    path = os.path.join(tempfile.mkdtemp(), "baseline_model.py")
    with open(path, "w") as fh:
        fh.write(src)

    # Both modules register "argonne2" on the Auto* classes at import; the second one would raise.
    saved = (transformers.AutoConfig.register, transformers.AutoModel.register,
             transformers.AutoModelForCausalLM.register)
    transformers.AutoConfig.register = lambda *a, **k: None
    transformers.AutoModel.register = lambda *a, **k: None
    transformers.AutoModelForCausalLM.register = lambda *a, **k: None
    try:
        spec = importlib.util.spec_from_file_location("baseline_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        (transformers.AutoConfig.register, transformers.AutoModel.register,
         transformers.AutoModelForCausalLM.register) = saved
    return mod


BASE_KWARGS = dict(
    vocab_size=256, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=128, max_position_embeddings=64,
    use_flash_attention=True,
)
BATCH, SEQ = 2, 32


def _build(module, seed=0, **overrides):
    torch.manual_seed(seed)
    cfg = module.ArgonneConfig(**{**BASE_KWARGS, **overrides})
    m = module.ArgonneModel(cfg).eval()
    return m


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, BASE_KWARGS["vocab_size"], (BATCH, SEQ), generator=g)


results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


print("\n=== 1. PARITY: defaults must reproduce argonne4.0 exactly ===")
baseline = _load_baseline()
x = _inputs()
old = _build(baseline)
new = _build(new_model)
new.load_state_dict(old.state_dict())
with torch.no_grad():
    lo = old(x).logits
    ln = new(x).logits
delta = (lo - ln).abs().max().item()
check("default config is bit-identical to argonne4.0", delta == 0.0, f"max|delta|={delta:.3e}")

n_old = sum(p.numel() for p in old.parameters())
n_new = sum(p.numel() for p in new.parameters())
check("default param count unchanged", n_old == n_new, f"{n_old} vs {n_new}")

print("\n=== 2. Finding A: the sliding window must actually mask ===")
# All-local layers with a tight window: a token further back than the window cannot influence the
# last position. On argonne4.0's model.py this test fails -- the window is dropped on the SDPA path.
win = 4
m = _build(new_model, attn_pattern="L", sliding_window_size=win)
x2 = x.clone()
x2[:, 0] = (x2[:, 0] + 7) % BASE_KWARGS["vocab_size"]  # perturb far outside the window
with torch.no_grad():
    a = m(x).logits[:, -1]
    b = m(x2).logits[:, -1]
check("perturbation outside window does not reach the last token",
      torch.equal(a, b), f"max|delta|={(a - b).abs().max().item():.3e}")

with torch.no_grad():
    c = m(x).logits[:, 1]
    d = m(x2).logits[:, 1]
check("perturbation INSIDE window does reach a nearby token",
      not torch.equal(c, d), f"max|delta|={(c - d).abs().max().item():.3e}")

old_win = _build(baseline, interleaved_local_attention=True, local_attention_window=win)
old_win.load_state_dict(_build(new_model, interleaved_local_attention=True,
                               local_attention_window=win).state_dict())
with torch.no_grad():
    e = old_win(x).logits[:, -1]
    f = old_win(x2).logits[:, -1]
check("(control) argonne4.0 leaks across the window -- the bug this fixes",
      not torch.equal(e, f), f"max|delta|={(e - f).abs().max().item():.3e}")

print("\n=== 3. Attention layout: LLLG + NoPE on global layers ===")
m = _build(new_model, num_hidden_layers=8, attn_pattern="LLLG",
           sliding_window_size=8, nope_global=True)
windows = [blk.attn.sliding_window for blk in m.blocks]
ropes = [blk.attn.use_rope for blk in m.blocks]
check("LLLG gives 3 sliding : 1 global per group",
      windows == [8, 8, 8, None] * 2, str(windows))
check("nope_global strips RoPE from exactly the global layers",
      ropes == [True, True, True, False] * 2, str(ropes))

m_rope = _build(new_model, num_hidden_layers=8, attn_pattern="LLLG",
                sliding_window_size=8, nope_global=False)
m_rope.load_state_dict(m.state_dict())
with torch.no_grad():
    check("NoPE changes the computation", not torch.equal(m(x).logits, m_rope(x).logits))

print("\n=== 4. Document masking ===")
m = _build(new_model, attn_pattern="G", doc_mask=True)
doc_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
doc_ids[:, SEQ // 2:] = 1                      # two packed documents per row
x3 = x.clone()
x3[:, 0] = (x3[:, 0] + 3) % BASE_KWARGS["vocab_size"]   # perturb inside document 0 only
with torch.no_grad():
    a = m(x, document_ids=doc_ids).logits[:, -1]
    b = m(x3, document_ids=doc_ids).logits[:, -1]
check("doc 1 is unaffected by a change in doc 0", torch.equal(a, b),
      f"max|delta|={(a - b).abs().max().item():.3e}")
with torch.no_grad():
    c = m(x).logits[:, -1]
check("without document_ids the mask is off (cross-doc attention returns)",
      not torch.equal(a, c))

print("\n=== 5. ReLU^2 MLP ===")
m = _build(new_model, mlp_type="relu2")
check("relu2 block has no gate_proj",
      not hasattr(m.blocks[0].mlp, "gate_proj") and hasattr(m.blocks[0].mlp, "up_proj"))
mlp_params = sum(p.numel() for p in m.blocks[0].mlp.parameters())
swiglu_params = sum(p.numel() for p in _build(new_model).blocks[0].mlp.parameters())
check("relu2 FFN is 2/3 the params of SwiGLU at equal width",
      mlp_params * 3 == swiglu_params * 2, f"{mlp_params} vs {swiglu_params}")
with torch.no_grad():
    check("relu2 forward produces finite logits", torch.isfinite(m(x).logits).all().item())

print("\n=== 6. Gated attention ===")
m = _build(new_model, attn_gate=True)
check("gate_proj exists on attention", hasattr(m.blocks[0].attn, "gate_proj"))
with torch.no_grad():
    check("gated forward is finite", torch.isfinite(m(x).logits).all().item())

print("\n=== 7. Real MTP module ===")
m_off = _build(new_model)
m_on = _build(new_model, mtp_module_layers=1, mtp_loss_weight=0.3)
missing = m_on.load_state_dict(m_off.state_dict(), strict=False).missing_keys
check("MTP adds only mtp_modules.* parameters",
      all(k.startswith("mtp_modules.") for k in missing), f"{len(missing)} new tensors")

m_on.train()
m_off.train()
labels = _inputs(seed=2)
loss_on = m_on(x, labels=labels).loss
loss_off = m_off(x, labels=labels).loss
check("MTP loss is added on top of the trunk CE", loss_on.item() > loss_off.item(),
      f"{loss_on.item():.4f} vs {loss_off.item():.4f}")

loss_on.backward()
grads = [p.grad is not None and p.grad.abs().sum().item() > 0
         for n, p in m_on.named_parameters() if n.startswith("mtp_modules.")]
check("gradients reach every MTP parameter", all(grads), f"{sum(grads)}/{len(grads)}")

m_on.eval()
with torch.no_grad():
    check("MTP is training-only: eval logits match the no-MTP model",
          torch.equal(m_on(x).logits, m_off.eval()(x).logits))

print("\n=== 8. Combined 4.5 candidate config ===")
m = _build(new_model, num_hidden_layers=8, attn_pattern="LLLG", sliding_window_size=8,
           nope_global=True, attn_gate=True, mlp_type="relu2", doc_mask=True,
           mtp_module_layers=1, mtp_loss_weight=0.3, rope_theta=500000.0)
m.train()
out = m(x, labels=labels, document_ids=doc_ids)
check("all flags together: finite loss", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
out.loss.backward()
n_nograd = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
check("all flags together: every parameter receives a gradient", not n_nograd, str(n_nograd[:3]))

failed = [n for n, ok, _ in results if not ok]
print(f"\n{'=' * 60}\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
