# Staged HF config fixes: READ-ONLY plan, nothing pushed
Generated 2026-09-17 02:5x CDT from the LIVE Hub configs (`hf_hub_download` of each `config.json`).
No write, create, delete or upload call was made. `auto_map` is **excluded** per the owner's
2026-09-16 decision ("it's okay without automap as long as they can use the models").

## (a) `eos_token_id: null` on a chat-shaped repo → `.generate()` never stops: 5 repos
Set `eos_token_id: 151645` (`<|im_end|>`, the token these were trained to emit).

| repo | now | after |
|---|---|---|
| Argonne-1.0-Instruct | key **ABSENT** (not `null`; same effect on `.generate()`) | `151645` |
| Argonne-2.5-ctx13568-instruct | `null` | `151645` |
| Argonne-2.5-think | `null` | `151645` |
| Argonne2.5-instruct | `null` | `151645` |
| argonne-3.0-instruct | `null` | `151645` |

## (b) a sliding window the weights never saw → silently wrong attention on flash-attn-2: 5 repos
Set `interleaved_local_attention: false` and `local_attention_window: null`.
**CORRECTED 2026-09-17 12:4x CDT from a live re-audit:** this line first read
`enable_interleaved_local_attention`, a key that exists in NO config and in no version of the arch.
`ArgonneConfig.__init__` (model.py:69-70) declares `interleaved_local_attention: bool = True` and
`local_attention_window: Optional[int] = 256`, and those are the two keys the live configs carry.
Editing the prefixed name would have added a dead field and left the real one `True`, so the plan
would have reported success while the defect survived. The window was
dead code in every pretrain (no `flash_attn_interface` ⇒ SDPA ⇒ dropped), so the weights are
full-attention on all 24 layers; a user whose env HAS flash-attn-2 gets attention the model never
trained with.

| repo | now | after |
|---|---|---|
| argonne-3.0-base | `True` / `256` | `false` / `null` |
| argonne-3.0-instruct | `True` / `256` | `false` / `null` |
| Argonne-3.0-think | `True` / `256` | `false` / `null` |
| argonne-3.5-base | `True` / `256` | `false` / `null` |
| Argonne-3.5-think | `True` / `256` | `false` / `null` |

## (c) NOT in the ask: one extra field the audit flags, for a yes/no
`Argonne-2.0` is a **base** model carrying the **chat** eos `151645`; it should be `151643` or null.
One field, same mechanism. Say if you want it included.

## Scope
- **9 unique repos** (`argonne-3.0-instruct` appears in both (a) and (b)), plus `Argonne-2.0` if (c).
- `config.json` only. No weights, no tokenizer, no README.
- Deliberately unchanged: `eos_token_id: null` on `argonne-3.0-base` / `argonne-3.5-base`, null eos is
  correct for a base model, and the ask is about the chat repos.

## Re-audit 2026-09-17 12:4x CDT
Re-ran against the live Hub because the ask had been open 1d 20h and a stale ask is worse than none.
Both tables still name the RIGHT repos, and two details were wrong; both are corrected above.
Confirmed unchanged: the 5 eos repos, and `interleaved_local_attention=True` / `local_attention_window=256`
on exactly the 5 repos in (b). `Argonne-3.0-think` and `Argonne-3.5-think` already carry
`eos_token_id=151645`, which is why they appear only in (b). `argonne-3.0-base` and `argonne-3.5-base`
also have a null eos and are excluded from (a) on purpose: they are base models, not chat.
Still nothing pushed; no write, create, delete or upload call was made.
