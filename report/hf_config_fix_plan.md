# Staged HF config fixes: READ-ONLY plan, nothing pushed
Generated 2026-09-17 02:5x CDT from the LIVE Hub configs (`hf_hub_download` of each `config.json`).
No write, create, delete or upload call was made. `auto_map` is **excluded** per the owner's
2026-09-16 decision ("it's okay without automap as long as they can use the models").

## (a) `eos_token_id: null` on a chat-shaped repo → `.generate()` never stops: 5 repos
Set `eos_token_id: 151645` (`<|im_end|>`, the token these were trained to emit).

| repo | now | after |
|---|---|---|
| Argonne-1.0-Instruct | `null` | `151645` |
| Argonne-2.5-ctx13568-instruct | `null` | `151645` |
| Argonne-2.5-think | `null` | `151645` |
| Argonne2.5-instruct | `null` | `151645` |
| argonne-3.0-instruct | `null` | `151645` |

## (b) a sliding window the weights never saw → silently wrong attention on flash-attn-2: 5 repos
Set `enable_interleaved_local_attention: false` and `local_attention_window: null`. The window was
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
