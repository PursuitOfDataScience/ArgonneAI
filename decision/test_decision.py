"""CPU tests for the Argonne decision model (tiny random trunk, real tokenizer). ~1 min.

    python decision/test_decision.py

(1) the encoder reads the answer at "Answer:"'s last token and each option at its closing newline;
(2) right padding is exact: an example's logits do not change when it shares a batch with a longer one;
(3) the KV-cache path (state read once, question block on top) gives the same hidden states as one full pass;
(4) a few optimizer steps on a learnable toy task drive the loss well below ln(K);
(5) save -> load reproduces the logits bit for bit, and the checkpoint keeps the tied embedding;
(6) system_one returns Jev's JSON shape for Choice / Score / Noul, respects 255 options, rejects 256.
"""
import json
import math
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(4)
from decision.modeling import DecisionHead, DecisionModel  # noqa: E402
from decision.schema import Choice, Encoder, NL, Noul, Score  # noqa: E402
from model import ArgonneConfig, ArgonneModel  # noqa: E402

from decision.modeling import local_dir  # noqa: E402
BASE = local_dir(os.environ.get("DECISION_BASE", "PursuitOfDataScience/argonne-4.5-base-ctx13568"))
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)
    return ok


def tiny(vocab=151669):
    cfg = ArgonneConfig(vocab_size=vocab, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, intermediate_size=128, max_position_embeddings=2048, block_size=2048,
                        use_flash_attention=True, tie_word_embeddings=True, qk_norm=True, v_norm=True, sandwich_norm=True,
                        logit_softcap=15.0, rope_theta=1e6, mlp_type="swiglu")
    torch.manual_seed(0)
    trunk = ArgonneModel(cfg)
    return DecisionModel(trunk, DecisionHead(64, 16))


def main():
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(BASE, "tokenizer.json"))
    enc = Encoder(tok, max_len=2048)

    print("=== (1) encoder positions")
    q = Choice("Which team should handle this ticket?", ["billing", "technical", "other"])
    e = enc.encode("My card was charged twice for one order.", q)
    ids = e["input_ids"]
    check("answer read at the ':' of 'Answer:'", tok.decode([ids[e["ans_pos"]]]) == ":" and tok.decode(ids[e["ans_pos"] - 1:e["ans_pos"] + 1]) == "Answer:")
    check("each option read at its closing newline", all(ids[p] == NL for p in e["opt_pos"]) and len(e["opt_pos"]) == 3)
    check("option text precedes its read position", tok.decode(ids[e["opt_pos"][0] - 3:e["opt_pos"][0]]).strip().endswith("billing"))
    s = enc.encode("x " * 5000, q)
    check("an over-long state is clipped to fit max_len, head and tail kept", len(s["input_ids"]) <= 2048 and s["truncated"])
    sc = enc.encode("", Score("How angry?", {2: "Very angry", 0: "Calm", 1: "Frustrated"}))
    check("Score legend laid out in ascending level order", "(0) Calm" in tok.decode(sc["input_ids"]) and
          tok.decode(sc["input_ids"]).index("(0) Calm") < tok.decode(sc["input_ids"]).index("(2) Very angry"))

    print("=== (2) right padding is exact")
    m = tiny().eval()
    a = enc.encode("short state", Noul("The sky is green."))
    b = enc.encode("a much longer state " * 30, Choice("Pick one", ["red", "green", "blue", "cyan"]))
    def batch(exs):
        t = max(len(x["input_ids"]) for x in exs)
        k = max(len(x["opt_pos"]) for x in exs)
        ids = torch.full((len(exs), t), 151643)
        opt = torch.zeros(len(exs), k, dtype=torch.long)
        mask = torch.zeros(len(exs), k, dtype=torch.bool)
        for j, x in enumerate(exs):
            ids[j, :len(x["input_ids"])] = torch.tensor(x["input_ids"])
            opt[j, :len(x["opt_pos"])] = torch.tensor(x["opt_pos"])
            mask[j, :len(x["opt_pos"])] = True
        ans = torch.tensor([x["ans_pos"] for x in exs])
        return ids, opt, mask, ans
    with torch.no_grad():
        alone = m(*batch([a]))[0, :2]
        both = m(*batch([a, b]))[0, :2]
    check("logits of a short example unchanged when batched with a longer one", torch.allclose(alone, both, atol=1e-5),
          f"max diff {float((alone - both).abs().max()):.2e}")

    print("=== (3) KV cache: state read once, question block on top")
    with torch.no_grad():
        full = m.hidden(torch.tensor([b["input_ids"]]))
        split = b["n_state"] + 1                          # state + "\n\n"
        _, cache = m.hidden(torch.tensor([b["input_ids"][:split]]), use_cache=True)
        tail, _ = m.hidden(torch.tensor([b["input_ids"][split:]]), past_key_values=cache, use_cache=True)
    check("cached question-block hidden states == one full pass", torch.allclose(full[0, split:], tail[0], atol=1e-4),
          f"max diff {float((full[0, split:] - tail[0]).abs().max()):.2e}")

    print("=== (4) the head learns a toy task")
    m = tiny().train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    colors = ["red", "green", "blue", "yellow"]
    exs = [enc.encode(f"The ball is {c}.", Choice("What color is the ball?", colors)) for c in colors]
    ids, optp, mask, ans = batch(exs)
    labels = torch.arange(4)
    first = None
    for it in range(60):
        logits = m(ids, optp, mask, ans)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    check("loss starts near ln(4) and falls below 0.2", abs(first - math.log(4)) < 0.2 and loss.item() < 0.2,
          f"{first:.3f} -> {loss.item():.4f}")

    print("=== (5) save -> load")
    d = tempfile.mkdtemp()
    try:
        out = os.path.join(d, "final")
        m.eval()
        m.save(out, extra_config={"temperatures": {"choice": 1.0}}, trunk_dtype=torch.float32, tokenizer_dir=BASE)
        m2 = DecisionModel.load(out, dtype=torch.float32).eval()
        with torch.no_grad():
            l1, l2 = m(ids, optp, mask, ans), m2(ids, optp, mask, ans)
        check("reloaded model gives identical logits", torch.equal(l1, l2), f"max diff {float((l1 - l2).abs().max()):.2e}")
        check("embedding and lm_head still tied after reload", m2.trunk.lm_head.weight.data_ptr() == m2.trunk.embed_tokens.weight.data_ptr())
        check("tokenizer files travel with the model", os.path.exists(os.path.join(out, "tokenizer.json")))

        print("=== (6) system_one output")
        from decision.api import ArgonneDecision
        dm = ArgonneDecision(out, device="cpu")
        r = dm.system_one("I was charged twice and I'm really annoyed. Please refund me.", {
            "dept": Choice("Which team should handle this ticket?", ["billing", "technical", "other"]),
            "refund": Noul("The customer is asking for a refund."),
            "anger": Score("How angry is the customer?", {0: "Calm", 1: "Frustrated but civil", 2: "Very angry"}),
            "http": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]},
        })
        a = r["answers"]
        check("top-level keys model / answers / usage", set(r) >= {"model", "answers", "usage"} and
              set(r["usage"]) >= {"input_tokens", "output_tokens"})
        check("choice answer: a listed option, probabilities over exactly the options, confidence in [0, 1]",
              a["dept"]["choice"] in ["billing", "technical", "other"] and set(a["dept"]["probabilities"]) == {"billing", "technical", "other"}
              and abs(sum(a["dept"]["probabilities"].values()) - 1) < 1e-3 and 0 <= a["dept"]["confidence"] <= 1)
        check("noul answer: a probability", a["refund"]["type"] == "noul" and 0 <= a["refund"]["noul"] <= 1)
        sc = a["anger"]
        exp = sum(float(k) * v for k, v in sc["probabilities"].items())
        check("score answer: expected level over the legend, legend echoed", abs(sc["score"] - exp) < 2e-3 and
              set(sc["legend"]) == {"0", "1", "2"} and 0 <= sc["score"] <= 2)
        check("dict-form questions (HTTP/CLI) work", a["http"]["choice"] in ("a", "b"))
        # same answers whether the state is shared (cache) or re-read per question
        solo = dm.system_one("I was charged twice and I'm really annoyed. Please refund me.",
                             {"refund": Noul("The customer is asking for a refund.")})
        check("shared-state answers == one-question-per-call answers", solo["answers"]["refund"]["noul"] == a["refund"]["noul"])
        big = dm.system_one("x", {"q": Choice("Pick", [f"option {i}" for i in range(255)])})
        check("255 options accepted", len(big["answers"]["q"]["probabilities"]) == 255)
        try:
            dm.system_one("x", {"q": Choice("Pick", [f"o{i}" for i in range(256)])})
            check("256 options rejected", False)
        except ValueError:
            check("256 options rejected", True)
        print(json.dumps(r, indent=1)[:900])
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
