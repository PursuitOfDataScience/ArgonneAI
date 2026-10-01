"""Jev-style decision API for the Argonne decision model.

    from decision.api import ArgonneDecision, Choice, Score, Noul
    m = ArgonneDecision(snapshot_download("PursuitOfDataScience/argonne-4.5-decision"))
    m.system_one(state=ticket, questions={
        "dept":   Choice(instructions="Which team should handle this ticket?", criteria=["billing", "technical", "other"]),
        "refund": Noul(instructions="The customer is asking for a refund."),
        "anger":  Score(instructions="How angry is the customer?", legend={0: "Calm", 1: "Frustrated but civil", 2: "Very angry"}),
    })
    -> {"model": "argonne-decision-...", "answers": {
          "dept":   {"type": "choice", "choice": "billing", "confidence": 0.93, "probabilities": {...}},
          "refund": {"type": "noul", "noul": 0.97},
          "anger":  {"type": "score", "score": 1.37, "confidence": 0.41, "legend": {...}, "probabilities": {...}}},
        "usage": {"input_tokens": 411, "output_tokens": 68}}

The state is read ONCE (a KV cache), then each question block runs on top of it, so ten questions about one
document cost one pass over the document. Nothing is generated: every answer is a softmax over the options the
caller supplied, divided by a per-type temperature fitted on held-out data (decision/calibrate.py).
`confidence` = 1 - H(p)/ln(K), the normalised certainty of the distribution (1.0 when all mass is on one option).

CLI:   python decision/api.py --model DIR --state "..." --questions '{"q": {"type": "noul", "instructions": "..."}}'
HTTP:  python decision/api.py --model DIR --serve 8008     (POST /v1/system_one {"state": ..., "questions": {...}})
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision.modeling import DecisionModel  # noqa: E402
from decision.schema import Choice, Encoder, MAX_LEN, NLNL, Noul, Score, question_options  # noqa: E402,F401


def _confidence(p):
    k = len(p)
    if k < 2:
        return 1.0
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, 1.0 - h / math.log(k))


def _as_question(q):
    """Accept Choice/Score/Noul objects or plain dicts {"type": ..., ...} (the HTTP/CLI form)."""
    if isinstance(q, (Choice, Score, Noul)):
        return q
    t = q.get("type")
    if t == "choice":
        return Choice(q["instructions"], list(q["criteria"]))
    if t == "score":
        return Score(q["instructions"], {float(k): v for k, v in q["legend"].items()})
    if t == "noul":
        return Noul(q["instructions"])
    raise ValueError(f"unknown question type {t!r}")


class ArgonneDecision:
    def __init__(self, model_dir, device=None, dtype=torch.bfloat16, name=None):
        from transformers import PreTrainedTokenizerFast
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            dtype = torch.float32
        self.model = DecisionModel.load(model_dir, dtype=dtype).to(self.device).eval()
        cfg = self.model.decision_config
        self.temps = {k: float(v) for k, v in (cfg.get("temperatures") or {}).items()}
        self.name = name or cfg.get("name") or f"argonne-decision-{os.path.basename(os.path.normpath(model_dir))}"
        self.tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
        self.max_len = int(cfg.get("max_len") or MAX_LEN)
        self.enc = Encoder(self.tok, self.max_len)

    @torch.no_grad()
    def system_one(self, state, questions, model=None):
        qs = {name: _as_question(q) for name, q in questions.items()}
        blocks = {name: self.enc.question_block(q) for name, q in qs.items()}
        longest = max((len(b[0]) for b in blocks.values()), default=0)
        state_ids = self.enc.enc(state.strip()) if state and state.strip() else []
        budget = self.max_len - longest - 1
        truncated = len(state_ids) > budget
        if truncated:
            marker = self.enc.enc(" [...] ")
            keep = max(budget - len(marker), 0)
            head = keep // 2
            state_ids = (state_ids[:head] + marker + state_ids[len(state_ids) - (keep - head):])[:budget] if keep else []
        prefix = state_ids + ([NLNL] if state_ids else [])
        cache = None
        if prefix:
            ids = torch.tensor([prefix], device=self.device)
            _, cache = self.model.hidden(ids, use_cache=True)
        answers, n_in = {}, len(prefix)
        for name, q in qs.items():
            block, opt_pos, ans = blocks[name]
            ids = torch.tensor([block], device=self.device)
            h, _ = self.model.hidden(ids, past_key_values=cache, use_cache=True)
            h = h[0]
            k = len(opt_pos)
            logits = self.model.head(h[ans][None], h[torch.tensor(opt_pos, device=self.device)][None],
                                     torch.ones(1, k, dtype=torch.bool, device=self.device))[0]
            p = torch.softmax(logits / self.temps.get(q.type, 1.0), -1).tolist()
            n_in += len(block)
            opts = question_options(q)
            if q.type == "noul":
                answers[name] = {"type": "noul", "noul": round(p[0], 4)}
            elif q.type == "choice":
                best = max(range(k), key=lambda i: p[i])
                answers[name] = {"type": "choice", "choice": opts[best], "confidence": round(_confidence(p), 4),
                                 "probabilities": {o: round(x, 4) for o, x in zip(opts, p)}}
            else:
                levels = sorted(q.legend, key=float)
                answers[name] = {"type": "score", "score": round(sum(float(lv) * x for lv, x in zip(levels, p)), 4),
                                 "confidence": round(_confidence(p), 4),
                                 "legend": {_key(lv): q.legend[lv] for lv in levels},
                                 "probabilities": {_key(lv): round(x, 4) for lv, x in zip(levels, p)}}
        out_tokens = len(self.tok.encode(json.dumps(answers, separators=(",", ":")), add_special_tokens=False))
        usage = {"input_tokens": n_in, "output_tokens": out_tokens}
        if truncated:
            usage["truncated_state"] = True
        return {"model": model or self.name, "answers": answers, "usage": usage}


def _key(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def _serve(dm, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/system_one":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                t0 = time.time()
                with lock:
                    out = dm.system_one(body.get("state", ""), body["questions"], body.get("model"))
                out["usage"]["latency_ms"] = round(1000 * (time.time() - t0), 1)
                data, code = json.dumps(out).encode(), 200
            except Exception as e:
                data, code = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), 400
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"serving {dm.name} on :{port} (POST /v1/system_one)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--state", default="")
    ap.add_argument("--questions", default=None, help="JSON object {name: {type, instructions, criteria|legend}}")
    ap.add_argument("--serve", type=int, default=0)
    args = ap.parse_args()
    dm = ArgonneDecision(args.model)
    if args.serve:
        _serve(dm, args.serve)
    else:
        print(json.dumps(dm.system_one(args.state, json.loads(args.questions)), indent=2))


if __name__ == "__main__":
    main()
