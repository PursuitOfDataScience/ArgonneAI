"""Evaluate a decision model on packed splits: accuracy, log loss, Brier, ECE per question type and per source.

    torchrun --standalone --nproc_per_node=8 decision/evaluate.py --model DIR --data PACKED --splits val,test,ood \\
        --out eval.json

Writes eval.json (summary + per-source tables) and eval.npz (per-example logits, targets, types, sources for the
val and test splits) that decision/calibrate.py fits temperatures on. Confidence for ECE is the probability of
the predicted option (Jev reports ECE per question type the same way). Score also gets the mean absolute error of
the expected level against the gold level.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision.modeling import DecisionModel  # noqa: E402
from decision.train import Packed, ece, make_micros  # noqa: E402

TYPES = ["choice", "score", "noul"]


def metrics(logits_list, target, label, levels=None, temp=1.0):
    """logits_list: list of 1-D arrays (variable K). Returns dict of acc / nll / brier / ece (+ mae for score)."""
    n = len(logits_list)
    if n == 0:
        return {}
    acc, nll, brier, conf, ok, mae = 0.0, 0.0, 0.0, [], [], []
    for i, lg in enumerate(logits_list):
        z = lg / temp
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        t = target[i][:len(p)]
        pred = int(p.argmax())
        c = pred == label[i]
        acc += c
        nll += float(-(t[t > 0] * np.log(np.clip(p[t > 0], 1e-12, 1))).sum())
        brier += float(((p - t) ** 2).sum())
        conf.append(float(p.max()))
        ok.append(c)
        if levels is not None:
            lv = np.asarray(levels[i][:len(p)], dtype=float)
            mae.append(abs(float((p * lv).sum()) - lv[label[i]]))
    out = {"n": n, "acc": acc / n, "nll": nll / n, "brier": brier / n, "ece": ece(conf, ok)}
    if mae:
        out["mae"] = float(np.mean(mae))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="val,test,ood")
    ap.add_argument("--out", required=True)
    ap.add_argument("--micro_tokens", type=int, default=32768)
    ap.add_argument("--micro_max_ex", type=int, default=256)
    ap.add_argument("--levels_from", default=None, help="built/ dir with the JSONL, to read Score levels for MAE")
    args = ap.parse_args()

    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = DecisionModel.load(args.model, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval()

    result = {"model": args.model, "summary": {}, "per_source": {}}
    raw = {}
    for split in args.splits.split(","):
        d = os.path.join(args.data, split)
        if not os.path.exists(os.path.join(d, "index.npy")):
            continue
        data = Packed(d)
        mine = np.arange(rank, len(data), world, dtype=np.int64)
        micros = make_micros(data.lengths, mine, args.micro_tokens, args.micro_max_ex, np.random.default_rng(0))
        local_rows = []
        with torch.no_grad():
            for m in micros:
                b = data.collate(m, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    lg = model(b["input_ids"], b["opt_pos"], b["opt_mask"], b["ans_pos"]).float()
                lg = lg.cpu().numpy()
                k = data.index[m][:, 4]
                tg = b["target"].cpu().numpy()
                for j, ex in enumerate(m):
                    local_rows.append((int(ex), lg[j, :k[j]].tolist(), tg[j, :k[j]].tolist()))
        if dist.is_initialized():
            gathered = [None] * world
            dist.all_gather_object(gathered, local_rows)
            rows = [r for g in gathered for r in g]
        else:
            rows = local_rows
        if rank != 0:
            continue
        rows.sort(key=lambda r: r[0])
        idx = np.array([r[0] for r in rows])
        logits = [np.asarray(r[1], dtype=np.float64) for r in rows]
        target = [np.asarray(r[2], dtype=np.float64) for r in rows]
        meta = data.index[idx]
        label, typ, src = meta[:, 5], meta[:, 6], meta[:, 7]
        # Score levels: the tag of each option is its level; the packed data keeps levels implicit (0..K-1 order
        # of ascending levels), so read the real level values from the JSONL when given, else use 0..K-1
        levels = [np.arange(len(l)) for l in logits]
        if args.levels_from and os.path.exists(os.path.join(args.levels_from, f"{split}.jsonl")):
            lv_by_id = {}
            with open(os.path.join(args.levels_from, f"{split}.jsonl")) as f:
                for line in f:
                    if '"score"' in line:
                        r = json.loads(line)
                        if r["type"] == "score":
                            lv_by_id[r["id"]] = r["levels"]
            ids = data.meta["ids"]
            levels = [np.asarray(lv_by_id.get(ids[i], np.arange(len(logits[j])))) for j, i in enumerate(idx)]
        summary = {}
        for t, tn in enumerate(TYPES):
            sel = np.where(typ == t)[0]
            if len(sel):
                summary[tn] = metrics([logits[i] for i in sel], [target[i] for i in sel], label[sel],
                                      levels=[levels[i] for i in sel] if tn == "score" else None)
        sel_long = np.where(meta[:, 1] > 2048)[0]
        if len(sel_long):
            summary["long_inputs(>2048 tok)"] = metrics([logits[i] for i in sel_long], [target[i] for i in sel_long], label[sel_long])
        result["summary"][split] = summary
        per = {}
        for s, name in enumerate(data.meta["sources"]):
            sel = np.where(src == s)[0]
            if len(sel):
                t0 = TYPES[int(typ[sel[0]])]
                per[name] = dict(metrics([logits[i] for i in sel], [target[i] for i in sel], label[sel],
                                         levels=[levels[i] for i in sel] if t0 == "score" else None), type=t0)
        result["per_source"][split] = per
        if split in ("val", "test", "ood"):
            kmax = max(len(l) for l in logits)
            L = np.full((len(logits), kmax), -np.inf, dtype=np.float32)
            T = np.zeros((len(logits), kmax), dtype=np.float32)
            for j, (l, t) in enumerate(zip(logits, target)):
                L[j, :len(l)] = l
                T[j, :len(t)] = t
            raw[split] = {"logits": L, "target": T, "label": label, "type": typ, "source": src,
                          "ids": np.array([data.meta["ids"][i] for i in idx]),
                          "source_names": np.array(data.meta["sources"])}
        print(f"[eval] {split}: " + " | ".join(f"{k}: acc {v['acc']:.3f} ece {v['ece']:.3f} n {v['n']}"
                                              for k, v in summary.items()), flush=True)
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        npz = os.path.splitext(args.out)[0] + ".npz"
        np.savez_compressed(npz, **{f"{s}_{k}": v for s, d in raw.items() for k, v in d.items()})
        print(f"[eval] wrote {args.out} and {npz}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
