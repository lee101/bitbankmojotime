"""Quality dial on validation rows only: how much each extra unit of inference compute buys, and whether a post-hoc quantile calibration
helps. Levels are increasing compute (forwards per row); every level is scored on the same rows.

Calibration (per task x horizon, in benchmark scale units u = q / scale): u'_j = m + a (u_j - m) + b_j, m = the median knot, a on a grid,
b_j the closed-form pinball-optimal shift (the tau_j-quantile of the residual), knots re-sorted. It is fitted on the first half of each
fold's validation window (chronological) and scored on the second half, so every number in the table is out of sample.

  python -m train.dial            -> runs/dial/dial.json, printed table
"""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from bmtbench.metrics import Boot, ci
from train import search as S
from train.model2 import BMT2
from train.model import QLEV

TAU = np.array(QLEV)
W4 = os.path.join(D.RUNS, "wave4", "ckpt")
LEVELS = [("single", ["ohlc+cal@t-ens4-s0"], False), ("single-sym", ["ohlc+cal@t-ens4-s0"], True),
          ("ens4", [f"ohlc+cal@t-ens4-s{i}" for i in range(4)], False), ("ens4-sym", [f"ohlc+cal@t-ens4-s{i}" for i in range(4)], True),
          ("ens8-sym", [f"ohlc+cal@t-ens8-s{i}" for i in range(8)], True)]
EXTRA = [os.path.join(D.RUNS, "search", "ckpt", "ohlc+cal")]           # size-s seed 0 of the input search (same training setup)


def load(path, fi):
    c = torch.load(f"{path}_{D.FOLDS[fi][0]}.pt", map_location="cuda", weights_only=False)
    m = BMT2(horizons=S.HZ, **c["cfg"]).cuda()
    m.load_state_dict(c["sd"])
    return m


def pin(u, y):
    d = y[:, None] - u
    return np.maximum(TAU * d, (TAU - 1) * d).mean(1)


def fit(u, y, grid=np.linspace(0.85, 1.15, 31)):
    best = None
    for a in grid:
        v = u[:, 10:11] + a * (u - u[:, 10:11])
        b = np.array([np.quantile(y - v[:, j], TAU[j]) for j in range(21)])
        l = pin(np.sort(v + b, 1), y).mean()
        if best is None or l < best[0]:
            best = (l, a, b)
    return best[1], best[2]


def apply(u, a, b):
    return np.sort(u[:, 10:11] + a * (u - u[:, 10:11]) + b, 1)


if __name__ == "__main__":
    R = S.Runner("cuda", vram_gb=2.5)
    out_dir = os.path.join(D.RUNS, "dial")
    os.makedirs(out_dir, exist_ok=True)
    levels = LEVELS + [("ens8-sym+s", [os.path.join(W4, f"ohlc+cal@t-ens8-s{i}") for i in range(8)] + EXTRA, True)]
    rows = {}                                                         # (task, h) -> dict(level -> per-row loss on the B halves)
    meta = {}
    for task, k in (("hourly", 1), ("daily", 24)):
        v = R.val[task]
        for fi in range(4):
            i = np.flatnonzero(v["fold"] == fi)
            i = i[np.argsort(v["tkey"][i], kind="stable")]
            half = v["tkey"][i] < np.median(v["tkey"][i])
            cache = {}
            for name, mem, sym in levels:
                qs = []
                for mname in mem:
                    path = mname if os.sep in mname else os.path.join(W4, mname)
                    key = (path, sym)
                    if key not in cache:
                        m = load(path, fi)
                        cache[key] = R.predict(m, v["a"][i], v["t"][i], k, sym=sym, fold=fi).cpu().numpy()
                        del m
                    qs.append(cache[key])
                q = np.mean(qs, 0)
                for n, h in enumerate(D.TASKS[task]["horizons"]):
                    u = q[:, S.HZ.index(h)] / v["scale"][i][:, None]
                    y = v["y"][i][:, n] / v["scale"][i]
                    a, b = fit(u[half], y[half])
                    d = rows.setdefault((task, h), {})
                    for lv, uu in ((name, u), (name + "+cal", apply(u, a, b))):
                        d.setdefault(lv, []).append(pin(uu[~half], y[~half]))
                    d.setdefault("_asset", {}).setdefault(name, None)
                    meta.setdefault((task, h), {}).setdefault("asset", {}).setdefault(fi, v["asset"][i][~half])
                    meta[(task, h)].setdefault("blk", {}).setdefault(fi, v["blk"][i][~half])
                    meta[(task, h)].setdefault("cal", {}).setdefault(name, []).append([float(a), float(np.abs(b).max())])
            torch.cuda.empty_cache()
        print(task, "predicted", flush=True)
    names = [n for n, _, _ in levels]
    cost = {n: len(m) * (2 if s else 1) for n, m, s in levels}
    table, res = [], {}
    for th, d in rows.items():
        asset = np.concatenate([meta[th]["asset"][f] for f in range(4)])
        blk = np.concatenate([meta[th]["blk"][f] for f in range(4)])
        boot = Boot(asset, blk)
        base = np.concatenate(d["single"]).astype(np.float64)
        for lv in [x for n in names for x in (n, n + "+cal")]:
            cur = np.concatenate(d[lv]).astype(np.float64)
            res[f"{th[0]}|h{th[1]}|{lv}"] = dict(spl=float(np.mean([cur[asset == x].mean() for x in np.unique(asset)])),
                                                 vs_single=ci(1 - boot.asset_mean(cur) / boot.asset_mean(base)))
    json.dump(dict(res=res, cost=cost, cal={f"{k[0]}|h{k[1]}": v["cal"] for k, v in meta.items()}), open(os.path.join(out_dir, "dial.json"), "w"), indent=1)
    hdr = "| level | forwards / row | " + " | ".join(f"{t} h{h}" for t, h in rows) + " |"
    print(hdr)
    for lv in [x for n in names for x in (n, n + "+cal")]:
        cells = [f"{100 * res[f'{t}|h{h}|{lv}']['vs_single'][0]:+.3f} [{100 * res[f'{t}|h{h}|{lv}']['vs_single'][1]:+.3f}, {100 * res[f'{t}|h{h}|{lv}']['vs_single'][2]:+.3f}]" for t, h in rows]
        print(f"| {lv} | {cost[lv.replace('+cal', '')]} | " + " | ".join(cells) + " |")
