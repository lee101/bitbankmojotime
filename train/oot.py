"""Out-of-time sanity read (not a test fold): fold-F4 checkpoints on origins after the archive the benchmark was built from.
Extends the hourly cache with a later kline CSV dump (ts, open, high, low, close, qv), builds hourly rows on [start, end) with the
test-row rules (stride 5, real bars, history, fill share), scores sPL h 1 / 6 / 24 per group, paired 1-day-block bootstrap.

  BMT_DATA=<ext cache> python -m train.oot --extend <csv dir> --start 2026-11-12 --end 2026-11-24
"""
import argparse, datetime, json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from bmtbench.metrics import Boot, ci
from train.bank import Bank, flip
from train.model import QLEV
from train.model2 import BMT2
from train.wave4 import Ens

SRC = "/nvme0n1-disk/code/bitbankmojotime/data/cache/hourly"


def extend(csv_dir, out_dir, syms):
    os.makedirs(out_dir, exist_ok=True)
    rep = {}
    for x in syms:
        z = dict(np.load(os.path.join(SRC, x + ".npz")))
        f = os.path.join(csv_dir, x + ".csv")
        if os.path.exists(f):
            c = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
            t = (c[:, 0] // 3600).astype(np.int64)
            ov = np.intersect1d(t, z["t"])
            if len(ov):
                i, j = np.searchsorted(z["t"], ov), np.searchsorted(t, ov)
                rep[x] = dict(overlap=len(ov), max_close_rel=float(np.abs(c[j, 4] / z["c"][i] - 1).max()), new=int((t > z["t"][-1]).sum()))
            k = t > z["t"][-1]
            if k.any():
                n = int(k.sum())
                for key, col in (("o", 1), ("h", 2), ("l", 3), ("c", 4)):
                    z[key] = np.concatenate([z[key], c[k, col]])
                z["qv"] = np.concatenate([z["qv"], c[k, 5].astype(np.float32)])
                z["n"] = np.concatenate([z["n"], np.zeros(n, np.float32)])
                z["tb"] = np.concatenate([z["tb"], np.full(n, np.nan, np.float32)])
                z["t"] = np.concatenate([z["t"], t[k]])
        np.savez(os.path.join(out_dir, x + ".npz"), **z)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", default="/sdb-disk/tradingbots/binance/univ/h1")
    ap.add_argument("--start", default="2026-11-12")
    ap.add_argument("--end", default="2026-11-24")
    ap.add_argument("--ckpt", default="/nvme0n1-disk/code/bitbankmojotime/runs/search/ckpt")
    ap.add_argument("--models", default="bmt2-ohlc+cal=ohlc+cal,bmt2-ohlc+cal@t=ohlc+cal@t")
    ap.add_argument("--kd-dir", default="/nvme0n1-disk/code/bitbankmojotime/runs/wave5/ckpt")
    ap.add_argument("--out", default=os.path.join(D.RUNS, "oot"))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    u = D.universe()
    rep = extend(a.extend, os.path.join(D.CACHE, "hourly"), u["eval"])
    P = D.load_hourly(u["eval"])
    eh = lambda s: int(datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp() // 3600)
    lo, hi = eh(a.start), eh(a.end)
    r = D.build_rows(P, "hourly", [("OOT", lo, hi)])
    assert (r["t"] + P.t0 >= lo).all() and (r["t"] + P.t0 + 24 < hi).all()
    bank = Bank(u["eval"], [x for x in u["seen"] if x in u["eval"]], "cpu")
    ai = np.array([bank.ix[s] for s in P.syms])[r["a"]]
    tb = r["t"] + P.t0 - bank.t0
    lev = torch.tensor(QLEV)
    H = (1, 6, 24)
    HZ = (1, 5, 6, 24)
    preds = {}
    for spec in a.models.split(","):
        nm, mem = spec.split("=")
        ms = []
        for mn in mem.split(":"):
            d = a.kd_dir if "-s" in mn and "@t-" in mn else a.ckpt
            c = torch.load(os.path.join(d, f"{mn}_F4.pt"), map_location="cpu", weights_only=False)
            assert c["limit_abs"] <= D.FOLDS[3][1] - 168
            m = BMT2(horizons=HZ, **c["cfg"]); m.load_state_dict(c["sd"]); m.eval(); ms.append(m)
        m = Ens(ms) if len(ms) > 1 else ms[0]
        for sym in (False, True):
            out = []
            with torch.no_grad():
                for i in range(0, len(tb), 2048):
                    b = bank.gather(torch.as_tensor(ai[i:i + 2048]), torch.as_tensor(tb[i:i + 2048]), 1, fold=3)
                    q = m.quantiles(b)
                    if sym:
                        q = 0.5 * (q - m.quantiles(flip(b)).flip(-1))
                    out.append(q[:, [HZ.index(h) for h in H]].float())
            preds[nm + ("-sym" if sym else "")] = torch.sort(torch.cat(out), -1).values
    cr = D.ctx_ret(P, r["a"], r["t"])
    sd = torch.tensor(cr.std(1), dtype=torch.float32)
    z = torch.distributions.Normal(0, 1).icdf(lev)
    preds["naive"] = sd[:, None, None] * torch.tensor([math.sqrt(h) for h in H])[None, :, None] * z[None, None, :]
    y = torch.tensor(r["y"])
    sc = r["scale"].astype(np.float64)
    pl = {}
    for k, q in preds.items():
        d = y[:, :, None] - q
        pl[k] = (torch.maximum(lev * d, (lev - 1) * d).mean(-1).numpy().astype(np.float64) / sc[:, None])
    unseen = np.array([D.is_unseen(s) for s in P.syms])[r["a"]]
    blk = (r["t"] + P.t0) // 24
    res = dict(rows=int(len(tb)), assets=int(len(np.unique(r["a"]))), unseen_assets=int(len(np.unique(r["a"][unseen]))), window=[a.start, a.end],
               extend_check={k: v for k, v in list(rep.items())[:3]}, max_close_rel_all=max(v["max_close_rel"] for v in rep.values()),
               new_bars_min=min(v["new"] for v in rep.values()), syms_extended=len(rep), table={})
    ref = "bmt2-ohlc+cal-sym"
    for g, msk in (("all", np.ones(len(tb), bool)), ("seen", ~unseen), ("unseen", unseen)):
        bt = Boot(r["a"][msk], blk[msk])
        for j, h in enumerate(H):
            base = bt.asset_mean(pl[ref][msk, j])
            for k in pl:
                v = bt.asset_mean(pl[k][msk, j])
                res["table"][f"{g}|h{h}|{k}"] = dict(sPL=ci(v), skill_vs_naive_pct=ci(100 * (1 - v / bt.asset_mean(pl["naive"][msk, j]))),
                                                     d_vs_promoted=ci(v - base))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "oot.json"), "w"), indent=1)
    for k, v in res["table"].items():
        print(k, " ".join(f"{n}={x[0]:.4f}[{x[1]:.4f},{x[2]:.4f}]" for n, x in v.items()))
    print(json.dumps({k: v for k, v in res.items() if k != "table"}))
