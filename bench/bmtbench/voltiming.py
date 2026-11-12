"""Volatility-timing value of the quantile forecast on the unseen folds (daily task, horizon 1).  python -m bmtbench.voltiming

Each day and symbol: exposure w = clip(target / sigma_hat, 0, 3), sigma_hat = (q84 - q16) / 2 of the model's one-day forecast
(interpolated from the 21 levels), target = the symbol's training-period daily mean |r| x 1.2533 (so w is about 1 on average and uses
no test information). Portfolio = equal weight over the symbols with a row that day, rebalanced daily, long only.
Cost = half the round-trip cost on every unit of |change in w|. Reported: annualised Sharpe (365 days) with the 28-day block
bootstrap, and its paired difference to constant exposure (w = 1) and to GARCH sizing.
"""
import argparse, json, os
import numpy as np
from . import data as D
from .metrics import Boot

MODELS = ["constant", "naive", "ewma", "garch", "timestral-official", "timestral-max", "chronos-2-ret", "bmt-t", "bmt-s-sym"]


def sig(q):
    lev = D.QLEV
    f = lambda p: q[:, 3] + (q[:, 4] - q[:, 3]) * (p - lev[3]) / (lev[4] - lev[3]) if p < 0.5 else q[:, 16] + (q[:, 17] - q[:, 16]) * (p - lev[16]) / (lev[17] - lev[16])
    return np.maximum((f(0.8413) - f(0.1587)) / 2, 1e-8)


def series(rows, w, cost):
    """daily portfolio return net of turnover cost: [days], day keys."""
    o = np.lexsort((rows["t_abs"], rows["a"]))
    a, t, y, w = rows["a"][o], rows["t_abs"][o], rows["y"][o, 0].astype(np.float64), w[o]
    prev = np.where((np.r_[-1, a[:-1]] == a) & (np.r_[-9, t[:-1]] == t - 1), np.r_[0, w[:-1]], w)      # first day of a run: no entry cost charged
    pnl = w * y - np.abs(w - prev) * cost / 2
    days, inv = np.unique(t, return_inverse=True)
    return days, np.bincount(inv, pnl) / np.bincount(inv)


def run(models, costs=(0.0, 4e-4, 20e-4)):
    rows = D.load_rows("daily")
    out, ser = {}, {}
    for c in costs:
        for m in models:
            if m == "constant":
                w = np.ones(len(rows["t"]))
            elif os.path.exists(D.pred_path("daily", m)):
                w = np.clip(rows["scale"] * 1.2533 / sig(np.asarray(np.load(D.pred_path("daily", m), mmap_mode="r")[:, 0], np.float64)), 0, 3)
            else:
                continue
            days, r = series(rows, w, c)
            ser[m] = r
            out.setdefault(m, {})["mean_w"] = float(w.mean())
        b = Boot(np.zeros(len(days), int), days // 28)
        sh = {}
        for m, r in ser.items():
            mu, m2 = b.pooled(r), b.pooled(r * r)
            sh[m] = mu / np.sqrt(np.maximum(m2 - mu * mu, 1e-18)) * np.sqrt(365)
        for m in ser:
            q = lambda v: [float(v[0])] + [float(x) for x in np.percentile(v[1:], [2.5, 97.5])]
            out[m][f"sharpe@{c * 1e4:.0f}"] = q(sh[m])
            out[m][f"vs_constant@{c * 1e4:.0f}"] = q(sh[m] - sh["constant"])
            if "garch" in sh:
                out[m][f"vs_garch@{c * 1e4:.0f}"] = q(sh[m] - sh["garch"])
    return out


def table(out):
    f = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
    md = ["| sizing by | mean exposure | Sharpe, no cost | vs constant | vs GARCH sizing | Sharpe at 4 bp | vs constant | Sharpe at 20 bp | vs constant |", "|---|---|---|---|---|---|---|---|---|"]
    for m, r in out.items():
        md.append(f"| {m} | {r['mean_w']:.2f} | {f(r['sharpe@0'])} | {f(r['vs_constant@0'])} | {f(r['vs_garch@0']) if 'vs_garch@0' in r else 'n/a'} | {f(r['sharpe@4'])} | {f(r['vs_constant@4'])} | "
                  f"{f(r['sharpe@20'])} | {f(r['vs_constant@20'])} |")
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    out = run(a.models.split(","))
    json.dump(out, open(os.path.join(D.RUNS, "voltiming.json"), "w"), indent=1)
    open(os.path.join(D.RUNS, "voltiming.md"), "w").write(table(out))
    print(table(out))
