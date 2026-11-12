"""Does the forecast have tradable value on the unseen folds, after costs?  python -m bmtbench.tradable [--models a,b,c]

Three tests per (task, horizon), all on the benchmark's test rows with the protocol's calendar-block bootstrap:
  IC      mean cross-sectional Spearman correlation between the forecast median and the realised return (origins with >= 10 symbols).
  L/S     at each origin: long the top fifth of symbols by forecast median, short the bottom fifth, equal weight, hold h bars.
          Gross = mean(long) - mean(short) in bp of one leg's notional; each leg pays a round trip, so net = gross - 2 x cost.
  Rule    long-only: buy a symbol at the origin when its forecast median exceeds the round-trip cost, hold h bars, sell.
          Net per trade = realised return - cost. The random-walk alternative never trades (0).
Cost scenarios (round trip, assumptions, not measured fills): 0; 4 bp (maker-tier fees, no spread paid); 20 bp (taker: 2 x 7.5 bp
fee + 2 x 2.5 bp half-spread / slippage). No market impact, no funding, shorting assumed possible for the L/S test.
"""
import argparse, json, os
import numpy as np
from . import data as D
from .metrics import Boot, ci

COSTS = (0.0, 4e-4, 20e-4)
DEFAULT = ["bmt-t", "bmt-s-sym", "timestral-official", "timestral-official-2048", "timestral-max", "timestral-fast", "chronos-2-ret", "chronos-2"]


def rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def by_time(med, y, tkey, min_n=10):
    """per origin: (time, IC, long-short spread). Origins with fewer than min_n symbols or a constant median are skipped."""
    o = np.argsort(tkey, kind="stable")
    med, y, tk = med[o], y[o], tkey[o]
    cuts = np.flatnonzero(np.diff(tk)) + 1
    T, IC, LS = [], [], []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(tk)]):
        if b - a < min_n:
            continue
        m, v = med[a:b], y[a:b]
        if m.std() == 0:
            continue
        k = max(1, (b - a) // 5)
        i = np.argsort(m, kind="stable")
        T.append(tk[a]); LS.append(v[i[-k:]].mean() - v[i[:k]].mean())
        IC.append(np.corrcoef(rank(m), rank(v))[0, 1] if v.std() > 0 else 0.0)
    return np.array(T), np.array(IC), np.array(LS)


def run(models):
    out = {}
    for task in D.TASKS:
        rows, cfg = D.load_rows(task), D.TASKS[task]
        for m in models:
            if not os.path.exists(D.pred_path(task, m)):
                continue
            q = np.load(D.pred_path(task, m), mmap_mode="r")
            for j, h in enumerate(cfg["horizons"]):
                if j >= q.shape[1]:
                    continue
                med, y = np.asarray(q[:, j, 10], np.float64), rows["y"][:, j].astype(np.float64)
                T, IC, LS = by_time(med, y, rows["t_abs"])
                r = {"origins": int(len(T))}
                if len(T):
                    b = Boot(np.zeros(len(T), int), T // cfg["block"])
                    r["ic"] = ci(b.pooled(IC))
                    r["ls_gross_bp"] = ci(b.pooled(LS) * 1e4)
                    for c in COSTS[1:]:
                        r[f"ls_net_bp@{c * 1e4:.0f}"] = ci(b.pooled(LS - 2 * c) * 1e4)
                for c in COSTS:
                    k = med > c
                    key = f"rule@{c * 1e4:.0f}"
                    if k.sum() >= 30:
                        bb = Boot(np.zeros(int(k.sum()), int), rows["t_abs"][k] // cfg["block"])
                        r[key] = dict(trades=int(k.sum()), share=float(k.mean()), net_bp=ci(bb.pooled(y[k] - c) * 1e4), hit=float((y[k] > c).mean()))
                    else:
                        r[key] = dict(trades=int(k.sum()), share=float(k.mean()), net_bp=None, hit=None)
                r["all_long_bp"] = float(y.mean() * 1e4)
                out[f"{task}|h{h}|{m}"] = r
    return out


def f3(x, d=2):
    return "n/a" if x is None else f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]"


def table(out):
    md = []
    for task in D.TASKS:
        for h in D.TASKS[task]["horizons"]:
            ks = [k for k in out if k.startswith(f"{task}|h{h}|")]
            if not ks:
                continue
            md += [f"\n### {task}, horizon {h} (mean return of holding every row: {out[ks[0]]['all_long_bp']:+.2f} bp)\n",
                   "| model | rank IC | L/S gross bp | L/S net bp at 4 bp | L/S net bp at 20 bp | rule at 0 cost: trades %, net bp | rule at 4 bp: trades %, net bp | rule at 20 bp: trades %, net bp |",
                   "|---|---|---|---|---|---|---|---|"]
            for k in ks:
                r = out[k]
                rule = lambda c: f"{100 * r[f'rule@{c}']['share']:.1f} %, {f3(r[f'rule@{c}']['net_bp'])}"
                md.append(f"| {k.split('|')[2]} | {f3(r.get('ic'), 3)} | {f3(r.get('ls_gross_bp'))} | {f3(r.get('ls_net_bp@4'))} | {f3(r.get('ls_net_bp@20'))} | {rule(0)} | {rule(4)} | {rule(20)} |")
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT))
    a = ap.parse_args()
    out = run(a.models.split(","))
    json.dump(out, open(os.path.join(D.RUNS, "tradable.json"), "w"), indent=1)
    open(os.path.join(D.RUNS, "tradable.md"), "w").write(table(out))
    print(table(out)[:5000])
