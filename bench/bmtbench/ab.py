"""Offline A/B of two forecasters under ONE decision rule, PnL with costs on the unseen folds (docs/INTEGRATION.md).

  python -m bmtbench.ab --a bmt2-ohlc+cal-sym --b timestral-official [--others garch,...] [--topfrac 0.125]

Rule (a stand-in with the shape of a daily rotation; a consumer's own rule plugs in through `decide`): every UTC day rank the symbols
by score = forecast median / forecast sigma (daily task, horizon 1), hold the top `topfrac` of them equal-weight for one day, long only.
PnL per day = mean return of the held names - mean return of all names (the no-rotation alternative) - cost x turnover, where
turnover = share of the book replaced since the previous day and cost = round-trip cost of a replaced name (0 / 4 / 20 bp scenarios).
Same rows, same rule, same calendar-block bootstrap for both forecasters; the difference A - B is paired.
"""
import argparse, json, os
import numpy as np
from . import data as D
from .metrics import Boot
from .voltiming import sig

COSTS = (0.0, 4e-4, 20e-4)


def decide(score, k):
    """the decision rule: indices of the names to hold."""
    return np.argsort(-score, kind="stable")[:k]


def daily_pnl(rows, q, topfrac):
    """-> days, gross excess return per day, turnover per day (nan-free); None if the forecast has no ranking information."""
    score = np.asarray(q[:, 0, 10], np.float64) / sig(np.asarray(q[:, 0], np.float64))
    o = np.lexsort((rows["a"], rows["t_abs"]))
    t, a, y, s = rows["t_abs"][o], rows["a"][o], rows["y"][o, 0].astype(np.float64), score[o]
    cuts = np.flatnonzero(np.diff(t)) + 1
    days, gross, turn, prev, prev_day = [], [], [], None, None
    for i, j in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if j - i < 8 or s[i:j].std() == 0:
            prev = None
            continue
        k = max(1, int(round((j - i) * topfrac)))
        held = set(a[i:j][decide(s[i:j], k)].tolist())
        days.append(t[i]); gross.append(np.mean([y[i:j][a[i:j] == x][0] for x in held]) - y[i:j].mean())
        turn.append(1.0 if prev is None or prev_day != t[i] - 1 else 1 - len(held & prev) / k)
        prev, prev_day = held, t[i]
    return (np.array(days), np.array(gross), np.array(turn)) if days else None


def run(models, topfrac):
    rows = D.load_rows("daily")
    res = {m: daily_pnl(rows, D.open_pred("daily", m), topfrac) for m in models if (m in D.models_of("daily"))}
    res = {m: r for m, r in res.items() if r is not None}
    days = sorted(set.intersection(*[set(r[0].tolist()) for r in res.values()]))
    idx = {m: np.searchsorted(r[0], days) for m, r in res.items()}
    b = Boot(np.zeros(len(days), int), np.array(days) // 28)
    out = {"days": len(days), "topfrac": topfrac, "models": {}}
    q = lambda v: [float(v[0])] + [float(x) for x in np.percentile(v[1:], [2.5, 97.5])]
    vec = {}
    for m, r in res.items():
        g, tn = r[1][idx[m]], r[2][idx[m]]
        out["models"][m] = {"turnover": float(tn.mean())}
        for c in COSTS:
            vec[(m, c)] = b.pooled(g - c * tn) * 1e4
            out["models"][m][f"net_bp@{c * 1e4:.0f}"] = q(vec[(m, c)])
    a0, b0 = models[0], models[1]
    if a0 in res and b0 in res:
        out["a_minus_b"] = {f"@{c * 1e4:.0f}": q(vec[(a0, c)] - vec[(b0, c)]) for c in COSTS}
    return out


def table(out, a, b):
    f = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
    md = [f"Daily rotation rule, top {out['topfrac']:.1%} of names by score, {out['days']} unseen days. Excess return over holding every name, bp per day:", "",
          "| forecaster | turnover / day | no cost | 4 bp | 20 bp |", "|---|---|---|---|---|"]
    for m, r in out["models"].items():
        md.append(f"| {m} | {r['turnover']:.2f} | {f(r['net_bp@0'])} | {f(r['net_bp@4'])} | {f(r['net_bp@20'])} |")
    if "a_minus_b" in out:
        d = out["a_minus_b"]
        md += ["", f"Paired difference `{a}` minus `{b}`, bp per day: no cost {f(d['@0'])}, 4 bp {f(d['@4'])}, 20 bp {f(d['@20'])}."]
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--others", default="")
    ap.add_argument("--topfrac", type=float, default=0.125)
    a = ap.parse_args()
    ms = [D.resolve(x) for x in [a.a, a.b] + [x for x in a.others.split(",") if x]]
    a.a, a.b = ms[0], ms[1]
    out = run(ms, a.topfrac)
    json.dump(out, open(os.path.join(D.RUNS, "ab.json"), "w"), indent=1)
    open(os.path.join(D.RUNS, "ab.md"), "w").write(table(out, a.a, a.b))
    print(table(out, a.a, a.b))
