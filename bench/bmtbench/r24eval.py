"""PREREG-24H section 4: hourly h = 24 test table of the selected 24 h ranking arm.  python -m bmtbench.r24eval --cand <arm> [--inc inc]"""
import argparse, json, os
import numpy as np
from . import data as D
from .metrics import Boot, ci
from .tradable import by_time

REFS = ["bmt2-ohlc+cal", "bmt2-ohlc+cal-sym", "timestral-official", "timestral-official-2048"]


def series(rows, med, sel):
    T, IC, LS = by_time(med[sel].astype(np.float64), rows["y"][sel, 2].astype(np.float64), rows["t_abs"][sel])
    return dict(T=T, IC=IC, LS=LS)


def run(cand, inc):
    rows = D.load_rows("hourly")
    u = D.universe()
    unseen = np.isin(np.array(rows["syms"])[rows["a"]], u["unseen"])
    names = {f"{cand}-s{s}": D.resolve(f"bmt2-r24-{cand}-s{s}") for s in (0, 1)}
    names.update({f"{inc}-s{s}": D.resolve(f"bmt2-r24-{inc}-s{s}") for s in (0, 1)})
    names.update({r: D.resolve(r) for r in REFS})
    meds = {k: np.asarray(D.open_pred("hourly", v)[:, 2, 10]) for k, v in names.items()}
    out = dict(rows=names)
    for grp, sel in (("all", np.ones(len(unseen), bool)), ("seen", ~unseen), ("unseen", unseen)):
        S = {k: series(rows, m, sel) for k, m in meds.items()}
        T = S[f"{cand}-s0"]["T"]
        for k in S:
            keep = np.isin(S[k]["T"], T)
            S[k] = {x: v[keep] for x, v in S[k].items()}
        T = np.array(sorted(set.intersection(*[set(S[k]["T"].tolist()) for k in S])))
        for k in S:
            keep = np.isin(S[k]["T"], T)
            S[k] = {x: v[keep] for x, v in S[k].items()}
            assert np.array_equal(S[k]["T"], T)
        b = Boot(np.zeros(len(T), int), T // 168)
        avg = lambda a, x: 0.5 * (S[f"{a}-s0"][x] + S[f"{a}-s1"][x])
        arms = {cand: {x: avg(cand, x) for x in ("IC", "LS")}, inc: {x: avg(inc, x) for x in ("IC", "LS")}}
        arms.update({f"{cand}-s{s}": S[f"{cand}-s{s}"] for s in (0, 1)})
        arms.update({r: S[r] for r in REFS})
        g = dict(origins=int(len(T)), level={}, diff={})
        for k, v in arms.items():
            g["level"][k] = dict(ic=ci(b.pooled(v["IC"])), ls_gross=ci(b.pooled(v["LS"]) * 1e4), ls_net4=ci(b.pooled(v["LS"] - 8e-4) * 1e4),
                                 ls_net20=ci(b.pooled(v["LS"] - 40e-4) * 1e4))
        for k in [inc] + REFS:
            g["diff"][k] = dict(d_ic=ci(b.pooled(arms[cand]["IC"] - arms[k]["IC"])), d_ls=ci(b.pooled(arms[cand]["LS"] - arms[k]["LS"]) * 1e4))
        for s in (0, 1):
            k = f"{cand}-s{s}"
            g["diff"][f"{k} vs {inc}"] = dict(d_ic=ci(b.pooled(arms[k]["IC"] - arms[inc]["IC"])), d_ls=ci(b.pooled(arms[k]["LS"] - arms[inc]["LS"]) * 1e4))
        out[grp] = g
    return out


def table(o, cand, inc):
    f = lambda x, d=2: f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]"
    md = []
    for grp in ("all", "seen", "unseen"):
        g = o[grp]
        md += [f"\n**{grp}** ({g['origins']} origins, hourly h = 24, test folds F1-F4)\n", "| model | rank IC | L/S gross bp | L/S net at 4 bp | L/S net at 20 bp | d IC: `" + cand + "` minus row | d L/S bp: `" + cand + "` minus row |", "|---|---|---|---|---|---|---|"]
        for k, v in g["level"].items():
            d = g["diff"].get(k) or g["diff"].get(f"{k} vs {inc}")
            lab = f"`{k}`" + (" (2 seeds)" if k in (cand, inc) else "")
            dd = (f"{f(d['d_ic'], 4)} | {f(d['d_ls'])}" + (" (seed vs inc)" if k.startswith(cand + "-s") else "")) if d else " | "
            md.append(f"| {lab} | {f(v['ic'], 4)} | {f(v['ls_gross'])} | {f(v['ls_net4'])} | {f(v['ls_net20'])} | {dd} |")
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--inc", default="inc")
    a = ap.parse_args()
    o = run(a.cand, a.inc)
    json.dump(o, open(os.path.join(D.RUNS, "r24", "test.json"), "w"), indent=1)
    t = table(o, a.cand, a.inc)
    open(os.path.join(D.RUNS, "r24", "test.md"), "w").write(t)
    print(t)
