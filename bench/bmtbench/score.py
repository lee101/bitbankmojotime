"""Score every prediction file and write runs/scores_<task>.json + bench/LEADERBOARD.md.  python -m bmtbench.score"""
import glob, json, os, sys
import numpy as np
from . import data as D
from .metrics import Boot, summarize, ci

REFS = ("naive", "timestral-fast")
GROUPS = ("all", "seen", "unseen", "BTCUSDT", "ETHUSDT", "F1", "F2", "F3", "F4")


def masks(rows):
    syms = list(rows["syms"])
    m = {"all": np.ones(len(rows["t"]), bool), "seen": ~rows["unseen"], "unseen": rows["unseen"]}
    for s in ("BTCUSDT", "ETHUSDT"):
        m[s] = rows["a"] == syms.index(s)
    for f in range(4):
        m[f"F{f + 1}"] = rows["fold"] == f
    return m


def score_task(task, only=None):
    rows = D.load_rows(task)
    cfg = D.TASKS[task]
    block = rows["t_abs"] // cfg["block"]
    models = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(D.RUNS, "preds", task, "*.npy")))
    if only:
        models = [m for m in models if m in only or m in REFS]
    preds = {m: np.load(D.pred_path(task, m), mmap_mode="r") for m in models}
    preds = {m: q for m, q in preds.items() if q.shape[0] == len(rows["t"])}
    res = {}
    for g, mk in masks(rows).items():
        if mk.sum() == 0:
            continue
        boot = Boot(rows["a"][mk], block[mk])
        for j, h in enumerate(cfg["horizons"]):
            vecs, summ = {}, {}
            for m, q in preds.items():
                if j >= q.shape[1]:
                    continue
                summ[m], vecs[m] = summarize(np.asarray(q[:, j])[mk], rows["y"][mk, j], rows["scale"][mk], rows["a"][mk], block[mk], rows["t_abs"][mk], boot)
            for m in summ:
                for r in REFS:
                    if r in summ:
                        summ[m]["skill_" + r] = ci(1 - vecs[m]["sPL"] / vecs[r]["sPL"])
                        summ[m]["win_" + r] = float((summ[m]["asset_pin"] < summ[r]["asset_pin"]).mean())
                        summ[m]["dpin_" + r] = ci(vecs[m]["pinball"] - vecs[r]["pinball"])
            for m in summ:
                summ[m].pop("asset_pin")
                res[f"{g}|h{h}|{m}"] = summ[m]
    json.dump(res, open(os.path.join(D.RUNS, f"scores_{task}.json"), "w"))
    return res


def fmt(x, d=4, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    k = 100 if pct else 1
    if isinstance(x, (tuple, list)):
        return f"{x[0] * k:.{d}f} [{x[1] * k:.{d}f}, {x[2] * k:.{d}f}]"
    return f"{x * k:.{d}f}"


def tables(task, res, info):
    cfg, out = D.TASKS[task], []
    names = sorted({k.split("|")[2] for k in res})
    for h in cfg["horizons"]:
        rk = sorted((m for m in names if f"all|h{h}|{m}" in res), key=lambda m: res[f"all|h{h}|{m}"]["sPL"][0])
        n = res[f"all|h{h}|{rk[0]}"]
        out.append(f"\n### {task}, horizon {h} ({n['n']} rows, {n['assets']} assets, all folds)\n")
        out.append("| model | params | sPL [95% CI] | skill vs naive % | skill vs timestral-fast % (paired) | pinball bp | CRPS bp | win % vs naive / tsl-fast |")
        out.append("|---|---|---|---|---|---|---|---|")
        for m in rk:
            s = res[f"all|h{h}|{m}"]
            out.append(f"| {m} | {info.get(m, {}).get('params', '-')} | {fmt(s['sPL'])} | {fmt(s.get('skill_naive'), 2, True)} | {fmt(s.get('skill_timestral-fast'), 2, True)} | "
                       f"{fmt(s['pinball'], 3)} | {fmt(s['crps'], 3)} | {fmt(s.get('win_naive'), 0, True)} / {fmt(s.get('win_timestral-fast'), 0, True)} |")
        out.append("\n| model | MAE bp | relMAE vs zero | MASE | cov 80 / 90 / 98 % | calib | DA % (non-zero medians %) | IC |")
        out.append("|---|---|---|---|---|---|---|---|")
        for m in rk:
            s = res[f"all|h{h}|{m}"]
            out.append(f"| {m} | {fmt(s['mae'], 3)} | {fmt(s['relmae'], 4)} | {fmt(s['mase'], 4)} | {s['c80'][0] * 100:.1f} / {s['c90'][0] * 100:.1f} / {s['c98'][0] * 100:.1f} | "
                       f"{s['calib']:.4f} | {fmt(s['da'][0], 2, True)} ({s['nz_median'] * 100:.0f}) | {fmt(s['ic'], 4)} |")
        out.append(f"\nSkill vs timestral-fast %, paired 95 % CI, by group (positive = better than Timestral fast), horizon {h}:\n")
        gs = [g for g in GROUPS[1:] if any(k.startswith(g + f"|h{h}|") for k in res)]
        out.append("| model | " + " | ".join(gs) + " |")
        out.append("|---|" + "---|" * len(gs))
        for m in rk:
            out.append(f"| {m} | " + " | ".join(fmt(res.get(f"{g}|h{h}|{m}", {}).get("skill_timestral-fast"), 2, True) for g in gs) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    info = json.load(open(os.path.join(D.REPO, "bench", "models.json"))) if os.path.exists(os.path.join(D.REPO, "bench", "models.json")) else {}
    md = []
    for task in D.TASKS:
        if "--cached" in sys.argv:
            res = json.load(open(os.path.join(D.RUNS, f"scores_{task}.json")))
        else:
            res = score_task(task)
        md.append(tables(task, res, info))
    open(os.path.join(D.RUNS, "leaderboard_tables.md"), "w").write("\n".join(md))
    print("\n".join(md)[:6000])
