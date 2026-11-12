"""Paired sPL skill of chosen models against chosen references on the test rows (all folds; also unseen assets and the last fold).

  python -m bmtbench.paired --models bmt2-x-sym,bmt-t --refs timestral-official,timestral-official-2048,garch,bmt-s-sym
"""
import argparse, json, os
import numpy as np
from . import data as D
from .metrics import Boot, summarize, ci


def run(models, refs):
    out = {}
    for task in D.TASKS:
        rows, cfg = D.load_rows(task), D.TASKS[task]
        blk = rows["t_abs"] // cfg["block"]
        P = {m: np.load(D.pred_path(task, m), mmap_mode="r") for m in set(models) | set(refs) if os.path.exists(D.pred_path(task, m))}
        for g, mk in (("all", np.ones(len(blk), bool)), ("unseen", rows["unseen"]), ("F4", rows["fold"] == 3)):
            boot = Boot(rows["a"][mk], blk[mk])
            for j, h in enumerate(cfg["horizons"]):
                v = {m: summarize(np.asarray(q[:, j])[mk], rows["y"][mk, j], rows["scale"][mk], rows["a"][mk], blk[mk], rows["t_abs"][mk], boot)[1]["sPL"]
                     for m, q in P.items() if j < q.shape[1]}
                for m in models:
                    for r in refs:
                        if m in v and r in v and m != r:
                            out[f"{task}|h{h}|{g}|{m}|{r}"] = ci(1 - v[m] / v[r])
    return out


def table(out, models, refs):
    f = lambda x: "n/a" if x is None else f"{100 * x[0]:+.2f} [{100 * x[1]:+.2f}, {100 * x[2]:+.2f}]"
    md = []
    for g, title in (("all", "all rows"), ("unseen", "never-trained-on assets"), ("F4", "fold F4 (2026)")):
        md += [f"\nPaired sPL skill, % (95 %), {title}:\n", "| task, horizon | model | " + " | ".join("vs " + r for r in refs) + " |", "|---|---|" + "---|" * len(refs)]
        for task in D.TASKS:
            for h in D.TASKS[task]["horizons"]:
                for m in models:
                    md.append(f"| {task} h{h} | {m} | " + " | ".join(f(out.get(f"{task}|h{h}|{g}|{m}|{r}")) for r in refs) + " |")
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--refs", default="timestral-official,timestral-official-2048,garch,timestral-fast,bmt-s-sym")
    a = ap.parse_args()
    ms, rs = a.models.split(","), a.refs.split(",")
    out = run(ms, rs)
    json.dump(out, open(os.path.join(D.RUNS, "paired.json"), "w"), indent=1)
    open(os.path.join(D.RUNS, "paired.md"), "w").write(table(out, ms, rs))
    print(table(out, ms, rs)[:3000])
