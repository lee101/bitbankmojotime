"""Shuffled-time placebo (bench/PROTOCOL.md section 6.3).  python -m bmtbench.placebo --name bmt-s-placebo [--seed 0]

The candidate was retrained and scored on time-permuted returns (train.train --placebo). Pass = the upper end of the 95 % interval of
its sPL skill vs the `empirical` baseline is below 1 % at every horizon.
"""
import argparse, json, os
import numpy as np
from . import data as D, baselines as B
from .metrics import Boot, summarize, ci

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    u = D.universe()
    E = D.shuffle_time(D.load_hourly(u["eval"]), a.seed)
    out = {}
    for task in D.TASKS:
        p = E if task == "hourly" else D.to_daily(E)
        z = np.load(os.path.join(D.RUNS, "placebo", f"{a.name}_{task}.npz"))
        rows = {k: z[k] for k in ("a", "t", "fold", "y", "scale")}
        assert np.allclose(rows["y"][:, 0], p.logp[rows["a"], rows["t"] + 1] - p.logp[rows["a"], rows["t"]], atol=1e-6)
        cfg = D.TASKS[task]
        blk = (rows["t"] + p.t0) // cfg["block"]
        boot = Boot(rows["a"], blk)
        emp, nai = B.empirical(p, rows, task), B.naive(p, rows, task)
        for j, h in enumerate(cfg["horizons"]):
            v = {n: summarize(q[:, j], rows["y"][:, j], rows["scale"], rows["a"], blk, rows["t"], boot)[1]["sPL"] for n, q in (("cand", z["q"]), ("emp", emp), ("naive", nai))}
            s = ci(1 - v["cand"] / v["emp"])
            out[f"{task}|h{h}"] = dict(skill_vs_empirical=s, skill_naive_vs_empirical=ci(1 - v["naive"] / v["emp"]), sPL=float(v["cand"][0]), sPL_empirical=float(v["emp"][0]),
                                       rows=int(len(rows["t"])), passed=bool(s[2] < 0.01))
            print(f"{task} h{h}: candidate skill vs empirical {100 * s[0]:+.2f}% [{100 * s[1]:+.2f}, {100 * s[2]:+.2f}]  ->  {'PASS' if s[2] < 0.01 else 'FAIL'}")
    json.dump(out, open(os.path.join(D.RUNS, "placebo", f"{a.name}.json"), "w"), indent=1)
