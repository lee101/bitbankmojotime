"""Shuffled-time placebo for a search config (registered form, PROTOCOL.md 6.3): retrain on time-permuted bars, score on the permuted
test periods against the better unconditional baseline.   python -m train.placebo2 --groups ohlc,har
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D, baselines as B
from bmtbench.metrics import Boot, summarize, ci
from train.search import Runner, OUT, HZ, make_cfg, name_of
from train.model2 import BMT2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="")
    ap.add_argument("--size", default="s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use5m", action="store_true")
    a = ap.parse_args()
    groups = [g for g in a.groups.split(",") if g]
    R = Runner("cuda", placebo_seed=a.seed, vram_gb=3.0, use5m=a.use5m)
    b, u = R.bank, D.universe()
    cfg, nm = make_cfg(groups, a.size), "placebo-" + name_of(groups, a.size)
    models = []
    for fi in range(4):
        R.train_fold(cfg, fi, 1500, save=nm)
        c = torch.load(os.path.join(OUT, "ckpt", f"{nm}_{D.FOLDS[fi][0]}.pt"), map_location="cuda", weights_only=False)
        m = BMT2(horizons=HZ, **c["cfg"]).cuda()
        m.load_state_dict(c["sd"])
        models.append(m)
    ev = [x for x in u["eval"] if x in b.ix]
    ei = np.array([b.ix[x] for x in ev])
    lp = b.g["logp"][ei].cpu().numpy()
    live = np.zeros_like(lp, bool)
    for i, x in enumerate(ei):
        live[i, b.first[x]:b.last[x] + 1] = True
    Ph = D.Panel(ev, b.t0, np.where(live, lp, np.nan), live.copy())
    out = {}
    for task, P, k in (("hourly", Ph, 1), ("daily", D.to_daily(Ph), 24)):
        rows = D.build_rows(P, task)
        tb = rows["t"] if task == "hourly" else (rows["t"] + P.t0) * 24 + 23 - b.t0
        hs = [HZ.index(h) for h in D.TASKS[task]["horizons"]]
        q = np.zeros((len(tb), len(hs), 21), np.float32)
        for fi in range(4):
            i = np.flatnonzero(rows["fold"] == fi)
            q[i] = R.predict(models[fi], ei[rows["a"][i]], tb[i], k, fold=fi)[:, hs].cpu().numpy()
        q = np.sort(q, 2)
        blk = (rows["t"] + P.t0) // D.TASKS[task]["block"]
        boot = Boot(rows["a"], blk)
        emp, nai = B.empirical(P, rows, task), B.naive(P, rows, task)
        for j, h in enumerate(D.TASKS[task]["horizons"]):
            v = {n: summarize(x[:, j], rows["y"][:, j], rows["scale"], rows["a"], blk, rows["t"], boot)[1]["sPL"] for n, x in (("cand", q), ("emp", emp), ("naive", nai))}
            ref = "naive" if v["naive"][0] < v["emp"][0] else "emp"
            s = ci(1 - v["cand"] / v[ref])
            out[f"{task}|h{h}"] = dict(skill_vs_best_unconditional=s, reference=ref, rows=int(len(tb)), passed=bool(s[2] < 0.01))
            print(f"{task} h{h}: vs {ref} {100 * s[0]:+.2f}% [{100 * s[1]:+.2f}, {100 * s[2]:+.2f}] {'PASS' if s[2] < 0.01 else 'FAIL'}", flush=True)
    os.makedirs(os.path.join(D.RUNS, "placebo"), exist_ok=True)
    json.dump(out, open(os.path.join(D.RUNS, "placebo", nm + ".json"), "w"), indent=1)
