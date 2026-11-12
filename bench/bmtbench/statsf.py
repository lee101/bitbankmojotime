"""AutoETS / AutoARIMA baselines (Nixtla StatsForecast) on the log-price series: refit every 30 days on the last 2000 bars, the fitted
model is applied (`forward`) at every origin in between; Gaussian quantiles from the model's h-step prediction interval.

  python -m bmtbench.statsf --task hourly --model autoarima --jobs 10
"""
import argparse, concurrent.futures as cf, os, time, warnings
import numpy as np
from . import data as D
from .baselines import Z

WIN, LEVEL = 2000, 68.27


def run_asset(job):
    kind, logp, first, t, fold, H, refit = job
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    from statsforecast.models import AutoARIMA, AutoETS
    hm = max(H)
    out = np.zeros((len(t), len(H), D.NQ), np.float32)
    model, nxt, cur_fold, nfit = None, -1, -1, 0
    for i in range(len(t)):
        y = logp[max(first, t[i] - WIN + 1):t[i] + 1]
        base = y[-1]
        y = y - base
        if model is None or t[i] >= nxt or fold[i] != cur_fold:
            model = (AutoARIMA(season_length=1) if kind == "autoarima" else AutoETS(season_length=1, model="ZZN")).fit(y)
            nxt, cur_fold, nfit = t[i] + refit, fold[i], nfit + 1
        try:
            f = model.forward(y, h=hm, level=[LEVEL])
            mu, sd = f["mean"], (f[f"hi-{LEVEL}"] - f[f"lo-{LEVEL}"]) / 2
            if not (np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 0).all()):
                raise ValueError
        except Exception:                                            # numerical failure: random-walk fallback on the context
            r = np.diff(y)
            mu, sd = np.zeros(hm), np.sqrt((r ** 2).mean() * np.arange(1, hm + 1))
        for j, h in enumerate(H):
            out[i, j] = mu[h - 1] + sd[h - 1] * Z
    return out, nfit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hourly")
    ap.add_argument("--model", required=True, choices=["autoarima", "autoets"])
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--limit-assets", type=int, default=0)
    a = ap.parse_args()
    u = D.universe()
    p = D.panel(a.task, u["eval"])
    rows = D.load_rows(a.task)
    cfg = D.TASKS[a.task]
    refit = 30 * 24 // cfg["step"]
    jobs, idxs = [], []
    for ai in np.unique(rows["a"])[:a.limit_assets or None]:
        i = np.flatnonzero(rows["a"] == ai)
        i = i[np.argsort(rows["t"][i], kind="stable")]
        idxs.append(i)
        jobs.append((a.model, p.logp[ai], int(p.first[ai]), rows["t"][i], rows["fold"][i], cfg["horizons"], refit))
    q = np.zeros((len(rows["t"]), len(cfg["horizons"]), D.NQ), np.float32)
    t0, nfit = time.time(), 0
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        for i, (o, n) in zip(idxs, ex.map(run_asset, jobs)):
            q[i] = o
            nfit += n
    meta = dict(seconds=round(time.time() - t0, 1), fits=nfit, rows=len(rows["t"]), window=WIN, refit_bars=refit)
    print(a.model, a.task, meta, flush=True)
    if not a.limit_assets:
        D.save_pred(a.task, a.model, q, meta)
