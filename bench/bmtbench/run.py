"""Produce prediction files runs/preds/<task>/<model>.npy for the baselines.  python -m bmtbench.run --task hourly --models naive,ewma"""
import argparse, time
import numpy as np
from . import data as D, baselines as B

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hourly")
    ap.add_argument("--models", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rows only (no file written)")
    a = ap.parse_args()
    u = D.universe()
    p = D.panel(a.task, u["eval"])
    rows = D.load_rows(a.task)
    assert list(rows["syms"]) == p.syms
    if a.limit:
        rows = {k: (v[:: max(1, len(rows["t"]) // a.limit)] if k != "syms" else v) for k, v in rows.items()}
    seen_idx = [p.ix[s] for s in u["seen"]]
    starts = [s for _, s, _ in D.fold_bounds(a.task)]
    for m in a.models.split(","):
        t0, meta = time.time(), {}
        if m == "naive":
            q = B.naive(p, rows, a.task)
        elif m == "drift":
            q = B.drift(p, rows, a.task)
        elif m == "empirical":
            q = B.empirical(p, rows, a.task)
        elif m == "seasonal-naive":
            q = B.seasonal_naive(p, rows, a.task)
        elif m == "ewma":
            q, lam = B.ewma(p, rows, a.task, seen_idx)
            meta["lambda_per_fold"] = lam
        elif m == "garch":
            q = B.garch(p, rows, a.task, starts)
        elif m in ("timestral-fast", "timestral-max"):
            from . import foundation as F
            q = F.timestral_kernel(p, rows, a.task, m.split("-")[1])
        elif m in ("chronos-2", "chronos-2-ret"):
            from . import foundation as F
            q = F.chronos2(p, rows, a.task, "ret" if m.endswith("ret") else "logp")
        elif m == "timestral-official":
            from . import foundation as F
            q = F.timestral_official(p, rows, a.task)
        else:
            raise SystemExit("unknown model " + m)
        meta.update(seconds=round(time.time() - t0, 1), rows=len(rows["t"]))
        if q is None:
            print(m, "cannot produce every row: no entry"); continue
        print(m, q.shape, meta, flush=True)
        if not a.limit:
            D.save_pred(a.task, m, q, meta)
