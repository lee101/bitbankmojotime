"""Score models on the preregistered subsample with the benchmark's own metric code restricted to those rows.
  python score_sub.py [--models a,b] [--rows secondary] [--validate]"""
import argparse, json
import numpy as np
from common import *


def score(models, rows_idx, r):
    mk = np.asarray(rows_idx)
    blk = (r["t_abs"] // D.TASKS[TASK]["block"])[mk]
    boot = Boot(r["a"][mk], blk)
    out, vec = {}, {}
    for m in models:
        q = load_pred(m)
        for j, h in enumerate(HOR):
            if j >= q.shape[1]:
                continue
            s, v = summarize(np.asarray(q[mk, j]), r["y"][mk, j], r["scale"][mk], r["a"][mk], blk, r["t_abs"][mk], boot)
            s.pop("asset_pin")
            out[f"{m}|h{h}"] = s
            vec[(m, h)] = v["sPL"]
    for m in models:
        vec[(m, "mean")] = np.mean([vec[(m, h)] for h in HOR if (m, h) in vec], 0)
        out[f"{m}|hmean"] = dict(sPL=ci(vec[(m, "mean")]))
    for m in models:
        for ref in (BMT2, "naive"):
            if ref in models and m != ref:
                for h in list(HOR) + ["mean"]:
                    if (m, h) not in vec:
                        continue
                    out[f"{m}|h{h}"][f"dsPL_{ref}"] = ci(vec[(m, h)] - vec[(ref, h)])
                    out[f"{m}|h{h}"][f"skill_{ref}"] = ci(1 - vec[(m, h)] / vec[(ref, h)])
    return out


def validate(r):
    """full-set sPL of naive at h1 must reproduce the leaderboard (0.2697 [0.2587, 0.2820])."""
    mk = np.arange(len(r["t"]))
    blk = r["t_abs"] // D.TASKS[TASK]["block"]
    boot = Boot(r["a"], blk)
    q = load_pred("naive")
    s, _ = summarize(np.asarray(q[:, 0]), r["y"][:, 0], r["scale"], r["a"], blk, r["t_abs"], boot)
    print("naive h1 full set sPL", s["sPL"], "MASE", s["mase"], "IC", s["ic"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(REF_MODELS))
    ap.add_argument("--rows", default="rows")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = D.load_rows(TASK)
    if a.validate:
        validate(r); raise SystemExit
    sub = json.load(open(SUB))
    res = score(a.models.split(","), sub[a.rows if a.rows != "rows" else "rows"] if a.rows != "secondary" else sub["secondary_rows"], r)
    f = lambda x: f"{x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]"
    for k, s in res.items():
        if "hmean" in k:
            print(k, "sPL", f(s["sPL"]), "dsPL_bmt2", f(s.get(f"dsPL_{BMT2}", (np.nan,) * 3)))
        else:
            print(k, "sPL", f(s["sPL"]), "MASE", f(s["mase"]), "IC", f"{s['ic']:.4f}", "c90", f"{s['c90'][0]:.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"))
