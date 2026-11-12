"""Build the universe (data/universe.json) and the benchmark row tables (runs/rows_<task>.npz) from the fetched archive."""
import json, os, sys
import numpy as np
from . import data as D

if __name__ == "__main__":
    up = os.path.join(D.REPO, "data", "universe.json")
    if not os.path.exists(up) or "--universe" in sys.argv:
        json.dump(D.build_universe(), open(up, "w"), indent=0)
    u = D.universe()
    print("pool", len(u["pool"]), "eval", len(u["eval"]), "seen", len(u["seen"]), "unseen", len(u["unseen"]), "train", len(u["train"]))
    os.makedirs(D.RUNS, exist_ok=True)
    for task in D.TASKS:
        p = D.panel(task, u["eval"])
        r = D.build_rows(p, task)
        r["t_abs"] = r["t"] + p.t0
        r["unseen"] = np.isin(r["a"], [p.ix[s] for s in u["unseen"]])
        np.savez(D.rows_path(task), syms=np.array(p.syms), **r)
        f = np.bincount(r["fold"], minlength=4)
        print(task, "rows", len(r["t"]), "per fold", f.tolist(), "assets", len(np.unique(r["a"])), "unseen rows", int(r["unseen"].sum()),
              "filled bars in panel", int((~p.real & p.live).sum()))
