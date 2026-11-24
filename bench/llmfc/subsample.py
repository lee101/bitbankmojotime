"""Preregistered stratified row subsample (bench/llmfc/PREREG.md section 2). Writes subsample.json."""
import json
import numpy as np
from common import *

N_TS_PER_FOLD, PER_TS = 10, dict(btceth=2, seen=7, unseen=6)


def main():
    r = D.load_rows(TASK)
    rng = np.random.default_rng(SEED)
    syms = list(r["syms"])
    be = np.isin(r["a"], [syms.index("BTCUSDT"), syms.index("ETHUSDT")])
    bucket = np.where(be, 0, np.where(r["unseen"], 2, 1))
    block = r["t_abs"] // D.TASKS[TASK]["block"]
    rows = []
    for f in range(4):
        mf = r["fold"] == f
        ts = np.unique(r["t_abs"][mf])
        cnt = {t: 0 for t in ts}
        a_t, t_t = r["a"][mf], r["t_abs"][mf]
        ok_ts = [t for t in ts if (t_t == t).sum() >= 15]
        chosen, used_blocks = [], set()
        for t in rng.permutation(ok_ts):
            b = int(t // D.TASKS[TASK]["block"])
            if b in used_blocks:
                continue
            idx = np.flatnonzero((r["t_abs"] == t) & mf)
            pick = []
            for name, k in zip(("btceth", "seen", "unseen"), (0, 1, 2)):
                cand = idx[bucket[idx] == k]
                pick += list(rng.choice(cand, min(PER_TS[name], len(cand)), replace=False))
            if len(pick) < 15:
                continue
            chosen += pick[:15]; used_blocks.add(b)
            if len(used_blocks) == N_TS_PER_FOLD:
                break
        rows += chosen
    rows = np.sort(np.array(rows))
    sec = np.sort(rng.choice(rows, 150, replace=False))
    out = dict(seed=SEED, task=TASK, rows=rows.tolist(), secondary_rows=sec.tolist(),
               n=len(rows), per_fold=np.bincount(r["fold"][rows], minlength=4).tolist(),
               per_bucket=np.bincount(bucket[rows], minlength=3).tolist(), n_timestamps=len(np.unique(r["t_abs"][rows])),
               n_blocks=len(np.unique(block[rows])), n_assets=len(np.unique(r["a"][rows])))
    json.dump(out, open(SUB, "w"))
    print({k: v for k, v in out.items() if k not in ("rows", "secondary_rows")})


if __name__ == "__main__":
    main()
