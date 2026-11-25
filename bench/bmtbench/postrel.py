"""Post-release column for the leaderboard (PMHOLD-FCST task 4).  BMT_RUNS=<runs dir> python -m bmtbench.postrel [task]
Timestral-209M (public checkpoint, repo commit 2026-10-15) and Chronos-2 (release 2025-10-20) have undisclosed / partly disclosed
pretraining corpora, so every leaderboard fold before their release may be in-sample for them. This re-scores the existing prediction
files (no model is re-run) on the rows whose origin is after each release, next to the pre-release part of the same folds:
  W_pre   = F4 origins 2026-01-01 .. 2026-10-14 (pre-Timestral, latest unseen fold)
  W_post  = origins 2026-10-15 .. 2026-10-31 (post-Timestral; F4 tail, ~2.4 weeks)
  W_c2pre = F3 + F4 origins 2025-01-01 .. 2025-10-19,  W_c2post = origins 2025-10-20 .. 2026-10-31 (post-Chronos-2)
sPL, paired skill vs timestral-official / chronos-2 / garch (95 % weekly-block bootstrap, protocol section 5) and median rank IC."""
import os, sys, json
import numpy as np
from . import data as D
from .metrics import Boot, summarize, ci

task = sys.argv[1] if len(sys.argv) > 1 else "hourly"
rows = D.load_rows(task); cfg = D.TASKS[task]; th = rows["t_abs"] if task == "hourly" else rows["t_abs"]
step = cfg["step"]; tab = rows["t_abs"] * step                         # epoch hour of the origin (daily rows are in days)
W = [("pre-Timestral F4 2026-01-01..10-14", D.eh(2026), D.eh(2026, 10, 15)), ("post-Timestral 2026-10-15..10-31", D.eh(2026, 10, 15), D.eh(2026, 11)),
     ("pre-Chronos-2 2025-01-01..10-19", D.eh(2025), D.eh(2025, 10, 20)), ("post-Chronos-2 2025-10-20..2026-10-31", D.eh(2025, 10, 20), D.eh(2026, 11))]
MODELS = ["timestral-official", "timestral-official-2048", "timestral-fast", "timestral-max", "chronos-2", "garch", "empirical", "naive",
          "bmt2-ohlc+cal-sym", "bmt2-ohlc+cal@t-kd", "bmt2-ohlc+cal@t-ens4-sym", "bmt2-syn-ohlc+cal@t-sym"]
REFS = ["timestral-official", "chronos-2", "garch"]
preds = {m: np.load(D.pred_path(task, m), mmap_mode="r") for m in MODELS if os.path.exists(D.pred_path(task, m))}
print(f"# Post-release column ({task}); rows by origin window; sPL lower = better; skill = 1 - sPL / sPL_ref (95 % CI)\n")
out = {}
for lab, lo, hi in W:
    mk = (tab >= lo) & (tab < hi)
    if mk.sum() == 0: print(f"{lab}: no rows"); continue
    block = rows["t_abs"][mk] // cfg["block"]; boot = Boot(rows["a"][mk], block)
    print(f"## {lab}: {mk.sum()} rows, {len(np.unique(rows['a'][mk]))} assets, {len(np.unique(block))} blocks\n")
    for j, h in enumerate(cfg["horizons"]):
        summ, vecs = {}, {}
        for m, q in preds.items():
            if j < q.shape[1]: summ[m], vecs[m] = summarize(np.asarray(q[mk, j]), rows["y"][mk, j], rows["scale"][mk], rows["a"][mk], block, rows["t_abs"][mk], boot)
        print(f"h{h}: | model | sPL | IC | " + " | ".join(f"skill vs {r}" for r in REFS) + " |")
        print("|---|---|---|" + "---|" * len(REFS))
        for m in summ:
            sk = []
            for r in REFS:
                if r in vecs and r != m: s = ci(1 - vecs[m]["sPL"] / vecs[r]["sPL"]); sk.append(f"{100 * s[0]:+.2f} [{100 * s[1]:+.2f}, {100 * s[2]:+.2f}]")
                else: sk.append("-")
            print(f"| {m} | {summ[m]['sPL'][0]:.4f} | {summ[m]['ic']:+.4f} | " + " | ".join(sk) + " |")
            out[f"{lab}|h{h}|{m}"] = dict(sPL=summ[m]["sPL"], ic=summ[m]["ic"], n=summ[m]["n"])
        print()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f"postrel_{task}.json"), "w"), indent=0)
