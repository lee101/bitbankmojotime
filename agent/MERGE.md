# Branch `agent/search`: what it is and how to merge it

Worked in the worktree `/nvme0n1-disk/code/bitbankmojotime-agent-wt`, output root `runs-agent/` (git-ignored). Nothing here writes
to the main checkout. Branched from main at `c52f22d`.

## Run it in isolation

```bash
export BMT_RUNS=$PWD/runs-agent                                   # predictions, checkpoints, tables of this branch
export BMT_DATA=/nvme0n1-disk/code/bitbankmojotime/data/cache     # shared market-data cache, read only
export PYTHONPATH=bench:.                                         # registry: agent/models.json, report: agent/LEADERBOARD.md
python -m train.final_eval --names ohlc+cal --ckpt-dir runs-agent/ckpt_search    # writes bmt2-ohlc+cal~<hash> and registers it
python -m bmtbench.score && python -m bmtbench.report
```

## What changed in shared code (possible conflicts)

| file | change |
|---|---|
| `bench/bmtbench/data.py` | appended: model registry (`register`, `open_pred`, `resolve`, `models_of`, `model_hash`, `file_sha`) |
| `bench/bmtbench/score.py`, `paired.py`, `tradable.py`, `ab.py`, `voltiming.py` | predictions are loaded only through `D.open_pred` (hash-checked); models come from the registry, not from a directory listing; short names are resolved |
| `bench/bmtbench/report.py` | reads `agent/leaderboard_head.md`, writes `agent/LEADERBOARD.md` (`BMT_REPORT_DIR` overrides), row table from the registry |
| `train/final_eval.py` | row name = `<prefix><name>~<8 hex of sha256(config + fold checkpoints)>`, registers the row; `--tier`, `--note` |
| `tools/research_log.py` | `BMT_RUNS` / `BMT_REPORT_DIR`, wave 3 |
| `tests/test_registry.py` | new |

Main still globs `runs/preds/` and uses `bench/models.json`; after the merge it needs a registry (`agent/models.json` can be
moved to `bench/models.json` format-compatibly: it is a superset with `pred_sha256` per task) or `BMT_REGISTRY` pointed at one.

## Files of this branch that supersede files on main (copy over on merge, do not three-way merge)

| this branch | supersedes on main | why |
|---|---|---|
| `agent/LEADERBOARD.md` (+ `agent/leaderboard_head.md`) | `bench/LEADERBOARD.md`, `bench/leaderboard_head.md` | main's copy (commit c52f22d) was generated while `runs/preds/*/bmt2-ohlc+cal*.npy` held a different model, see below |
| `agent/models/research/MODEL_CARD.md`, `agent/models/permissive/MODEL_CARD.md` | `models/*/MODEL_CARD.md` | hash names; the permissive card's "vs the research tier" row on main is wrong (below) |
| `agent/PLAN.md` | `docs/PLAN.md` | search outcome, ranker not confirmed |
| `agent/INTEGRATION.md` | `docs/INTEGRATION.md` | 256-bar model, corrected A/B statement |
| `agent/RESEARCH_LOG.md` | `docs/RESEARCH_LOG.md` | wave 3 added |
| `agent/models.json` | `bench/models.json` | registry with hashes |

## Statements on main that are wrong and are corrected here

Cause: between 22:12 and 00:09 UTC on 2026-11-12/13 the files `runs/preds/{hourly,daily}/bmt2-ohlc+cal.npy` and `-sym.npy` in the
main checkout held predictions of the wave-2 retrain of `ohlc+cal` (47 training symbols, `runs/search2/ckpt`), not of the promoted
wave-1 model (333 symbols, `runs/search/ckpt`). Everything computed from them in that window is about the wrong model:
- `bench/LEADERBOARD.md` at c52f22d: the paired table rows of `bmt2-ohlc+cal-sym`, the synthetic tier's "vs research" numbers
  (-0.34 / -0.23 / -0.42, -0.39 / -0.61 should be -0.38 / -0.30 / -0.54, -0.54 / -0.69), the tradable and A/B rows of
  `bmt2-ohlc+cal-sym`, and the sentence about a threshold rule with "+60 bp [14, 96] on 994 trades" (that was the 47-symbol
  model; the promoted model gives +27 bp [-7, 52] on 2,265 trades, not separated).
- `models/permissive/MODEL_CARD.md` at c52f22d: the "vs the research tier" row (same correction).
- `README.md` at c52f22d: "two of about 150 net-of-cost intervals exclude zero" should read "one of 240".

## State left in the main checkout by this agent

- Restored at 00:09 UTC on 2026-11-13 (before the instruction to stop writing there), from `runs/search/ckpt/ohlc+cal_F{1..4}.pt`:
  `runs/preds/hourly/bmt2-ohlc+cal.npy`, `runs/preds/hourly/bmt2-ohlc+cal.json`, `runs/preds/hourly/bmt2-ohlc+cal-sym.npy`,
  `runs/preds/hourly/bmt2-ohlc+cal-sym.json`, and the same four names under `runs/preds/daily/`. They now hold the wave-1 promoted
  model again. If the owner's session wants its own `ohlc+cal` (wave-2) predictions there, rerun its command, ideally under
  another name.
- Tables in `runs/` last written by this agent while the mix-up was in place (stale, regenerate or ignore):
  `runs/scores_hourly.json`, `runs/scores_daily.json`, `runs/leaderboard_tables.md` (00:02-00:03 UTC), `runs/paired.json`,
  `runs/paired.md` (00:05), `runs/tradable.json`, `runs/tradable.md`, `runs/ab.json`, `runs/ab.md` (00:06).
- Uncommitted edits of this agent in the main working tree: `bench/models.json` (added rows) and `docs/PLAN.md` (the text now in
  `agent/PLAN.md`). Safe to discard with `git checkout -- bench/models.json docs/PLAN.md`.
- `runs/preds/*/bmt2-ohlc+cal+rv5*.npy` were written by the other session: a config the search did not promote, so by the
  registered search rule it has no leaderboard row in this branch.
- Everything else under `runs/` that this agent wrote earlier (checkpoints, ledgers, baselines) is untouched since and was
  copied, not moved, into `runs-agent/`.
