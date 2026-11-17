# Pre-registration: confirmation of the trade-day tilt (PREREG-GATE form (a)) on universes the lead never touched

Date 2026-11-18 (UTC). Committed before any forecast for the new coins or any harness run on R9-R24. Code: `tools/r24gate.py`
(forms below, universe set option), `train/rank24.py botpred --coins`. Outputs `runs-agent/r24/confirm/`.

## Data

- Universes R9-R24 of the SIGNAL confirmation (`bgm-signal-wt/tools/signal/conf.py`, its seeded coin lists, panels and SEL books
  in `/sdb-disk/tradingbots/binance/signal/shift/u`, read only; imported, not copied). Base arms for them are simulated here with the
  same binary as the forms (paired), saved under our CORRVOL_D.
- Forecast: the frozen `mkt+r24` checkpoints (seed-averaged 24 h median, daily 00:00 UTC, 2023-01-01 .. 2026-10-31). Coins of R9-R24
  not in the earlier forecast file are predicted with the same checkpoints (prediction only, no training).
- Post-2026-10 bars: the forecaster's input cache ends 2026-10-31 23:00 UTC, so no post-October forecast exists on disk; the tiny-n
  check is not possible now and is folded into the February 2027 forward scoring (PREREG-GATE section 4). Nothing is fetched.

## Forms (co-primary 1 and 2; same pooled test as PREREG-GATE)

1. `gt` frozen as run: identical code and parameters to the lead (replica band 0.05, rows scaled down only).
2. `gtr` risk-matched: the same trade-day multipliers, then each day the tilted row is rescaled to the base row's gross (up or down),
   so the book's gross equals the base book's every day.
Secondary (sensitivity, 10 placebo draws): `gtb` = form 1 with the replica band set to each configuration's own band. The harness keys
weight files by (pv, cap, regime, top); for this run the key is extended with the band (a monkey-patch of `slib.wkey` inside our
process; base weights do not depend on the band, so the base arm is unchanged).

Statistic: `al.pooled` on R9-R24 restricted to 2023-01 .. 2026-10, costs c7 and c10; placebo 20 draws per co-primary (forecast ranks
shuffled across coins within each day); per-year d Sharpe; turnover change vs base (median); ruin view on M11 (the ruin points need M11;
M11 is a lead universe, so the ruin row is descriptive).

## Bar

A co-primary confirms iff at both costs: class INNOVATION under `al.klass` with its 20-draw placebo band; at least 10 of 16 universes
with positive family-mean d Sharpe; turnover change <= +2 %. "Confirmed" if either co-primary passes (two looks, stated); "lean" if
either has the d Sharpe interval above zero at both costs without passing; otherwise "not confirmed". The ruin view and the
secondary are reported, not gated.

If confirmed: a producer integration design on a branch of binancegomojo (worktree `/nvme0n1-disk/code/bgm-fcast-wt`), flag default
off, not installed, with the licence blocker stated. If not: no integration work.
