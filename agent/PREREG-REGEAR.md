# Pre-registration: gearing of the confirmed trade-day tilt (track REGEAR)

Date 2026-11-18 (UTC). Committed before any run at k != 1 and before any pure-gearing run. Run before this commit: a k = 1 identity
smoke test on R9 (it reproduces a stored result, see "Identity"; no new number was seen). Code `tools/r24regear.py`; outputs
`runs-agent/regear/`.

## What this is and is not

`gt` (PREREG-GATE form (a)) is confirmed (PREREG-GATE-CONFIRM, wave 4c). It only scales rows down, so it lowers gross and return.
The owner's objective is maximum CAGR under a worst-month drawdown limit (30 %). This run asks at what constant multiplier k of the
book's risk budget `gt` has the risk of the untilted base, and what CAGR that buys. It is a gearing calibration: a move along the
frontier of an already confirmed tilt. It makes no new significance claim, has no placebo, and uses the same data that selected and
confirmed `gt` (the 14 lead universes, and R9-R24, which are no longer untouched). k is chosen in sample; the forward check is the
February 2027 scoring already registered.

## Arms

- `base`: no tilt, k = 1 (the stored base arms, read only).
- `gt` at k in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30}.
- Control, pure gearing: `base` at k in {0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30}, no tilt. The three values below 1
  are added to the requested grid because `gt`(k*) is by rule at or below base risk, so its pure-gearing match sits at k <= 1.

Everything else is frozen as in wave 4c: forecast files (`runs-agent/r24/botpred.npz`, `botpred_conf.npz`), `tilt_trade_days` with
replica band 0.05 and rows scaled down only, costs c7 / c10, window 2023-01 .. 2026-10, the 3,003-configuration family.

## How k is applied (harness = producer)

Producer (`bgm trend-targets`, `wf_core`): stitched book -> portfolio vol target and gross cap (`--pvt`, `--cap`; bear rows `--pvt-bear`,
`--cap-bear`; multiplier ceiling 4) -> forecast tilt -> holdings simulation with gross cap `--cap`. k multiplies `--pvt`, `--cap`,
`--pvt-bear`, `--cap-bear`.

Harness: a configuration c = (pv, cap, stop, cool, band, regime, top) becomes c_k = (round(pv k, 6), round(cap k, 6), rest unchanged)
and goes through the unchanged pipeline: `cv.make_wfun` computes `slib.pscale(raw book, raw equity, pv k, cap k)`, bear rows with
(pv k x bear pv mult, cap k x bear cap mult), then the arm's `post` = `r24gate.tilt_trade_days(W_k, P, f, k0, 0.05, both=False)`. So
the tilt's "never above the untilted row's gross" refers to the k-scaled untilted row, and the replica's 0.05 band is absolute (more
days cross it in a larger book, as in the producer). The trend-sim cell takes its gross cap `g` from c_k (cap k); borrow on negative
cash costs the harness's 10 % APR. The vol-target ceiling (4) and the tilt band are not scaled. For k = 1, c_k is c itself.

Identity: `gt` at k = 1 must reproduce the stored `gt` run bit-for-bit: per universe and cost the `al.summarize` arrays and the
turnover median against the wave 4c resume cache (R9-R24), and the pooled statistics against `runs-agent/r24/gate/gate_gt.json`
(lead, including its ruin row) and `runs-agent/r24/confirm/gate_gt.json`. Any mismatch stops the run.

## Universes

1. The two live configurations on M11, each reported alone, at both costs: Binance settings (pv .25, cap 1.875, stop .06, cool 72,
   band .05, MA200 bear rows pv .1875 cap 1.5625, top 10) and KuCoin settings (pv .20, cap 1.5, cool 96, bear pv .15 cap 1.25, rest
   equal; simulated here because it is not in the family sample). Research panel (Binance bars) for both.
2. Pooled family: CORE + R1-R8 + R9-R24 = 30 universes x 3,003 configurations, c7 and c10. Lead (14) and R9-R24 (16) also separately.

## Statistics (per arm, paired with base k = 1, evaluation window only)

Per configuration: CAGR, Sharpe, annual volatility (daily, x sqrt 365), worst calendar-month return, max drawdown (all `slib.full`),
worst 21-day and 60-day drawdown (largest loss over any holding period of up to 21 / 60 days on daily closes), mean gross (mean row
gross of the target book fed to the sim), turnover (sim total; the gearing changes the whole simulated history, the tilt only the
window, so it is reported against base and against pure gearing at the same k). Pooled figure = median over the 90,090
configuration-universe cells of the level and of the paired difference (arm - base). Share of cells with higher CAGR at not-higher
worst-month drawdown. Paired month-block bootstrap (2,000 shared month draws, as `al.pooled`) of the mean over universes of the
family-mean d Sharpe and d CAGR; for a live configuration its own d Sharpe and d CAGR. Ruin view: `algos/ruin.py` on M11, median
over the ruin points of the worst S1-S5 P(max drawdown > 60 %), base -> arm; for the live configurations their own value.

## Pick rule (fixed now)

Condition at a grid k: (pooled) the median paired difference of the worst-month drawdown and of the 60-day drawdown are both <= 0,
at c7 and at c10; (live) for each of the two live configurations, its worst-month drawdown and its 60-day drawdown are both not
above its base's, at c7 and at c10. k* = the largest grid k such that the condition holds at k and at every smaller grid k. If it
fails at 1.00, k* = 1.00 (no regear). Nothing above 1.30 is extrapolated: if 1.30 passes, k* = 1.30 and that is said.

Secondaries, reported, not deciding: the pooled-only and the live-only k; k matched on volatility (largest grid k whose pooled median
vol ratio to base is <= 1 at both costs, and the interpolated crossing); the owner's 30 % worst-month limit read on the two live rows.

## Tilt versus pure gearing at equal drawdown

For a risk measure D (worst month, 60-day drawdown; also max drawdown and volatility), the pure-gearing path k' -> (median d D,
median d CAGR) over the control grid, piecewise linear. For `gt`(k): k' = the smallest k' at which pure gearing reaches the d D of
`gt`(k); the tilt's contribution at equal drawdown = median d CAGR of `gt`(k) minus the path's d CAGR at k'. Outside the control grid:
reported as such, not extrapolated. Live rows: the same with the configuration's own values. Also the same-k difference `gt`(k) - base(k).

## Production statement (descriptive, no deploy)

For k*: the producer flags / environment variables per bot, and whether venue margin or the executors' own limits make it infeasible.
