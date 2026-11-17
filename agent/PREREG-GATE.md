# Pre-registration: the frozen 24 h forecast used only where the trend bots already trade (PLAN item 2), and forward scoring (item 1)

Date 2026-11-17, committed before any run of these forms. Forecast: the frozen `mkt+r24` checkpoints (`bmt2-r24-mkt+r24-s0~bec5b0cd`,
`-s1~3db2cd50`, seed-averaged 24 h median, `runs-agent/r24/botpred.npz`, daily 00:00 UTC, 2023-01-01 .. 2026-10-31). No retraining,
no GPU. Code `tools/r24gate.py`; outputs `runs-agent/r24/gate/`. Research evaluation only (research-tier weights, NC data).

## 1. Harness and statistic (unchanged from PREREG-24H section 5)

ALGOS pooled harness, 14 universes (CORE + R1-R8), the 3,000-configuration family, costs c7 and c10, off arm = `base`. The rank r is
the forecast median's cross-sectional rank among the universe's non-PAXG coins with a forecast, as a fraction f = rank / (m - 1)
in [0, 1]; outside the forecast window every form equals `base`. Statistic: `al.pooled` restricted to 2023-01 .. 2026-10 (mean d Sharpe
with month bootstrap, share, d bp/day, per-year d Sharpe), class by `al.klass` with the placebo band. Placebo: 10 draws per form with the
forecast permuted across coins within each day. Ruin view: `algos/ruin.py`'s CORRVOL scenarios (median worst 60-day drawdown over the
ruin points, S1-S5) on M11, form vs base, restricted window. Turnover: total turnover per configuration from the sim, form minus base
(pre-2023 is identical, so the difference is the window's), median over configurations.

## 2. Forms (weights fed to the sim; the sim's stops, cooldown and band are unchanged; no form re-grosses other coins)

Let W be the base live book after the vol target and regime rows, h the coin's book weight drifted with price since its last trade
(Python replica of the sim's rule: trade when |target - current| > band of the configuration, exits when the target is 0, entries when
current is 0; stops are not replicated).

- (a) `gt`, tilt only on trade days: on a day the replica says coin j trades anyway, its multiplier is set to 1 + (f - 0.5) (range
  0.5 .. 1.5) and held until its next trade day; between trade days the multiplier does not change. Row gross never exceeds the base
  row gross (scaled down if it would).
- (b) `ge`, entry gate: when the base book opens coin j (W was 0, is > 0) and f < 1/3 (bottom tercile), the entry is delayed: target 0
  while W stays > 0 and f < 1/3; the base target is taken on the first day f >= 1/3; if W returns to 0 first, the entry is skipped. Never
  adds a trade the base book does not make (a delayed entry is the same trade later).
- (c) `gx`, exit hint: the requested "exit one day early" needs tomorrow's exit decision (look-ahead), so the causal form is: while the
  base book holds coin j, if f is in the bottom decile (f <= 0.1) and its trend score fell today (tscore[d] < tscore[d-1]), the target
  becomes 0 from that day until the base book itself exits; then base again. Each such exit replaces the base exit (one exit, earlier).

## 3. Decision

Per form and cost, the harness class. "Adds value without turnover" requires: class NEUTRAL (+) or INNOVATION at both costs, the mean
d Sharpe above the placebo maximum at both costs, at least 3 of 4 years positive, the ruin view not worse (median worst dd60 not up by
more than 0.5 points), and median turnover change <= +2 % of base. Three forms are tested, so a single NEUTRAL (+) is read as a lead,
not a finding. Nothing is staged in either case.

## 4. Item 1: forward scoring of the frozen checkpoints (registered now, run once the data exist)

- Frozen: F4 checkpoints of `mkt+r24` s0 / s1 and `inc` s0 / s1 (`runs-agent/r24/ckpt/*_F4.pt`, sha256 in `agent/models.json`).
- Data: Binance spot 1 h archives from 2026-11-01, same pipeline and universe; scored once when 2026-11-01 .. 2027-01-31 is complete
  (archives for January appear in early February 2027). No retraining, no other config.
- Rows: hourly task, stride 5, 64 symbols, h = 24. Primary: d rank IC (`mkt+r24` minus `inc`, seed-averaged), paired 7-day block
  bootstrap; confirmed if the 95 % interval is above zero. Secondary: d L/S net at 4 bp (90 days give an interval of roughly +-15 bp, so
  a null there is expected and not a refutation). Also reported against the registered `bmt2-ohlc+cal~8bf4ed9e` and Timestral-2048 if
  its forecasts are produced for the same rows.
