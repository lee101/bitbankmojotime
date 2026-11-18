# Pre-registration: the 168 h forecast as a weekly SELECTION signal on a wide point-in-time universe (track FSEL)

Date 2026-11-18 (UTC). Branch `agent/horizon`. Committed before any forecast for the wide universe, any book and any statistic of
this test. Follows wave 5 (`agent/PREREG-HORIZON.md`): `mkt+r168` ranks at IC 0.106-0.115 with lag-1-day rank autocorrelation 0.57,
and every form tested there used it as a multiplier on the M11 trend book. Here it selects names, at a holding period equal to its
horizon. Code `train/rankh.py wide`, `tools/fsel.py`, `tools/fsel_trend.py`; outputs `runs-agent/fsel/`; evidence `agent/fsel/`.
Research only: nothing is staged or deployed; every input directory is read only.

## 0. What was read first (so known failures are not repeated)

binancegomojo ledger and failedexperiments; UNIV. Wider point-in-time universes were worse for the trend book (UNIV: -0.27 to
-0.68 Sharpe; the average liquid alt loses 2-3 % a month; M11 carries a hindsight premium of about 0.27). Trailing-return
selection is no better than a random coin over the same spells (MOM1); relative strength mean-reverts (RSUNI, RELSTR); low-vol /
trailing-Sharpe universe picks have no IC for the trend-cell score (DYNUNI); a low-vol sleeve was bull beta (LVBH, H32); learned
cross-sectional rankers had IC 0.09-0.12 that a circular-shift placebo reproduced, i.e. a static asset effect, and their
re-weighting of the held set failed (RERANK, HRK1); weekly cadence of the trend book lost (CADENCE). Consequences here: the
circular-shift placebo is registered next to the two requested ones; momentum ranks are the baseline to beat; selection is tested
as selection (who is held), not as re-weighting of a held set.

## 1. Data, universe, forecasts

- Universe: the UNIV construction, read only (`/sdb-disk/tradingbots/binance/univ`, `bgm-univ-wt` `tools/univ/uv.py`): 699 Binance
  USDT spot names from the public archive with delisted pairs, stable / wrapped / leveraged names removed, ticker reuse split into
  segments; ranking on the 1st of each month from earlier daily bars only; eligible = listed >= 180 days, >= 27 bars in the last
  30 days, a bar on the previous day; members(N) = top N by median daily quote volume of the last 30 days; a member is tradable on
  decision day d if it had a daily bar on day d - 1. N in (30, 50); the stored union panels X30 / X50 (no gold column) on the M11
  hourly grid. No hand-picked list.
- Forecasts: `mkt+r168` (primary; 168 h median, seed mean) and `mkt+r24` (comparison; 24 h median), out of sample only: each
  date's fold model (training labels end >= 2,496 h before the fold's first origin), 2023-01-01 .. 2026-10-31, origin = the hourly
  bar closing at 00:00 UTC. 79 of the 222 top-50 names of this window are not in the forecaster's kline cache (listed after
  2022-12); their hourly bars come from the UNIV hourly archive (the same public source; bars identical where both exist), cut at
  the cache's last hour. They are never in the training pool or the market-state series. A member without a forecast at a decision
  is not ranked that week; coverage is reported. Prediction only, no training.
- Baseline ranks: momentum over 7 and 28 days (log return of the decision-time price). Comparison rank: `mkt+r24`.

## 2. Schedule and segments

Weekly decisions on Mondays 00:00 UTC from 2023-01-02; the forecast origin is the bar that closes at the decision time (index
strictly before the decision bar); membership uses days before the decision day. **Lead segment** 2023-01-02 .. 2025-05-01 00:00
(28 calendar months): every choice below is made on it, from inputs cut at its end. **Confirmation segment** 2025-05-01 ..
2026-11-01 00:00 (18 months): opened once, for the single picked candidate, after the pick and the minimum detectable effect are
committed.

## 3. Forms (five, fixed), each for N in (30, 50) and K in (5, 10)

Top-K = the K tradable members with the highest rank value.
1. `sel`: weekly long-only top-K, equal weight, as a standalone book under the producer's risk rules: target gross
   g = min(pv / sigma, cap), sigma = annualised stdev of the last 60 daily returns of the equal-weight basket about to be held
   (a book that starts in 2023 has no realised history, so the basket's trailing vol replaces the producer's own-book vol), live
   settings pv 0.25, cap 1.875; regime gate = the producer's bear rows (equal-weight index of the N members below its 200-day mean:
   pv 0.1875, cap 1.5625). Rebalanced to target every Monday, no per-name stop, no trade between Mondays.
2. `tsel`: the trend book restricted to the forecast top-K: the UNIV masked-signal trend arm (trend-wf selector, live
   configuration, daily decisions, stops, cooldown, band, vol target, bear rows from the N-member index) with eligibility = tradable
   member AND in the top-K of the latest Monday. Trend votes decide exposure, the forecast decides eligibility. Before 2023-01-02
   eligibility is plain membership (selector and vol-target warm-up, not evaluated).
3. `blend`: 50 / 50 daily-rebalanced mix of form 1 (same N, K) and the live book.
4. `selh`: form 1 with rank hysteresis: a held name stays while its rank is within the top 2K; free slots (K in total) are
   filled from the best-ranked names not held.
5. `ls`: +0.5 / K on each top-K name and -0.5 / K on each bottom-K name, weekly, no risk rules: an information check only (not
   tradable as registered: borrow, squeezes), with the long leg and the short leg each measured against the equal-weight
   universe. It cannot be picked.
Simulation: forms 1, 3, 4, 5 on daily decision-time prices of the union panel (weights drift between Mondays; cost = rate x traded
notional; 10 % a year on gross above 1); the engine is checked against `bgm trend-sim` on one weight file (reported). Form 2 and
the live book run through the producer harness (`trend-wf` / `trend-sim`). A delisted name is marked at its last price.
Live book (reference): M11 trend book with the 24 h trade-day tilt, risk budget x 1.10 (`gt` at k 1.10, PREREG-REGEAR), live
Binance settings (pv 0.25, cap 1.875, stop 6 %, cooldown 72 h, band 0.05, MA200 bear rows, top 10), frozen `mkt+r24` forecasts.
Costs: 7 and 10 bp per unit traded on both sides of every comparison; stress: the form at 20 bp against the live book at 10 bp.

## 4. Placebos (20 draws each, per form 1 / 2 / 3 / 4 and (N, K); seeds [20261118, 6, family, draw])

- shuffled ranks: the forecast values permuted across the tradable members, independently every Monday;
- random K names: one persistent random key per name; the form is driven by the key (changes only when membership changes);
- circular shift (the RERANK lesson): the forecast panel shifted in time by a random number of weeks (at least 13 from either
  end, wrapping inside the segment), names kept: static preferences and rank persistence stay, timing goes.
For form 2 (each arm is a trend-wf run) placebos are run only for candidates that meet conditions (a), (c), (d) of section 6;
the others cannot become eligible.

## 5. Statistics

Per book and segment: Sharpe (daily, x sqrt 365), CAGR, worst calendar month, worst 60-day drawdown (any window up to 60 days),
max drawdown, turnover (traded notional per year in units of equity), mean gross. Paired against the live book: d Sharpe and
d CAGR with a month-block bootstrap (2,000 draws of calendar months, the same draws for both books), 95 % interval. Shown beside
every forecast form: the same form with `mkt+r24`, with momentum 7 d / 28 d, and the three placebo ranges. Descriptive: weekly
rank IC of each rank against the next-week return on the members, with its circular-shift value; names per week with a forecast.
Capacity note: at 5,000 USDT equity, order notional against the name's 30-day median daily quote volume and against the 5.5 USDT
minimum order, for the picked candidate (if none: form 1, N 50, K 10).

## 6. Pick rule (lead segment only, `mkt+r168` rank, forms 1-4: 16 candidates)

A candidate (form, N, K) is eligible iff at 7 bp AND at 10 bp:
(a) its Sharpe is above the live book's (paired d Sharpe > 0) and its worst calendar month is not worse than -30 %;
(b) its Sharpe is above every draw of each of the three placebo families of the same form, N, K;
(c) its Sharpe is above the same form, N, K driven by momentum 7 d and by momentum 28 d;
(d) at 20 bp its Sharpe is above the live book's at 10 bp.
The eligible candidate with the largest d Sharpe against the live book at 10 bp is picked. No eligible candidate: stop,
registered null at the lead search; the confirmation segment stays closed. 16 candidates are searched without correction; the
confirmation segment is the guard.

## 7. Confirmation (picked candidate, frozen; only after a commit that states the pick and the minimum detectable effect)

Minimum detectable d Sharpe on the confirmation segment (80 % power) = 2.8 x (lead bootstrap SE of the picked candidate's d Sharpe)
x sqrt(28 / 18), stated before the segment is read; if it exceeds the lead's d Sharpe, that is said then: the segment cannot
resolve an effect of the lead's size and the strongest possible verdict is "lean".
On the confirmation segment, at 7 and 10 bp:
- CONFIRMED iff d Sharpe against the live book has its 95 % interval above zero, the Sharpe is above every draw of the three
  placebo families and above both momentum versions, the worst month is not worse than -30 %, and (d) holds;
- LEAN iff all of that holds with the point d Sharpe > 0 in place of the interval;
- otherwise NOT CONFIRMED.
Reported for the picked candidate, not gated: the other six weekday phases of the weekly schedule (CADENCE lesson: phase luck),
the live stop cell applied to forms 1 / 4, full-window numbers.

## 8. Leak checks (run with the lead search, reported; any failure stops the run)

1. Origin: for every ranked (name, decision) the forecast's origin bar closes exactly at the decision time.
2. Membership: the ranking of a sample month recomputed from daily bars truncated at the ranking day equals the stored ranking.
3. Truncation: (i) forecasts at three decision dates recomputed from a feature bank whose every value after the origin is
   replaced by noise are identical; (ii) form-1 weights up to a date recomputed with every input after that date removed are
   identical; (iii) if the confirmation segment is opened, the full-window run reproduces the lead-segment books exactly.
Sanity: the wide prediction pass reproduces the wave-5 bot-panel forecasts of the 45 archive coins.

## 9. Limits

One 46-month history; Binance bars only; the 2023-26 window was used by waves 4-5 for the M11 forms, so only the wide universe and
the selection use are new. CPU <= 12 threads, nice 10; GPU through the repo wrapper, prediction only; artifacts small and resume
caches deleted after the result files exist. Anything added after results is labelled post hoc.
