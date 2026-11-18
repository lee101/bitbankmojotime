# Pre-registration: does a longer forecast horizon (72 h, 168 h) rank better and improve the trend book beyond the 24 h `gt` tilt

Date 2026-11-18 (UTC). Branch `agent/horizon` (from `agent/search` 42f753c). Committed before any training run of the new models, any
forecast from them and any harness run of the forms below. Code: `train/rankh.py` (training, predictions), `tools/hzstep0.py` (step 0,
CPU), `tools/hzgate.py` (steps 1-2). Outputs `runs-agent/hz/`. Data: the existing Binance kline cache (owner decision 2026-11-18:
research tier may be trained on freely); nothing is fetched. Steps run in order and stop at the first registered stop.

Incumbent: the frozen `mkt+r24` checkpoints (`runs-agent/r24/ckpt`, seeds 0 / 1) and the confirmed form `gt` (PREREG-GATE (a),
PREREG-GATE-CONFIRM form 1). Called `gt24` below.

## Models (fixed list: two)

`mkt+r72`, `mkt+r168`: the `mkt+r24` recipe with the ranking horizon H = 72 or 168 hours. Unchanged: inputs ohlc+cal+mkt, size s,
1,500 steps, i.i.d. batch 1024 with bar sizes 1 h / 24 h at 0.7 / 0.3 plus a grouped batch of 32 origins x 32 symbols on 1 h bars,
AdamW lr 1e-3, cosine, EMA 0.998, final EMA weights (no checkpoint selection), seeds 0 and 1, the four folds F1-F4 and their training
limits. Changed, and nothing else:
- the horizon set is (1, 5, 6, 24, H) bars; slot H exists for 1 h bars only (24 h-bar rows carry no loss at that slot);
- the correlation term (weight 0.1) is at slot H: within-origin Pearson correlation between the H-hour median in return units and
  the realised H-hour log return winsorised at +-10 model sigma (sigma x sqrt(H)). No other slot has a ranking term (the 24 h slot is
  pinball only).
Overlapping labels: every 1 h-bar training row needs its H-hour label to end before the fold's training limit (limit = test start
- 168 h - 90 d - 168 h), so the last training label ends at least 168 h (>= H for both models) before the first validation origin and
2,496 h before the first test origin. Validation labels end at least 24 h before the test start. Labels of neighbouring origins overlap
inside the training set (sampling is i.i.d.; the ranking term is cross-sectional inside one origin).

## Step 0: forecast quality (GPU for training and prediction, then CPU)

Scores: s24 = 24 h median of `mkt+r24`; s72 = 72 h median of `mkt+r72`; s168 = 168 h median of `mkt+r168`. Descriptive extra rows:
each long model's own 24 h slot, and the equal-weight rank blends s24 + sH.
Row sets:
- V: the PREREG-24H validation rows (four folds' 90-day windows, seen evaluation symbols, every hourly origin).
- T: the benchmark's hourly test rows on the unseen folds (stride 5, 64 symbols, F1-F4).
- B: the bots' panel: daily 00:00 UTC origins 2023-01-01 .. 2026-10-31 from each date's fold model, the 45 non-PAXG coins of the
  harness archive (CORE, R1-R24 coins), seed-averaged median as the trading forms use it.
Realised return at horizon h in (24, 72, 168): log close(t + h) - log close(t) from the cache; a row without a bar h hours ahead is
dropped for that h. Rank IC = per-origin Spearman (origins with >= 10 symbols), mean over origins; on V and T the per-origin IC is
averaged over the two seeds (PREREG-24H convention), on B the seed-averaged median is ranked. Intervals: paired calendar-block
bootstrap with 28-day blocks (four times the longest label, so overlapping labels stay inside a block except at its edges), 1,000
draws, the same draws for every score. V touches fewer than 20 such blocks: its intervals are indicative.
Persistence on B: lag-1-day and lag-7-day rank autocorrelation (Spearman between the cross-sections of day d and day d - k on common
coins) and the implied turnover of a daily rank tilt = mean over coins and days of |f(d) - f(d-1)|, f = rank fraction in [0, 1] (the
tilt multiplier is 0.5 + f, so this is the one-way daily change of an equal-weight book in units of its gross).
Sanity checks (not gates): s24 reproduces the registered validation IC (0.0745) and test IC (0.0845) and the stored bot forecasts.
Also reported, no gate: validation sPL at h 1 / 6 / 24 of the long models against `mkt+r24`.

Proceed rule, per H in (72, 168); dIC_H = IC at horizon H of sH minus IC at horizon H of s24:
- (a) ranks better: the 95 % interval of dIC_H on T is above zero and the point dIC_H on V is positive; or
- (b) more persistent at equal IC: the IC at horizon H of sH on T has its interval above zero, the interval of dIC_H on T is not
  entirely below zero, and on B the lag-1-day rank autocorrelation of sH exceeds that of s24 by at least 0.10 with the paired
  interval of the difference above zero.
If neither horizon proceeds: stop, registered null (no harness run). If both proceed, one horizon H* is carried: the larger mean of
the B-panel ICs at 72 h and 168 h of its own score. The other horizon gets one descriptive arm in step 1 (`gt` with its forecast, no
placebo, cannot be picked). Minimum detectable dIC (80 % power, two-sided 5 %) = 2.8 x bootstrap SE, reported with the table.

## Step 1: trading lead search (CPU)

ALGOS pooled harness as in PREREG-GATE: 14 universes CORE + R1-R8, the 3,000-configuration family, costs c7 and c10, statistic
`al.pooled` restricted to 2023-01 .. 2026-10. Two reference arms simulated by the same binary on the same configurations: `base`
(untilted) and `gt24`. f = rank fraction in [0, 1] of the forecast among the universe's non-PAXG coins with a forecast.
Four forms with H = H*:
1. `gtH`: `gt` (identical code and parameters, replica band 0.05, rows scaled down only) with sH instead of s24.
2. `gtB`: `gt` with the blend rank: f = rank fraction of (f24 + fH) / 2.
3. `dtH`: the daily continuous tilt of PREREG-24H section 5.2 (the form that failed for 24 h): held weights x (0.5 + fH), row
   gross kept (`al.mult_post`).
4. `dbH`: daily tilt with a no-trade band: coin j's multiplier is set to 0.5 + fH on the first day the base book holds it and is
   reset to the current 0.5 + fH only on days when it differs from the held multiplier by more than 0.25; forgotten when the base
   weight is 0; row gross kept (`al.mult_post`).
Placebo: 10 draws per form, sH permuted across coins within each day (for `gtB` only the H component; the 24 h ranks stay).
Reported per form and cost, paired against BOTH references: mean d Sharpe with month-bootstrap 95 % interval, share of
configurations improved, d bp/day, per-year d Sharpe, universes positive, placebo range, class by `al.klass` (vs base), median
turnover change vs base and vs `gt24`, ruin view on M11 (vs base).
Pick rule: a form is eligible iff at both costs its paired d Sharpe over `gt24` has the interval above zero, the mean above every
placebo draw of that form (measured against `gt24`) and at least 80 % of configurations improved over `gt24`. The eligible form with
the largest c7 mean d Sharpe over `gt24` is picked. No eligible form: stop, registered null (the incumbent `gt24` stays).

## Step 2: confirmation (only if a form is picked)

16 new universes R25-R40, never built or simulated before: the R9-R24 procedure (BTC, ETH, PAXG + 7 coins: 4 by `rng.choice`
without replacement from the 15 archive coins never used in R1-R8 in alphabetical order, then 3 from the remaining non-M11 archive
coins in order of first archive bar; this reproduces all 16 R9-R24 sets from rng [20261115, 9, i]) with rng [20261118, 25, i],
i = 0..15, none equal to an R1-R24 set:
R25 ATOM AVAX COMP DASH ONE SAND UNI | R26 AAVE DOT ICX IOST UNI ZEC ZRX | R27 AVAX DASH ENJ IOTA ONE QTUM SOL |
R28 BAND ICX MANA ONE SAND SOL ZEC | R29 AAVE AVAX IOTA MANA ONT UNI ZEN | R30 BAT COMP ENJ ONE QTUM ZEN ZRX |
R31 ATOM AVAX BAND BAT DOT IOTA ZEC | R32 AAVE BAT DOGE SOL TFUEL VET ZEC | R33 COMP DOT IOTA ONT SAND ZEC ZEN |
R34 ATOM DOT NEO ONE UNI ZEC ZIL | R35 AVAX COMP ICX IOTA SAND ZEN ZIL | R36 ATOM BAT DOT IOTA ONE UNI ZEN |
R37 AVAX BAND CHZ COMP ENJ IOTA ZEC | R38 AAVE CHZ IOTA ONE QTUM SOL ZEN | R39 BAT CHZ DOT IOST NEO QTUM UNI |
R40 AVAX ONE ONT QTUM SNX UNI XTZ.
Built with `bgm-signal-wt/tools/shift/prep.py` unchanged into our own data directory (read-only links to the existing archive); a
universe that fails to build is dropped and reported. Limit of "new": the same 46-coin archive and period as R1-R24; only the
combinations are new (R9-R24 have now been used in three waves, hence the new draw).
Frozen picked form, `base` and `gt24` on the same binary, 20 placebo draws. Bar (all three): class INNOVATION against `base` at c7
and c10 (`al.klass` with the placebo band); paired d Sharpe over `gt24` with the 95 % interval above zero at both costs; at least
12 of 16 universes with positive family-mean d Sharpe over `gt24` at both costs. Reported, not gated: the risk-matched arm (tilted
row rescaled to the base row's gross every day, against `base` and against risk-matched `gt24`), turnover, per-year, the ruin view
on M11 (a lead universe: descriptive). Minimum detectable paired gain over `gt24` (80 % power) = 2.8 x the month-bootstrap SE of
that paired statistic; the step-1 SE is stated before step 2 runs and the step-2 SE with the result.

## Limits

GPU: the repo's wrapper path (`bmtbench.gpu`: wait for head-room, cap 3.2 GB, blackout 15:20-16:15 America/New_York on weekdays),
retry after out-of-memory, no signal to any other process. CPU <= 12 threads, nice 10. Artifacts under 10 GB; resume caches deleted
once the result JSONs exist. Research evaluation only: nothing is staged or deployed; the harness's own directories are read only.
