# Benchmark protocol (v1, fixed before any result was produced)

The git history is the record: this file was committed before the first prediction file existed. Changes after that
date are listed under "Amendments" at the bottom with the reason, and never silently.

## 1. Data

- Source: Binance spot 1h klines, monthly archives on `data.binance.vision` (public, no key), 2017-08 .. 2026-10,
  sha256-verified against the archive `.CHECKSUM` files (`data/fetch_binance.py`). No data is stored in git.
- Price = bar close. Gaps inside a symbol's listed life (exchange outages) are forward-filled (no trade = price
  unchanged) and flagged. Hourly log return `r_t = log(c_t / c_{t-1})`.
- Daily bars are derived from the hourly closes at 23:00 UTC (the close of the UTC day).
- Pool `P`: every `*USDT` spot symbol whose first 1h archive month is <= 2022-12, minus stablecoins, fiat, pegged /
  wrapped duplicates and leveraged tokens (list in `data/fetch_binance.py`). Delisted symbols stay in (no survivorship
  filter).
- Evaluation universe `E`: the 64 symbols of `P` with first archive month <= 2020-12 and the highest total quote
  volume over calendar 2021 (a period before every test fold).
- Held-out assets `E_unseen`: symbols of `E` with `int(sha256(symbol)[:8], 16) % 4 == 0`, except BTCUSDT and ETHUSDT
  which are forced into `E_seen`. No learned candidate is ever trained on an `E_unseen` symbol, in any fold.
- Training pool for learned candidates: `P \ E_unseen`, restricted in time per fold (section 2).

## 2. Unseen folds

Walk-forward, expanding window, by calendar period:

| fold | training data strictly before | test period (origins) |
|---|---|---|
| F1 | 2023-01-01 | 2023-01-01 .. 2023-12-31 |
| F2 | 2024-01-01 | 2024-01-01 .. 2024-12-31 |
| F3 | 2025-01-01 | 2025-01-01 .. 2025-12-31 |
| F4 | 2026-01-01 | 2026-01-01 .. 2026-10-31 |

- Purge: the last bar any training target may touch is `train_end - 168 h` (hourly) / `train_end - 10 d` (daily).
- Validation (early stopping, hyper-parameters) = the last 90 days of the purged training period, itself separated
  from the rest of the training data by the same purge. The test period is never used for selection: every
  configuration that was scored on test appears in the leaderboard.
- Every result is reported for two groups: `seen` (asset was in the training pool, test period unseen) and `unseen`
  (asset never trained on AND test period unseen). BTCUSDT and ETHUSDT are also reported alone.
- Zero-shot foundation models (Timestral, Chronos-2) are not retrained; their pretraining corpora and cut-off dates are
  not fully disclosed, so their older folds may be in-sample for them. F4 is the cleanest fold for them.

## 3. Forecast origins, targets, rows

- Hourly task: origins are bars whose epoch hour is divisible by 5 (stride 5 is coprime with 24, so every hour of day
  is sampled equally). Daily task: every UTC day.
- An origin `t` is a row iff: bar `t` and the target bars `t+1 .. t+Hmax` are real (not filled) and inside the test
  period; the symbol has at least 6200 hourly bars (hourly task) / 600 daily bars (daily task) of history at `t`; at
  most 1 % of the last 512 context bars are filled.
- Targets: cumulative log return `y_h = log(c_{t+h} / c_t)`; hourly `h in {1, 6, 24}`, daily `h in {1, 5}`.
- Every model is scored on exactly the same rows. A model that cannot produce a row gets no leaderboard entry for
  that table.
- Context available to a model: anything at or before bar `t` (the protocol standard is the last 512 bars of the
  target symbol; cross-asset models may also use other symbols' bars at or before `t`).

## 4. Forecast format

21 quantiles of `y_h` at levels 0.01, 0.05, 0.10, ..., 0.95, 0.99 per (row, horizon). Point forecast = the 0.5
quantile. Quantiles are sorted before scoring (a crossing model is not rescued in any other way). Models with only a
mean and a variance get Gaussian quantiles.

## 5. Metrics

Per task, horizon and group; `scale_a` = mean |r| of asset `a` over its training period of that fold (the in-sample
one-step naive error of the log-price series, i.e. the MASE denominator).

- `MAE` of the median, pooled, in bp (1e-4 log return). `relMAE` = MAE / MAE(zero-return forecast).
- `MASE` = mean over assets of `MAE_a / scale_a` (naive = random walk on log price).
- `pinball` = mean pinball loss over the 21 levels, pooled, in bp.
- `sPL` (PRIMARY, used for ranking) = mean over assets of `pinball_a / scale_a`: asset-equal, scale-free.
- `CRPS` = `2 * integral of pinball over tau`, trapezoid rule over the 21 levels on [0.01, 0.99], normalised by 0.98,
  pooled, in bp.
- Coverage of the central 80 % (q10..q90), 90 % (q05..q95) and 98 % (q01..q99) intervals; `calib` = mean over the 21
  levels of |P(y <= q_tau) - tau|.
- Directional accuracy `DA` = P(sign(median) = sign(y)) over rows where both are non-zero, and the share of rows with
  a non-zero median. `IC` = mean over timestamps (with >= 10 assets) of the cross-sectional Spearman correlation
  between median and `y`.
- Skill vs a reference `R`: `1 - sPL / sPL_R`. References: `naive` (zero-mean Gaussian random walk with the context
  standard deviation) and `timestral-fast`.
- Per-asset win rate vs a reference: share of assets with lower pinball than the reference.
- Confidence intervals: 95 % percentile intervals from a moving-block bootstrap over CALENDAR blocks (7 days hourly,
  28 days daily), resampling whole blocks jointly across assets (rows at one timestamp are cross-sectionally
  dependent), 1000 resamples, fixed seed. Differences between two models are bootstrapped paired on the same resample.
  "Beats" means the paired 95 % interval of the sPL difference excludes zero.

## 6. Leakage checks (must pass before a learned model's numbers are published)

1. Timestamp audit (asserted at run time and in `tests/`): max training-target bar <= `train_end - purge`; min test
   origin >= `train_end`; no (asset, bar) is both a training target and a test target; `E_unseen` symbols absent from
   every training set.
2. Causality test: changing any bar after the origin leaves every model input of that row bit-identical.
3. Shuffled-time placebo: the returns of each asset are permuted in time (independently inside the training period
   and inside each test period), the learned candidate is retrained and re-scored on the permuted data. Temporal
   structure is gone, so nothing can honestly beat an unconditional forecast. Pass = the upper end of the 95 % interval
   of the candidate's sPL skill vs the `empirical` baseline (context quantiles) is below 1 %. The 1 % allowance is
   for legitimately better tail estimation than 512-sample empirical quantiles; a real leak shows up as tens of
   percent.

## 7. Baselines

`naive` (zero-mean Gaussian, context std, sqrt(h)), `drift` (context mean x h + same spread), `empirical` (context
quantiles; h > 1 from overlapping h-sums), `seasonal-naive` (hourly: same hour yesterday; daily: same weekday),
`ewma` (EWMA variance, lambda chosen on validation, filtered-historical-simulation quantiles), `garch` (GARCH(1,1)-t
refit at each fold start, filtered forward), `autoets` / `autoarima` (StatsForecast; refit every 30 days on the last
2000 bars, Gaussian quantiles), `chronos-2` (official `chronos-forecasting` pipeline in its own process; input = last
512 log prices, native multi-step quantiles minus the last log price; also returns-fed for h = 1), `timestral-fast` and
`timestral-max` (the production `tsfm16` kernel in a separate process: one-step quantiles of the last 256 / flip-averaged
512 log returns clipped to +-0.3, the contract the bitbank.nz service exposes; h > 1 by temporal aggregation = one
step on non-overlapping h-bar returns ending at the origin), `timestral-official` if the reference PyTorch code is
available (log-price input, native multi-step).

## 8. Speed

Per model, on this host: rows/s and p50 / p99 latency per call at batch 1, 64, 4096 (>= 30 timed calls after 5
warm-up calls; one row = one 512-bar context -> all horizons x 21 quantiles), peak VRAM of the process, weights and
binary size. GPU = one RTX 5090 shared with production servers (numbers are lower bounds and vary +-30 % with the
co-tenants' load, the load is recorded next to each run). CPU = 1 thread and 12 threads. Host-device transfer is
included, data loading is not.

## Amendments

(none yet)
