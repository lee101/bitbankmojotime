---
# DRAFT model card, not published. Fields marked TBD wait for owner decisions (docs/PLAN.md section 7).
license: cc-by-nc-sa-4.0          # research tier (trained on Binance Vision archives). Permissive tier: TBD
pipeline_tag: time-series-forecasting
tags: [time series, forecasting, quantile, crypto, mojo]
---

# bmt (working name): compact quantile forecaster for financial returns

A small patch transformer that maps the last 512 log returns of one series to 21 quantiles (0.01, 0.05 .. 0.95, 0.99) of the
cumulative log return at horizons 1, 5, 6 and 24 bars. Trained directly on returns, inference in Mojo on CPU or GPU.
More at https://bitbank.nz (TBD: wording).

| size | parameters | layers x width | weights |
|---|---|---|---|
| t | 0.11M | 2 x 64 | 0.4 MB |
| s | 0.81M | 4 x 128 | 3.2 MB |

## Intended use and limits
- Research and evaluation of probabilistic return forecasts (volatility and tail shape). Not investment advice.
- The forecast is about spread and tails. The median is only marginally better than zero (hourly relMAE 0.997, directional
  accuracy 51-53 %, rank IC 0.04-0.065 on unseen folds; not separated from noise on daily bars) and has not been tested
  against trading costs.
- Research-tier weights are non-commercial (CC BY-NC-SA 4.0, derived from Binance Vision data): no live trading, no paid
  signals, no commercial products.
- Evaluated on hourly and daily Binance spot USDT pairs, 2023-2026. Not evaluated on stocks, FX or other bar sizes.

## Input and output
- Input: `r[N, 512]` float32 log returns, oldest first, bars of any constant size. Scale-free: the model divides by the context RMS.
- Output: `q[N, 4, 21]` float32 quantiles of `log(P[t+h] / P[t])`, h = 1, 5, 6, 24 bars; never crossing.
- Optional flip averaging (2 passes) for a symmetric forecast.

## Training
- Data: Binance spot 1h klines from `data.binance.vision`, 333 USDT pairs, resampled to 1h / 4h / 24h bars; 17 evaluation symbols
  are never trained on. Walk-forward: one model per test year, trained on data ending 104 days before the test period.
- Loss: pinball over the 21 levels in scale units; mirror augmentation; 1,500 steps of batch 1024, AdamW, weight EMA.
- The published file is the fold-F4 model (training data before 2025-09-19). TBD: retrain on all data for release.

## Evaluation
`bench/LEADERBOARD.md` at commit TBD: asset-equal scaled pinball with block-bootstrap 95 % intervals on unseen periods and unseen
assets, against naive, EWMA, GARCH, AutoETS / AutoARIMA, Chronos-2 and Timestral-209M, plus the speed table and the leakage checks.

## Files
`bmt.bin` (16 int32 header + float32 tensors), `selftest_in.f32`, `selftest_ref.f32` (fixed input and PyTorch fp32 reference;
`bmt selftest <dir>` must report PASS, tolerance 2e-4 scale units).

## Attribution
Market data: Binance Vision, CC BY-NC-SA 4.0. Not affiliated with or endorsed by Binance. Baselines: Timestral-209M (Mistral AI),
Chronos-2 (Amazon), StatsForecast (Nixtla).
