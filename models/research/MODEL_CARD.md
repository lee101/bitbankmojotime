---
# DRAFT, not published.
license: cc-by-nc-sa-4.0
pipeline_tag: time-series-forecasting
tags: [time series, forecasting, quantile, crypto, mojo]
---

# bmt2 research tier (`bmt2-ohlc+cal`, sizes s and t)

Patch transformer that maps the last 512 bars of one series (log return, log high/low range, Garman-Klass sigma) plus the start
hour and weekday of the next bar to 21 quantiles (0.01, 0.05 .. 0.95, 0.99) of the cumulative log return at 1, 5, 6 and 24 bars.
Inference in Mojo on CPU (`build/bmt2`) or GPU (`src/bmt2_gpu.mojo`). More at https://bitbank.nz.

| file | parameters | sha256 (first 16) |
|---|---|---|
| `weights/bmt2-ohlc+cal@t/bmt2.bin` | 0.11M (2 x 64) | b628c1ae7a64afb1 |
| `weights/bmt2-ohlc+cal/bmt2.bin` | 0.82M (4 x 128) | 443fcf460eef9552 |
| `weights/bmt2-ohlc+cal@t-kd/bmt2.bin` | 0.11M, distilled from the 4-seed ensemble (recommended single model) | 530ba3365d36e5ed |
| `weights/bmt2-ohlc+cal@t-ens4/s{0,1,2,3}/bmt2.bin` | 4 x 0.11M, seed ensemble (s0 = the single model) | b628c1ae7a64afb1, ddb7f956e3617279, d241d69b60da3b7d, eb3c06cc35a68d7c |

The distilled model scores the same as the ensemble on the unseen folds (+4.56 % vs +4.57 % sPL against the random walk, hourly h 1)
at one forward per row. The ensemble is run by passing the four files comma-separated (`bmt2 serve s0/bmt2.bin,s1/bmt2.bin,...`); the forecast is the
quantile average. On the unseen folds, flip-averaged, it adds 0.04-0.06 % sPL over the single seed at hourly h 1 / 6 / 24 (paired
intervals exclude zero; daily not separated): +1.12 % [0.94, 1.31] / +0.81 % / +1.17 % against the Timestral reference pipeline.

## Licence and limits
- CC BY-NC-SA 4.0. Trained on Binance Vision archives, which are non-commercial and whose terms cover derived models: no live
  trading, no paid signals, no commercial product. Attribution: Binance Vision. Not affiliated with or endorsed by Binance.
- Research and evaluation only; not investment advice. Evaluated on hourly and daily Binance spot USDT pairs, 2023-2026.
- The forecast is about spread and tails. The median has a small rank signal (cross-sectional rank IC about 0.06) that shows no
  robust edge after trading costs in the tests of `bench/LEADERBOARD.md`.

## Training
333 USDT pairs (17 evaluation symbols never trained on), 1 h and 24 h bars, one model per walk-forward fold (training data ends
104 days before the fold), pinball loss in scale units, mirror augmentation, 1,500 steps of batch 1024, weight EMA. Inputs were
chosen by a validation-only search (`docs/RESEARCH_LOG.md`). The shipped files are the fold-F4 models (data before 2025-09-19).

## Evaluation (unseen folds, paired 95 % intervals, `bench/LEADERBOARD.md`)
Hourly sPL skill of the size-s model with flip averaging: +1.04 % [0.88, 1.21] / +0.75 % / +1.12 % (h 1 / 6 / 24) against the
Timestral-209M reference pipeline, +0.91 % / +0.43 % / +0.36 % against Timestral with a 2,048-bar context, +1.0 to +1.3 % against
GARCH(1,1)-t; daily h 1 +0.65 % [0.40, 0.90] against Timestral. Shuffled-time placebo: pass at every horizon.
Self-test: `bmt2 selftest <dir>` (tolerance 2e-4 scale units; measured 2e-6 CPU, 9e-6 GPU).
