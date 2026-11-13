---
# DRAFT, not published. Licence intended, owner to confirm.
license: apache-2.0
pipeline_tag: time-series-forecasting
tags: [time series, forecasting, quantile, synthetic data, mojo]
---

# bmt2 permissive tier (`bmt2-syn-ohlc+cal~0ddf348c` size s, `bmt2-syn-ohlc+cal@t~2c0a3b61` size t, `bmt2-syn-ohlc+cal-ctx256@t~b5523fb3` 256-bar context)

The same model as the research tier (512 bars of log return, log high/low range and Garman-Klass sigma + start hour and weekday
-> 21 quantiles at 1, 5, 6, 24 bars), trained ONLY on synthetic bars. This is the tier a trading bot or a paid API may use.

| file | parameters | sha256 (first 16) |
|---|---|---|
| `weights/bmt2-syn-ohlc+cal@t~2c0a3b61/bmt2.bin` | 0.11M (2 x 64), 512 bars | 6320d4473389200f |
| `weights/bmt2-syn-ohlc+cal~0ddf348c/bmt2.bin` | 0.82M (4 x 128), 512 bars | fa04d2ba4bfdba8c |
| `weights/bmt2-syn-ohlc+cal-ctx256@t~b5523fb3/bmt2.bin` | 0.11M, 256 bars (for consumers with short candle history) | a5a9b54d52a1cf2c |

## What touched the weights
- Training data: `train/synth.py`, a domain-randomised simulator (multi-scale stochastic volatility with leverage and jumps,
  Student-t returns, intraday / weekend seasonality, tick-size rounding, MA(1) bounce, high / low from Brownian-bridge extremes).
  Every parameter is drawn from broad priors written down from textbook stylised facts. No parameter was fitted to a dataset.
- Checkpoint selection: a held-out synthetic set. The simulator was written once and trained once; its priors were not tuned
  against market data afterwards.
- No market data of any exchange or vendor entered training.

Two things a reader should know before calling it clean:
1. The ARCHITECTURE and INPUT LIST (range + calendar) were chosen by a search validated on Binance Vision data. The weights hold
   no such data, but the design decision was informed by it.
2. The accuracy numbers below were measured on Binance Vision data; those tables are non-commercial derived data. The weights are not.

## Licence
Intended Apache-2.0 (as the code). The owner confirms before any use or release.

## What the tier costs (same unseen folds, paired 95 % intervals, `agent/LEADERBOARD.md`)
| hourly h 1 / h 6 / h 24, daily h 1 / h 5 | sPL skill of `bmt2-syn-ohlc+cal-sym~0ddf348c` |
|---|---|
| vs the research tier | -0.38 % [-0.48, -0.27] / -0.30 % / -0.54 % [-0.70, -0.38], -0.54 % / -0.69 % [-1.32, -0.01] |
| vs Timestral-209M reference pipeline | +0.66 % [0.47, 0.86] / +0.45 % / +0.58 %, +0.11 % [-0.22, 0.43] / -0.35 % [-0.82, 0.07] |
| vs Timestral, 2,048-bar context (hourly) | +0.53 % / +0.13 % / -0.18 % [-0.33, -0.02] |
| vs GARCH(1,1)-t | +0.96 % / +0.69 % / +0.53 %, +0.56 % / +0.41 % |
| vs the production Timestral kernel (fast) | +1.48 % / +1.51 % / +1.13 %, +0.81 % (daily h 1) |

The synthetic tier gives up 0.3-0.6 % hourly and 0.5-0.7 % daily against the research weights and still beats the production
forecast by 1.1-1.5 % hourly. The size-t file scores the same within 0.05 %. The 256-bar file costs a further 0.1-0.2 %
(vs research -0.45 % / -0.44 % / -0.76 % hourly; vs the production kernel +1.41 % / +1.38 % / +0.92 %).

## Other data routes considered for this tier (none used)
| source | status |
|---|---|
| Candles the bots collect themselves from Poloniex public endpoints | terms not established (the terms page could not be retrieved); not used |
| Candles collected from KuCoin public endpoints | KuCoin Terms of Use Art. 90-92 forbid commercial use of, derivatives of and systematic collection of platform content, and no API licence was found; treated as not cleared |
| Binance | Vision archives are non-commercial; live API terms not checked from this host |
| Public time-series corpora | per-subset licences not checked; none used |

Self-test: `bmt2 selftest <dir>` (measured 1e-5 CPU, 4e-5 GPU against the 2e-4 tolerance).
