# Pre-registration: `mkt+r24` retrained on the owner's venue API bars (KuCoin 1 h klines), tilt test on R9-R24

Date 2026-11-18 (UTC). Committed before any KuCoin or OKX bar is downloaded, before any training on them and before any harness
run. Follows `agent/PREREG-LIC.md` (wave 4d: Kraken retrain kept the forecast quality, tilt NEUTRAL because Kraken lists 21 of the 30
R9-R24 coins). Seen before this commit: the KuCoin and OKX pair lists (`/api/v2/symbols`, `/api/v5/public/instruments`; metadata, no
prices), the endpoints' HTTP status, and the terms pages. Code committed with this file: `tools/kucoin_cache.py`, `tools/lic2.py`,
`train/rank24.py botpred --end` (new option, default path unchanged). Recipe and forms unchanged: `train/rank24.py`,
`tools/r24gate.py`. Outputs `runs-agent/lic2/` (git-ignored, never published).

## Source

KuCoin public REST `GET /api/v1/market/candles?type=1hour` (no key, no signed call), USDT spot pairs. The owner already trades on
KuCoin through its API. Second source, only for harness coins KuCoin does not serve (secondary S3): OKX public REST
`GET /api/v5/market/history-candles?bar=1H`. Fetch etiquette: 4 connections, 5 requests/s in total, shared exponential back-off on
429, 1,400-hour windows (under the 1,500-row cap; a full 1,500-row answer aborts), resumable per pair, generic User-Agent.

## Data mapping (mechanical, fixed now)

- Pair list: the `/api/v2/symbols` snapshot taken when the fetch starts; `quoteCurrency == USDT`; canonical key `<baseCurrency>USDT`
  (a name only, as in PREREG-LIC); bars requested by `symbol`. At the time of writing 808 USDT pairs; of the 30 R9-R24 coins RVN is
  not listed, of the 36 ALGOS coins RVN and ICX are not. For harness coins missing from the list the candle endpoint is asked for
  `<BASE>-USDT` anyway and whatever it serves is used under the same rules.
- Excluded bases: the PREREG-LIC stablecoin / gold / fiat sets, the Binance fetcher's exclusion list (pegged and wrapped
  duplicates), KuCoin's market "Stocks", leveraged-token names (`3L` / `3S` style or an UP / DOWN / BULL / BEAR suffix on a base longer
  than 4 characters), bases outside `[A-Z0-9]+`.
- Bars: rows `[open time, open, close, high, low, base volume, quote turnover]`; `t` = epoch hour of the open time; `qv` = turnover;
  taker-buy share unknown, set to 0.5; trade count unknown, set to 0 (no arm uses either). Only completed bars: open time before the
  hour in which the fetch started (`dl/state.json: end`). Same row filter as PREREG-LIC (close > 0, high >= low > 0, one bar per
  hour). Hours without a bar are missing bars (KuCoin publishes none when there is no trade).
- First bar: located with one weekly-candle request; hourly windows start 8 days before the first weekly bar.
- Liquidity filter, as PREREG-LIC: a pair enters the pool iff its first bar is on or before 2022-12-31 and at least 50 % of the hours
  from its first bar to 2022-12-31 (last 365 days of that span at most) have a bar. To save requests the fetcher downloads the 2022
  window first and the rest only if the 2022 fill is at least 45 % (margin) or the coin is a harness coin; the exact 50 % filter is
  applied when the cache is built.
- Universe by the protocol's rule (`bmtbench.data.build_universe`): pool = the filtered pairs; evaluation set = the 64 highest 2021
  quote volumes among pairs listed by 2020-12; held-out = the same sha256 rule; train = pool minus held-out.
- Known limit, stated now: the pair list holds only pairs KuCoin still lists, so pairs it delisted are missing from the pool
  (survivorship in the training pool and in the KuCoin validation IC). The harness test is not affected: fixed universes, the
  SIGNAL books and returns, read only.
- No pair is added or dropped because of the sanity comparison below; it is reported only.

## Training (identical recipe)

`train/rank24.py train --arms mkt+r24 --seeds 0,1` with `BMT_DATA=runs-agent/lic2/cache`, `BMT_UNIVERSE=runs-agent/lic2/universe.json`,
`BMT_R24_OUT=runs-agent/lic2/r24`: same folds F1-F4, same training limits, 1,500 steps, final EMA, loss weight 0.1, same model size,
batch 1,024 plus 32 x 32 groups. Nothing is tuned. GPU cap 3.2 GB, the head-room wait and the 15:20-16:15 ET blackout apply; no
other GPU process is touched. If the card cannot host the run: wait and retry. The only permitted change, after 30 minutes of
failing, is batch 512 (groups unchanged), marked as a deviation in the table. If it still cannot run, that is reported as blocked
and the secondaries (prediction only) are still run.

## Forecast for the harness

`train/rank24.py botpred --arms mkt+r24` for the 30 R9-R24 coins and the 36 ALGOS coins that are in the KuCoin pool, daily 23:00 UTC
origins 2023-01-01 .. 2026-10-31, inputs from KuCoin bars only (the live-use situation). A forecast needs a bar at the origin hour
and 6,200 h of history; coins or days without a forecast get multiplier 1 in the forms.

## Test (same harness, same bar as PREREG-GATE-CONFIRM and PREREG-LIC)

R9-R24 (`GATE_UNIS=CONF`), 2023-01 .. 2026-10, costs c7 / c10, forms `gt` (frozen, co-primary 1) and `gtr` (risk-matched,
co-primary 2), 20 placebo draws each (same seeds as `tools/r24gate.py`), base arms from a private copy of
`runs-agent/r24/confirm/{off,turn,u}` (same binary, same configs).

Bar, unchanged: a co-primary passes iff at both costs it is INNOVATION under `al.klass` with its 20-draw placebo band, at least 10 of
16 universes have positive family-mean d Sharpe, and the turnover change is <= +2 %. "Survives" if either co-primary passes; "lean"
if either has the d Sharpe interval above zero at both costs without passing; otherwise "does not survive".

## Registered secondaries (form `gt`, 20 placebos each; the same bar is computed and reported; none can change the verdict)

- S1, isolates coverage: the Kraken-trained weights of wave 4d, frozen (`runs-agent/lic/r24/ckpt`, read only). Inputs: Kraken bars
  for every coin in the Kraken pool, KuCoin bars (cut at Kraken's last hour, 2026-09-30 23:00) for the harness coins outside the
  Kraken pool; market series from the Kraken training pool, as trained. October 2026 stays empty, as in wave 4d.
- S2, one live source: the same Kraken weights, every input from KuCoin bars (the Kraken pool symbols that KuCoin has in full, the
  market series from the Kraken training symbols among them, harness coins added), origins to 2026-10-31.
- S3, OKX fill, run only if a R9-R24 coin is outside the KuCoin pool and OKX serves bars for it: the KuCoin-trained weights, OKX 1 h
  bars as that coin's inputs (not part of the market series), added to the primary forecast.
- Reference, as in PREREG-LIC: the Binance-trained forecast restricted to the primary's (coin, day) cells. If the primary fails and
  the reference fails too, the loss is coverage; if the reference passes and the primary fails, the KuCoin model lost the effect.

Reading: if the primary does not survive and S1 or S2 meets the bar, the result is written "does not survive (primary); secondary
meets the bar and needs its own confirmation", never "survives".

## Coverage (registered report)

For each forecast: how many of the 30 R9-R24 and the 36 ALGOS coins have a forecast, each coin's first forecast date (and how many
start on the first day, 2023-01-01), and its (coin, day) cells as a share of the Binance forecast file (42,000 and 50,341 cells).

## Also reported, not gated

Validation 24 h rank IC on the KuCoin seen symbols with the registered Binance `ohlc+cal` on the same rows; next-day IC and rho with
the trend score on the ALGOS harness (`tools/r24bots.py corr`) for the KuCoin forecast and for the Binance forecast on the same
cells; per-day rank correlation of the two forecasts; pool symbols per year. Data sanity for symbols in both caches, 2023-01 ..
2026-10, against the Binance cache and the Kraken cache: correlation of hourly log returns on hours where both have a bar, share of
the other source's bar hours that KuCoin lacks, median absolute log price gap.

## October 2026 and after

- October 2026 is inside the primary window here (KuCoin bars cover it); in wave 4d it had no tilt.
- After October: a separate read, consistency only, which cannot change the verdict. Forecasts with the F4 checkpoints (their
  training data ends before 2026) for origins 2026-11-01 23:00 UTC up to the last origin whose 24 h outcome is complete in the cache
  (`botpred --end`). Statistic: per-day Spearman IC of the forecast against the next 24 h KuCoin close-to-close return over the
  R9-R24 and ALGOS coins with a forecast; reported as mean, number of days and share of days above zero, next to the same statistic
  on 2023-01 .. 2026-10 for scale. About 16 days, so no interval is claimed. No tilt simulation: the harness statistic is defined on
  2023-01 .. 2026-10 and is not re-windowed.

## What a pass would and would not mean

A pass says the tilt is reproducible from the API bars of the venue the owner trades on. It is not a legal clearance; the terms text
is recorded in `agent/LICENCE-ROUTES.md` (section 5, added with the results) and the reading is the owner's or counsel's. In short:
KuCoin's Terms of Use (updated 2026-10-29, articles 89-92) reserve the rights in all platform content, bar "commercial use of the
Platform or any content thereof", collecting "prices", data collection tools "for other commercial interests" and systematically
building databases from platform content without written permission; the public site shows no API-specific terms and no
machine-learning clause. OKX's API Agreement (28 July 2026, 9.4) limits Market Data to "your own personal, non-commercial trading and
account management purposes" and bars training models "intended to be commercialized or made available to third parties".
