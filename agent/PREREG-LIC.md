# Pre-registration: does the `mkt+r24` trade-day tilt survive retraining on a licence-route data source?

Date 2026-11-18 (UTC). Committed before any bar of the new source is downloaded, before any training on it and before any harness
run. Survey and quotes: `agent/LICENCE-ROUTES.md`. Code: `tools/kraken_cache.py` (new), `train/rank24.py` (unchanged recipe),
`tools/r24gate.py` (unchanged forms). Outputs `runs-agent/lic/`.

## Source

Kraken's downloadable OHLCVT archive (`assets.kraken.com/marketing/institutions/Kraken_OHLCVT_Full.zip.part00..04`, checked
against `OHLCVT_Full_PARTS_SHA256SUMS.txt`), 60-minute files only, every pair from its first trade to 2026-09-30. Chosen because it
is the only free source in the survey whose terms text has no machine-learning or trading-use prohibition and that reaches a
cross-section of the needed size hourly from 2020 (Coinbase bars ML use outright; Alpaca, Bitstamp, CryptoCompare/CoinDesk are
personal / non-commercial or retired; Gemini has too little history). It is fetched without keys; no API call, no signed call.
Nothing is published; the cache stays under `runs-agent/lic/cache/` (git-ignored).

## Data mapping (mechanical, fixed now)

- Pairs quoted in USD only (Kraken file names `<BASE>USD_60.csv`, with Kraken's legacy names normalised: XBT -> BTC, XDG -> DOGE,
  and the `X...Z...` four-letter forms). Excluded: fiat bases, stablecoin bases (USDT, USDC, DAI, PYUSD, TUSD, USDP, EURT, USDE,
  RLUSD, USDG, USD1, USDS, FDUSD, USDQ, USDR, GUSD, BUSD, UST), tokenised stocks / ETFs (names containing `x` suffix forms such as
  `AAPLxUSD`), and gold-backed tokens (PAXG, XAUT).
- Each pair is stored under the canonical key `<BASE>USDT` so that the harness and the held-out-asset hash keep working unchanged;
  the key is a name only, the bars are Kraken USD bars. `qv = volume x close`; taker-buy share is unknown and set to 0.5 (no arm
  here uses it). Hours without trades are missing bars, exactly like missing Binance bars (close forward-filled, `real` False).
- Liquidity filter: a pair enters the pool if its first bar is on or before 2022-12-31 and at least 50 % of the hours from its first
  bar to 2022-12-31 (capped at the last 365 days of that span) have a trade.
- Universe by the protocol's rule (`bmtbench.data.build_universe`): pool = the filtered pairs; evaluation set = the 64 highest 2021
  quote volumes among pairs listed by 2020-12 (all of them if fewer); held-out = the same sha256 rule; train = pool minus held-out.

## Training (identical recipe)

`train/rank24.py train --arms mkt+r24 --seeds 0,1` with `BMT_DATA=runs-agent/lic/cache`, `BMT_UNIVERSE=runs-agent/lic/universe.json`,
`BMT_R24_OUT=runs-agent/lic/r24`: same folds F1-F4, same training limits, 1,500 steps, final EMA, same loss weight 0.1, same model
size. No hyper-parameter is changed or tuned. GPU cap 3.2 GB, the 15:20-16:15 ET blackout and the head-room wait apply.

## Forecast for the harness

`train/rank24.py botpred --arms mkt+r24` for the union of the 36 ALGOS coins and the 30 R9-R24 coins that exist in the Kraken pool
or cache, daily 23:00 UTC origins, inputs from Kraken bars only (the live-use situation). Coins Kraken does not list get no forecast
(multiplier 1 in the forms, as for coins without history before). Kraken bars end 2026-09-30, so 2026-10 has no forecast.

## Test (same harness, same bar as PREREG-GATE-CONFIRM)

R9-R24 (`GATE_UNIS=CONF`), 2023-01 .. 2026-10, costs c7 / c10, forms `gt` (frozen, co-primary 1) and `gtr` (risk-matched,
co-primary 2), 20 placebo draws each, base arms reused read-only from `runs-agent/r24/confirm` (same binary, same configs).

Bar, unchanged: a co-primary passes iff at both costs it is INNOVATION under `al.klass` with its 20-draw placebo band, at least 10 of
16 universes have positive family-mean d Sharpe, and the turnover change is <= +2 %. "Survives" if either co-primary passes; "lean"
if either has the d Sharpe interval above zero at both costs without passing; otherwise "does not survive".

Reference (reported, not gated): the original Binance-trained `mkt+r24` forecast restricted to exactly the (coin, day) cells where
the Kraken forecast exists, form `gt`, 20 placebos. It separates "less coverage" from "worse model": if the reference also fails,
the loss is coverage; if the reference passes and Kraken fails, the model lost the effect.

Also reported: per-day Spearman IC of each forecast vs the next-day return on the harness coins (`tools/r24bots.py corr`), number of
pool symbols per year, and Kraken-vs-Binance forecast rank correlation on shared cells.

## What a pass would and would not mean

A pass says the effect does not depend on Binance Vision bars. It does not clear live use by itself: Kraken's terms grant use of its
content "only for your own benefit", forbid commercial exploitation / making it available to third parties, and contain broad
anti-automation wording; whether bot sizing of the owner's own account is inside that is the owner's (or counsel's) call, recorded in
`agent/LICENCE-ROUTES.md`. The live inference inputs would also have to come from a source under usable terms.
