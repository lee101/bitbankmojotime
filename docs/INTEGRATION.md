# Integration: rotation consumer (bitbankgomojo)

PRIVATE NOTE: this file describes the interface of a private consumer. Remove or generalise it before the repository goes public
(`docs/RELEASE_CHECKLIST.md`). It was written from read-only inspection of the consumer's deployed directory; nothing there was
changed or executed.

## What the consumer does today (interface only)

- It is a daily rotation job over 8 USDT pairs. Its "forecast" is one cross-sectional score per pair from a tree model; the state
  file says so itself: "Ranks, not expected returns". It does not consume quantile forecasts or any forecast service today.
- Input: a candle history file, schema `rotation-completed-history-v1`:
  `{"pairs": [...], "hours": [epoch hours], "ohlcv": [[[open, high, low, close, volume] per pair] per hour]}`, completed hourly
  candles only, source "Poloniex public candles". The current file holds about 317 consecutive hours.
- Output: a state file, schema `rotation-forecast-state-v1`: `pairs`, `issued_hour`, `usable_execution_hour` (= issued + 1),
  `raw_scores` (one per pair, cross-sectionally standardised), `smoothed_scores`, bookkeeping (observations, model file hashes,
  receipt).
- Call pattern and latency: one batch of 8 pairs per run; the daily job has a 60 s freshness budget and a 200 s start timeout,
  an hourly job refreshes the inputs. Latency is not a constraint for a forecaster (milliseconds needed, seconds available).
- Not established: the decision rule that turns scores into positions and the score smoothing (compiled binary / sources outside
  that directory, not read), whether more than 317 hours of history are kept anywhere, and Poloniex's terms for the candles.

## Drop-in contract

```
bmt2 rotation <bmt2.bin> <history.json> [threads]        # BMT_SYM=1 for the flip-averaged forecast
```

reads the consumer's own history file and prints one JSON object:

| field | meaning |
|---|---|
| `schema` | `bmt-rotation-forecast-v1` |
| `pairs` | copied verbatim from the history file, same order |
| `issued_hour`, `usable_execution_hour` | last completed hour in the file, and that + 1 (the consumer's convention) |
| `horizon_hours` | 24 |
| `raw_scores` | per pair: forecast median / forecast sigma at 24 h, standardised across the pairs (mean 0, sd 1). Ranks, not expected returns |
| `median_logret`, `sigma_logret`, `q05_logret`, `q95_logret` | per pair, 24 h cumulative log return: median, (q85 - q15) / 2.0728, 5 % and 95 % quantiles |

`raw_scores` has the semantics of the consumer's own `raw_scores`, so it can be fed to the same smoothing and decision code as an
alternative score vector, or the sigma / quantile fields can be used as extra features. The CLI builds the model inputs itself
(log returns, log high/low range, Garman-Klass sigma, volume if the model uses it, start hour and weekday of the next bar) and
refuses models that need inputs a candle file cannot give (other symbols, 5-minute data, long-memory scalars) and files with gaps
or bad candles. `tests/test_mojo_parity.py` checks the CLI against the PyTorch reference on a synthetic history file.

Requirements and gaps:
- History length: the model needs context + 1 completed hourly candles per pair. The 512-bar models need 513; the consumer's file
  has about 317. Either the collector keeps 513 hours, or the 256-bar variant is used (`ctx=256`, needs 257; see the leaderboard
  row `bmt2-*-ctx256` if present, otherwise it is the next thing to train).
- Weights: only the permissive tier (`models/permissive/`) may be used by a bot. The research tier is non-commercial.
- Speed: 8 pairs take about 3 ms on one CPU thread with the size-t model; the weight file is under 0.5 MB.

## Offline A/B recipe (old vs new forecast, same decision rule, PnL with costs, unseen folds)

1. Freeze the decision rule: the function that maps a score vector at an origin to holdings. `bench/bmtbench/ab.py` has a
   stand-in with the same shape (`decide`: hold the top eighth by score for one day, long only); replace it with the consumer's
   rule without changing anything else.
2. Produce both score histories on the SAME origins: the old forecaster's scores as it issued them (or re-issued by its own
   offline replay), the new ones from `bmt2 rotation` on the history file as of each origin (never a later file).
3. Run both through the rule on the unseen folds only (models trained before the fold, `bench/PROTOCOL.md`), with costs:
   PnL per day = return of the held names - return of the no-rotation alternative - cost x turnover; scenarios 0 / 4 / 20 bp round
   trip per replaced name.
4. Compare with the paired calendar-block bootstrap: `python -m bmtbench.ab --a <new> --b <old>` prints each forecaster's net
   excess return and the paired difference with a 95 % interval. Ship nothing on a difference whose interval includes zero.
5. Then shadow: log the new scores next to the live ones and repeat step 4 on live data before any switch.

What the stand-in rule says today is in `bench/LEADERBOARD.md` (section "Tradable value"): under a daily top-eighth rotation no
forecaster's median, old or new, earns a positive excess return on the unseen folds, and no pair of them is separated.
