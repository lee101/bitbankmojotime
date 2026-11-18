# fcast24: daily 24 h forecast file for the trend producers (FCAST-TILT)

PRIVATE NOTE: describes a private consumer (binancegomojo `trend-targets --fcast-tilt`). Remove before the repository goes public.

`tools/fcast24.py` writes the `bmt-fcast24 v1` file the producers read. Design: binancegomojo
`docs/experiments/FCAST-TILT-integration.md`; staging and runbook: binancegomojo `deploy/staged/fcast-tilt/README.md`.

## Contract

```
# bmt-fcast24 v1 model=<name>~<hash8> seeds=<n> horizon_h=24 created=<epoch s>
<decision ts, 00:00 UTC, epoch s>\t<PAIR as BTC_USDT>\t<median 24 h log return, %.8e>
```
Sorted by (time, pair), newline terminated, atomic replace. A row's inputs are the 512 hourly bars closing at or before its decision ts
(origin = the 23:00 bar). `hash8` = sha256 over the exported weight files in seed order: weight-agnostic, any weight set can be dropped in;
a new weight set starts a new file (`seed` its history) and the producer's `--fcast-tilt-model` pin must be updated with it.

## Path

Production host has python3 without numpy or torch, so the run path is standard library + the Mojo kernel:

1. `refresh`: public REST klines of the venue (KuCoin tested here; Binance implemented, returns 451 from this box). One small file per
   symbol (`#range=from,to` + bars), atomic, resumable (a rerun fetches only symbols not yet at the origin), <= 4 in flight, paced
   (8 requests/s), 429 / 5xx / venue rate codes push every worker back (Retry-After or exponential, 7 tries). Closed bars only.
2. inputs (pure Python, float32 / float16 exact): per coin `r`, `hl`, `gk`; market series over the model's pool `mkt`, `disp`, `brd`;
   calendar. Same arithmetic as `train/bank.py`.
3. inference: `build/bmt2 serve <weights.bin>` once per seed (the existing CPU kernels), seed mean in float32.
4. gates, then atomic replace: every `--require` coin (default the M10 coins) has a row, at least `--min-pool` pool symbols alive on every bar of
   the window, finite values. A day that fails a gate is not written, the exit code is 3 and the file is untouched (the producer then freezes
   its multipliers). Missing earlier days (job down) are caught up, at most `--max-catchup` days.

`run` = refresh + write; schedule it hourly at :02 (after the day's row exists it exits without a network call). `--also` writes atomic
copies for readers in other directories. `export` (torch) turns a rank24 checkpoint into the weight file; `seed` (numpy) builds the history
from a `rank24 botpred` file; `check` validates a file with the producer's rules.

Build: `./build.sh fcast` (`TARGET_CPU=x86-64-v3` default; host-native builds emit AMD-only instructions that SIGILL on Intel hosts).
Checked with binancegomojo `deploy/isa_check.py`: 0 forms outside the baseline.

## Evidence (2026-11-18, `tests/test_fcast24.py`, `python tests/test_fcast24.py parity 14`)

501 (day, coin) rows: 14 days of 2026 x every coin of the research forecast file, fold F4 weights, research cache (Binance Vision).

| check | result |
|---|---|
| inputs vs `Bank.gather` (bank liveness): r, hl, gk, mkt, disp, brd, calendar | bit-identical, all 501 x 512 values per channel (hl needs the bank's float64 -> float32 -> float16 double rounding) |
| Mojo engine vs PyTorch CPU, same rows | max 7.3e-7 / 6.1e-7 of the forecast scale (seed 0 / 1), 4.8e-8 absolute; bar 2e-5 |
| writer inputs vs bank inputs, PyTorch | identical forecasts |
| vs the research file (written on a GPU with TF32 matmuls) | <= 2.5e-4 of scale; ensemble max abs 1.6e-5 (median abs forecast 1.6e-3); min per-day rank correlation 0.9997; 2 of 140 M10 ranks differ |
| online liveness rule (48 h forward fill, no knowledge of later bars) vs bank liveness | <= 1.4e-3 of scale; 0 of 140 M10 ranks differ |

Unit tests (mock venue, always run): at most 4 requests in flight, nine 429s retried, hard-failing symbols reported and the rest kept, resume
fetches only what is missing, next day fetches only new bars, no open bar stored, no partial files; file round trip and strict reader;
inputs vs an independent numpy version; end to end refresh -> Mojo -> file with catch-up, idempotent re-run, gates and model pin.

Real KuCoin run from this box (`data/universe.json` pool, 343 symbols asked, 165 listed on KuCoin, pool alive 155): one daily run
22.3 s wall (166 requests, 21.2 s paced refresh; 0.8 s inputs + inference for 43 coins x 2 seeds), 3.5 s CPU, no 429. Cold start with 48 days
of history: 41.7 s, 331 requests. Two independent caches give the same file.

## Known gaps

- Venue transfer: the weights in hand were trained on Binance klines with a 333-symbol pool. Fed with KuCoin klines (155 pool symbols alive),
  October 2026 forecasts vs the research file: rank correlation 0.87 over 43 coins, 0.94 on the M10 coins (min 0.77 over 30 days), 119 of 300
  M10 ranks differ, mean multiplier change 0.06. This is not the confirmed signal bit for bit until weights trained on KuCoin klines
  (branch `agent/lic2`) replace these; the writer and file do not change then (new weight files, new hash, new seeded history).
- The 6200-hour listing age the research rows required is not checked (a 561-bar window cannot see it); irrelevant for the coins the bots trade.
- A pool symbol whose first bar inside the fetched window follows a gap longer than 48 h is treated as newly listed.
- The pool aggregation runs in Python, not in the Mojo CLI: the kline refresh has to be Python anyway and the aggregation is 80 k additions
  (0.2 s). `bmt2 serve` is unchanged.
