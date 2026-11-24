# PREREG: general LLM at low reasoning as a frontier forecaster (id PMLLMFC, 2026-11-24)

Sealed before the first gateway call (see the commit that adds this file; every call is logged to
`~/research_data/llm_spend/spend.jsonl`). Worktree `research/llm-frontier`; the live tree, its configs and its
`runs/` are read only. No orders, no GPU, CPU nice 10, <= 6 threads.

## 1. Question

Is a general LLM at `reasoning_effort: low`, given only the last 256 hourly log returns of an unnamed asset, a frontier
probabilistic forecaster against the purpose-trained `bmt2` model on the same held-out benchmark rows? Second question
(owner's agents-on-top idea): does the same LLM improve `bmt2` when it receives the `bmt2` quantiles as a hint ("refine")?

## 2. Rows (hourly task only; daily is out of this USD 5 slot)

Full hourly set = 393,407 rows. Preregistered subsample (`subsample.py`, seed 20261124, written to `subsample.json` before
any call): per fold F1-F4, 10 origin timestamps in 10 distinct weekly bootstrap blocks; at each timestamp 15 assets
(BTCUSDT + ETHUSDT, 7 random seen, 6 random unseen) => 600 rows, 150 per fold, 80 / 280 / 240 per bucket, 40 timestamps,
40 blocks, 62 assets. Timestamps rather than scattered rows so IC (>= 10 assets per timestamp) is defined.
Secondary subset: 150 rows drawn uniformly from the 600 (same seed), for the expensive model.

Every model is scored on exactly these rows with the benchmark's own `bmtbench.metrics.summarize` and `Boot` restricted to
the subsample (`score_sub.py`); the paired block bootstrap (1000 resamples, weekly calendar blocks, seed 20261112) is shared
across models so differences are paired. Validation of the wrapper: naive h1 on the full set reproduces the leaderboard
sPL 0.2697 [0.2587, 0.2820] and MASE 0.7337 [0.7036, 0.7676] exactly.

Reference rows on the subsample (`refs_sub.json`, computed before any call): `bmt2` = `bmt2-ohlc+cal-sym` (the promoted
model). sPL h1 / h6 / h24 = 0.2419 / 0.8204 / 1.2239, horizon mean 0.7621 [0.6070, 0.9305]; naive mean 0.8002,
timestral-official 0.7765, timestral-fast 0.7869, chronos-2 0.8131. Paired: bmt2 beats naive by 0.038 [0.021, 0.055] and
timestral-official by 0.0145 [0.0040, 0.0244] in mean sPL, so the subsample has power for paired differences of this size.

## 3. Models and prompt

- Gateway: openpaths `POST /v1/chat/completions`, `routing_strategy: config`, `reasoning_effort: low`, temperature 0,
  `max_tokens` 2500. Key from the environment only, never printed.
- Primary cheap model: `deepseek-v4-flash` (config.yaml 0.15 / 0.60 USD per M, peak 0.30 / 1.20; cheaper than
  `gpt-6.1-terra` at 1 / 5). Costs are logged at the PEAK prices (conservative). Secondary: `gpt-6.1-sol` (2 / 10).
  `grok-4.7` is excluded (search = look-ahead).
- Prompt (`llm.py`): system "quantitative forecaster, JSON only"; user = the last 256 hourly log returns in integer basis
  points, oldest first, clipped to +-3000 bp, plus the horizon list {1, 6, 24} bars and the 21 quantile levels; asks for
  `{"1": [21], "6": [21], "24": [21]}` in bp. No asset name, no venue, no dates (PIT clean to the extent that a return
  string does not identify the asset; BTC / ETH could in principle be recognised, reported as a caveat, not controlled).
- Variants: `llm-flash` (prompt only), `llm-flash-refine` (same prompt + the `bmt2` quantiles of that row as "a strong prior
  to refine"), `llm-sol` (prompt only, secondary 150 rows, run only if the gate in section 5 opens).
- Parsing: strict; a JSON object with the three keys, 21 finite numbers each, |q| <= 1e5 bp; sorted; bp / 1e4. One retry
  on malformed output or HTTP error. A row that still fails is marked failed and scored with the `naive` forecast of that
  row; failure rate reported separately. Rows outside the subsample hold the `naive` forecast so the array has the
  benchmark shape (393407, 3, 21) and the standard scorer also accepts it.

## 4. Cost rule (hard cap USD 5 for the slot, running total in spend.jsonl, client stops when it is reached)

Estimate: prompt ~2.1k chars ~ 1.0-1.2k tokens in, <= 2.5k out. Flash at peak prices <= 0.0036 per call, 1,200 calls
(600 + 600 refine, plus retries) <= ~2.7 USD if outputs run long, ~1.3 USD at 1k out. Sol at 1.2k in / 1.5k out ~ 0.017
per call. Procedure: (a) pilot = the first 5 rows of `llm-flash` (they are part of the subsample, not extra); (b) from the
measured mean cost per flash call, if 1,200 x cost > 2.8 USD the refine variant is cut to the secondary 150 rows (stated in
RESULTS); (c) after the flash runs, the sol row count = min(150, floor((4.70 - spent) / measured-or-estimated cost per sol
call)); below 50 affordable rows sol is skipped and reported as skipped. Nothing else is adjusted after seeing scores.

## 5. Decision rules (sealed)

Primary statistic: sPL (asset-equal scaled pinball) per horizon and its mean over h = 1, 6, 24; MASE and IC reported.

- Rule A (owner's rule, as given): the LLM is a frontier candidate only if its mean-horizon sPL on the subsample is
  <= bmt2's upper 95 % bootstrap bound on the same rows (0.9305), else FAIL. Note recorded before any call: on 40 blocks
  this unpaired bound is wide enough that `naive` (0.8002) also satisfies it, so Rule A alone cannot certify frontier status;
  it can only fail a model.
- Rule B (supplementary, preregistered now): "frontier-equivalent" requires in addition that the paired 95 % interval of
  sPL(LLM) - sPL(bmt2) (mean over horizons) includes or is below zero, and that the LLM beats `naive` paired (interval below
  zero). An LLM that is paired-worse than `naive` is reported as "below the random walk".
- Refine: a lead only if `llm-flash-refine` beats `bmt2` alone on mean-horizon sPL with the paired interval excluding zero
  (upper bound < 0). Otherwise "no lead"; if it is paired-worse than bmt2, "degrades".
- Gate for the secondary model: `gpt-6.1-sol` runs on the 150 secondary rows only if `llm-flash` mean sPL is within 10 % of
  bmt2's (sPL_flash <= 1.10 x 0.7621 = 0.8383). Its verdict uses Rules A / B on the 150 rows (references re-scored on those rows).
- Failure rate is reported per variant; failed rows carry the naive forecast and are not dropped.

## 6. Outputs

`runs/preds/hourly/llm-*.npy|json` (gitignored), `bench/llmfc/RESULTS.md`, `bench/llmfc/REGISTER_ROWS.md`,
`~/research_data/llm_spend/spend.jsonl`.

## Amendments (before any score was computed)

- A1 (2026-11-24, after the 5-row pilot + 4 debug calls, USD 0.043 spent, nothing scored). At `reasoning_effort: low` and
  `minimal` the gateway's `deepseek-v4-flash` runs an unbounded chain of thought (it hand-computes the context standard
  deviation, 6,000-7,700 characters of reasoning, hits `max_tokens` 2500 with empty content, 0 / 7 parsed). Raising the cap
  to cover it (~8k reasoning tokens) costs ~USD 0.01 per call, i.e. ~USD 12 for the two 600-row arms, over the slot. Changes:
  (1) the flash arms (`llm-flash`, `llm-flash-refine`) run in non-thinking mode (`reasoning_effort: none`, 324 output tokens
  and a valid JSON in the debug call) and the prompt additionally states the mean / std of the last 256, 64 and 24 bars and
  the 24-bar sum computed from the same returns (PIT clean, same information), with the instruction not to recompute by hand;
  (2) because flash no longer tests "low thinking", `gpt-6.1-sol` at `reasoning_effort: low` (bounded on OpenAI models) on the
  150 secondary rows becomes the low-thinking arm and runs regardless of the 10 % gate, subject only to the cost rule of
  section 4 (pilot of 5 rows first; `max_tokens` 2500; the cost rule caps its row count). Decision rules of section 5 are
  unchanged and apply to each arm on its rows.
- A2 (2026-11-24, after the 5-row sol pilot, USD 0.070 spent, nothing scored). Sol at low parsed 5 / 5 at ~870 in / ~270 out
  tokens (~USD 0.0045 per call), so the 150-row sol arm costs ~USD 0.7 and the whole plan ~USD 1.7. The slack is used for two
  more preregistered arms on the same 150 secondary rows, both under the same rules: `llm-terra` = `gpt-6.1-terra` at
  `reasoning_effort: low`, prompt only (is the low-thinking result model specific?), and `llm-sol-refine` = `gpt-6.1-sol` at
  low with the bmt2 hint (the owner's agents-on-top idea with a thinking model). Nothing else changes; cap USD 5 stands.
