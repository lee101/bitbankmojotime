# Scope, provenance and licences (phase 0, 2026-11-12)

Working name `bitbankmojotime`: a fast probabilistic forecaster for crypto and stock returns with Mojo inference, benchmarked on
unseen folds against Timestral-209M and Chronos-2. This file records what each existing model and data source is, what may be
adapted or redistributed, and what limits speed today. It is an engineering reading of the licences, not legal advice.

## 1. Models

### Timestral-209M (third-party weights: Mistral AI)

| | |
|---|---|
| Origin | `mistralai/Timestral-209M` on Hugging Face (family: 5M / 28M / 123M / 209M). Local copy: HF cache snapshot `9874d252`, `config.json` + `model.safetensors`, downloaded 2026-10. |
| Reference code | `github.com/mistralai/timestral`, Apache-2.0, inference only (no training or fine-tuning code). Commit `a534bdb` is the one pinned in this benchmark. |
| Architecture (from `config.json` and the model card) | Causal patch transformer, 24 blocks (`time, time, time, time_variate` x 6), d_model 768, 12 heads, MLP hidden 2048 (gated), patch 32, RoPE (theta 1e4, partial 0.5), sigmoid-gated attention, norm eps 1e-6. Context up to 32,768 steps, up to 1,024 forecast steps per pass (longer horizons block by block, median fed back). Univariate / multivariate, past and known-future covariates. |
| Input scaling | Per-patch location / scale from causal running statistics, `asinh` normalisation (`use_asinh: true`, `minimum_scale 1e-6`); missing values as NaN with an observation mask. No tokenisation into bins: patches are embedded from values + mask. |
| Output | 21 quantiles (0.01, 0.05 .. 0.95, 0.99). The reference `forecast()` averages the series and its sign-flipped copy by default (`symmetric_averaging=True`). |
| Training data | NOT disclosed. The model card has no data section and the tech report is "coming soon". Consequence: any historical fold may be in-sample for it; the benchmark flags this. |
| Weights licence | "Modified MIT License", attribution notice "2026 - Mistral AI". Grants use, copy, modify, merge, publish, distribute, sublicense, sell. Conditions: (1) keep the attribution + permission notice in all copies or substantial portions; (2) no rights at all for a company (or the employer of the user) with more than USD 20M global consolidated MONTHLY revenue, and this restriction "applies to the Model and any derivatives, modifications, or combined works based on it". |
| The owner's parts | `timestral-serve/` (`tsfm16` Mojo fp16 kernel, `serve.py`) is in-house code. `mojo/data/*.f16` (176 files, 381 MB) is a format conversion of Mistral's weights, i.e. still "the Model". |

What this allows:
- Fine-tuning / LoRA / last-layer adaptation: allowed.
- Redistribution of adapted weights (or of the fp16 conversion): allowed, under the SAME Modified MIT licence with Mistral's
  attribution and the revenue restriction passed on. Such weights cannot be relicensed as Apache-2.0 or MIT, so they must live in
  a separate artifact with their own LICENSE, not under this repo's licence.
- Distillation into a student: not addressed explicitly. The student contains none of the weights, but the licence reaches
  "derivatives ... or combined works based on it". Treated here as UNCLEAR: distilled students stay private until the owner (or
  Mistral) decides. The default public models use no teacher.

### Chronos-2 (third-party weights: Amazon)

- `amazon/chronos-2`, Apache-2.0 (weights and the `chronos-forecasting` code). 120M-parameter T5-style encoder: 12 layers, d_model
  768, 12 heads, d_ff 3072, input/output patch 16, group attention across series, `arcsinh` instance scaling, REG token, 21
  quantiles, context up to 8192, up to 64 output patches. Training data per the model card: `autogluon/chronos_datasets`,
  `Salesforce/GiftEvalPretrain` and synthetic data.
- Fine-tuning and redistribution of derivatives are allowed under Apache-2.0 (keep the licence / NOTICE, state changes).
- `bitbankmojo` (github `lee101/bitbank-mojo`) is the owner's Mojo port of the OHLC sidecar around it. That repo has no LICENSE
  file; it should get one before anything here links to it as open source.

### mojo-statsforecast, mojo-neuralforecast (the owner's repos)

- `mojo-statsforecast`: MIT, (c) 2026 Lee Penkman, public at `lee101/mojo-statsforecast`. A partial port of Nixtla StatsForecast
  (additive ETS, non-seasonal CSS ARIMA / AutoARIMA grid, Theta) with a ctypes Python API. No built library on this host and no
  pixi env. Built against the toolchain installed here (mojo 1.2.0.dev2026100805) the compile reports errors and AutoARIMA fails at
  run time, so it is NOT used in the benchmark; the Nixtla reference implementation is. Upstream StatsForecast is Apache-2.0: a
  NOTICE crediting it is advisable.
- `mojo-neuralforecast`: MIT, same holder, no git remote, the code itself is untracked. Losses, N-BEATS basis functions and N-BEATS
  inference on weights taken from a trained torch model. No training loop, no PatchTST. Not usable as a baseline.

## 2. Data

### Binance public archives (`data.binance.vision`): NON-COMMERCIAL

The "Binance Vision Dataset Terms" v1.0 (last updated 2026-08-26, `TERMS_AND_CONDITIONS.md` in `binance/binance-public-data`)
put the archives under CC BY-NC-SA 4.0 plus express prohibitions:

- 2.4: "Use of any data derived from, calculated with, or extracted from the Datasets (including charts, models, and derivative
  analytical indicators) is strictly governed by these Vision Dataset Terms."
- 4.1: permitted non-commercial uses are "academic research, non-monetized open educational projects, algorithmic historical
  backtesting for purely personal non-production research, and non-commercial open-source data science evaluations."
- 4.2: no use "for live proprietary trading execution, automated commercial order generation, or signal distribution to third
  parties for direct or indirect compensation."
- 4.4: no integration "into commercial financial products ... or commercial trading bot platforms."
- 4.5: a redistributed derivative work must credit Binance Vision and stay under CC BY-NC-SA 4.0.
- 8: no implied endorsement, no use of the Binance name to market commercial applications.

Consequences for this project:
1. The benchmark (a non-commercial open-source evaluation) may use the archives. The repo ships fetch scripts and checksums, never
   the data. Tables derived from it carry the attribution and CC BY-NC-SA 4.0.
2. Weights trained on these archives are "models" derived from the Datasets: they can only be published as CC BY-NC-SA 4.0
   research weights, and they may NOT be used in the owner's live trading bots or in a paid bitbank.nz API. Everything trained in
   this phase falls in that class.
3. A repo whose README promotes a commercial site is a grey area for "non-monetized": owner / legal call before going public.
### Other sources

| Source | Status |
|---|---|
| Alpaca / SIP US stock data (the owner's) | Vendor and exchange (CTA / UTP) agreements restrict redistribution and derived data. Private evaluation only; no public weights, no public tables. Not touched in this phase: no private stock table was produced. |
| Exchange market data pulled through the owner's own API accounts (e.g. the candles bitbank.nz already serves) | Governed by each exchange's API terms. Candidate for commercially usable training data, terms to be read per exchange before use. |
| Public time-series corpora on Hugging Face (`autogluon/chronos_datasets`, `Salesforce/GiftEvalPretrain`, `Salesforce/lotsa_data`) | Mixed per-subset licences. Not used here. Each subset must be checked before it is used for public weights. |
| Synthetic returns (stochastic-volatility / GARCH / jump processes) | Fully owned. The only source that gives unencumbered Apache / MIT weights today. |

### Proposed default

- Public benchmark: Binance Vision archives via fetch script, attribution, CC BY-NC-SA 4.0 on the result tables, protocol and
  harness under the repo licence. The harness is source-agnostic so the same protocol can be rerun on a licensed source.
- Public weights, tier A ("research"): trained on Binance Vision data, CC BY-NC-SA 4.0, labelled non-commercial.
- Public weights, tier B ("permissive") and anything the bots use: trained on synthetic data plus data the owner holds commercial
  rights to; Apache-2.0 or MIT. This tier does not exist yet; see `docs/PLAN.md`.
- Repo code licence: Apache-2.0 (`LICENSE` and `NOTICE` were set in commit 6e6ba68; before that this file proposed it).

### Bot-usable tier: which data could a trading bot's weights be trained on (2026-11-12, second pass)

Checked without calling any exchange API from this host; "not established" means exactly that.

| route | finding |
|---|---|
| Synthetic bars from a simulator whose parameters come from priors, not from a dataset | Clean. Used: `models/permissive/`, cost 0.2-0.6 % sPL against the research weights. Caveat: the input list was chosen with validation on Binance Vision data; the weights hold none of it. |
| Candles a bot collected itself from Poloniex public endpoints (the rotation job's source) | Not established: the terms page returned no text. Not used. |
| Candles collected from KuCoin public endpoints | KuCoin Terms of Use Art. 90-92 bar commercial use of, derivatives of, and systematic collection of platform content; no separate API licence found. Treated as not cleared. |
| Binance | Vision archives: non-commercial (above). Live-API terms not read from this host. |
| Public corpora on Hugging Face | Per-subset licences not checked; none used. |
| A purchased commercial market-data licence | Would clear real-data training; owner decision. |

## 3. Existing kernels and what limits speed

Measured on this host (RTX 5090 shared with production servers, 85-100 % utilised by other tenants during the runs):

- Timestral `tsfm16` (Mojo, fp16 tensor-core GEMMs from the Modular kernel library + custom fused kernels, fp32 residual stream): a
  private copy of the production binary reproduces the stored self-test vectors (max abs 2.6e-4 fast, 4.6e-5 max) and forwards
  4,096 ctx-256 rows in 0.25 s end to end; the live service's own probe reports 17.8k series/s. `quality=max` costs 4 such units.
  At about 3.8 GFLOP per row that is roughly 65 TFLOP/s effective, within about 2x of the 130 TFLOP/s bf16 GEMM rate the
  bitbankmojo README reports for this card. The kernel is GEMM-bound: at most about 2x is left in it.
- Chronos-2 in Mojo (`bitbankmojo`, README numbers, not re-measured): 424 req/s at concurrency 1 to 2,462 req/s at 256, one request
  = 4 OHLC series, context 720, horizon 24; 77 % of the time in cuBLAS GEMMs; 1,284 MiB. The reference PyTorch pipeline in this
  benchmark ran 393k rows (context 512, horizon 24, fp32, batch 256) in 355 s = 1.1k rows/s.
- Conclusion: both foundation models are already near the GEMM ceiling of the card. The remaining lever is FLOPs per row. A 0.8M
  parameter model needs about 40 MFLOP per row, 100x less than Timestral, and runs on CPU (section "Speed" of
  `bench/LEADERBOARD.md`).
- For small batches the cost is not arithmetic: a one-shot process pays seconds of start-up and weight load, and with a resident
  process kernel launch plus the pipe round trip dominate batch-1 latency.

Mojo / MAX GPU API, as read on 2026-11-12 for this design:
- Toolchain installed and used here: mojo 1.2.0.dev2026100805 with max-core 26.7.0.dev2026100805
  (`/nvme0n1-disk/code/mojo-p2m-toolchain`); `bitbankmojo` pins 1.2.0.dev2026092605.
- Nightly API reference for `max.gpu.host.DeviceContext` (the documentation site is mid-migration: `docs.modular.com` redirects to
  `mojolang.org` and `max.modular.com`, several pages 404): kernels are launched with `ctx.enqueue_function(...)`, either a named
  function as a parameter or a capturing closure whose captured device pointers are encoded for the device; passing a function
  directly costs 50-500 ns per enqueue, `compile_function` once + enqueue of the handle removes that; host buffers from
  `enqueue_create_host_buffer` are pinned; copies are asynchronous until `synchronize()`; `get_memory_info()` returns (free, total).
- Working reference for the pinned version: the owner's `tsfm16.mojo` (capturing-closure kernels, `TileTensor` + `_matmul_gpu` /
  `multistage_gemm` tensor-core GEMMs, shared-memory attention kernel).
- Source changes needed on 100805 versus older notes: `Pointer.load` -> `unsafe_load`, `bitcast` -> `unsafe_bitcast`, positional
  pointer indexing deprecated in favour of `unsafe_offset=`, `external_call` lives in `std.ffi`, `stderr` in `std.sys`, `out` is a
  reserved argument convention, and a libc `external_call` must use the same signature as the standard library's own (`write` with
  an `Int` fd).

## 4. Interface

A drop-in replacement for the current forecast service needs one contract: `rows x ctx float32` log returns in,
`rows x 21 float32` quantiles out (the public `/api/v1/timestral/forecast` endpoint). `bmt serve` speaks the same shape over a
pipe. How downstream systems consume forecasts is deliberately not documented in this repository.
