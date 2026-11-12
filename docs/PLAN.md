# Plan after phase 0-2 (2026-11-12)

Numbers quoted here are in `bench/LEADERBOARD.md`; licences in `docs/SCOPE.md`.

## 1. Where things stand

- The benchmark is hard and flat: on hourly crypto returns every serious model sits within about 4 % (asset-equal scaled
  pinball) of a zero-mean Gaussian random walk, and GARCH(1,1)-t is statistically level with Timestral. Point forecasts are
  almost pure noise: the foundation models' medians are no better than zero, the candidates' are 0.2-0.3 % better hourly
  (directional accuracy 51-53 %, rank IC 0.04-0.065, possibly microstructure). The contest is volatility and tail shape.
- A 0.1M-0.8M parameter model trained in about one minute per fold beats the reference Timestral pipeline by 0.4-0.9 % sPL on
  unseen folds (paired intervals exclude zero; 0.1-0.5 % against Timestral with a 2,048-bar context) at 17x-87x the GPU throughput
  of the production `tsfm16` kernel, and runs on CPU. The size and data learning curves are flat: neither more parameters
  (0.1M -> 0.8M -> 4.8M) nor more training symbols (5 % -> 25 % -> 100 % of the pool) moves the score beyond bootstrap noise.
  Capacity and data volume are not the bottleneck; the information in a 512-bar close-to-close context is.
- So "pushing it" means better inputs, not a bigger network. That decides the search space below.
- Everything trained so far is CC BY-NC-SA research weight (Binance Vision terms) and may not be used by the bots.

## 2. Autoresearch loop

Objective: validation sPL (asset-equal scaled pinball), averaged over the four folds' validation windows (the 90 days before each
test period) with horizons weighted equally, subject to a latency budget (rows/s at batch 4096 and batch-1 p99 from
`bmtbench.speed`). The test folds are never an objective.

Search space, in the order of expected value:
1. Inputs. OHLC range per bar (Parkinson / Garman-Klass: intrabar range is a far less noisy volatility proxy than |close-to-close|),
   volume and trade count, long-memory scalars (RMS ratios over 1 d / 1 w / 1 m / 3 m / 9 m, HAR style), longer or multi-resolution
   context (512 fine + coarse tokens up to 6,000 bars), calendar (hour, weekday, known event flags), cross-asset state (BTC / ETH
   returns, market-wide realised volatility, cross-sectional dispersion).
2. Output. Tail parametrisation (spliced generalised-Pareto tails beyond the 5 % knots), sign-equivariant head (exact instead of
   2-pass flip averaging), horizon set, joint multi-horizon consistency.
3. Training. Frequency mix, loss weighting across horizons, asset sampling, steps / lr / EMA, seed ensembles, synthetic
   pre-training (stochastic-volatility + jump simulators), which is also the route to permissively licensed weights.
4. Architecture. d / layers / heads / patch, attention vs conv / MLP-mixer / state-space readout. Low priority while the curves
   are flat.
5. Teachers. Distillation from Timestral-max / Chronos-2 quantiles and LoRA / last-layer adaptation of Timestral (research tier
   only, licence in SCOPE.md; earlier in-house fine-tuning overfit after 250-500 steps and improved calibration, not ranking).

Trial budget and scheduling: one trial = 4 folds x 1,500 steps x batch 1024 (about 5 minutes on the shared local GPU for 0.8M
parameters). Successive halving: 500 steps on one fold for every proposal, the best third get 1,500 steps on two folds, the best
third of those the full four folds with two seeds. A JSONL ledger records config, git hash, validation scores and seconds for every
trial, including failures (`runs/lc.jsonl` is the seed of it).

Promotion rule: a config replaces the incumbent only if the paired block-bootstrap 95 % interval of its validation sPL improvement
excludes zero at two or more horizons and it stays inside the latency budget. Validation noise is about +-0.3 %, so single-run
differences smaller than that are ignored.

Test policy: only promoted configs are scored on the test folds, at most one per day, and every scored config is added to the
leaderboard with its interval, winners and losers alike. The shuffled-time placebo is rerun for each promoted architecture.

Stop rules: a branch stops after 20 consecutive trials without a promotion; the loop stops when the best promoted improvement over
the trailing 100 trials is below 0.1 % sPL, when the budget is spent, or immediately on any leakage-check failure.

"Pushed to its limit" means all of: 100 trials without a promotion; learning curve flat in parameters and in data at the final
input set (doubling either gains < 0.1 % sPL); the placebo passes; the remaining gap to the unseen-asset score is inside the
interval. At that point the number is the number.

## 3. Compute and cost

Local first. Measured here: 1,500 steps of the 0.8M model take 60-105 s per fold on the shared RTX 5090 (other tenants at 85-100 %
utilisation, 2-6 GB free, my cap 4 GB, blackout 15:20-16:15 ET), so about 10 four-fold trials per hour. No pod was started in
this phase and nothing was spent: a pod could not have beaten a 5-minute local trial once start-up and data sync are counted.

When a pod becomes worth it (inputs with long context or OHLC features at 5M+ parameters, or the Timestral adaptation arms, each
10-30 minutes per trial locally), on-demand prices read on 2026-11-12: RTX 4090 USD 0.34 (community) / 0.89 (secure) per hour,
RTX 5090 0.69 / 1.19, billed per second.

| stage | work | pod hours | cost (5090 secure) |
|---|---|---|---|
| A | 300 input-search trials at 0.8M-5M parameters, four folds each | 25-40 | USD 30-48 |
| B | Timestral LoRA / last-layer arms, 5 arms x 4 folds x <= 500 steps | 3-4 | USD 4-5 |
| C | synthetic pre-training + fine-tune for the permissive tier, 3 runs | 10-15 | USD 12-18 |

Proposed first tranche: stage A only, capped at USD 40 in total and USD 20 per pod, to be confirmed before launch. First step of
that tranche is a 10-minute paired timing run (same trial locally and on the pod) to show the pod's trials-per-dollar; if the
pod is not at least 3x the local rate the tranche stays local.

Pod rules (workspace cost guard): generic names, never `keep-*`; terminate in a `finally` block and arm a hard self-termination
timer on the pod at start; inputs are the public archives fetched on the pod (or the 247 MB compressed cache through an R2 token
minted for one bucket); logs and checkpoints streamed to R2 so a killed pod loses
nothing; no trading credentials or private data on a pod. Note: the existing infra only launches serverless endpoints, so a pod launcher must be written and its credential source agreed.

## 4. Inference kernels: next measured steps

Current state: `src/bmt.mojo` (CPU, fp32, 8-lane SIMD, 4-row register blocking, last-token-only final layer) and
`src/bmt_gpu.mojo` (fp32 cuBLAS GEMMs + six fused kernels), both matching PyTorch fp32 within 2e-5 scale units, tolerance 2e-4.

GPU, in order: last-token-only final layer (about 19 % fewer FLOPs at four layers); fp16 / bf16 tensor-core GEMMs through the
Modular kernel library as `tsfm16` does (the fp32 path leaves an order of magnitude of GEMM rate unused; parity tolerance becomes
about 1e-3); fuse bias + LayerNorm into the GEMM epilogues; `compile_function` once per kernel instead of per-enqueue closures;
a shared-memory request ring instead of pipes for batch-1 latency; builds for sm86 / sm89.
CPU: 2x4 output-row blocking, an AVX-512 build for newer hosts, fp16 weights, a `shared-lib` build with a C ABI so Go and Python
call it in process.
Each step lands with before / after rows in the speed table and the self-test still passing.

## 5. Plugging into the bots (after the licence gate)

Gate: weights the bots use must come from the permissive tier (SCOPE.md section 2). Nothing trained on Binance Vision archives is
deployed.

1. Same wire contract as `/api/v1/timestral/forecast` (`rows x ctx float32 -> rows x 21 float32`, h = 1 head): `bmt serve` is the
   resident worker; a `model=` switch in the service's batcher selects it. Self-test vectors gate serving exactly as for `tsfm16`.
2. Shadow first: log the candidate's quantiles next to the production forecast for at least three months and judge it with a
   pre-registered significance rule on ranking and PnL metrics (earlier holdout gains from fine-tuning did not survive).
   MAE / pinball decide what reaches shadow, PnL decides what ships.
3. Consumers that read forecasts from a cache get a small writer (median / low / high / confidence from the quantiles).
4. A daily / overnight stock variant needs the owner's stock data: private evaluation only.

## 6. RL environment

`bmt serve` already is a batched rollout primitive: N contexts in, N x H x 21 quantiles out, 100k+ rows/s on the GPU and tens of
thousands on CPU for the small sizes. Two uses:
- Observation model: the quantiles (or the last-token embedding) are features for the policy; one forward per environment step for
  the whole batch of environments.
- Generative simulator: sample the next return by inverse-CDF interpolation between the 21 knots (generalised-Pareto beyond the
  ends), append to each environment's 512-return ring buffer, repeat. An in-process Mojo loop (no pipe) should reach 50k-100k
  environment steps per second per GPU. Cross-asset dependence comes from the market-covariate variant plus a rank copula on the
  uniforms. Policies trained there are validated on real walk-forward data only; the simulator has no order book and no impact.

## 7. Open decisions for the owner

1. Name: `bitbankmojotime` (working), repo under `lee101/`, Hugging Face id.
2. Code licence: Apache-2.0 (proposed) or MIT.
3. Data policy. Binance Vision archives are CC BY-NC-SA 4.0 with trading / commercial prohibitions (terms of 2026-08-26). Options:
   (a) public benchmark + non-commercial research weights only; (b) add a permissive tier trained on synthetic data and on market
   data held under commercial terms, which is also the only tier the bots may use; (c) buy a commercial data licence.
   Related: whether linking a commercial site from a benchmark built on NC data is acceptable.
4. Timestral derivatives: publish adapted weights under Mistral's Modified MIT licence, keep them private, or skip; whether
   distillation from Timestral is acceptable.
5. Private stock evaluation on the Alpaca / SIP data: go / no-go (stays private either way).
6. RunPod tranche of section 3 and where the pod credential comes from.
7. Hugging Face model card: draft in `docs/MODEL_CARD.md`; which tier is published first.
8. Housekeeping found on the way: `lee101/bitbank-mojo` has no LICENSE file; `mojo-statsforecast` has no NOTICE for its Nixtla
   origin and does not build cleanly on the toolchain installed here.
