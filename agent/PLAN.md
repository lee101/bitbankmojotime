# Plan (agent branch version, updated 2026-11-12 after the input search; supersedes `docs/PLAN.md` on merge)

Numbers quoted here are in `bench/LEADERBOARD.md`; licences in `docs/SCOPE.md`.

## 1. Where things stand (updated after the input search, 2026-11-12 evening)

- The benchmark is hard and flat: every serious model sits within about 4.5 % (asset-equal scaled pinball) of a zero-mean
  Gaussian random walk; GARCH(1,1)-t is level with Timestral. The contest is volatility and tail shape.
- Three search waves, 90 configurations, 3.8 GPU-hours of search (about 4.5 with refits, placebo, synthetic training and scoring) of the 12 budgeted, validation only (`docs/RESEARCH_LOG.md`):
  - promoted for pinball: bar range + Garman-Klass sigma, then calendar inputs. On the unseen folds the promoted model beats
    the Timestral reference pipeline by 1.0 % / 0.75 % / 1.1 % (hourly h 1 / 6 / 24), the 2,048-context Timestral by
    0.9 % / 0.4 % / 0.4 %, GARCH by 1.0-1.3 %, all separated;
  - not promoted (each under 0.1 % on validation): volume, taker-buy share, long-memory volatility scalars, vol-of-vol, a daily
    token stream of 256 days, BTC / ETH returns, market breadth, peer baskets, BTC range, realised variance from 5-minute
    returns and its downside share;
  - ranking: a cross-sectional correlation loss on the median was promoted on validation (rank IC +0.018 at h 1, +0.016 at
    h 6, pinball cost 0.16 %) and did NOT confirm on the unseen folds (IC 0.068 / 0.060 / 0.043 against 0.066 / 0.061 / 0.053
    without it at size s; pinball 0.35-0.5 % worse; worse under the rotation rule). Market-state inputs raised the validation
    24 h IC by a further 0.016 while lowering the 1 h IC, so the registered rule did not promote them: an untested lead.
- Learning curves: flat for pinball in parameters (0.11M = 0.82M = 4.8M) and in data (17 symbols = 333). For ranking the
  size is flat and the data is not quite (5 % of the symbols costs 0.003 IC at h 1).
- Permissive tier: the promoted model trained on synthetic bars only loses 0.3-0.6 % hourly and 0.5-0.7 % daily against the
  research weights and still beats Timestral hourly. That is the tier the bots may use.
- Tradable value: rank IC 0.05-0.07 is real and two to three times Timestral's, but of 240 net-of-cost intervals one is above
  zero (Timestral with 2,048 context, 24 h long-short at maker-tier costs).

### What "at its limit" looks like now

- For pinball with inputs derivable from klines (hourly and 5-minute OHLCV, taker flow, cross-asset bars, calendar) the proposal
  list is exhausted: after the two promotions, 26 further configurations and both learning curves moved nothing beyond noise.
  The formal stop rule (100 trials without a promotion) has not been reached; the honest statement is "no untested kline input
  left", not "proven optimal".
- New information would have to come from data that is not in klines: order-book and trade-level flow, perpetual funding / basis
  / open interest, options-implied volatility, scheduled events. Each needs a data source with usable terms first.
- For trading value the limit is costs, not the forecaster: a perfect copy of today's signal earns 5-14 bp gross per round trip.

### Wave 4 outcome (2026-11-17, `agent/PREREG-24H.md`, `agent/RESEARCH_LOG.md` wave 4)

- Items 1 and 2 of the previous list are done. Fixed set-up: i.i.d.-trained incumbent, final EMA weights, two seeds, the loop's
  validation windows. Under it `mkt+r24` (market-state inputs + a 24 h within-origin correlation loss, weight 0.1) raised the
  validation 24 h IC by +0.043 [+0.030, +0.056] but its L/S Delta (+8.9 bp [-3.2, +22.1]) did not clear the gate; scored once on
  test as the largest-Delta arm (not promoted): IC 0.085 vs 0.045 (incumbent retrain) / 0.050 (registered) / 0.043 (Timestral-2048),
  24 h L/S net at 4 bp +16.4 bp [+9.2, +23.2], +10.6 [+5.5, +15.4] over the incumbent, both seeds separate, no pinball cost.
- Bots: rho with the binancegomojo trend score -0.14; next-day IC 0.070 on the 36 harness coins where the trend score has none; a
  daily (1 + r) tilt of the trend book beats random-rank placebos by ~0.09 Sharpe but is still below the plain book (-0.09 / -0.12 d
  Sharpe at c7 / c10) because of the extra turnover.

### Wave 5 outcome (2026-11-18, `agent/PREREG-HORIZON.md`, research log wave 5)

- Forecast: `mkt+r72` / `mkt+r168` (same recipe, ranking term at 72 h / 168 h) rank better than `mkt+r24` at every horizon on the
  unseen folds (IC at 168 h 0.106 against 0.067, d +0.039 [+0.022, +0.057]; at 24 h 0.091-0.093 against 0.085) and their ranks are
  more persistent (lag-1-day autocorrelation 0.41 / 0.57 against 0.21). No pinball cost.
- Trading: registered null. With the 168 h forecast the trade-day tilt is +0.166 Sharpe vs base and +0.021 [-0.033, +0.078] over
  the incumbent `gt24` (57 % of configurations, 8/14 universes): not separated, minimum detectable 0.08. Blend +0.009. The daily tilt
  becomes neutral (+0.03) instead of harmful but stays 0.11 below `gt24`; a no-trade band does not help. `gt24` stays; no
  confirmation run, R25-R40 (seeded in the prereg) remain unbuilt.

### Next, in order of expected value

1. Forward scoring registered in `agent/PREREG-GATE.md` section 4: frozen F4 checkpoints, 2026-11-01 .. 2027-01-31, scored once in
   early February 2027; primary d rank IC vs `inc`. Earlier wording: confirm `mkt+r24` forward: it failed its own validation gate, so treat the test result as one strong draw. Score it on data after
   2026-10-31 as it arrives (frozen checkpoints, same L/S convention), and rerun the validation with longer windows (e.g. 180 days)
   so a 24 h L/S gate has power; register that rule before looking.
2. Done 2026-11-17 (`agent/PREREG-GATE.md`, research log wave 4b): tilting only on the days the trend book trades anyway is
   INNOVATION on the pooled harness at c7 / c10 (+0.145 / +0.141 d Sharpe, placebo max +0.037, 4/4 years, turnover -4.9 %). The entry
   gate is neutral; the exit hint is de-risking that the placebo matches. Next: a gross-matched tilt on post-October data and on
   R9-R24, registered first; then the licence question (the research-tier weights cannot go to the bots).
   CONFIRMED 2026-11-18 on R9-R24 (`agent/PREREG-GATE-CONFIRM.md`): +0.141 / +0.138 d Sharpe, 15/16 universes, turnover -6.6 %;
   risk-matched co-primary +0.065, not separated. Producer design (default off, not installed): binancegomojo
   `research/fcast-tilt`. Blocker: research-tier weights (CC BY-NC-SA data) cannot size live trades; owner decision on data licence.
   Earlier wording: bot use without the turnover penalty: use the forecast only at the bot's own rebalance points (entry ranking among new trend
   candidates, or a tie-break on top-k selection) instead of a daily tilt; and a rank-change band. One prereg on the ALGOS harness.
3. Permissive tier for the ranker: the gain needs cross-sections (market-state inputs + grouped loss), which synthetic independent
   series cannot supply; needs commercially licensed cross-sectional data (KuCoin / Poloniex terms bar commercial use).
   Done 2026-11-18 (`agent/LICENCE-ROUTES.md`, `agent/PREREG-LIC.md`): survey of 9 sources. No free source with usable terms has
   100+ symbols hourly from 2020; Coinbase bars ML training outright; Kraken (own-benefit grant, no ML / trading clause) is the
   largest free one (27 symbols by 2020, 156 by 2022-12). `mkt+r24` retrained on Kraken keeps its quality (val IC +0.063; harness
   next-day IC +0.060 = Binance model on the same cells), but the R9-R24 tilt does NOT survive by the registered bar (`gt` +0.065
   [-0.049, +0.180], NEUTRAL) because Kraken lists only 21/30 coins; the Binance model on the same coverage also fails (+0.038).
   Owner decisions: (a) whether Kraken's "own benefit" covers bot sizing; (b) a source for the missing coins' bars; (c) or Tardis
   (Business tier for 2019+ history; Solo USD 900 / month reaches 2022-11) / CoinGecko Analyst (USD 129, close-only, needs a
   no-range variant). Next if (c) is bought: rebuild the cache from it and rerun `runs-agent/lic/run.sh` + `gate.sh` unchanged.
4. Permissive tier: the 256-bar variant for the rotation consumer exists (`bmt2-syn-ohlc+cal-ctx256@t~b5523fb3`); next is a
   shadow run per `agent/INTEGRATION.md`. A ranking head cannot be trained on independent synthetic series; it needs cross-sections of real data
   under commercial terms.
5. Kernel work of section 4 (the promoted model runs at 218k rows/s on the shared GPU and 24k rows/s on 12 CPU threads; speed is
   not the bottleneck for any current consumer).

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
2. Code licence: `LICENSE` is Apache-2.0 since commit 6e6ba68; confirm that this covers the permissive-tier weights too.
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
