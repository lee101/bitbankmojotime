# bitbankmojotime (working name)

A fast probabilistic forecaster for crypto and stock returns: a compact quantile model with Mojo inference on CPU and GPU, and a
benchmark that scores it on unseen calendar folds and unseen assets against Timestral-209M, Chronos-2 and the classical
baselines. Built for the forecasts behind [bitbank.nz](https://bitbank.nz).

Status: private pre-release. Licence not chosen (`LICENSE`), nothing published, research weights only (`docs/SCOPE.md`).

## What is here

- `bench/PROTOCOL.md`: the evaluation protocol, committed before any result, with dated amendments.
- `bench/LEADERBOARD.md`: every model on the same 393k hourly / 80k daily rows, with block-bootstrap intervals, paired skills,
  seen vs unseen assets, the speed table, the learning curve and the leakage checks (including one that failed as written).
- `train/`: the reference model (`bmt`, 0.11M-4.8M parameters) and the walk-forward trainer.
- `src/bmt.mojo`, `src/bmt_gpu.mojo`: inference with a self-test vector, bench and pipe-server modes.
- `docs/`: scope and licences, plan, model card draft, release checklist.

## Headline (unseen folds 2023-2026, 62 Binance USDT pairs, details and caveats in the leaderboard)

| hourly, horizon 1 | parameters | scaled pinball vs random walk | vs Timestral reference (paired 95 %) | rows/s, batch 4096 |
|---|---|---|---|---|
| bmt-t | 0.11M | +4.03 % | +0.56 % [+0.38, +0.73] | 489k GPU, 30k on 12 CPU threads |
| bmt-s-sym (2 passes) | 0.81M | +4.07 % | +0.60 % [+0.44, +0.77] | 44k GPU |
| Timestral-209M, reference pipeline | 209M | +3.50 % | 0 | 1.1k GPU (PyTorch) |
| Timestral-209M, production kernel (max / fast) | 209M | +3.34 % / +2.70 % | -0.16 % / -0.83 % | 5.6k / 15k GPU |
| GARCH(1,1)-t | 4 per asset | +3.21 % | -0.30 % [-0.47, -0.12] | CPU |
| Chronos-2 (returns in / prices in) | 120M | +2.54 % / -1.13 % | -0.99 % / -4.79 % | 1.5k GPU (PyTorch) |

The whole field sits within about 4 % of a Gaussian random walk; the candidates' edge is small, consistent across horizons and
assets, and comes with a 17x-87x faster forward. Speed was measured on a GPU shared with production workloads.

## Quick start

See `CONTRIBUTING.md` for the commands (tests, data fetch, baselines, training, scoring, Mojo build and self-test).

## Data and third-party models

Market data is fetched by the user from Binance Vision (`data.binance.vision`), CC BY-NC-SA 4.0, non-commercial; the result
tables and `data/universe.json` are derived from it and carry the same licence and attribution. Not affiliated with or endorsed
by Binance. Timestral-209M is Mistral AI's model (Modified MIT licence), Chronos-2 is Amazon's (Apache-2.0); neither is
redistributed here. Nothing in this repository is investment advice.
