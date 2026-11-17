# Pre-registration: 24 h cross-sectional ranking objective (PLAN items 2 then 1), committed before any training run

Date 2026-11-17. Branch `agent/search`, outputs under `runs-agent/r24/`. Code: `train/rank24.py` (training, validation, one test
scoring), `bench/bmtbench/r24eval.py` (test tables), `tools/r24bots.py` (bot overlap). Validation rows only until section 4.

## 1. Fixed validation set-up (PLAN item 2)

Problem it fixes: wave 3 compared candidates with an incumbent trained on grouped batches whose validation IC (0.046) was below what
the same inputs trained on i.i.d. windows reach on the unseen folds (0.066), and it picked each arm's best of six checkpoints by the
very IC it then compared, with one seed.

- Incumbent: `ohlc+cal` (the promoted inputs), trained exactly as in wave 1: i.i.d. windows, bar sizes 1 h / 24 h with probability
  0.7 / 0.3, batch 1024, 1,500 steps, AdamW lr 1e-3, cosine, EMA 0.998, size s. Retrained here with seeds 0 and 1 under the rules
  below. Sanity check (reported, not a gate): seed-0 retrain vs the registered checkpoints `runs-agent/ckpt_search/ohlc+cal_F*.pt`.
- Every arm (incumbent and candidates) is evaluated with its final EMA weights at step 1,500. No checkpoint selection.
- Two seeds (0, 1) per arm; seeds change initialisation and every sampler draw. Per-origin series are averaged over the two seeds.
- Validation rows: the autoresearch loop's windows, i.e. the 90 days before each fold's purged training end (protocol section 2),
  seen evaluation symbols, hourly task, every hourly origin, four folds pooled. Test rows are not loaded before section 4.
- Validation metrics at h = 24 h: rank IC (per-origin Spearman of the forecast median vs the realised 24 h log return, origins with
  >= 10 symbols); long-short = mean of the top fifth minus mean of the bottom fifth by median, net = gross - 2 x 4 bp (maker tier,
  the `bmtbench.tradable` definition). Hourly sPL at h 1 / 6 / 24 is reported for the pinball cost (no gate).
- Intervals: paired moving-block bootstrap over 7-day calendar blocks (`bmtbench.metrics.Boot`, 1,000 draws).

## 2. Candidates (fixed list; nothing is added after the first result)

| id | inputs | loss |
|---|---|---|
| `inc` | ohlc+cal | pinball (incumbent) |
| `mkt` | ohlc+cal+mkt (pool mean return, cross-sectional dispersion, breadth) | pinball |
| `r24` | ohlc+cal | pinball + 24 h ranking loss, weight 0.1 |
| `mkt+r24` | ohlc+cal+mkt | pinball + 24 h ranking loss, weight 0.1 |
| `mkt+r24x3` | ohlc+cal+mkt | pinball + 24 h ranking loss, weight 0.3 |

24 h ranking loss: each step adds a grouped batch to the unchanged i.i.d. batch: 32 origins x 32 training symbols live at that origin,
1 h bars only; minus weight x mean over groups (>= 8 usable rows) of the Pearson correlation, within the group, between the 24 h median
in return units and the realised 24 h log return winsorised at +-10 model sigma. Groups are mirrored as a whole. Pinball applies to
both batches. No other horizon gets a ranking term.

## 3. Promotion rule (validation only)

Delta = per-origin (seed-averaged candidate L/S net) - (seed-averaged incumbent L/S net), all validation origins of the four folds.
A candidate is promoted iff the 95 % interval of mean Delta is above zero AND each of its two seeds alone has a positive point Delta
against the seed-averaged incumbent. Among promoted candidates the largest point Delta wins. Since the cost term is the same for every
origin, Delta net equals Delta gross; the absolute net level of each arm is reported as well.

If no candidate is promoted, the candidate with the largest point Delta is still scored once on test (the owner asked for the test
comparison), labelled "not promoted" everywhere, and added to the leaderboard like every scored config.

## 4. One test scoring

Only the selected candidate (both seeds) and the incumbent retrain (both seeds) are predicted on the unseen folds, once. Hourly task,
h = 24, the benchmark's test rows (stride 5, 64 symbols, F1-F4). Reported with paired 95 % intervals (same bootstrap, 7-day blocks):
rank IC, L/S gross, net at 4 bp and at 20 bp, for the candidate (seed-averaged series and each seed) against (a) the incumbent retrain
(seed-averaged), (b) the registered `bmt2-ohlc+cal~8bf4ed9e` and its `-sym`, (c) `timestral-official` and `timestral-official-2048`.
Seen / unseen split reported. Primary test number: Delta L/S net at 4 bp vs (a). All rows go to the registry and `agent/LEADERBOARD.md`.

## 5. Does it add to the bots' trend signal (research evaluation only; nothing is staged or deployed)

Forecasts: for every coin of the ALGOS pooled harness (`/nvme0n1-disk/code/bgm-signal-wt/tools/algos`, universes CORE + R1-R8,
36 coins without PAXG, all in the training pool's symbol set), daily at 00:00 UTC from 2023-01-01 to 2026-10-31: origin = the hourly
bar closing at 00:00, 24 h median from the fold model of that date (seed-averaged), for the selected candidate and the incumbent.

1. Correlation: per day, cross-sectional Spearman between the forecast median and the harness trend score (`cv.F.tscore`, mean of the
   12 base signals) on the coins of each universe; mean over days and universes, with a 7-day block interval. Same for the forecast's
   and the trend score's IC against the realised next-day return on those coins.
2. Blend arm on the harness: held weights x max(0, 1 + b x r), r = cross-sectional rank of the forecast median among the universe's
   non-PAXG coins with a forecast, scaled to [-0.5, +0.5]; b = 1; row gross kept (`al.mult_post`); r = 0 outside the forecast window.
   Off arm = `base`. Statistic = the harness's pooled paired test (`al.pooled`) restricted to the months 2023-01 .. 2026-10, both
   costs (c7, c10); placebo = 10 draws with the forecast ranks permuted across coins within each day (family mean d Sharpe per draw);
   class by `al.klass` with the placebo band. Harness outputs go to `runs-agent/r24/algos` (CORRVOL_D); the harness's own data
   directories are read only.
3. If the harness cannot be run unchanged, a clearly labelled proxy instead: equal-weight long book of coins with trend score > 0.5,
   weights x (1 + b r), next-day return, same window, paired block bootstrap.

## 6. Limits and constraints

GPU: <= 4 GB (cap 3.2 GB), waits for head-room, no GPU 15:20-16:15 America/New_York on weekdays (`BMT_GPU_BLACKOUT`). CPU <= 12
threads, nice 10. Research tier: weights trained on Binance Vision archives (CC BY-NC-SA) are not for the bots; section 5 is an
evaluation of whether the signal would add, not an integration.
