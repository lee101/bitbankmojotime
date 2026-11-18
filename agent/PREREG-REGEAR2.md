# Pre-registration: confirmation of k = 1.20 on new universes (track REGEAR2)

Date 2026-11-18 (UTC). Committed before any R41-R56 panel is built and before any R41-R56 simulation or statistic. Done before this
commit, all on stored universes or without returns: the draw of the coin lists below, the merge of stored forecasts, a rebuild-identity
check and a code-path test on R9 (see "Checks"). Code `tools/r24regear.py` (`draw`, `fcmerge`, `build`, `run`, `ana2`); outputs
`runs-agent/regear2/`.

## Why

Wave 4f (PREREG-REGEAR) registered k* = 1.10; production switches the tilt on there. In that in-sample table every pooled risk
measure of `gt` is at or below base up to k 1.20 (worst month -1.0, 21-day -0.9, 60-day -1.5, maxDD -1.1, ruin 39.4 against 41.1) with
median d CAGR +2.7; the rule stopped at 1.10 because one live path (KuCoin settings) had a worst month 0.2 points above its base at
1.15. This run tests the k = 1.20 reading on universes the calibration never used. One look, no placebo, a gearing statement only.

## Universes R41-R56

The R9-R24 procedure (`bgm-signal-wt` SIGNAL CONFIRM-PREREG section 2): BTC, ETH, PAXG + 7 coins, 4 by `rng.choice` without
replacement from the 15 archive coins never used in R1-R8 (alphabetical), then 3 from the remaining non-M11 archive coins in order of
first archive bar. `draw_sets` reproduces all 16 R9-R24 sets from rng [20261115, 9, i] and all 16 R25-R40 sets from rng
[20261118, 25, i]. New seed: rng [20261119, 41, i], i = 0..15; a set equal to any R1-R40 set would be redrawn (none was). R25-R40 are
reserved by the HORIZON track and are not built or used here.

R41 AVAX COMP DASH ONE QTUM SOL XTZ | R42 DOT IOTA QTUM TFUEL UNI ZEC ZEN | R43 AVAX COMP DOGE DOT ONE SNX SOL |
R44 COMP IOTA RVN SOL ZEC ZEN ZRX | R45 AAVE AVAX QTUM SNX SOL ZEC ZIL | R46 BAT DASH DOT MANA ONE SAND ZIL |
R47 AAVE BCH DASH DOT QTUM SOL ZEN | R48 BAT DOGE DOT IOST SAND SOL ZEC | R49 ALGO DASH IOTA MANA QTUM UNI ZIL |
R50 AAVE ALGO AVAX BAT COMP DOT ZIL | R51 BAND IOTA NEO SNX SOL TFUEL ZEC | R52 COMP IOTA MANA ONE TFUEL VET ZEC |
R53 AAVE AVAX DOT MANA UNI ZIL ZRX | R54 AVAX BAND BCH COMP IOTA ONE SNX | R55 ATOM AVAX BCH COMP SNX ZEC ZEN |
R56 COMP DOT MANA ONE SNX SOL VET.

Built with `bgm-signal-wt/tools/shift/prep.py` unchanged (R-universe rules, live grid, trend-wf trail 1095, top 5 / 10 / 20) into our
own data directory (read-only links to the archive). A universe that fails to build is dropped and reported; the universe count
below then becomes 62.5 % of those built, rounded up.

Limits of "new": the same 46-coin archive and period as every earlier universe, all share BTC / ETH / PAXG, and every coin was already
in a calibration universe; only the combinations are new (each new set shares at most 5 of its 7 coins with any earlier set).

## Forecast

No new prediction is run. The frozen `mkt+r24` checkpoints already have stored predictions for every non-PAXG archive coin:
`runs-agent/r24/botpred.npz` (36 coins) and `botpred_conf.npz` (30 coins, wave 4c). `fcmerge` takes the first file and adds the nine
coins only in the second (`runs-agent/regear2/botpred_r41.npz`, all 31 needed coins present). The 21 coins in both files agree to
1.9e-8 (two prediction runs of the same weights), so the choice of file does not matter. Prediction only, nothing trained.

## Arms

base k 1 (simulated here with the same binary, paired); `gt` at k 1.00, 1.10, 1.20, 1.30; pure gearing (no tilt) at 1.10, 1.20, 1.30
and, for the equal-drawdown path, 0.75, 0.80, 0.85, 0.90, 0.95. Costs c7 and c10, window 2023-01 .. 2026-10, the 3,003-configuration
family. k is applied exactly as in PREREG-REGEAR (same functions: pv and cap x k before the tilt, tilt band 0.05, rows down only).

## Statistics

Definitions as PREREG-REGEAR, pooled over R41-R56 (16 x 3,003 = 48,048 cells): medians of levels and of paired differences against
base k 1 for CAGR, Sharpe, gross, volatility, worst month, 21-day, 60-day and max drawdown; turnover; mean over universes of the
family-mean d CAGR and d Sharpe with the shared month-block bootstrap (2,000 draws). Per universe: median paired d CAGR and median
paired d worst month.

Ruin view: the `algos/ruin.py` scenarios (S1-S5, P(max drawdown > 60 %)) on each new universe at 27 of the 99 ruin points (every
fourth of the first 96 sample points and the three anchors; fewer points for compute), for base and the `gt` arms. Pooled ruin
statistic = median over the 16 x 27 universe-point cells of the worst S1-S5 probability; the mean is reported too. The M11 ruin row of
wave 4f is in sample and not part of this rule.

## Rule (fixed now)

Evaluated at a given k, at c7 and at c10:
- (risk) the pooled median paired difference of worst month, of 60-day drawdown and of max drawdown are each <= 0, and the pooled ruin
  statistic of `gt`(k) is not above base's;
- (gain) the mean paired d CAGR is > 0 (point estimate; the interval is reported), and at least 10 of the 16 universes have a positive
  median paired d CAGR together with a median paired d worst month <= 0.

The share criterion first proposed (>= 60 % of configurations with higher CAGR at not-higher worst month) is replaced by the universe
count above before the run: on the calibration data itself that share was 52 % / 51 % at k 1.20 and never above 52 % on the grid (one
configuration's worst month is a single noisy month), so 60 % could not be met even by a perfect replication. The universe count is
its sensible form (wave 4f at 1.20: 24 of 30 universes with median CAGR up, 27 of 30 with median worst month not up). The
configuration share is still reported.

Verdict: **"k 1.20 confirmed"** if (risk) and (gain) hold at k 1.20 at both costs; otherwise **"keep 1.10"** if (risk) holds at
k 1.10 at both costs (1.10 was registered on risk not above base; its CAGR gain is small and is not required here); otherwise
**"lower"**. k 1.10 and k 1.30 are reported on the same footing (risk and gain). k 1.30 is not adopted from this run whatever it
shows: in wave 4f its max drawdown and ruin view were above base.

## Also reported, not deciding

- The frozen `gt` at k 1 on R41-R56 as one more out-of-sample look at the tilt: `al.pooled` mean d Sharpe with its interval, share,
  universes positive, years, the sign placebo. No shuffled-rank placebo is run here, so the `al.klass` class is given with the sign
  placebo only and labelled as such.
- Tilt minus pure gearing at equal worst month / 60-day drawdown / max drawdown / volatility (PREREG-REGEAR definition, control grid
  0.75 .. 1.30).
- Production for k 1.20 on Binance (descriptive): the trader-script change the executor's margin-level floor needs, and how many days
  the floor would clip without it.

## Checks (before this commit, no new-universe statistic)

- `buildcheck`: R9 rebuilt through our build path is byte-identical to the stored R9 (panel, lows, grid, the three SEL books and raw
  equities).
- Code-path test on that R9 copy (reduced grid): the base arm simulated here equals the stored base bit-for-bit, `gt` k 1 equals the
  wave 4c resume cache. R9's numbers are wave 4c / 4f data, nothing new.

Compute: <= 8 threads, no GPU, caches deleted after the JSONs are written.

## Amendment (2026-11-18, about 21:15 UTC): timing versus static preference of the frozen `gt`

Committed before any run of it, after the REGEAR2 result (wave 4g). It does not change the rule above. Reason: every placebo so far for
`gt` shuffled the forecast ranks across coins within a day, which removes the day-level timing and the forecaster's static coin
preference together; a parallel track found most of its weekly rank IC to be a static asset effect.

Frozen `gt` at k 1, c7 and c10, window as before. Universes: the 14 lead + R9-R24 (stored; primary, "stored30"), and R41-R56
(rebuilt deterministically, base re-simulated). Code `tools/r24regear.py` `run3`, `ana3`, `pref`; outputs `runs-agent/regear2/timing.json`.

1. Circular-shift placebo. Inside the forecast window (L days) every coin's forecast series is shifted by the same offset with
   wrap-around (day d gets the forecasts of day d + s) and ranks are recomputed per day, so each day's cross-section is a real one from
   another date: every coin's forecast distribution and the static cross-sectional preference are kept, the alignment with the day is
   destroyed. 20 offsets s_j = round((j + 1) L / 21), j = 0..19 (each at least L / 21, about 66 days, from the true alignment on both
   sides), the same for all universes; a pooled placebo value = `gt` with offset j on all universes. For compute the placebo arms run
   on every third configuration of the family (1,001; on M11 also the ruin points); the real `gt` is reported on the full family and,
   for every paired comparison, on the same 1,001. Reported: placebo mean d Sharpe distribution (mean, sd, min .. max, p95), the real
   value's rank among the 21, real minus placebo mean with the paired month-block bootstrap interval and with the range over offsets.
2. Static-preference arm. The daily rank is replaced by each coin's expanding-window mean rank fraction over the forecast days before
   the day (no look-ahead), on the same (day, coin) cells: `sr` = its cross-sectional rank fraction among those coins (the multiplier
   spread of `gt`; primary), `st` = the mean rank fraction itself as written in the request (compressed towards 0.5, so a weaker tilt).
   Full family, same form. With it a description of the preference (`pref`): mean rank per coin, between-coin share of the rank
   variance, half-sample persistence, rank correlation with trailing 30-day volatility, 30-day return and 30-day quote volume (size
   proxy; no market capitalisation in the archive).
3. The same arms for the pooled risk measures: median paired worst month, 21-day, 60-day and max drawdown, volatility ratio, and the
   ruin view on M11 (27 of the 99 ruin points for base, real, every offset and the static arms).

Wording of the result, on stored30 at each cost: timing share X = 1 - (placebo mean) / real; static-preference share Y = (placebo mean
- within-day-shuffle mean) / real; the remainder Z = within-day-shuffle mean / real is the form's mechanics (stored shuffle means of
waves 4b / 4c, 10 and 20 draws, weighted 14 : 16). "Timing carries X of the gain, static preference Y". On R41-R56 there is no
within-day shuffle, so only X is given there. The `sr` arm is the no-look-ahead check of Y + Z. No pass / fail bar: a decomposition.
If the deadline cuts the R41-R56 part, the offsets completed on all 16 universes are reported with their count.
