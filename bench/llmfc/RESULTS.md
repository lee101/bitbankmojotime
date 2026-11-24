# RESULTS: general LLM at low thinking vs the purpose-trained forecaster (PMLLMFC, 2026-11-24)

## Headline

FAIL. On the preregistered 600-row hourly subsample (bench/llmfc/PREREG.md; same rows, same metric code, paired block
bootstrap) a general LLM given the last 256 log returns is not a frontier forecaster: `deepseek-v4-flash` (non-thinking,
A1) is paired-worse than `bmt2-ohlc+cal-sym` by +0.0248 sPL [+0.0095, +0.0439] (mean of h 1 / 6 / 24; 3.3 % worse) and is
not separated from the zero-mean random walk (-0.0133 [-0.0339, +0.0131]). The two low-thinking arms on the 150 secondary
rows are the same: `gpt-6.1-sol` +0.0305 [+0.0154, +0.0497] and `gpt-6.1-terra` +0.0312 [+0.0175, +0.0497] worse than bmt2,
and at h 24 both are paired-worse than the random walk (+0.054 / +0.053, intervals exclude zero). At h 1 every LLM lands
between `naive` and `timestral-official` (sPL about timestral-fast level); at h 24 their medians are 3x too large
(mean |median| 36 bp vs 12 bp for bmt2) and the 90 % interval covers 83-89 % instead of 90-91 %.

Rule A (owner's sealed rule, sPL <= bmt2's unpaired upper bound 0.9305 on 600 rows / 1.2083 on 150 rows) is satisfied by
every arm, but as recorded before any call it is also satisfied by `naive`, so it certifies nothing; Rule B (paired) fails
for every arm. Rule B's second clause (beats naive paired) is not met by any arm either.

Refine (agents on top of the forecaster): `llm-flash-refine` returned the bmt2 hint verbatim in 600 / 600 rows (mean
|delta| 2e-6 bp): identical scores, no lead, no harm. `llm-sol-refine` (thinking model, 150 rows) edits the hint (mean
|delta| 12.6 bp, 95 % of rows changed) and DEGRADES it: +0.0185 [+0.0071, +0.0309] sPL vs bmt2 alone, driven by h 24
(+0.045 [+0.018, +0.073]); h 1 is unchanged (+0.0007 [-0.0022, +0.0040]). Verdict: "no lead" (flash), "degrades" (sol).

Cost: USD 2.44 of the 5.00 cap (peak prices, incl. 5 + 4 + 5 pilot / debug calls); every call is in
`~/research_data/llm_spend/spend.jsonl` (slot PMLLMFC). Model-output failure rate 0 % in every arm after the one allowed
retry; gateway 429 rate limits hit 22-47 % of first-pass rows at 2 workers and were re-queued at 1 worker (A3) until served.

## Design recap

- Rows: 600 = 4 folds x 10 weekly blocks x 15 assets (BTC, ETH, 7 seen, 6 unseen) at one stride-5 origin per block; seed
  20261124; 150-row secondary subset. Scorer = `bmtbench.metrics.summarize` + `Boot` on the subsample rows
  (`score_sub.py`); full-set validation reproduced the leaderboard `naive` h1 sPL 0.2697 [0.2587, 0.2820] exactly.
- Prompt: 256 returns in integer bp, horizons {1, 6, 24}, 21 levels, JSON out; no asset, venue or dates. A1 added the
  context mean / std (256 / 64 / 24 bars) and the 24-bar sum because flash at `low` spent > 2,500 tokens hand-computing
  the std (0 / 7 parsed); flash arms therefore run with `reasoning_effort: none`, sol / terra at `low` (bounded).
- Failures: a row that cannot be parsed after one retry would carry the naive forecast (none occurred).
- IC is undefined on the 150-row subset (< 10 assets per timestamp); it is reported on the 600 rows (15 per timestamp).
- Caveats: 40 bootstrap blocks (wide unpaired intervals, adequate paired power: bmt2 vs timestral-official is separated
  on these rows as on the full set); BTC / ETH return strings could in principle be recognised by the model (not
  controlled); deepseek's pretraining cut-off may cover F1-F3, which only favours the LLM; one prompt design, temperature 0.

## Run summary

| arm | model | effort | rows | served / parsed | model failure rate | tokens in / out | cost USD |
|---|---|---|---|---|---|---|---|
| llm-flash | deepseek-v4-flash | none (A1) | 600 | 600 / 600 | 0.0 % | 521,521 / 194,308 | 0.390 |
| llm-flash-refine | deepseek-v4-flash + bmt2 hint | none (A1) | 600 | 600 / 600 | 0.0 % | 632,180 / 111,413 | 0.323 |
| llm-sol | gpt-6.1-sol | low | 150 | 150 / 150 | 0.0 % | 130,538 / 43,800 | 0.699 |
| llm-terra | gpt-6.1-terra | low | 150 | 150 / 150 | 0.0 % | 130,538 / 39,671 | 0.329 |
| llm-sol-refine | gpt-6.1-sol + bmt2 hint | low | 150 | 150 / 150 | 0.0 % | 158,376 / 33,904 | 0.656 |

Token counts include the re-queued 429 attempts the gateway billed (usage reported with the 429 is zero). The gate for
the secondary model (flash within 10 % of bmt2: 0.7869 / 0.7621 = 1.033) would have opened anyway; A2 made it unconditional.

## Tables

sPL = asset-equal scaled pinball (lower is better), paired dsPL = sPL(model) - sPL(reference) on the shared bootstrap
(negative = better than the reference; interval excluding zero = separated), MASE asset-equal, IC cross-sectional Spearman,
coverage of the q05-q95 interval. Full reference rows (empirical, chronos-2, chronos-2-ret, timestral-fast / -max,
bmt2-ohlc+cal@t-kd) are in `results_tables.md` / `scores_sub.json`.
### Preregistered 600-row subsample (600 rows)

| model | h | sPL [95 %] | paired dsPL vs bmt2 | paired dsPL vs naive | MASE [95 %] | IC | cov 90 % |
|---|---|---|---|---|---|---|---|
| naive | 1 | 0.2617 [0.2088, 0.3241] | +0.0198 [+0.0121, +0.0277] | ref | 0.6882 [0.5442, 0.8596] | nan | 93.0 |
| naive | 6 | 0.8419 [0.6265, 1.1148] | +0.0215 [+0.0003, +0.0409] | ref | 2.2384 [1.6946, 2.9185] | nan | 87.0 |
| naive | 24 | 1.2970 [1.0567, 1.5583] | +0.0731 [+0.0436, +0.1030] | ref | 3.4615 [2.7523, 4.2192] | nan | 91.3 |
| naive | mean | 0.8002 [0.6526, 0.9618] | +0.0381 [+0.0208, +0.0548] | ref | | | |
| timestral-official | 1 | 0.2499 [0.1943, 0.3148] | +0.0080 [+0.0045, +0.0117] | -0.0118 [-0.0190, -0.0044] | 0.6932 [0.5484, 0.8604] | -0.0269 | 92.5 |
| timestral-official | 6 | 0.8208 [0.6061, 1.1015] | +0.0003 [-0.0120, +0.0110] | -0.0212 [-0.0366, -0.0045] | 2.2300 [1.6927, 2.8960] | -0.0186 | 87.2 |
| timestral-official | 24 | 1.2590 [1.0032, 1.5316] | +0.0351 [+0.0099, +0.0604] | -0.0380 [-0.0682, -0.0055] | 3.4621 [2.7749, 4.2164] | 0.0318 | 91.5 |
| timestral-official | mean | 0.7765 [0.6232, 0.9443] | +0.0145 [+0.0040, +0.0244] | -0.0237 [-0.0387, -0.0072] | | | |
| bmt2-ohlc+cal-sym | 1 | 0.2419 [0.1874, 0.3073] | ref | -0.0198 [-0.0277, -0.0121] | 0.6851 [0.5390, 0.8568] | 0.0746 | 93.5 |
| bmt2-ohlc+cal-sym | 6 | 0.8204 [0.6014, 1.1052] | ref | -0.0215 [-0.0409, -0.0003] | 2.2399 [1.6996, 2.9166] | 0.0243 | 86.0 |
| bmt2-ohlc+cal-sym | 24 | 1.2239 [0.9769, 1.4940] | ref | -0.0731 [-0.1030, -0.0436] | 3.4164 [2.7135, 4.1591] | 0.1095 | 90.7 |
| bmt2-ohlc+cal-sym | mean | 0.7621 [0.6070, 0.9305] | ref | -0.0381 [-0.0548, -0.0208] | | | |
| llm-flash | 1 | 0.2510 [0.1989, 0.3139] | +0.0091 [+0.0023, +0.0161] | -0.0107 [-0.0179, -0.0020] | 0.6954 [0.5520, 0.8608] | -0.0482 | 92.0 |
| llm-flash | 6 | 0.8318 [0.6115, 1.1189] | +0.0114 [-0.0080, +0.0340] | -0.0101 [-0.0317, +0.0165] | 2.2629 [1.7289, 2.9495] | -0.0388 | 82.3 |
| llm-flash | 24 | 1.2779 [1.0200, 1.5801] | +0.0540 [+0.0210, +0.0943] | -0.0191 [-0.0635, +0.0365] | 3.5586 [2.8618, 4.3236] | -0.1139 | 85.3 |
| llm-flash | mean | 0.7869 [0.6335, 0.9628] | +0.0248 [+0.0095, +0.0439] | -0.0133 [-0.0339, +0.0131] | | | |
| llm-flash-refine | 1 | 0.2419 [0.1874, 0.3074] | +0.0000 [-0.0000, +0.0001] | -0.0198 [-0.0277, -0.0121] | 0.6849 [0.5389, 0.8564] | 0.0787 | 93.5 |
| llm-flash-refine | 6 | 0.8204 [0.6013, 1.1052] | +0.0000 [-0.0000, +0.0000] | -0.0215 [-0.0408, -0.0002] | 2.2400 [1.6997, 2.9168] | 0.0394 | 86.0 |
| llm-flash-refine | 24 | 1.2239 [0.9769, 1.4940] | +0.0000 [-0.0000, +0.0000] | -0.0731 [-0.1030, -0.0436] | 3.4162 [2.7134, 4.1587] | 0.1007 | 90.7 |
| llm-flash-refine | mean | 0.7621 [0.6070, 0.9305] | +0.0000 [-0.0000, +0.0000] | -0.0381 [-0.0548, -0.0208] | | | |

### Secondary 150-row subset (150 rows)

| model | h | sPL [95 %] | paired dsPL vs bmt2 | paired dsPL vs naive | MASE [95 %] | IC | cov 90 % |
|---|---|---|---|---|---|---|---|
| naive | 1 | 0.3761 [0.2220, 0.5112] | +0.0147 [+0.0046, +0.0238] | ref | 0.9645 [0.5893, 1.2928] | nan | 88.0 |
| naive | 6 | 1.1420 [0.6673, 1.5440] | -0.0052 [-0.0259, +0.0192] | ref | 2.9827 [1.8964, 3.8990] | nan | 85.3 |
| naive | 24 | 1.4321 [1.1536, 1.6641] | +0.0282 [+0.0045, +0.0714] | ref | 4.1903 [3.3160, 4.8949] | nan | 92.0 |
| naive | mean | 0.9834 [0.7096, 1.2126] | +0.0126 [+0.0012, +0.0305] | ref | | | |
| timestral-official | 1 | 0.3714 [0.2101, 0.5125] | +0.0099 [+0.0030, +0.0148] | -0.0048 [-0.0163, +0.0066] | 0.9679 [0.5952, 1.3032] | nan | 88.7 |
| timestral-official | 6 | 1.1363 [0.6514, 1.5411] | -0.0108 [-0.0258, +0.0081] | -0.0057 [-0.0272, +0.0132] | 2.9416 [1.8603, 3.8202] | nan | 85.3 |
| timestral-official | 24 | 1.4312 [1.1499, 1.6614] | +0.0273 [-0.0069, +0.0603] | -0.0009 [-0.0417, +0.0246] | 4.1269 [3.2496, 4.7589] | nan | 90.7 |
| timestral-official | mean | 0.9796 [0.6962, 1.2090] | +0.0088 [-0.0050, +0.0218] | -0.0038 [-0.0229, +0.0100] | | | |
| bmt2-ohlc+cal-sym | 1 | 0.3615 [0.2028, 0.5004] | ref | -0.0147 [-0.0238, -0.0046] | 0.9653 [0.5859, 1.2951] | nan | 90.7 |
| bmt2-ohlc+cal-sym | 6 | 1.1472 [0.6590, 1.5577] | ref | +0.0052 [-0.0192, +0.0259] | 2.9872 [1.8941, 3.8917] | nan | 84.0 |
| bmt2-ohlc+cal-sym | 24 | 1.4039 [1.1099, 1.6419] | ref | -0.0282 [-0.0714, -0.0045] | 4.1005 [3.2290, 4.8205] | nan | 90.7 |
| bmt2-ohlc+cal-sym | mean | 0.9708 [0.6882, 1.2083] | ref | -0.0126 [-0.0305, -0.0012] | | | |
| llm-flash | 1 | 0.3693 [0.2129, 0.5072] | +0.0078 [-0.0031, +0.0181] | -0.0068 [-0.0153, +0.0033] | 0.9679 [0.5868, 1.2929] | nan | 88.0 |
| llm-flash | 6 | 1.1681 [0.6797, 1.5829] | +0.0209 [-0.0068, +0.0553] | +0.0261 [+0.0004, +0.0595] | 3.0332 [1.9497, 3.9820] | nan | 80.7 |
| llm-flash | 24 | 1.5026 [1.1644, 1.7903] | +0.0987 [+0.0443, +0.1696] | +0.0704 [+0.0022, +0.1342] | 4.2687 [3.3692, 5.0346] | nan | 82.7 |
| llm-flash | mean | 1.0133 [0.7133, 1.2607] | +0.0425 [+0.0155, +0.0752] | +0.0299 [+0.0006, +0.0607] | | | |
| llm-flash-refine | 1 | 0.3615 [0.2028, 0.5004] | +0.0000 [-0.0000, +0.0001] | -0.0146 [-0.0237, -0.0045] | 0.9647 [0.5852, 1.2949] | nan | 90.7 |
| llm-flash-refine | 6 | 1.1472 [0.6589, 1.5576] | -0.0000 [-0.0001, +0.0000] | +0.0052 [-0.0193, +0.0260] | 2.9872 [1.8940, 3.8923] | nan | 84.0 |
| llm-flash-refine | 24 | 1.4039 [1.1099, 1.6420] | +0.0001 [+0.0000, +0.0001] | -0.0282 [-0.0714, -0.0045] | 4.1006 [3.2289, 4.8208] | nan | 90.7 |
| llm-flash-refine | mean | 0.9709 [0.6882, 1.2083] | +0.0000 [-0.0000, +0.0001] | -0.0126 [-0.0305, -0.0012] | | | |
| llm-sol | 1 | 0.3684 [0.2111, 0.5101] | +0.0069 [+0.0006, +0.0134] | -0.0077 [-0.0159, +0.0027] | 0.9653 [0.5926, 1.2934] | nan | 88.0 |
| llm-sol | 6 | 1.1493 [0.6664, 1.5644] | +0.0022 [-0.0110, +0.0219] | +0.0073 [-0.0136, +0.0284] | 2.9829 [1.8959, 3.9233] | nan | 86.0 |
| llm-sol | 24 | 1.4864 [1.1910, 1.7287] | +0.0826 [+0.0409, +0.1298] | +0.0543 [+0.0040, +0.0893] | 4.3310 [3.4361, 5.0386] | nan | 89.3 |
| llm-sol | mean | 1.0014 [0.7139, 1.2413] | +0.0305 [+0.0154, +0.0497] | +0.0180 [-0.0042, +0.0367] | | | |
| llm-terra | 1 | 0.3697 [0.2117, 0.5124] | +0.0082 [+0.0024, +0.0144] | -0.0065 [-0.0150, +0.0038] | 0.9672 [0.5917, 1.2946] | nan | 88.7 |
| llm-terra | 6 | 1.1516 [0.6702, 1.5701] | +0.0044 [-0.0093, +0.0268] | +0.0096 [-0.0117, +0.0313] | 2.9972 [1.9094, 3.9128] | nan | 86.0 |
| llm-terra | 24 | 1.4848 [1.2014, 1.7292] | +0.0809 [+0.0447, +0.1234] | +0.0526 [+0.0084, +0.0867] | 4.3044 [3.4350, 5.0260] | nan | 88.7 |
| llm-terra | mean | 1.0020 [0.7146, 1.2416] | +0.0312 [+0.0175, +0.0497] | +0.0186 [-0.0013, +0.0364] | | | |
| llm-sol-refine | 1 | 0.3622 [0.2041, 0.5013] | +0.0007 [-0.0022, +0.0040] | -0.0140 [-0.0232, -0.0028] | 0.9650 [0.5862, 1.2907] | nan | 90.7 |
| llm-sol-refine | 6 | 1.1570 [0.6684, 1.5719] | +0.0098 [+0.0001, +0.0220] | +0.0150 [-0.0104, +0.0377] | 2.9928 [1.9038, 3.9061] | nan | 84.0 |
| llm-sol-refine | 24 | 1.4489 [1.1558, 1.6862] | +0.0450 [+0.0179, +0.0729] | +0.0168 [-0.0308, +0.0437] | 4.2325 [3.3663, 4.9442] | nan | 89.3 |
| llm-sol-refine | mean | 0.9894 [0.7013, 1.2284] | +0.0185 [+0.0071, +0.0309] | +0.0059 [-0.0169, +0.0214] | | | |

## Verdict against the sealed rules

| arm | rows | Rule A (<= bmt2 upper bound) | Rule B paired vs bmt2 (mean h) | paired vs naive (mean h) | verdict |
|---|---|---|---|---|---|
| llm-flash | 600 | pass (0.7869 <= 0.9305; naive also passes) | +0.0248 [+0.0095, +0.0439] worse | -0.0133 [-0.0339, +0.0131] n.s. | FAIL, not frontier |
| llm-sol (low) | 150 | pass (1.0014 <= 1.2083) | +0.0305 [+0.0154, +0.0497] worse | +0.0180 [-0.0042, +0.0367] n.s.; h24 worse than naive | FAIL, not frontier |
| llm-terra (low) | 150 | pass (1.0020 <= 1.2083) | +0.0312 [+0.0175, +0.0497] worse | +0.0186 [-0.0013, +0.0364] n.s.; h24 worse than naive | FAIL, not frontier |
| llm-flash-refine | 600 | - | +0.0000 (hint copied verbatim, 600 / 600) | = bmt2 | no lead |
| llm-sol-refine | 150 | - | +0.0185 [+0.0071, +0.0309] worse | +0.0059 [-0.0169, +0.0214] | degrades bmt2 |

What would change the picture: a thinking budget large enough to let the model do real volatility estimation costs
~USD 0.01+ per row (A1), i.e. the LLM is both worse and 10,000x more expensive per forecast than the 0.11M bmt2 model;
the agents-on-top idea needs a model that improves calibrated tails at h 24, which is exactly where both thinking arms
lose. Not a lead for the Alpaca or crypto stacks; the register row is in `REGISTER_ROWS.md`.

Paths: worktree `/nvme0n1-disk/code/bmt-llmfc-wt` (branch `research/llm-frontier`), scripts `bench/llmfc/{common,subsample,
score_sub,llm,results}.py`, predictions `runs/preds/hourly/llm-*.npy|json` (gitignored, naive outside the subsample),
subsample `bench/llmfc/subsample.json`, scores `bench/llmfc/scores_sub.json`, spend `~/research_data/llm_spend/spend.jsonl`.
