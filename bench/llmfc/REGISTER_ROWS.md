# Register rows for predictionmojo (to merge; this worktree does not touch the register)

## experiments/ledger.csv (12 columns)

```
PMLLMFC,pm_llmfc_general_llm_low_thinking_as_frontier_forecaster,"general LLM over openpaths (deepseek-v4-flash non-thinking, gpt-6.1-sol / -terra reasoning low) given the last 256 hourly log returns in bp (no asset / dates) asked for the 21 quantiles at h 1 / 6 / 24, scored with the bitbankmojotime bench metric code on a preregistered 600-row subsample (4 folds x 10 weekly blocks x 15 assets, seed 20261124; 150-row secondary) against bmt2-ohlc+cal-sym / naive / timestral, paired block bootstrap (prereg research/llm-frontier 21f259b + A1-A3 e7bb12f 6346862 8b320cf): flash mean-h sPL 0.7869 vs bmt2 0.7621, paired +0.0248 [+0.0095, +0.0439] worse, vs naive -0.0133 [-0.0339, +0.0131] n.s.; sol low +0.0305 [+0.0154, +0.0497], terra low +0.0312 [+0.0175, +0.0497] worse than bmt2, both worse than the random walk at h24 (+0.054 / +0.053 separated; medians 3x too large, 90% cov 83-89%); refine with the bmt2 quantiles as hint: flash copies the hint verbatim 600/600 (no lead), sol edits it and degrades +0.0185 [+0.0071, +0.0309]; owner's Rule A (<= bmt2 upper unpaired bound) passed by every arm and by naive, non-discriminating; model-output failures 0%, gateway 429 re-queued; USD 2.44 of 5; at low effort deepseek burns > 2.5k tokens hand-computing the std (0/7 parsed), ~USD 0.01/row for real thinking vs the 0.11M bmt2; bitbankmojotime bench/llmfc/RESULTS.md",FAIL,,,,,,,,FAIL
```

## docs/innovations.md, "Tried on the other bots" table

```
| 2026-11-24 | crypto forecaster (bitbankmojotime bench, 64 Binance USDT names, hourly h 1 / 6 / 24, folds 2023-26), 600-row preregistered subsample + 150-row secondary | general LLM as the forecaster (deepseek-v4-flash non-thinking; gpt-6.1-sol / -terra reasoning low; 256 returns in bp, no asset / dates) and as a "refine" agent on top of the bmt2 quantiles | FAIL as frontier (paired-worse than bmt2 in every arm, not better than the random walk, worse than it at h 24); refine: no lead (flash copies the hint 600/600) / degrades (sol +0.0185 [+0.0071, +0.0309] sPL) | flash mean-h sPL 0.7869 vs bmt2 0.7621 (+0.0248 [+0.0095, +0.0439]); sol +0.0305 [+0.0154, +0.0497]; terra +0.0312 [+0.0175, +0.0497]; h 24 vs naive +0.054 / +0.053 separated; 0 % parse failures; USD 2.44 | bitbankmojotime branch research/llm-frontier, bench/llmfc/PREREG.md (21f259b, A1-A3) and RESULTS.md; ~/research_data/llm_spend/spend.jsonl slot PMLLMFC |
```
