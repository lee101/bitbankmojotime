# Release checklist (licence, NOTICE and history scan done 2026-11-12; weights and CI items still open)

Owner decisions first (docs/PLAN.md section 7): name, code licence, data policy / weight tier, Timestral derivatives.

## Before the repository goes public
- [ ] `LICENSE` replaced by the chosen licence; `NOTICE` with: Timestral (Mistral AI, Modified MIT, weights not included),
      Chronos-2 (Amazon, Apache-2.0), StatsForecast (Nixtla, Apache-2.0), Binance Vision attribution + CC BY-NC-SA 4.0 for
      the derived tables and `data/universe.json`, "not affiliated with or endorsed by Binance".
- [ ] `gitleaks git --redact -c .gitleaks.toml` clean over the WHOLE history, run on the exact commit that will be pushed.
- [ ] `git log --format='%an %ae' | sort -u` shows only the identity already public in `lee101/bitbank-mojo`.
- [ ] `git ls-files` reviewed: no data, no trained weights (only `tests/vectors/bmt-t-random`, random weights), no host paths
      that matter, no strategy code or parameters, no keys or key hashes.
- [ ] Legal read of "non-monetized" (Binance Vision terms 4.1) against a README that links bitbank.nz; decide wording.
- [ ] CI green on GitHub (gitleaks + tests); branch protection on `main`.
- [ ] Benchmark reproducible from a clean clone: fetch -> prep -> baselines -> train -> score reproduces the leaderboard within
      the stated intervals; the placebo and leakage tests pass.
- [ ] Speed table regenerated on an idle GPU (the published one states the co-tenant load).

## Before any weights are published (Hugging Face or elsewhere)
- [ ] Tier decided: research (CC BY-NC-SA 4.0, Binance-trained) and / or permissive (synthetic + commercially licensed data).
- [ ] Model card (`docs/MODEL_CARD.md`) completed for that tier, with its own LICENSE file in the model repo.
- [ ] Self-test vector shipped with the weights; `bmt selftest` passes on CPU and on each GPU arch build.
- [ ] Weights hash recorded in the model card and in the repo release notes.
- [ ] No Timestral-derived weights unless under Mistral's Modified MIT licence with its notice.
- [ ] gitleaks over the model repo as well.

## After
- [ ] bitbank.nz link and benchmark claims match the leaderboard commit they cite.
- [ ] Autoresearch ledger published with every scored configuration.
