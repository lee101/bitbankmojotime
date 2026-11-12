# Contributing

Status: private pre-release (code under Apache-2.0, see `LICENSE` / `NOTICE`); outside contributions open once the repository is public.

## Ground rules

- No secrets, account details, trading strategy code or strategy parameters. No market data files and no model weights in git
  (`.gitignore` blocks the usual extensions; the only binary files allowed are the small self-test vectors under `tests/vectors/`).
- `git config core.hooksPath hooks` once after cloning: the pre-commit hook scans staged changes with gitleaks, the pre-push hook
  scans the whole history. CI runs the same scan and the tests.
- The benchmark protocol (`bench/PROTOCOL.md`) is fixed before results. A change to it is a commit of its own, listed under
  "Amendments" with the reason, and every affected number is regenerated.
- A model enters `bench/LEADERBOARD.md` only through `bench/bmtbench/score.py`, on the same rows as every other model, with the
  leakage checks of protocol section 6 passing. Every configuration that was scored on the test folds is listed, including the
  ones that lost.
- Speed numbers state the hardware, the co-tenant load, batch size and whether data transfer is included.

## Layout

| path | what |
|---|---|
| `src/bmt.mojo`, `src/bmt_gpu.mojo` | inference (CPU SIMD, GPU), self-test, bench and pipe-server modes |
| `train/` | PyTorch reference model, walk-forward trainer, weight export + self-test vector |
| `bench/bmtbench/` | data panel, folds, rows, metrics, bootstrap, baselines, scoring, speed |
| `data/` | fetch scripts for public archives (nothing downloaded is committed) |
| `tests/` | metric and leakage tests on synthetic data, Mojo-vs-PyTorch parity vectors |
| `docs/` | scope / licences, plan, model card draft, release checklist |

## Running

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy scipy pytest          # tests only
uv pip install --python .venv/bin/python torch safetensors statsforecast arch chronos-forecasting   # full benchmark
.venv/bin/python -m pytest -q tests
python3 data/fetch_binance.py list && python3 data/fetch_binance.py fetch --symbols data/cache/fetch_syms.txt && python3 data/fetch_binance.py build
PYTHONPATH=bench .venv/bin/python -m bmtbench.prep                                    # universe + benchmark rows
PYTHONPATH=bench .venv/bin/python -m bmtbench.run --task hourly --models naive,ewma,garch
PYTHONPATH=bench .venv/bin/python -m train.train --name bmt-s --size s
PYTHONPATH=bench .venv/bin/python -m bmtbench.score
./build.sh gpu && ./build/bmt selftest tests/vectors/bmt-s
```
