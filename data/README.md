# Data

Nothing in this directory is market data. `fetch_binance.py` downloads public Binance spot klines from `data.binance.vision`
into `data/cache/` (git-ignored) and verifies every archive against its published SHA-256 checksum.

`universe.json` is the mechanical symbol selection of `bench/PROTOCOL.md` section 1 (pool, evaluation universe, held-out
assets), derived from the archive index and 2021 quote volumes.

Source and terms: Binance Vision (`data.binance.vision`), provided under the Binance Vision Dataset Terms, CC BY-NC-SA 4.0,
non-commercial. Anything derived from the archives (this symbol list, the benchmark tables, weights trained on them) carries the
same attribution and licence. This project is not affiliated with or endorsed by Binance. Keep the download concurrency modest.
See `docs/SCOPE.md` section 2 before using the data for anything other than research and evaluation.
