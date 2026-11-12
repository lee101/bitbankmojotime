"""Causality of the feature bank on the real archive (skipped without the data cache or a GPU-capable torch)."""
import os
import numpy as np
import pytest
from bmtbench import data as D

HAVE = os.path.exists(os.path.join(D.CACHE, "hourly", "BTCUSDT.npz"))


HAVE5 = os.path.exists(os.path.join(D.CACHE, "rv5", "BTCUSDT.npz"))


@pytest.mark.skipif(not HAVE, reason="no data cache")
@pytest.mark.parametrize("use5m", [False, True])
def test_no_input_depends_on_bars_after_the_origin(use5m):
    torch = pytest.importorskip("torch")
    from train.bank import Bank
    if use5m and not HAVE5:
        pytest.skip("no 5-minute cache")
    syms = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT", "ADAUSDT", "TRXUSDT", "XLMUSDT"]
    b = Bank(syms, syms[:6], "cpu", use5m=use5m)
    t0 = D.eh(2024, 6, 1) - b.t0
    a = torch.tensor([0, 2, 4, 3])
    t = torch.tensor([t0, t0 + 7, t0 + 100, t0 + 23])
    before = {k: {n: v.clone() for n, v in b.gather(a, t, k, fold=2).items()} for k in (1, 24)}
    if use5m:
        assert float(before[1]["fine"][:, 11].abs().sum()) > 0 and float(before[1]["fine"][:, 13].abs().sum()) > 0   # rv5 and peer are live
    cut = int(t.max()) + 1
    for n, g in b.g.items():                                         # poison everything strictly after the latest origin
        if g.dim() == 2:
            g[:, cut + (1 if n in ("cs2", "cs5", "csn") else 0):] = 7.0
        else:
            g[cut + (1 if n == "mkt" else 0):] = 7.0
    last = int(t.argmax())                                           # only the row whose origin is the latest must be unchanged for sure
    for k in (1, 24):
        after = b.gather(a, t, k, fold=2)
        for n in ("fine", "coarse", "scal", "cal"):
            assert torch.equal(before[k][n][last], after[n][last]), (k, n)
    y = b.gather(a, t, 1, (1, 6))["y"]
    assert not torch.equal(y[last], torch.zeros(2))                  # the target does see the poisoned future (the test has power)
