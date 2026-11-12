"""Leakage checks of bench/PROTOCOL.md section 6 on a synthetic panel (no market data needed)."""
import json, os
import numpy as np
import pytest
from bmtbench import data as D
from bmtbench import baselines as B
from bmtbench.metrics import Boot, summarize, ci


def synth(n_assets=4, seed=0, t0=None, t1=None):
    t0, t1 = t0 or D.eh(2021), t1 or D.eh(2026, 11)
    r = np.random.default_rng(seed)
    n = t1 - t0
    vol = 0.004 * np.exp(np.cumsum(r.normal(0, 0.05, (n_assets, n)), 1) * 0.3 - 1e-9)
    vol = np.clip(vol, 0.0005, 0.03)
    logp = np.cumsum(r.standard_t(4, (n_assets, n)) * vol, 1) + 5
    real = np.ones((n_assets, n), bool)
    real[1, 30000:30004] = False                                     # an outage on one asset
    logp[1, 30000:30004] = logp[1, 29999]
    return D.Panel(["BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT"][:n_assets], t0, logp, real)


@pytest.mark.parametrize("task", ["hourly", "daily"])
def test_rows_are_inside_test_periods_and_targets_real(task):
    ph = synth()
    p = ph if task == "hourly" else D.to_daily(ph)
    rows, cfg = D.build_rows(p, task), D.TASKS[task]
    assert len(rows["t"]) > 1000
    for fi, (_, s, e) in enumerate(D.fold_bounds(task)):
        m = rows["fold"] == fi
        ta = rows["t"][m] + p.t0
        assert ta.min() >= s and (ta + max(cfg["horizons"])).max() < e
        assert ((ta % cfg["stride"]) == 0).all()
    for j, h in enumerate(cfg["horizons"]):
        assert p.real[rows["a"], rows["t"] + h].all()
        assert np.allclose(rows["y"][:, j], p.logp[rows["a"], rows["t"] + h] - p.logp[rows["a"], rows["t"]], atol=1e-6)


def test_validation_is_purged_from_test():
    for task in D.TASKS:
        for (_, vs, ve), (_, s, _) in zip(D.val_bounds(task), D.fold_bounds(task)):
            assert ve == s - D.TASKS[task]["purge"] and vs < ve


def test_context_is_causal():
    """changing any bar after the origin leaves every model input of that row bit-identical."""
    p = synth()
    rows = D.build_rows(p, "hourly")
    i = np.arange(0, len(rows["t"]), 997)
    a, t = rows["a"][i], rows["t"][i]
    before = [D.ctx_ret(p, a, t), D.ctx_logp(p, a, t), D.ctx_agg(p, a, t, 6, 256), D.ctx_agg(p, a, t, 24, 256)]
    tmax = int(t.max())
    for k, tt in enumerate(t):                                       # poison everything after each origin, row by row
        lp = p.logp.copy()
        lp[a[k], tt + 1:] += 1.0
        q = D.Panel(p.syms, p.t0, lp, p.real)
        after = [D.ctx_ret(q, a[k:k + 1], t[k:k + 1]), D.ctx_logp(q, a[k:k + 1], t[k:k + 1]), D.ctx_agg(q, a[k:k + 1], t[k:k + 1], 6, 256),
                 D.ctx_agg(q, a[k:k + 1], t[k:k + 1], 24, 256)]
        for x, y in zip(before, after):
            assert np.array_equal(x[k:k + 1], y)
        if k > 40:
            break
    assert tmax < p.T


def test_unseen_split_is_deterministic_and_disjoint():
    assert not D.is_unseen("BTCUSDT") and not D.is_unseen("ETHUSDT")
    assert D.is_unseen("LTCUSDT") == D.is_unseen("LTCUSDT")
    up = os.path.join(D.REPO, "data", "universe.json")
    if os.path.exists(up):
        u = json.load(open(up))
        assert not set(u["unseen"]) & set(u["train"]) and set(u["unseen"]) <= set(u["eval"])
        assert all(D.is_unseen(s) for s in u["unseen"]) and not any(D.is_unseen(s) for s in u["seen"])
        assert len(u["eval"]) == 64


def test_training_sampler_respects_the_purge():
    torch = pytest.importorskip("torch")
    from train.train import Sampler, PURGE, VAL, KS
    p = synth()
    for name, s_abs, _ in D.FOLDS:
        limit = s_abs - PURGE - VAL - PURGE - p.t0
        smp = Sampler(p, limit, 512, 24, "cpu")
        assert smp.max_target < limit
        r, y, b, cal = smp.batch(4096, (1, 5, 6, 24), cov=True)
        assert r.shape == (4096, 512) and y.shape == (4096, 4) and torch.isfinite(r).all() and torch.isfinite(y).all()
        assert smp.max_target + p.t0 < s_abs - PURGE                 # no training target inside the purge or the test period


def test_shuffle_time_keeps_segment_distribution_and_kills_order():
    p = synth(2)
    q = D.shuffle_time(p, 1)
    cuts = [0] + [s - p.t0 for _, s, _ in D.FOLDS] + [p.T]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        x, y = np.diff(p.logp[0])[max(lo - 1, 0):hi - 1], np.diff(q.logp[0])[max(lo - 1, 0):hi - 1]
        assert np.allclose(np.sort(x), np.sort(y), atol=1e-9)
    ac = lambda r: np.corrcoef(np.abs(r[1:]), np.abs(r[:-1]))[0, 1]
    a0, a1 = ac(np.diff(p.logp[0])[:20000]), ac(np.diff(q.logp[0])[:20000])
    assert a0 > 0.05 and abs(a1) < a0 / 2


def test_placebo_check_has_power_against_a_leaking_model():
    """a model that peeks at the target must show a large skill vs the empirical baseline (so the check can fail), an honest
    unconditional model on shuffled data must not."""
    p = D.shuffle_time(synth(), 3)
    rows = D.build_rows(p, "hourly")
    i = np.arange(0, len(rows["t"]), 23)
    rows = {k: v[i] for k, v in rows.items()}
    emp = B.empirical(p, rows, "hourly")[:, 0]
    honest = B.naive(p, rows, "hourly")[:, 0]
    leak = emp * 0.5 + 0.5 * rows["y"][:, :1]
    blk = (rows["t"] + p.t0) // 168
    boot = Boot(rows["a"], blk)
    v = {n: summarize(q, rows["y"][:, 0], rows["scale"], rows["a"], blk, rows["t"], boot)[1]["sPL"] for n, q in (("emp", emp), ("honest", honest), ("leak", leak))}
    assert ci(1 - v["leak"] / v["emp"])[1] > 0.10                    # leak: far above the 1 % allowance
    assert ci(1 - v["honest"] / v["emp"])[2] < 0.01                  # honest: passes
