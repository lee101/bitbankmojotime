import numpy as np
from scipy import stats
from bmtbench.data import QLEV
from bmtbench.metrics import CRPS_W, Boot, pinball_levels, summarize, ci


def gauss(n=200000, seed=0):
    r = np.random.default_rng(seed)
    y = r.normal(0, 1, n).astype(np.float32)
    q = np.tile(stats.norm.ppf(QLEV), (n, 1)).astype(np.float32)
    return q, y


def test_levels_and_weights():
    assert len(QLEV) == 21 and QLEV[0] == 0.01 and QLEV[10] == 0.5 and QLEV[-1] == 0.99
    assert abs(CRPS_W.sum() - 1) < 1e-12


def test_pinball_matches_theory_for_perfect_gaussian():
    q, y = gauss()
    got = pinball_levels(q, y).mean(0)
    want = stats.norm.pdf(stats.norm.ppf(QLEV))                      # E[pinball_tau] of the true quantile of N(0,1)
    assert np.abs(got - want).max() < 5e-3


def test_summary_calibrated_and_wrong_scale_is_worse():
    q, y = gauss()
    n = len(y)
    a, blk, t = np.arange(n) % 7, np.arange(n) // 500, np.arange(n) // 7
    s, v = summarize(q, y, np.ones(n, np.float32), a, blk, t)
    assert abs(s["c80"][0] - 0.8) < 0.01 and abs(s["c98"][0] - 0.98) < 0.005 and s["calib"] < 0.005
    assert s["sPL"][1] <= s["sPL"][0] <= s["sPL"][2]
    b = Boot(a, blk)
    s2, v2 = summarize(q * 1.5, y, np.ones(n, np.float32), a, blk, t, b)
    s1, v1 = summarize(q, y, np.ones(n, np.float32), a, blk, t, b)
    d = ci(v2["sPL"] - v1["sPL"])
    assert d[1] > 0                                                  # paired interval excludes zero: the mis-scaled forecast loses
    assert np.all(v1["sPL"] - v1["sPL"] == 0)


def test_bootstrap_is_paired_and_seeded():
    a, blk = np.arange(1000) % 5, np.arange(1000) // 50
    b1, b2 = Boot(a, blk), Boot(a, blk)
    v = np.random.default_rng(1).normal(size=1000)
    assert np.array_equal(b1.pooled(v), b2.pooled(v))
    assert abs(b1.pooled(v)[0] - v.mean()) < 1e-12
    assert abs(b1.asset_mean(v)[0] - np.mean([v[a == i].mean() for i in range(5)])) < 1e-12
