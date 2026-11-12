"""Metrics and block-bootstrap confidence intervals (bench/PROTOCOL.md section 5). Pure numpy."""
import numpy as np
from .data import QLEV, NQ

_d = np.diff(QLEV)
CRPS_W = np.zeros(NQ)
CRPS_W[:-1] += _d / 2
CRPS_W[1:] += _d / 2
CRPS_W /= (QLEV[-1] - QLEV[0])                                       # trapezoid weights on [0.01, 0.99], sum to 1


def pinball_levels(q, y):
    """q[N, 21], y[N] -> pinball loss per level [N, 21]."""
    d = y[:, None].astype(np.float64) - q
    return np.maximum(QLEV * d, (QLEV - 1) * d)


def row_stats(q, y, scale):
    """per-row quantities every aggregate is built from."""
    pl = pinball_levels(q, y)
    med = q[:, NQ // 2].astype(np.float64)
    return dict(pin=pl.mean(1), crps=2 * (pl * CRPS_W).sum(1), ae=np.abs(y - med), ae0=np.abs(y).astype(np.float64),
                spin=pl.mean(1) / scale, sae=np.abs(y - med) / scale,
                c80=((y >= q[:, 2]) & (y <= q[:, 18])).astype(np.float64), c90=((y >= q[:, 1]) & (y <= q[:, 19])).astype(np.float64),
                c98=((y >= q[:, 0]) & (y <= q[:, 20])).astype(np.float64),
                hit=(y[:, None] <= q), da_ok=((med != 0) & (y != 0)).astype(np.float64), da=((np.sign(med) == np.sign(y)) & (med != 0) & (y != 0)).astype(np.float64))


def ic(med, y, tkey, min_n=10):
    """mean cross-sectional Spearman correlation over timestamps with >= min_n rows and a non-constant median."""
    o = np.argsort(tkey, kind="stable")
    med, y, tk = med[o], y[o], tkey[o]
    cuts = np.flatnonzero(np.diff(tk)) + 1
    out = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(tk)]):
        if b - a < min_n:
            continue
        m, v = med[a:b], y[a:b]
        if m.std() == 0 or v.std() == 0:
            continue
        rm, rv = np.argsort(np.argsort(m)).astype(np.float64), np.argsort(np.argsort(v)).astype(np.float64)
        out.append(np.corrcoef(rm, rv)[0, 1])
    return float(np.mean(out)) if out else float("nan")


class Boot:
    """moving-block bootstrap over calendar blocks, shared by every model scored on the same rows (so differences are paired).
    Blocks are resampled jointly across assets. Asset-equal metrics = mean over assets of (sum of row values / row count)."""

    def __init__(self, asset, block, n_boot=1000, seed=20261112):
        self.au, self.ai = np.unique(asset, return_inverse=True)
        self.bu, self.bi = np.unique(block, return_inverse=True)
        self.A, self.B = len(self.au), len(self.bu)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.B, size=(n_boot, self.B))
        self.W = np.zeros((n_boot + 1, self.B))
        self.W[0] = 1.0                                              # row 0 = the point estimate
        for r in range(n_boot):
            self.W[r + 1] = np.bincount(draws[r], minlength=self.B)
        self.cnt_ab = self._ab(np.ones(len(asset)))
        self.cnt_b = self.cnt_ab.sum(0)

    def _ab(self, v):
        return np.bincount(self.ai * self.B + self.bi, weights=v, minlength=self.A * self.B).reshape(self.A, self.B)

    def pooled(self, v):
        """[n_boot + 1] pooled row mean of v under each resample (index 0 = point estimate)."""
        return (self.W @ self._ab(v).sum(0)) / (self.W @ self.cnt_b)

    def asset_mean(self, v):
        """[n_boot + 1] mean over assets of the asset row mean."""
        num, den = self.W @ self._ab(v).T, self.W @ self.cnt_ab.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmean(np.where(den > 0, num / den, np.nan), 1)

    def per_asset(self, v):
        """point estimate per asset [A]."""
        return self._ab(v).sum(1) / self.cnt_ab.sum(1)


def ci(v):
    """(point, lo, hi) from a bootstrap vector whose element 0 is the point estimate."""
    if not np.isfinite(v[1:]).any():
        return float(v[0]), float("nan"), float("nan")
    lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
    return float(v[0]), float(lo), float(hi)


def summarize(q, y, scale, asset, block, tkey, boot=None):
    """all section-5 metrics for one (model, horizon, row subset). Returns (dict of point values / CIs, dict of bootstrap vectors)."""
    s = row_stats(q, y, scale)
    b = boot or Boot(asset, block)
    v = dict(sPL=b.asset_mean(s["spin"]), pinball=b.pooled(s["pin"]) * 1e4, crps=b.pooled(s["crps"]) * 1e4, mae=b.pooled(s["ae"]) * 1e4,
             mase=b.asset_mean(s["sae"]), c80=b.pooled(s["c80"]), c90=b.pooled(s["c90"]), c98=b.pooled(s["c98"]))
    v["relmae"] = b.pooled(s["ae"]) / b.pooled(s["ae0"])
    with np.errstate(invalid="ignore", divide="ignore"):
        v["da"] = b.pooled(s["da"]) / b.pooled(s["da_ok"])
    out = {k: ci(x) for k, x in v.items()}
    out["calib"] = float(np.abs(s["hit"].mean(0) - QLEV).mean())
    out["nz_median"] = float(s["da_ok"].sum() / max((y != 0).sum(), 1))
    out["ic"] = ic(q[:, NQ // 2], y, tkey)
    out["n"] = int(len(y))
    out["assets"] = int(b.A)
    out["asset_pin"] = b.per_asset(s["pin"])
    return out, v
