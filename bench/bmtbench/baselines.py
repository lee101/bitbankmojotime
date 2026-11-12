"""Statistical baselines (bench/PROTOCOL.md section 7). Each returns q[N, H, 21] quantiles of the cumulative log return."""
import numpy as np
from scipy import stats
from .data import QLEV, NQ, CTX, TASKS, ctx_ret, build_rows, val_bounds
from .metrics import pinball_levels

Z = stats.norm.ppf(QLEV)


def chunks(n, k=20000):
    return [slice(i, min(i + k, n)) for i in range(0, n, k)]


def hsums(x, h):
    """overlapping h-sums along the last axis: [N, L] -> [N, L-h+1]."""
    if h == 1:
        return x
    c = np.cumsum(np.concatenate([np.zeros((len(x), 1), x.dtype), x], 1, dtype=np.float64), 1)
    return (c[:, h:] - c[:, :-h]).astype(np.float32)


def naive(p, rows, task, drift=False):
    H = TASKS[task]["horizons"]
    out = np.zeros((len(rows["t"]), len(H), NQ), np.float32)
    for s in chunks(len(out)):
        x = ctx_ret(p, rows["a"][s], rows["t"][s]).astype(np.float64)
        sd = np.sqrt((x ** 2).mean(1)) if not drift else x.std(1)
        mu = x.mean(1) if drift else 0 * sd
        for j, h in enumerate(H):
            out[s, j] = (mu * h)[:, None] + np.sqrt(h) * sd[:, None] * Z[None, :]
    return out


def drift(p, rows, task):
    return naive(p, rows, task, drift=True)


def empirical(p, rows, task):
    H = TASKS[task]["horizons"]
    out = np.zeros((len(rows["t"]), len(H), NQ), np.float32)
    for s in chunks(len(out)):
        x = ctx_ret(p, rows["a"][s], rows["t"][s])
        for j, h in enumerate(H):
            out[s, j] = np.quantile(hsums(x, h), QLEV, axis=1).T
    return out


def seasonal_naive(p, rows, task):
    """point = the h-bar return one season ago (24 bars hourly, 7 bars daily); spread = in-context errors of that same rule."""
    H, m = TASKS[task]["horizons"], (24 if task == "hourly" else 7)
    out = np.zeros((len(rows["t"]), len(H), NQ), np.float32)
    for s in chunks(len(out)):
        x = ctx_ret(p, rows["a"][s], rows["t"][s])
        for j, h in enumerate(H):
            hs = hsums(x, h)                                         # hs[:, k] = return over context bars k..k+h-1
            L = x.shape[1]
            point = hs[:, L - m]                                     # bars (L-m)..(L-m+h-1) = target bars shifted back by m
            err = hs[:, m:] - hs[:, :-m]
            out[s, j] = point[:, None] + np.quantile(err, QLEV, axis=1).T
    return out


def ewma_sigma(x, lam):
    """one-step-ahead EWMA sigma for every context bar and for the next bar: ([N, L], [N])."""
    v = (x[:, :32].astype(np.float64) ** 2).mean(1)
    sig = np.zeros(x.shape, np.float64)
    for i in range(x.shape[1]):
        sig[:, i] = np.sqrt(v)
        v = lam * v + (1 - lam) * x[:, i].astype(np.float64) ** 2
    return np.maximum(sig, 1e-8), np.maximum(np.sqrt(v), 1e-8)


def ewma_q(x, lam, H):
    """filtered historical simulation: sigma_{t+1} x empirical quantiles of the standardised h-sums (first 32 bars are burn-in)."""
    sig, nxt = ewma_sigma(x, lam)
    z = (x / sig)[:, 32:].astype(np.float32)
    return np.stack([nxt[:, None] * np.quantile(hsums(z, h), QLEV, axis=1).T for h in H], 1)


LAMS = (0.90, 0.94, 0.97, 0.985)


def ewma(p, rows, task, seen_idx):
    """lambda per fold = argmin of validation pinball (h = first horizon) on the seen assets of that fold's validation period."""
    H = TASKS[task]["horizons"]
    val = build_rows(p, task, val_bounds(task))
    keep = np.isin(val["a"], seen_idx)
    rng = np.random.default_rng(0)
    lam_f = {}
    for f in np.unique(rows["fold"]):
        i = np.flatnonzero(keep & (val["fold"] == f))
        i = rng.choice(i, min(len(i), 20000), replace=False)
        x = ctx_ret(p, val["a"][i], val["t"][i])
        loss = [pinball_levels(ewma_q(x, l, H[:1])[:, 0], val["y"][i, 0]).mean() for l in LAMS]
        lam_f[int(f)] = LAMS[int(np.argmin(loss))]
    out = np.zeros((len(rows["t"]), len(H), NQ), np.float32)
    for f, lam in lam_f.items():
        idx = np.flatnonzero(rows["fold"] == f)
        for s in chunks(len(idx)):
            i = idx[s]
            out[i] = ewma_q(ctx_ret(p, rows["a"][i], rows["t"][i]), lam, H)
    return out, lam_f


def garch(p, rows, task, fold_starts, fit_len=8000):
    """GARCH(1,1) with Student-t innovations, zero mean: fitted once per (asset, fold) on the last fit_len real returns before the fold
    start, then filtered forward with fixed parameters. h-step cumulative variance from the analytic forecast recursion; quantiles are
    Student-t (same dof, unit variance) scaled by the cumulative sigma (exact for h = 1, an approximation for h > 1)."""
    from arch import arch_model
    H = TASKS[task]["horizons"]
    out = np.zeros((len(rows["t"]), len(H), NQ), np.float32)
    for f in np.unique(rows["fold"]):
        s0 = fold_starts[int(f)] - p.t0
        for a in np.unique(rows["a"][rows["fold"] == f]):
            idx = np.flatnonzero((rows["fold"] == f) & (rows["a"] == a))
            r = p.ret[a].astype(np.float64) * 100
            tr = r[max(p.first[a] + 1, s0 - fit_len):s0]
            try:
                res = arch_model(tr, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False)
                om, al, be, nu = (float(res.params[k]) for k in ("omega", "alpha[1]", "beta[1]", "nu"))
                if not (np.isfinite([om, al, be, nu]).all() and om > 0 and al >= 0 and be >= 0 and nu > 2.05):
                    raise ValueError
            except Exception:
                om, al, be, nu = 0.02 * tr.var(), 0.08, 0.90, 5.0
            per = min(al + be, 0.9995)
            lo, hi = max(p.first[a] + 1, s0 - 2000), int(rows["t"][idx].max()) + 1
            v = np.empty(hi - lo + 1)
            v[0] = tr.var()
            rr = r[lo:hi]
            for i in range(hi - lo):                                 # v[i+1] = variance forecast for bar lo+i+1 given bars <= lo+i
                v[i + 1] = om + al * rr[i] ** 2 + be * v[i]
            v1 = v[rows["t"][idx] + 1 - lo]
            vbar = om / max(1 - per, 5e-4)
            tq = stats.t.ppf(QLEV, nu) * np.sqrt((nu - 2) / nu)
            for j, h in enumerate(H):
                k = np.arange(h)
                cum = (vbar + (per ** k)[None, :] * (v1[:, None] - vbar)).sum(1)
                out[idx, j] = (np.sqrt(np.maximum(cum, 1e-12)) / 100)[:, None] * tq[None, :]
    return out
