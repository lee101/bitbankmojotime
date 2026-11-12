"""Panel, universe, folds and benchmark rows (bench/PROTOCOL.md sections 1-3). Pure numpy."""
import calendar, hashlib, json, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get("BMT_DATA", os.path.join(REPO, "data", "cache"))
RUNS = os.environ.get("BMT_RUNS", os.path.join(REPO, "runs"))
QLEV = np.array([0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99])
NQ = 21
CTX = 512


def eh(y, m=1, d=1):
    """epoch hour of a UTC date."""
    return calendar.timegm((y, m, d, 0, 0, 0)) // 3600


FOLDS = [("F1", eh(2023), eh(2024)), ("F2", eh(2024), eh(2025)), ("F3", eh(2025), eh(2026)), ("F4", eh(2026), eh(2026, 11))]
TASKS = {
    "hourly": dict(step=1, horizons=(1, 6, 24), stride=5, min_hist=6200, purge=168, block=168, val=90 * 24),
    "daily": dict(step=24, horizons=(1, 5), stride=1, min_hist=600, purge=10, block=28, val=90),
}


class Panel:
    """symbols aligned on one bar grid: logp[A, T] float64 (nan outside the listed life), real[A, T] (False = filled or missing)."""

    def __init__(self, syms, t0, logp, real, qv=None):
        self.syms, self.t0, self.logp, self.real, self.qv = list(syms), int(t0), logp, real, qv
        self.ix = {s: i for i, s in enumerate(self.syms)}
        r = np.diff(self.logp, axis=1, prepend=np.nan)
        self.ret = np.nan_to_num(r, nan=0.0).astype(np.float32)        # ret[a, i] = logp[i] - logp[i-1]; 0 where undefined
        self.live = np.isfinite(self.logp)
        self.first = np.where(self.live.any(1), self.live.argmax(1), self.logp.shape[1])   # first listed bar

    @property
    def T(self):
        return self.logp.shape[1]

    def sub(self, syms):
        i = [self.ix[s] for s in syms]
        return Panel(syms, self.t0, self.logp[i], self.real[i], None if self.qv is None else self.qv[i])


def load_hourly(syms):
    raw = {}
    for s in syms:
        f = os.path.join(CACHE, "hourly", s + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            if len(z["t"]) > 1:
                raw[s] = (z["t"], z["c"], z["qv"])
    syms = [s for s in syms if s in raw]
    t0, t1 = min(v[0][0] for v in raw.values()), max(v[0][-1] for v in raw.values())
    n = int(t1 - t0 + 1)
    logp, real, qv = np.full((len(syms), n), np.nan), np.zeros((len(syms), n), bool), np.zeros((len(syms), n), np.float32)
    for a, s in enumerate(syms):
        t, c, v = raw[s]
        i = (t - t0).astype(np.int64)
        real[a, i] = True
        qv[a, i] = v
        last = np.maximum.accumulate(np.where(real[a], np.arange(n), -1))
        ok = last >= 0
        ok[i[-1] + 1:] = False                                       # nothing after the last archived bar (delisting)
        lp = np.full(n, np.nan)
        lp[i] = np.log(c)
        logp[a, ok] = lp[last[ok]]
    return Panel(syms, t0, logp, real, qv)


def to_daily(p):
    """daily bars = the hourly close at 23:00 UTC; bar index = epoch day."""
    off = (23 - p.t0) % 24
    return Panel(p.syms, (p.t0 + off) // 24, p.logp[:, off::24].copy(), p.real[:, off::24].copy(), None)


def panel(task, syms):
    p = load_hourly(syms)
    return p if task == "hourly" else to_daily(p)


def is_unseen(sym):
    return sym not in ("BTCUSDT", "ETHUSDT") and int(hashlib.sha256(sym.encode()).hexdigest()[:8], 16) % 4 == 0


def build_universe(n_eval=64):
    idx = json.load(open(os.path.join(CACHE, "index_1h.json")))
    pool = sorted(s for s, m in idx.items() if m[0] <= "2022-12" and os.path.exists(os.path.join(CACHE, "hourly", s + ".npz")))
    cand = [s for s in pool if idx[s][0] <= "2020-12"]
    a, b = eh(2021), eh(2022)
    vol = {}
    for s in cand:
        z = np.load(os.path.join(CACHE, "hourly", s + ".npz"))
        m = (z["t"] >= a) & (z["t"] < b)
        vol[s] = float(z["qv"][m].astype(np.float64).sum())
    ev = sorted(sorted(cand, key=lambda s: -vol[s])[:n_eval])
    u = {"pool": pool, "eval": ev, "unseen": [s for s in ev if is_unseen(s)], "first_month": {s: idx[s][0] for s in pool},
         "last_month": {s: idx[s][-1] for s in pool}}
    u["seen"] = [s for s in ev if s not in u["unseen"]]
    u["train"] = [s for s in pool if s not in u["unseen"]]
    return u


def universe():
    return json.load(open(os.path.join(REPO, "data", "universe.json")))


def fold_bounds(task):
    """[(name, test_start, test_end)] in bar units of the task."""
    k = TASKS[task]["step"]
    return [(n, a // k, b // k) for n, a, b in FOLDS]


def val_bounds(task):
    """validation periods: the last 90 days of each fold's purged training period."""
    c = TASKS[task]
    return [(n, s - c["purge"] - c["val"], s - c["purge"]) for n, s, _ in fold_bounds(task)]


def build_rows(p, task, bounds=None):
    """benchmark rows for the symbols of panel p: dict(a, t (bar index into p), fold, y[N, H], scale[N]).
    bounds default to the test periods; val_bounds(task) gives the validation rows."""
    cfg = TASKS[task]
    H = cfg["horizons"]
    hm = max(H)
    filled = (~p.real & p.live).astype(np.int32)
    cf = np.concatenate([np.zeros((len(p.syms), 1), np.int32), np.cumsum(filled, 1)], 1)
    out = dict(a=[], t=[], fold=[], y=[], scale=[])
    for fi, (_, s, e) in enumerate(bounds or fold_bounds(task)):
        lo, hi = s - p.t0, min(e - p.t0, p.T)                        # test bars [lo, hi)
        for a in range(len(p.syms)):
            t = np.arange(max(lo, 0), hi - hm)
            t = t[(t + p.t0) % cfg["stride"] == 0]
            if not len(t):
                continue
            ok = p.real[a, t] & (t - p.first[a] >= cfg["min_hist"])
            for h in range(1, hm + 1):
                ok &= p.real[a, t + h]
            ok &= (cf[a, t + 1] - cf[a, np.maximum(t + 1 - CTX, 0)]) <= CTX // 100
            t = t[ok]
            if not len(t):
                continue
            tr = p.ret[a, p.first[a] + 1:max(lo, 0)][p.real[a, p.first[a] + 1:max(lo, 0)]]
            out["a"].append(np.full(len(t), a, np.int32)); out["t"].append(t.astype(np.int64)); out["fold"].append(np.full(len(t), fi, np.int8))
            out["y"].append(np.stack([p.logp[a, t + h] - p.logp[a, t] for h in H], 1).astype(np.float32))
            out["scale"].append(np.full(len(t), np.abs(tr).mean(), np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def ctx_ret(p, a, t, L=CTX):
    """[N, L] returns r_{t-L+1..t} of rows (a, t). Bars before the listing are 0."""
    i = t[:, None] + np.arange(-L + 1, 1)[None, :]
    return p.ret[a[:, None], np.maximum(i, 0)] * (i > p.first[a][:, None])


def ctx_logp(p, a, t, L=CTX):
    """[N, L] log prices at bars t-L+1..t."""
    i = t[:, None] + np.arange(-L + 1, 1)[None, :]
    return p.logp[a[:, None], np.maximum(i, 0)]


def ctx_agg(p, a, t, h, L):
    """[N, L] non-overlapping h-bar returns ending at t (needs L*h+1 listed bars)."""
    i = t[:, None] + (np.arange(-L, 1) * h)[None, :]
    lp = p.logp[a[:, None], np.maximum(i, 0)]
    return np.diff(lp, axis=1).astype(np.float32)


def rows_path(task):
    return os.path.join(RUNS, f"rows_{task}.npz")


def load_rows(task):
    z = np.load(rows_path(task), allow_pickle=False)
    return {k: z[k] for k in z.files}


def pred_path(task, model):
    return os.path.join(RUNS, "preds", task, model + ".npy")


def save_pred(task, model, q, meta=None):
    """q[N, H, 21] float32 in log-return units."""
    os.makedirs(os.path.dirname(pred_path(task, model)), exist_ok=True)
    assert q.ndim == 3 and q.shape[2] == NQ and np.isfinite(q).all(), (q.shape, np.isfinite(q).mean())
    np.save(pred_path(task, model), np.sort(q.astype(np.float32), axis=2))
    if meta:
        json.dump(meta, open(pred_path(task, model)[:-4] + ".json", "w"))


def shuffle_time(p, seed=0):
    """placebo panel (hourly): every asset's returns are permuted in time inside each calendar segment (before the first test period,
    and inside each test period). Temporal structure is destroyed; the unconditional distribution of each segment is kept."""
    rng = np.random.default_rng(seed)
    cuts = [0] + [min(max(s - p.t0, 0), p.T) for _, s, _ in FOLDS] + [p.T]
    logp = np.full_like(p.logp, np.nan)
    for a in range(len(p.syms)):
        if not p.live[a].any():
            continue
        f = int(p.first[a]); l = int(p.T - 1 - p.live[a, ::-1].argmax())
        r = np.diff(p.logp[a, f:l + 1])                             # r[i] = return of bar f+1+i
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            i0, i1 = max(lo, f + 1) - (f + 1), min(hi, l + 1) - (f + 1)
            if i1 - i0 > 1:
                r[i0:i1] = r[i0:i1][rng.permutation(i1 - i0)]
        logp[a, f:l + 1] = p.logp[a, f] + np.concatenate([[0.0], np.cumsum(r)])
    return Panel(p.syms, p.t0, logp, np.isfinite(logp), p.qv)
