"""PREREG-FSEL (CPU): the 168 h forecast as a weekly selection rank on the UNIV point-in-time panels X30 / X50 (read only).

  python3 tools/fsel.py books  <lead|full>     forms sel / selh / ls for every rank (forecasts, momentum, placebos) -> $BMT_FSEL_OUT/books_<seg>.npz
  python3 tools/fsel.py report <lead|conf>     tables, pick rule (lead) or the confirmation bar (conf, needs FSEL_PICK=form|N|K)
Segments: lead 2023-01-02 .. 2025-05-01, conf 2025-05-01 .. 2026-11-01. In `lead` mode every input is cut at the lead end on load.
Other entry points (used by tools/fsel_trend.py): Pan, ranks, held_daily.
"""
import json, math, os, sys
import numpy as np

U = "/sdb-disk/tradingbots/binance/univ"
OUT = os.environ.get("BMT_FSEL_OUT", "/nvme0n1-disk/code/bitbankmojotime-hz-wt/runs-agent/fsel")
NS, KS = (30, 50), (5, 10)
PV, CAP, PVB, CAPB, MA, VWIN = 0.25, 1.875, 0.1875, 1.5625, 200, 60
COST = {"c7": 7e-4, "c10": 10e-4, "c20": 20e-4}
NPL, PSEED, APR = 20, (20261118, 6), 0.10
SIG = ("f168", "f24", "mom7", "mom28"); FAM = ("shuf", "rand", "shift")
ts_ = lambda s: int(np.datetime64(s, "s").astype(np.int64))
T0, TL, T1 = ts_("2023-01-02"), ts_("2025-05-01"), ts_("2026-11-01")


class Pan:
    """decision-time prices, tradable membership and forecasts of one union panel, cut at the segment end."""
    def __init__(s, N, seg):
        d = f"{U}/shift/u/X{N}"; f = f"{d}/panel.bin"
        t, n = [int(x) for x in np.fromfile(f, "<i8", count=2, offset=8)]
        ts = np.fromfile(f, "<i8", count=t, offset=24)
        C = np.memmap(f, "<f8", "r", offset=24 + 8 * t + 8 * t * n, shape=(t, n))
        di = np.nonzero(ts % 86400 == 0)[0]; di = di[di > 0]; td = ts[di]
        kend = int(np.searchsorted(td, TL if seg == "lead" else T1))
        s.N, s.n, s.seg, s.cols = N, n, seg, json.load(open(f"{d}/cols.json"))
        s.td = td[:kend + 1].copy(); s.P = np.array(C[di[:kend + 1] - 1]); s.elig = np.load(f"{d}/elig.npy")[:kend + 1]
        s.k0, s.k1 = int(np.searchsorted(s.td, T0)), kend
        day = s.td // 86400
        s.mon = np.array([k for k in range(s.k0, s.k1) if day[k] % 7 == 4])            # Mondays (1970-01-05 = day 4)
        s.mid = s.td.astype("datetime64[s]").astype("datetime64[M]").astype(int)        # calendar month of each row
        s.kl = int(np.searchsorted(s.td, TL))                                          # first row of the confirmation segment
        with np.errstate(all="ignore"):
            s.ret = np.nan_to_num(s.P[1:] / s.P[:-1] - 1.0)                              # ret[d] = decision d -> d + 1
        s.F = {}

    def forecasts(s, path=None):
        z = np.load(path or f"{OUT}/wide.npz"); dec = (z["t_abs"] + 1) * 3600
        keep = dec < s.td[s.k1]; row = np.searchsorted(s.td, dec[keep]); assert (s.td[row] == dec[keep]).all()   # origin bar closes at the decision time
        sym = z["sym"][keep]; o = np.argsort(sym, kind="stable"); us, st = np.unique(sym[o], return_index=True); en = np.r_[st[1:], len(o)]
        loc = {x: o[a:b] for x, a, b in zip(us.tolist(), st, en)}
        for key in ("mkt+r168", "mkt+r24"):
            v = np.mean([z[f"{key}_s{k}"][keep] for k in (0, 1)], 0); M = np.full((len(s.td), s.n), np.nan)
            for j, nm in enumerate(s.cols):
                i = loc.get(nm.split("~")[0])
                if i is not None: M[row[i], j] = v[i]
            s.F[key] = M
        s.cand = s.elig & np.isfinite(s.F["mkt+r168"]) & np.isfinite(s.F["mkt+r24"]) & np.isfinite(s.P) & (s.P > 0)
        return int(keep.sum())

    def bear(s):
        """producer bear rows on the N-member index (uv.UV.bear): equal-weight index of tradable members below its 200-day mean."""
        with np.errstate(all="ignore"):
            r = np.diff(np.log(s.P), axis=0); r = np.where(np.isfinite(r) & s.elig[1:], r, np.nan); mu = np.nan_to_num(np.nanmean(r, 1))
        x = np.concatenate([[0.0], np.cumsum(mu)]); cs = np.cumsum(x); ma = np.full(len(x), np.nan); ma[MA:] = (cs[MA:] - cs[:-MA]) / MA
        return np.isfinite(ma) & (x < ma)


def ranks(p, name):
    """rank values [rows, n] (NaN = not rankable) for a signal or a placebo id 'fam:draw'."""
    V = np.full((len(p.td), p.n), np.nan)
    if name in ("f168", "f24"):
        V = p.F["mkt+r168" if name == "f168" else "mkt+r24"].copy()
    elif name.startswith("mom"):
        L = int(name[3:])
        with np.errstate(all="ignore"): V[L:] = np.log(p.P[L:] / p.P[:-L])
    else:
        fam, k = name.split(":"); rng = np.random.default_rng([*PSEED, FAM.index(fam), int(k), p.N]); F = p.F["mkt+r168"]
        if fam == "shuf":
            for d in p.mon:
                j = np.flatnonzero(p.cand[d]); V[d, j] = F[d, rng.permutation(j)]
        elif fam == "rand":
            V[:] = rng.random(p.n)[None, :]
        else:                                                         # circular shift inside each segment
            for a, b in ((p.k0, min(p.kl, p.k1)), (p.kl, p.k1)):
                m = p.mon[(p.mon >= a) & (p.mon < b)]
                if len(m) > 26:
                    sft = int(rng.integers(13, len(m) - 12)); V[m] = F[m[(np.arange(len(m)) + sft) % len(m)]]
    return np.where(p.cand, V, np.nan)


def held_sets(p, V, K, hyst=False, bottom=False):
    """per Monday: indices of the names held (top-K by V among the candidates; hysteresis: stay while inside the top 2K)."""
    out, H = [], np.array([], int)
    for d in p.mon:
        j = np.flatnonzero(np.isfinite(V[d]))
        o = j[np.argsort(V[d, j] if bottom else -V[d, j], kind="stable")]
        if hyst:
            keep = [x for x in H if x in set(o[:2 * K].tolist())]
            H = np.array(keep + [x for x in o if x not in set(keep)][:K - len(keep)], int)
        else:
            H = o[:K]
        out.append(H)
    return out


def held_daily(p, sets):
    """bool [rows, n]: held from its Monday to the next one (rows before the first Monday and from the segment end: False)."""
    M = np.zeros((len(p.td), p.n), bool); ends = np.r_[p.mon[1:], p.k1]
    for d, e, H in zip(p.mon, ends, sets): M[d:e, H] = True
    return M


def targets(p, sets, K, bear):
    """form 1 / 4 target weights at the Mondays: equal slots, gross min(pv / basket vol, cap), bear rows with the bear budget."""
    T = np.zeros((len(p.mon), p.n))
    for i, (d, H) in enumerate(zip(p.mon, sets)):
        if not len(H): continue
        sd = p.ret[d - VWIN:d][:, H].mean(1).std() * math.sqrt(365.0)
        pv, cap = (PVB, CAPB) if bear[d] else (PV, CAP)
        T[i, H] = (min(pv / sd, cap) if sd > 0 else cap) / K
    return T


def ls_targets(p, top, bot, K):
    T = np.zeros((len(p.mon), p.n))
    for i, (a, b) in enumerate(zip(top, bot)): T[i, a] += 0.5 / K; T[i, b] -= 0.5 / K
    return T


def simulate(p, T, borrow=True):
    """T [B, Mondays, n] -> gross daily returns, traded notional per day, gross per day, each [B, k1 - k0]; weights drift between Mondays."""
    B = T.shape[0]; D = p.k1 - p.k0; w = np.zeros((B, p.n)); r = np.zeros((B, D)); tv = np.zeros((B, D)); gg = np.zeros((B, D)); mi = {int(d): i for i, d in enumerate(p.mon)}
    for d in range(p.k0, p.k1):
        if d in mi:
            t = T[:, mi[d]]; tv[:, d - p.k0] = np.abs(t - w).sum(1); w = t.copy()
        x = p.ret[d]; g = w.sum(1)
        rp = w @ x - (np.maximum(g - 1.0, 0.0) * APR / 365.0 if borrow else 0.0)
        r[:, d - p.k0] = rp; gg[:, d - p.k0] = np.abs(w).sum(1)
        w = w * (1.0 + x)[None, :] / (1.0 + rp)[:, None]
    return r, tv, gg


def net(r, tv, c):
    return (1.0 + r) * (1.0 - COST[c] * tv) - 1.0


def stats(r, mid, tv=None, gg=None):
    n = len(r); E = np.concatenate([[1.0], np.cumprod(1.0 + r)]); pk = np.maximum.accumulate(E)
    wm = min(float(np.prod(1.0 + r[mid == m]) - 1.0) for m in np.unique(mid))
    d60 = max(float(1.0 - E[t] / E[max(0, t - 60):t + 1].max()) for t in range(len(E)))
    o = dict(sharpe=float(r.mean() / r.std(ddof=1) * math.sqrt(365.0)) if r.std() > 0 else 0.0, cagr=float(E[-1] ** (365.25 / n) - 1.0), worst_month=wm, dd60=d60, maxdd=float((1.0 - E / pk).max()))
    if tv is not None: o["turnover"] = float(tv.sum() / (n / 365.25))
    if gg is not None: o["gross"] = float(gg.mean())
    return o


def boot_pair(r1, r0, mid, nb=2000, seed=7):
    """paired month-block bootstrap of d Sharpe and d CAGR (the same month draws for both books)."""
    ms, ix = np.unique(mid, return_inverse=True); nm = len(ms); npm = np.bincount(ix, None, nm).astype(float)
    cnt = np.random.default_rng(seed).multinomial(nm, np.ones(nm) / nm, size=nb).astype(float); M = cnt @ npm
    def parts(r): return np.bincount(ix, r, nm), np.bincount(ix, r * r, nm), np.bincount(ix, np.log1p(r), nm)
    def sh(S, Q): m = (cnt @ S) / M; q = (cnt @ Q) / M; return m / np.sqrt(np.maximum(q - m * m, 1e-300) * M / (M - 1)) * math.sqrt(365.0)
    (S1, Q1, L1), (S0, Q0, L0) = parts(r1), parts(r0)
    dS = sh(S1, Q1) - sh(S0, Q0); dC = np.expm1(365.25 * (cnt @ L1) / M) - np.expm1(365.25 * (cnt @ L0) / M)
    pS = stats(r1, mid)["sharpe"] - stats(r0, mid)["sharpe"]; pC = stats(r1, mid)["cagr"] - stats(r0, mid)["cagr"]
    return dict(d_sharpe=[float(pS), float(np.percentile(dS, 2.5)), float(np.percentile(dS, 97.5))], se=float(dS.std()), d_cagr=[float(pC), float(np.percentile(dC, 2.5)), float(np.percentile(dC, 97.5))])


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 3 and ra.std() > 0 and rb.std() > 0 else np.nan


def week_ret(p, d):
    e = min(d + 7, p.k1)
    with np.errstate(all="ignore"): return p.P[e] / p.P[d] - 1.0


def info(p, V, K):
    """weekly IC and top-K / bottom-K excess over the equal-weight candidate universe (next-week simple return, no costs)."""
    ic, lo, sh, mm = [], [], [], []
    for d in p.mon:
        if d + 7 > p.k1: continue
        j = np.flatnonzero(np.isfinite(V[d])); y = week_ret(p, d)[j]; ok = np.isfinite(y); j, y = j[ok], y[ok]
        if len(j) < 2 * K: continue
        o = np.argsort(-V[d, j], kind="stable"); ic.append(spear(V[d, j], y)); lo.append(y[o[:K]].mean() - y.mean()); sh.append(y.mean() - y[o[-K:]].mean()); mm.append(p.mid[d])
    return np.array(ic), np.array(lo), np.array(sh), np.array(mm)


def mboot(v, mm, nb=2000, seed=7):
    """mean with a month-block interval."""
    ms, ix = np.unique(mm, return_inverse=True); nm = len(ms); S = np.bincount(ix, v, nm); c = np.bincount(ix, None, nm).astype(float)
    cnt = np.random.default_rng(seed).multinomial(nm, np.ones(nm) / nm, size=nb).astype(float); x = (cnt @ S) / (cnt @ c)
    return [float(np.nanmean(v)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def books(seg):
    out, meta = {}, dict(seg=seg, cover={}, info={}, cap={})
    for N in NS:
        p = Pan(N, seg); nrow = p.forecasts(); bear = p.bear()
        meta["cover"][N] = dict(forecast_rows=nrow, mondays=int(len(p.mon)), tradable=float(p.elig[p.mon].sum(1).mean()), ranked=float(p.cand[p.mon].sum(1).mean()), ranked_min=int(p.cand[p.mon].sum(1).min()),
                                bear_share=float(bear[p.mon].mean()), first=str(np.datetime64(int(p.td[p.mon[0]]), "s")), last=str(np.datetime64(int(p.td[p.mon[-1]]), "s")))
        names = list(SIG) + [f"{f}:{k}" for f in FAM for k in range(NPL)]
        V = {nm: ranks(p, nm) for nm in names}
        for K in KS:
            ids, T, bor = [], [], []
            for nm in names:
                top = held_sets(p, V[nm], K)
                ids.append(f"sel|{N}|{K}|{nm}"); T.append(targets(p, top, K, bear)); bor.append(True)
                ids.append(f"selh|{N}|{K}|{nm}"); T.append(targets(p, held_sets(p, V[nm], K, hyst=True), K, bear)); bor.append(True)
                if nm in SIG:
                    ids.append(f"ls|{N}|{K}|{nm}"); T.append(ls_targets(p, top, held_sets(p, V[nm], K, bottom=True), K)); bor.append(False)
                    ic, lo, sh, mm = info(p, V[nm], K)
                    meta["info"][f"{N}|{K}|{nm}"] = dict(ic=mboot(ic, mm), long_excess_bp=mboot(1e4 * lo, mm), short_excess_bp=mboot(1e4 * sh, mm), weeks=int(len(ic)))
            ics = [float(np.nanmean(info(p, V[f"shift:{k}"], K)[0])) for k in range(NPL)]
            meta["info"][f"{N}|{K}|shift_ic"] = [float(np.mean(ics)), float(np.min(ics)), float(np.max(ics))]
            T = np.stack(T); bor = np.array(bor)
            r, tv, gg = np.zeros((len(ids), p.k1 - p.k0)), np.zeros((len(ids), p.k1 - p.k0)), np.zeros((len(ids), p.k1 - p.k0))
            for flag in (True, False):
                i = np.flatnonzero(bor == flag)
                if len(i): r[i], tv[i], gg[i] = simulate(p, T[i], borrow=flag)
            for i, b in enumerate(ids): out[b] = np.stack([r[i], tv[i], gg[i]]).astype(np.float64)
            if (N, K) == (50, 10):                                   # engine check + capacity inputs for the reference cell
                i = ids.index("sel|50|10|f168"); W = np.full((len(p.td), p.n), np.nan); W[:p.k0] = 0.0; W[p.mon] = T[i]; W[p.k1:] = 0.0
                np.save(f"{OUT}/engine_w_{seg}.npy", W)
            print("books", N, K, len(ids), flush=True)
        if N == NS[0]: out["td"] = p.td[p.k0:p.k1 + 1].astype(np.float64); out["mid"] = p.mid[p.k0:p.k1].astype(np.float64)
        # truncation test (ii): form-1 targets up to a date from inputs cut at that date must equal the full ones
        if seg == "lead":
            kc = int(np.searchsorted(p.td, ts_("2024-03-04"))); q = Pan.__new__(Pan); q.__dict__.update(p.__dict__)
            q.td, q.P, q.elig, q.ret = p.td[:kc + 1], p.P[:kc + 1], p.elig[:kc + 1], p.ret[:kc]; q.k1 = kc; q.mon = p.mon[p.mon < kc]
            q.F = {k: np.where(np.arange(len(p.td))[:, None] < kc, v, np.nan)[:kc + 1] for k, v in p.F.items()}; q.cand = p.cand[:kc + 1] & np.isfinite(q.F["mkt+r168"])
            a = targets(q, held_sets(q, ranks(q, "f168"), 10), 10, q.bear()); b = targets(p, held_sets(p, V["f168"], 10), 10, bear)[:len(q.mon)]
            meta.setdefault("trunc_weights", {})[N] = float(np.abs(a - b).max())
    np.savez(f"{OUT}/books_{seg}.npz", **out); json.dump(meta, open(f"{OUT}/books_{seg}.json", "w"), indent=1)
    print(json.dumps(meta["cover"]), json.dumps(meta.get("trunc_weights")), "FSEL_BOOKS_DONE", flush=True)


def load_live(td, mode):
    z = np.load(f"{OUT}/live.npz"); k = np.searchsorted(z["td"], td); assert (z["td"][k] == td).all(); sg = "lead" if mode == "lead" else "conf"
    return {c: z[f"E_{c}"][k[1:]] / z[f"E_{c}"][k[:-1]] - 1.0 for c in COST}, {c: dict(turnover=float(z[f"turn_{sg}_{c}"]), gross=float(z[f"gross_{sg}"])) for c in COST}


def load_trend(seg, td):
    """form 2 arms from tools/fsel_trend.py: ({id: {cost: daily returns}}, {id: {cost: sim turnover}}, {id: mean gross})."""
    f = f"{OUT}/trend_{seg}.npz"; out, T, G = {}, {}, {}
    if not os.path.exists(f): return out, T, G
    z = np.load(f); k = np.searchsorted(z["td"], td); assert (z["td"][k] == td).all()
    for key in z.files:
        if key[:2] in ("E|", "T|"):
            kind, b, c = key.split("|", 2); b = b.replace("/", "|")
            if kind == "E": out.setdefault(b, {})[c] = z[key][k[1:]] / z[key][k[:-1]] - 1.0
            else: T.setdefault(b, {})[c] = float(z[key])
        elif key[:2] == "G|": G[key[2:].replace("/", "|")] = float(z[key])
    return out, T, G


def leak_membership(month="2024-06-01", top=50):
    """leak check 2: the stored ranking of one month recomputed from daily bars strictly before the ranking day."""
    import glob, pandas as pd
    STAB = set("USDC FDUSD TUSD BUSD DAI USDP PAX AEUR EUR USTC U XUSD USD1 RLUSD EURI BFUSD USDE WBTC WBETH BETH XAUT USDS USDSB SUSD".split())
    m = int(np.datetime64(month, "D").astype(np.int64)); L = []
    for f in sorted(glob.glob(f"{U}/d1/*.csv")):
        sym = os.path.basename(f)[:-4]
        if sym == "PAXGUSDT" or sym[:-4] in STAB: continue
        d = pd.read_csv(f, usecols=["ts", "qv"]); dy = (d.ts.values // 86400).astype(np.int64); qv = d.qv.values.astype(float); o = np.argsort(dy, kind="stable"); dy, qv = dy[o], qv[o]
        k = dy < m; dy, qv = dy[k], qv[k]                              # nothing from the ranking day on
        if not len(dy): continue
        cut = np.flatnonzero(np.diff(dy) > 7); a0 = cut[-1] + 1 if len(cut) else 0; dy, qv = dy[a0:], qv[a0:]
        if dy[0] > m - 180 or dy[-1] != m - 1: continue
        k0 = np.searchsorted(dy, m - 30)
        if len(dy) - k0 < 27: continue
        L.append((sym, float(np.median(qv[k0:]))))
    L.sort(key=lambda x: -x[1]); R = json.load(open(f"{U}/meta/rank.json"))[str(m)]
    a = [x[0] for x in L[:top]]; b = [x[0].split("~")[0] for x in R[:top]]
    return dict(month=month, top=top, same_names_in_order=bool(a == b), max_rel_score_diff=float(max(abs(x[1] / y[1] - 1) for x, y in zip(L[:top], R[:top]))), eligible_recomputed=len(L), eligible_stored=len(R))


def report(mode):
    seg = "lead" if mode == "lead" else "full"
    z = np.load(f"{OUT}/books_{seg}.npz"); td = z["td"].astype(np.int64); mid = z["mid"].astype(np.int64)
    a, b = (0, int(np.searchsorted(td, TL))) if mode == "lead" else (int(np.searchsorted(td, TL)), len(td) - 1)
    sl = slice(a, b); mid_s = mid[sl]; live, lx = load_live(td, mode); trend, tT, tG = load_trend(seg, td); yrs = (b - a) / 365.25
    L = {c: live[c][sl] for c in COST}
    res = dict(mode=mode, days=int(b - a), months=int(len(np.unique(mid_s))), live={c: dict(stats(L[c], mid_s), **lx[c]) for c in COST}, books={}, plac={})
    def series(form, N, K, nm, c):
        if form in ("sel", "selh", "ls"):
            x = z[f"{form}|{N}|{K}|{nm}"]; return net(x[0], x[1], c)[sl], x[1][sl], x[2][sl]
        if form == "blend":
            r, tv, gg = series("sel", N, K, nm, c); return 0.5 * r + 0.5 * live["c10" if c == "c20" else c][sl], None, None
        t = trend.get(f"tsel|{N}|{K}|{nm}")
        return (t[c][sl], None, None) if t and c in t else (None, None, None)
    for form in ("sel", "tsel", "blend", "selh", "ls"):
        for N in NS:
            for K in KS:
                for nm in SIG:
                    e = {}
                    for c in COST:
                        r, tv, gg = series(form, N, K, nm, c)
                        if r is None: continue
                        e[c] = stats(r, mid_s, tv, gg); e[c].update(boot_pair(r, L["c10" if c == "c20" else c], mid_s))
                        if form == "tsel" and mode == "lead" and f"tpre|{N}|0|x" in tT:   # sim total minus the pre-2023 warm-up book (each ends with one liquidation): approximate
                            e[c]["turnover"] = (tT[f"tsel|{N}|{K}|{nm}"][c] - tT[f"tpre|{N}|0|x"][c]) / yrs; e[c]["gross"] = tG.get(f"tsel|{N}|{K}|{nm}", float("nan"))
                    if e: res["books"][f"{form}|{N}|{K}|{nm}"] = e
                if form == "ls": continue
                for fam in FAM:
                    for c in COST:
                        v = [series(form, N, K, f"{fam}:{k}", c)[0] for k in range(NPL)]; v = [stats(x, mid_s)["sharpe"] for x in v if x is not None]
                        if v: res["plac"][f"{form}|{N}|{K}|{fam}|{c}"] = v
    # pick rule / confirmation bar
    def cond(form, N, K):
        bk = res["books"].get(f"{form}|{N}|{K}|f168"); o = dict(a=[], b=[], c=[], d=None, placebo_run=True)
        if not bk or not all(c in bk for c in COST): return None
        for c in ("c7", "c10"):
            o["a"].append(bool(bk[c]["d_sharpe"][0] > 0 and bk[c]["worst_month"] >= -0.30))
            pl = [res["plac"].get(f"{form}|{N}|{K}|{fam}|{c}") for fam in FAM]; o["placebo_run"] &= all(x is not None and len(x) >= NPL for x in pl)
            o["b"].append(bool(all(x is not None and len(x) >= NPL and bk[c]["sharpe"] > max(x) for x in pl)))
            o["c"].append(bool(all(bk[c]["sharpe"] > res["books"][f"{form}|{N}|{K}|{m}"][c]["sharpe"] for m in ("mom7", "mom28"))))
        o["d"] = bool(bk["c20"]["sharpe"] > res["live"]["c10"]["sharpe"])
        o["acd"] = bool(all(o["a"]) and all(o["c"]) and o["d"]); o["eligible"] = bool(o["acd"] and all(o["b"]))
        o["interval_above_zero"] = [bool(bk[c]["d_sharpe"][1] > 0) for c in ("c7", "c10")]; o["d_sharpe_c10"] = bk["c10"]["d_sharpe"][0]; o["se_c10"] = bk["c10"]["se"]
        return o
    res["rule"] = {f"{f}|{N}|{K}": cond(f, N, K) for f in ("sel", "tsel", "blend", "selh") for N in NS for K in KS}
    if mode == "lead":
        ok = [k for k, v in res["rule"].items() if v and v["eligible"]]
        res["picked"] = max(ok, key=lambda k: res["rule"][k]["d_sharpe_c10"]) if ok else None
        res["trend_placebo_needed"] = [k for k, v in res["rule"].items() if v and k.startswith("tsel") and v["acd"] and not v["placebo_run"]]
        if res["picked"]:
            se = res["rule"][res["picked"]]["se_c10"]; res["mde_conf"] = dict(se_lead=se, mde=2.8 * se * math.sqrt(28 / 18), lead_d_sharpe=res["rule"][res["picked"]]["d_sharpe_c10"])
    else:
        pk = os.environ["FSEL_PICK"]; v = res["rule"][pk]
        res["picked"] = pk; res["verdict"] = "CONFIRMED" if v["eligible"] and all(v["interval_above_zero"]) else "LEAN" if v["eligible"] else "NOT CONFIRMED"
    json.dump(res, open(f"{OUT}/report_{mode}.json", "w"), indent=1)
    pc = lambda x: f"{100 * x:+.1f}"
    for c in ("c7", "c10", "c20"):
        print(f"\n### {mode}, {c} (live book: Sharpe {res['live'][c]['sharpe']:.2f}, CAGR {pc(res['live'][c]['cagr'])} %, worst month {pc(res['live'][c]['worst_month'])} %, 60-day DD {pc(-res['live'][c]['dd60'])} %, max DD {pc(-res['live'][c]['maxdd'])} %)\n")
        print("| form | N | K | Sharpe: f168 | f24 | mom 7 d | mom 28 d | shuffled min .. max | random-K min .. max | shift min .. max | f168: CAGR % | worst month % | 60-day DD % | max DD % | turnover /yr | gross | d Sharpe vs live [95 %] | d CAGR vs live [95 %] |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for form in ("sel", "tsel", "blend", "selh", "ls"):
            for N in NS:
                for K in KS:
                    bk = res["books"].get(f"{form}|{N}|{K}|f168")
                    if not bk or c not in bk: continue
                    g = lambda m: f"{res['books'][f'{form}|{N}|{K}|{m}'][c]['sharpe']:+.2f}" if f"{form}|{N}|{K}|{m}" in res["books"] and c in res["books"][f"{form}|{N}|{K}|{m}"] else ""
                    pl = lambda fam: (lambda v: f"{min(v):+.2f} .. {max(v):+.2f}" if v else "")(res["plac"].get(f"{form}|{N}|{K}|{fam}|{c}"))
                    x = bk[c]
                    print(f"| {form} | {N} | {K} | {x['sharpe']:+.2f} | {g('f24')} | {g('mom7')} | {g('mom28')} | {pl('shuf')} | {pl('rand')} | {pl('shift')} | {pc(x['cagr'])} | {pc(x['worst_month'])} | {pc(-x['dd60'])} | {pc(-x['maxdd'])} | "
                          f"{x.get('turnover', float('nan')):.1f} | {x.get('gross', float('nan')):.2f} | {x['d_sharpe'][0]:+.2f} [{x['d_sharpe'][1]:+.2f}, {x['d_sharpe'][2]:+.2f}] | {pc(x['d_cagr'][0])} [{pc(x['d_cagr'][1])}, {pc(x['d_cagr'][2])}] |")
    print("\nrule", json.dumps(res["rule"])); print("picked", res.get("picked"), "verdict", res.get("verdict"), "mde", json.dumps(res.get("mde_conf")), "trend placebo needed", res.get("trend_placebo_needed"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "books": books(sys.argv[2])
    elif sys.argv[1] == "leak": r = leak_membership(); json.dump(r, open(f"{OUT}/leak_membership.json", "w"), indent=1); print(json.dumps(r))
    else: report(sys.argv[2])
