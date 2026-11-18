"""PREREG-LIC2 helpers (private artifacts under runs-agent/lic2; nothing here writes outside its <out> argument).

  python tools/lic2.py hyb s1 <kraken cache> <kraken universe> <kucoin cache> <out>    Kraken bars + KuCoin bars for harness coins outside the Kraken pool
  python tools/lic2.py hyb s2 <kraken universe> <kucoin cache> <out>                    the Kraken universe on KuCoin bars (BMT_DATA = the KuCoin cache)
  python tools/lic2.py hyb s3 <kucoin cache> <kucoin universe> <okx hourly dir> <out>   KuCoin universe + OKX bars for harness coins outside the KuCoin pool
  python tools/lic2.py cells <a.npz> <b.npz> <out.npz>                                  rows of b on the (sym, t_abs) cells of a
  python tools/lic2.py merge <a.npz> <b.npz> <out.npz>                                  a plus the rows of b whose symbol a lacks
  python tools/lic2.py cover <pred.npz> <binance pred.npz> conf|algo                    coverage vs the Binance forecast file
  python tools/lic2.py rankcorr <a.npz> <b.npz>                                         per-day rank correlation on shared cells
  python tools/lic2.py sanity <kucoin cache> <other cache>                              hourly-return agreement on shared symbols, 2023-01 .. 2026-10
  python tools/lic2.py val <r24 out dir>                                                validation 24 h rank IC (mkt+r24 2 seeds, registered, difference)
  python tools/lic2.py post <pred.npz> <cache>                                          post-October IC (consistency read)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D

CONF = "AAVE ALGO ATOM AVAX BAND BAT BTC COMP DOGE DOT ENJ ETH HBAR IOTA MANA NEO ONE QTUM RVN SAND SNX SOL TFUEL UNI VET XTZ ZEC ZEN ZIL ZRX".split()
ALGO = "ADA ALGO ATOM AVAX BAND BAT BCH BNB BTC CHZ DASH DOGE DOT ENJ ETC ETH HBAR ICX IOST IOTA KAVA LINK LTC NEO ONT RVN SOL TFUEL TRX VET XLM XRP XTZ ZEC ZIL ZRX".split()
HARN = sorted({c + "USDT" for c in CONF + ALGO})
day = lambda h: str(np.datetime64(int(h), "h"))


def hp(c, s):
    return os.path.join(c, "hourly", s + ".npz")


def fc(z):
    return np.mean([z[k] for k in z.files if k.startswith("mkt+r24")], 0)


def rk(x):
    return np.argsort(np.argsort(x)).astype(float)


def spear(a, b):
    return float(np.corrcoef(rk(a), rk(b))[0, 1]) if len(a) >= 4 and np.std(a) > 0 and np.std(b) > 0 else np.nan


def hyb(kind, *a):
    out = a[-1]
    os.makedirs(os.path.join(out, "cache", "hourly"), exist_ok=True)
    link = lambda src, s: os.path.lexists(hp(os.path.join(out, "cache"), s)) or os.symlink(os.path.abspath(src), hp(os.path.join(out, "cache"), s))
    if kind == "s1":
        kr, kru, ku = a[0], json.load(open(a[1])), a[2]
        last = 0
        for s in kru["pool"]:
            link(hp(kr, s), s); last = max(last, int(np.load(hp(kr, s))["t"][-1]))
        add = []
        for s in HARN:
            if s not in kru["pool"] and os.path.exists(hp(ku, s)):
                z = np.load(hp(ku, s)); m = z["t"] <= last
                if m.sum() > 1:
                    np.savez(hp(os.path.join(out, "cache"), s), **{k: z[k][m] for k in z.files}); add.append(s)
        u = dict(kru, pool=sorted(kru["pool"] + add))
    elif kind == "s2":
        kru, ku = json.load(open(a[0])), a[1]
        full = set(json.load(open(os.path.join(ku, "index_1h_all.json"))))
        add = [s for s in HARN if s in full and s not in kru["pool"]]
        u = {k: [s for s in kru[k] if s in full] for k in ("eval", "unseen", "seen", "train")}
        u["pool"] = sorted({s for s in kru["pool"] if s in full} | set(add))
        u["kraken_pool_missing"] = sorted(s for s in kru["pool"] if s not in full)
    else:
        ku, kuu, ox = a[0], json.load(open(a[1])), a[2]
        for s in kuu["pool"]:
            link(hp(ku, s), s)
        add = [s for s in HARN if s not in kuu["pool"] and os.path.exists(os.path.join(ox, s + ".npz"))]
        for s in add:
            link(os.path.join(ox, s + ".npz"), s)
        u = dict(kuu, pool=sorted(kuu["pool"] + add))
    json.dump(u, open(os.path.join(out, "universe.json"), "w"), indent=0)
    print(kind, "pool", len(u["pool"]), "train", len(u["train"]), "added", add, "kraken pool symbols without KuCoin bars", len(u.get("kraken_pool_missing", [])))


def cells(a, b, out, merge=False):
    A, B = np.load(a), np.load(b)
    if merge:
        have = set(A["sym"].tolist())
        m = np.array([s not in have for s in B["sym"].tolist()])
        keys = [k for k in A.files if k in B.files]
        np.savez(out, **{k: np.concatenate([A[k], B[k][m]]) for k in keys})
        print("merge", len(A["sym"]), "+", int(m.sum()), sorted(set(B["sym"][m].tolist())))
    else:
        ks = set(zip(A["sym"].tolist(), A["t_abs"].tolist()))
        m = np.array([(s, t) in ks for s, t in zip(B["sym"].tolist(), B["t_abs"].tolist())])
        np.savez(out, **{k: B[k][m] for k in B.files})
        print("cells", "a rows", len(A["sym"]), "b rows", len(B["sym"]), "shared", int(m.sum()))


def cover(pred, ref, which):
    z, r = np.load(pred), np.load(ref)
    coins = [c + "USDT" for c in (CONF if which == "conf" else ALGO)]
    sy, t = z["sym"], z["t_abs"]
    first = {s: int(t[sy == s].min()) for s in coins if (sy == s).any()}
    t0 = int(r["t_abs"].min())
    rs = set(zip(r["sym"].tolist(), r["t_abs"].tolist()))
    oct_ = (t >= D.eh(2026, 10)) & (t < D.eh(2026, 11))
    print(json.dumps(dict(which=which, coins=len(coins), with_forecast=len(first), missing=[s for s in coins if s not in first],
                          from_first_day=sum(1 for v in first.values() if v <= t0), later={s: day(v) for s, v in sorted(first.items()) if v > t0},
                          cells=int(len(sy)), binance_cells=int(len(r["sym"])), share=round(len(sy) / len(r["sym"]), 4),
                          cells_in_binance_file=int(sum((s, x) in rs for s, x in zip(sy.tolist(), t.tolist()))),
                          october_2026_cells=int(oct_.sum()), last_origin=day(t.max())), indent=0))


def rankcorr(a, b):
    A, B = np.load(a), np.load(b)
    fa, fb = fc(A), fc(B)
    ib = {k: i for i, k in enumerate(zip(B["sym"].tolist(), B["t_abs"].tolist()))}
    pa, pb, tt = [], [], []
    for i, k in enumerate(zip(A["sym"].tolist(), A["t_abs"].tolist())):
        if k in ib:
            pa.append(fa[i]); pb.append(fb[ib[k]]); tt.append(k[1])
    pa, pb, tt = np.array(pa), np.array(pb), np.array(tt)
    v = np.array([spear(pa[tt == d], pb[tt == d]) for d in np.unique(tt)])
    print(json.dumps(dict(shared_cells=int(len(tt)), days=int(np.isfinite(v).sum()), mean_rank_corr=round(float(np.nanmean(v)), 4))))


def sanity(ku, other):
    a0, a1 = D.eh(2023), D.eh(2026, 11)
    pool = set(json.load(open(os.path.join(ku, "index_1h.json"))))
    rows = {}
    for s in sorted(pool):
        if not os.path.exists(hp(other, s)):
            continue
        g = []
        for c in (ku, other):
            z = np.load(hp(c, s)); x = np.full(a1 - a0, np.nan)
            m = (z["t"] >= a0) & (z["t"] < a1)
            x[z["t"][m] - a0] = np.log(z["c"][m]); g.append((x, int(z["t"][0])))
        (x, fx), (y, fy) = g
        lo = max(fx, fy, a0) - a0
        x, y = x[lo:], y[lo:]
        if len(x) < 1000 or np.isfinite(y).sum() < 1000:
            continue
        rx, ry = np.diff(x), np.diff(y)
        m = np.isfinite(rx) & np.isfinite(ry)
        both = np.isfinite(x) & np.isfinite(y)
        rows[s] = dict(n=int(m.sum()), corr=float(np.corrcoef(rx[m], ry[m])[0, 1]) if m.sum() > 100 and rx[m].std() > 0 and ry[m].std() > 0 else np.nan,
                       ku_missing=float((np.isfinite(y) & ~np.isfinite(x)).sum() / max(np.isfinite(y).sum(), 1)),
                       other_missing=float((np.isfinite(x) & ~np.isfinite(y)).sum() / max(np.isfinite(x).sum(), 1)),
                       med_abs_dlogp_bp=float(1e4 * np.median(np.abs(x[both] - y[both]))) if both.any() else np.nan)
    q = lambda k: [round(float(v), 4) for v in np.nanpercentile([r[k] for r in rows.values()], [0, 10, 50, 90, 100])]
    hr = {s: {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[s].items()} for s in HARN if s in rows}
    print(json.dumps(dict(shared=len(rows), corr_min_p10_med_p90_max=q("corr"), ku_missing=q("ku_missing"), other_missing=q("other_missing"), med_abs_dlogp_bp=q("med_abs_dlogp_bp"),
                          harness_shared=len(hr), harness_corr_min=min((r["corr"] for r in hr.values()), default=None), harness_ku_missing_max=max((r["ku_missing"] for r in hr.values()), default=None),
                          worst=sorted(((round(r["corr"], 3), s) for s, r in rows.items()))[:8], harness=hr), indent=0))


def val(out):
    from bmtbench.metrics import Boot, ci
    from bmtbench.tradable import by_time
    rw = np.load(os.path.join(out, "val", "rows.npz"))
    y, tk = rw["y"][:, 2].astype(np.float64), rw["tkey"]
    ser = {}
    for n in ("mkt+r24_s0", "mkt+r24_s1", "registered"):
        T, IC, _ = by_time(np.load(os.path.join(out, "val", n + ".npz"))["med"][:, 2].astype(np.float64), y, tk)
        ser[n] = (T, IC)
    T = ser["mkt+r24_s0"][0]
    assert all(np.array_equal(T, v[0]) for v in ser.values())
    b = Boot(np.zeros(len(T), int), T // 168)
    ic = 0.5 * (ser["mkt+r24_s0"][1] + ser["mkt+r24_s1"][1])
    print(json.dumps(dict(origins=int(len(T)), symbols=int(len(np.unique(rw["asset"]))), ic=ci(b.pooled(ic)), ic_seed=[float(ser[f"mkt+r24_s{k}"][1].mean()) for k in (0, 1)],
                          registered=ci(b.pooled(ser["registered"][1])), d=ci(b.pooled(ic - ser["registered"][1])))))


def post(pred, cache):
    z = np.load(pred); f, sy, t = fc(z), z["sym"], z["t_abs"]
    y = np.full(len(f), np.nan)
    for s in np.unique(sy):
        c = np.load(hp(cache, s)); lp = dict(zip(c["t"].tolist(), np.log(c["c"]).tolist()))
        for i in np.flatnonzero(sy == s):
            if int(t[i]) + 24 in lp and int(t[i]) in lp:
                y[i] = lp[int(t[i]) + 24] - lp[int(t[i])]
    cut, out = D.eh(2026, 11), {}
    for nm, m in (("window_2023-01..2026-10", t < cut), ("post_october", t >= cut)):
        v = np.array([spear(f[m & (t == d) & np.isfinite(y)], y[m & (t == d) & np.isfinite(y)]) for d in np.unique(t[m])])
        v = v[np.isfinite(v)]
        out[nm] = dict(days=int(len(v)), mean_ic=round(float(v.mean()), 4) if len(v) else None, share_pos=round(float((v > 0).mean()), 3) if len(v) else None,
                       coins=int(len(np.unique(sy[m]))), first=day(t[m].min()) if m.any() else None, last=day(t[m].max()) if m.any() else None)
    print(json.dumps(out, indent=0))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "hyb":
        hyb(*a)
    elif c in ("cells", "merge"):
        cells(*a, merge=c == "merge")
    else:
        {"cover": cover, "rankcorr": rankcorr, "sanity": sanity, "val": val, "post": post}[c](*a)
