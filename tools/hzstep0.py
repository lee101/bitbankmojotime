"""PREREG-HORIZON step 0 (CPU): rank IC at 24 / 72 / 168 h and rank persistence of s24, s72, s168 on the row sets V, T, B; proceed rule.

  source tools/hz_env.sh; python3 tools/hzstep0.py        -> $BMT_HZ_OUT/step0.json, step0.md
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench.metrics import Boot, ci, pinball_levels

OUT = os.environ["BMT_HZ_OUT"]
R24 = os.path.join(os.environ["BMT_RUNS"], "r24")
HEV, BLK = (24, 72, 168), 28
HS, SEEDS = tuple(int(x) for x in os.environ.get("HZ_HS", "72,168").split(",")), tuple(int(x) for x in os.environ.get("HZ_SEEDS", "0,1").split(","))
SC = {"s24": "mkt+r24", **{f"s{H}": f"mkt+r{H}" for H in HS}, **{f"s{H}@24": f"mkt+r{H}@24" for H in HS}}


def grank(v, gid, start):
    o = np.lexsort((v, gid)); r = np.empty(len(v)); r[o] = np.arange(len(v))
    return r - start[gid]


def gcorr(x, ry, gid, ng, cnt, start):
    rx = grank(x, gid, start); mu = (cnt - 1) / 2
    dx, dy = rx - mu[gid], ry - mu[gid]
    with np.errstate(all="ignore"):
        return np.bincount(gid, dx * dy, ng) / np.sqrt(np.bincount(gid, dx * dx, ng) * np.bincount(gid, dy * dy, ng))


def frac(x, gid, cnt, start):
    return grank(x, gid, start) / np.maximum(cnt[gid] - 1, 1)


def ic_series(z, seedavg):
    """{h: (origin times, {score: per-origin IC})} on the origins with >= 10 rows that have a label at h."""
    out = {}
    for n, h in enumerate(HEV):
        m = np.isfinite(z["y"][:, n])
        ut, gid = np.unique(z["t_abs"][m], return_inverse=True); ng = len(ut)
        cnt = np.bincount(gid, minlength=ng).astype(float); start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        ry = grank(z["y"][m, n].astype(np.float64), gid, start); keep = cnt >= 10
        col = lambda k: z[k][m].astype(np.float64)
        ser = {}
        def score(fn):                                               # fn(seed or None) -> score vector on the masked rows
            if seedavg:
                return gcorr(fn(None), ry, gid, ng, cnt, start)[keep]
            return np.mean([gcorr(fn(s), ry, gid, ng, cnt, start)[keep] for s in SEEDS], 0)
        get = lambda nm, s: np.mean([col(f"{SC[nm]}_s{k}") for k in SEEDS], 0) if s is None else col(f"{SC[nm]}_s{s}")
        for nm in SC:
            ser[nm] = score(lambda s, nm=nm: get(nm, s))
        for H in HS:
            ser[f"s24+s{H}"] = score(lambda s, H=H: frac(get("s24", s), gid, cnt, start) + frac(get(f"s{H}", s), gid, cnt, start))
        out[h] = (ut[keep], ser)
    return out


def boot_of(t):
    return Boot(np.zeros(len(t), int), (t // 24) // BLK)


def persistence(z):
    """B panel: lag-1 / lag-7 rank autocorrelation and mean |d rank fraction| per day, per score, on the seed-averaged median."""
    syms = sorted(set(z["sym"].tolist())); days = np.unique(z["t_abs"]); di = np.searchsorted(days, z["t_abs"]); si = np.searchsorted(np.array(syms), z["sym"])
    res = {}
    def rkrow(x):
        return np.argsort(np.argsort(x)).astype(float)
    for nm in ["s24"] + [f"s{H}" for H in HS] + [f"s24+s{H}" for H in HS]:
        F = np.full((len(days), len(syms)), np.nan)
        def mat(k):
            M = np.full((len(days), len(syms)), np.nan); M[di, si] = np.mean([z[f"{SC[k]}_s{s}"] for s in SEEDS], 0); return M
        def fr(M):
            X = np.full_like(M, np.nan)
            for d in range(len(M)):
                m = np.isfinite(M[d])
                if m.sum() >= 2: X[d, m] = rkrow(M[d, m]) / (m.sum() - 1)
            return X
        if "+" in nm:
            a, b_ = nm.split("+"); F = fr(fr(mat(a)) + fr(mat(b_)))
        else:
            F = fr(mat(nm))
        r = {}
        for k in (1, 7):
            ac, to, tt = [], [], []
            for d in range(k, len(days)):
                if days[d] - days[d - k] != 24 * k: continue
                m = np.isfinite(F[d]) & np.isfinite(F[d - k])
                if m.sum() < 10: continue
                ac.append(np.corrcoef(rkrow(F[d, m]), rkrow(F[d - k, m]))[0, 1]); to.append(np.abs(F[d, m] - F[d - k, m]).mean()); tt.append(days[d])
            r[k] = (np.array(tt), np.array(ac), np.array(to))
        res[nm] = r
    return res


def spl_val():
    rw = np.load(os.path.join(R24, "val", "rows.npz")); out = {}
    def spl(path):
        q = np.load(path)["q"].astype(np.float32); r = []
        for n in range(3):
            pl = pinball_levels(np.sort(q[:, n], 1), rw["y"][:, n]).mean(1) / rw["scale"]
            r.append(float(np.mean([pl[rw["asset"] == x].mean() for x in np.unique(rw["asset"])])))
        return np.array(r)
    ref = np.mean([spl(os.path.join(R24, "val", f"mkt+r24_s{s}.npz")) for s in SEEDS], 0)
    for H in HS:
        x = np.mean([spl(os.path.join(OUT, "val", f"mkt+r{H}_s{s}.npz")) for s in SEEDS], 0)
        out[f"mkt+r{H}"] = [float(v) for v in 100 * (1 - x / ref)]
    return out


def sanity(Z):
    o = {}
    for s in SEEDS:
        ref = np.load(os.path.join(R24, "val", f"mkt+r24_s{s}.npz"))["med"][:, 2]
        o[f"V_s{s}_max_rel_diff"] = float(np.max(np.abs(Z["V"][f"mkt+r24_s{s}"] - ref)) / np.abs(ref).mean())
    key = {(a, int(b)): i for i, (a, b) in enumerate(zip(Z["B"]["sym"].tolist(), Z["B"]["t_abs"].tolist()))}
    for f in ("botpred.npz", "botpred_conf.npz"):
        b = np.load(os.path.join(R24, f)); ix = np.array([key.get((a, int(t)), -1) for a, t in zip(b["sym"].tolist(), b["t_abs"].tolist())])
        o[f"{f}_rows_missing"] = int((ix < 0).sum())
        for s in SEEDS:
            o[f"{f}_s{s}_max_rel_diff"] = float(np.max(np.abs(Z["B"][f"mkt+r24_s{s}"][ix[ix >= 0]] - b[f"mkt+r24_s{s}"][ix >= 0])) / np.abs(b[f"mkt+r24_s{s}"]).mean())
    return o


def main():
    Z = {k: dict(np.load(os.path.join(OUT, f"pred_{k}.npz"))) for k in "VTB"}
    res = dict(sanity=sanity(Z), spl_val_change_pct_h1_h6_h24=spl_val(), ic={}, dic={}, blocks={}, origins={})
    IC = {}
    for k in "VTB":
        IC[k] = ic_series(Z[k], seedavg=(k == "B"))
        for h in HEV:
            t, ser = IC[k][h]; b = boot_of(t)
            res["blocks"][f"{k}|{h}"] = int(b.B); res["origins"][f"{k}|{h}"] = int(len(t))
            for nm, v in ser.items():
                res["ic"][f"{k}|{h}|{nm}"] = ci(b.pooled(v))
            for H in HS:
                for nm in (f"s{H}", f"s24+s{H}"):
                    bv = b.pooled(ser[nm] - ser["s24"])
                    res["dic"][f"{k}|{h}|{nm}-s24"] = list(ci(bv)) + [float(np.std(bv[1:]))]
    P = persistence(Z["B"]); res["persist"] = {}
    for nm, r in P.items():
        for k in (1, 7):
            t, ac, to = r[k]; b = boot_of(t)
            e = dict(autocorr=ci(b.pooled(ac)), turn=ci(b.pooled(to)))
            if nm != "s24":
                t0, ac0, to0 = P["s24"][k]; assert np.array_equal(t, t0)
                e["d_autocorr"] = ci(b.pooled(ac - ac0)); e["turn_ratio"] = float(to.mean() / to0.mean())
            res["persist"][f"{nm}|lag{k}"] = e
    dec = {}
    for H in HS:
        dT, dV = res["dic"][f"T|{H}|s{H}-s24"], res["dic"][f"V|{H}|s{H}-s24"]; icT = res["ic"][f"T|{H}|s{H}"]; da = res["persist"][f"s{H}|lag1"]["d_autocorr"]
        a = bool(dT[1] > 0 and dV[0] > 0)
        b_ = bool(icT[1] > 0 and dT[2] >= 0 and da[0] >= 0.10 and da[1] > 0)
        dec[H] = dict(a=a, b=b_, proceed=a or b_, dIC_T=dT[:3], dIC_V=dV[:3], mde_dIC_T=2.8 * dT[3], mde_dIC_V=2.8 * dV[3], ic_T=icT, d_autocorr=da,
                      hold_ic_B=float(np.mean([res["ic"][f"B|{h}|s{H}"][0] for h in (72, 168)])))
    go = [H for H in HS if dec[H]["proceed"]]
    res["decision"] = dict(per_h=dec, proceed=go, h_star=(max(go, key=lambda H: dec[H]["hold_ic_B"]) if go else None), verdict="proceed to step 1" if go else "registered null at step 0")
    json.dump(res, open(os.path.join(OUT, "step0.json"), "w"), indent=1)
    f = lambda x, d=4: f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]"
    md = []
    for k, lab in (("T", "T: unseen folds, benchmark hourly test rows"), ("V", "V: validation rows"), ("B", "B: bots' panel, daily 00:00 UTC, 45 coins")):
        md += [f"\n{lab} ({res['origins'][f'{k}|24']} origins, {res['blocks'][f'{k}|24']} blocks of {BLK} days)\n", "| score | IC at 24 h | IC at 72 h | IC at 168 h |", "|---|---|---|---|"]
        for nm in list(SC) + [f"s24+s{H}" for H in HS]:
            md.append(f"| {nm} | " + " | ".join(f(res["ic"][f"{k}|{h}|{nm}"]) for h in HEV) + " |")
        for H in HS:
            for nm in (f"s{H}", f"s24+s{H}"):
                md.append(f"| {nm} minus s24 | " + " | ".join(f(res["dic"][f"{k}|{h}|{nm}-s24"]) for h in HEV) + " |")
    md += ["\nPersistence on B (seed-averaged median)\n", "| score | lag-1 d rank autocorr | diff vs s24 | lag-7 d | daily mean abs d rank fraction | ratio to s24 | 7-day |", "|---|---|---|---|---|---|---|"]
    for nm in P:
        e1, e7 = res["persist"][f"{nm}|lag1"], res["persist"][f"{nm}|lag7"]
        md.append(f"| {nm} | {f(e1['autocorr'], 3)} | {f(e1['d_autocorr'], 3) if 'd_autocorr' in e1 else ''} | {f(e7['autocorr'], 3)} | {f(e1['turn'], 3)} | {e1.get('turn_ratio', 1.0):.2f} | {f(e7['turn'], 3)} |")
    md.append("\n" + json.dumps(res["decision"]) + "\n\nsanity " + json.dumps(res["sanity"]) + "\nval sPL change % (h1, h6, h24; + = better than mkt+r24) " + json.dumps(res["spl_val_change_pct_h1_h6_h24"]))
    open(os.path.join(OUT, "step0.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
