"""PREREG-24H section 5: does the 24 h forecast add to the trend bots' signal? Research evaluation on the ALGOS pooled harness (read only).

  env: source <bgm-signal-wt>/tools/algos/env.sh; CORRVOL_D=runs-agent/r24/algos
  python tools/r24bots.py corr  <botpred.npz> <arm>[,<arm>]           correlation with the trend score, ICs
  python tools/r24bots.py blend <botpred.npz> <arm> [b] [n_placebo]    blend arm vs base, pooled paired test 2023-01 .. 2026-10
"""
import json, os, sys
import numpy as np
HARN = os.environ.get("ALGOS_DIR", "/nvme0n1-disk/code/bgm-signal-wt/tools/algos")
sys.path.insert(0, HARN)
import al, cv, slib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench.metrics import Boot, ci

T0, T1 = slib.ts_("2023-01-01T00:00"), slib.ts_("2026-11-01T00:00")
OUT = os.environ["CORRVOL_D"]


def restrict(Uo):
    Uo.k0 = int(np.searchsorted(Uo.td, T0)); Uo.y0 = 2023
    return Uo


def fmat(Uo, z, key):
    """forecast median [K, n] on the harness calendar (NaN where none): origin bar 23:00 -> decision 00:00 = (t_abs + 1) h."""
    F = np.full((Uo.K, Uo.n), np.nan)
    ks = [k for k in z.files if k == key or k.startswith(key + "_s")]
    v = np.mean([z[k] for k in ks], 0)
    d = np.searchsorted(Uo.td, (z["t_abs"] + 1) * 3600)
    ok = (d < Uo.K) & (Uo.td[np.minimum(d, Uo.K - 1)] == (z["t_abs"] + 1) * 3600)
    for j, s in enumerate(Uo.sy):
        m = ok & (z["sym"] == s)
        F[d[m], j] = v[m]
    return F


def rk(x):
    return np.argsort(np.argsort(x)).astype(float)


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return np.corrcoef(rk(a[m]), rk(b[m]))[0, 1]


def centred_rank(F, use):
    R = np.zeros_like(F)
    for d in range(len(F)):
        m = use & np.isfinite(F[d])
        if m.sum() >= 2:
            R[d, m] = rk(F[d, m]) / (m.sum() - 1) - 0.5
    return R


def corr(path, arms):
    z = np.load(path)
    out = {}
    for arm in arms:
        rows = []
        for u in al.POOL:
            Uo = slib.U(u); FE = cv.feats(Uo); ts = FE.tscore(); F = fmat(Uo, z, arm)
            nx = np.array([s not in slib.NOX for s in Uo.sy])
            y = np.full((Uo.K, Uo.n), np.nan); y[:-1] = FE.lr
            for d in np.flatnonzero((Uo.td >= T0) & (Uo.td < T1)):
                f, t, yy = F[d, nx], ts[d, nx], y[d, nx]
                rows.append((d // 7, spear(f, t), spear(f, yy), spear(t, yy), u))
        blk = np.array([r[0] for r in rows])
        res = {}
        for j, nm in ((1, "rho_forecast_trend"), (2, "ic_forecast"), (3, "ic_trend")):
            v = np.array([r[j] for r in rows]); m = np.isfinite(v)
            res[nm] = ci(Boot(np.zeros(m.sum(), int), blk[m]).pooled(v[m]))
            res[nm + "_m11"] = float(np.nanmean([r[j] for r in rows if r[4] == "M11"]))
        res["days_x_universes"] = int(len(rows))
        out[arm] = res
    return out


def blend(path, arm, b=1.0, nplac=10):
    z = np.load(path)
    al.GCAL = slib.Cal(restrict(slib.U("M11")))
    res = {c: [] for c in al.COSTS}
    plac = {c: [[] for _ in range(nplac)] for c in al.COSTS}
    for u in al.POOL:
        Uo = restrict(slib.U(u)); cal = slib.Cal(Uo); F = fmat(Uo, z, arm)
        nx = np.array([s not in slib.NOX for s in Uo.sy])
        def arm_of(R):
            return cv.Arm("fcr", post=lambda Uo_, FE, W, c, R=R: al.mult_post(W, np.maximum(1.0 + b * R, 0.0)))
        R0 = centred_rank(F, nx)
        wf = cv.make_wfun(Uo, arm_of(R0))
        for c in al.COSTS:
            E = al.run_cfgs(Uo, None, c, slib.arm_cfgs("base"), wfun=wf); E0 = al.off_E(Uo, c, "base")
            res[c].append(al.summarize(E, E0, cal))
        for k in range(nplac):
            rng = np.random.default_rng([20261117, k])
            Fp = F.copy()
            for d in range(len(Fp)):
                m = np.flatnonzero(nx & np.isfinite(Fp[d]))
                Fp[d, m] = Fp[d, rng.permutation(m)]
            wfp = cv.make_wfun(Uo, arm_of(centred_rank(Fp, nx)))
            for c in al.COSTS:
                E = al.run_cfgs(Uo, None, c, slib.arm_cfgs("base"), wfun=wfp); E0 = al.off_E(Uo, c, "base")
                plac[c][k].append(al.summarize(E, E0, cal))
        print(u, "done", flush=True)
    out = {}
    for c in al.COSTS:
        r = al.pooled(res[c])
        pm = [al.pooled(p, boot=False, sign=False)["mds"] for p in plac[c]] if nplac else []
        r["placebo_mds"] = pm
        r["class"] = al.klass(r, max(pm) if pm else None, min(pm) if pm else None)
        out[c] = {k: v for k, v in r.items() if k not in ("mds_u", "bp_u")}
    return out


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    if cmd == "corr":
        o = corr(path, sys.argv[3].split(","))
        json.dump(o, open(os.path.join(OUT, "corr.json"), "w"), indent=1)
    else:
        arm = sys.argv[3]; b = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0; n = int(sys.argv[5]) if len(sys.argv) > 5 else 10
        o = blend(path, arm, b, n)
        json.dump(o, open(os.path.join(OUT, f"blend_{arm}_b{b}.json"), "w"), indent=1)
    print(json.dumps(o, indent=1)[:4000])
