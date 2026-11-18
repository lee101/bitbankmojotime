"""PREREG-GATE: frozen 24 h forecast only where the trend bots already trade. Forms gt (tilt on trade days), ge (entry gate), gx (exit hint).

  env: source <bgm-signal-wt>/tools/algos/env.sh; CORRVOL_D=runs-agent/r24/gate
  python tools/r24gate.py <botpred.npz> <form> [n_placebo]
"""
import importlib.util, json, os, sys
import numpy as np
import numba
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r24bots as RB
import al, cv, slib

OUT = os.environ["CORRVOL_D"]
BAND = 0.05
UNIS = al.POOL
if os.environ.get("GATE_UNIS") == "CONF":
    _sp = importlib.util.spec_from_file_location("sigconf", os.path.join(RB.HARN, "..", "signal", "conf.py")); _cf = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_cf)
    UNIS = _cf.built()
    assert len(UNIS) == 16, UNIS
if os.environ.get("GATE_BANDKEY"):
    slib.wkey = lambda c: (c[0], c[1], c[5], c[6], c[4])


def frac(F, use):
    """rank fraction in [0, 1] among usable coins with a forecast, NaN elsewhere."""
    R = np.full_like(F, np.nan)
    for d in range(len(F)):
        m = use & np.isfinite(F[d])
        if m.sum() >= 2:
            R[d, m] = RB.rk(F[d, m]) / (m.sum() - 1)
    return R


@numba.njit(cache=True)
def tilt_trade_days(W, P, Fr, k0, band, both=False):
    K, n = W.shape
    out = W.copy()
    M = np.ones(n)
    h = W[k0 - 1].copy()
    for d in range(k0, K):
        if d > k0:
            port = 1.0 - h.sum()
            for j in range(n):
                g = P[d, j] / P[d - 1, j] if P[d - 1, j] > 0 and P[d, j] > 0 else 1.0
                h[j] *= g
                port += h[j]
            if port > 0:
                for j in range(n):
                    h[j] /= port
        g0 = 0.0
        for j in range(n):
            g0 += abs(W[d, j])
        tg = np.empty(n)
        for j in range(n):
            t = W[d, j] * M[j]
            if W[d, j] == 0.0:
                if h[j] != 0.0:
                    M[j] = 1.0
                tg[j] = 0.0
            elif h[j] == 0.0 or abs(t - h[j]) > band:
                M[j] = 1.0 + (Fr[d, j] - 0.5) if Fr[d, j] == Fr[d, j] else 1.0
                tg[j] = W[d, j] * M[j]
            else:
                tg[j] = t
        g1 = 0.0
        for j in range(n):
            g1 += abs(tg[j])
        s = g0 / g1 if g1 > 0 and (both or g1 > g0) else 1.0
        for j in range(n):
            out[d, j] = tg[j] * s
            if W[d, j] == 0.0 or h[j] == 0.0 or abs(W[d, j] * M[j] - h[j]) > band:
                h[j] = out[d, j]
    return out


@numba.njit(cache=True)
def entry_gate(W, Fr, k0):
    K, n = W.shape
    out = W.copy()
    for j in range(n):
        wait = False
        for d in range(k0, K):
            if W[d, j] <= 0.0:
                wait = False
                continue
            opening = W[d - 1, j] <= 0.0
            if opening and Fr[d, j] == Fr[d, j] and Fr[d, j] < 1.0 / 3.0:
                wait = True
            if wait:
                if Fr[d, j] == Fr[d, j] and Fr[d, j] < 1.0 / 3.0:
                    out[d, j] = 0.0
                else:
                    wait = False
    return out


@numba.njit(cache=True)
def exit_hint(W, Fr, TS, k0):
    K, n = W.shape
    out = W.copy()
    for j in range(n):
        off = False
        for d in range(k0, K):
            if W[d, j] <= 0.0:
                off = False
                continue
            if not off and W[d - 1, j] > 0.0 and Fr[d, j] == Fr[d, j] and Fr[d, j] <= 0.1 and TS[d, j] < TS[d - 1, j]:
                off = True
            if off:
                out[d, j] = 0.0
    return out


def make_arm(form, Uo, Fr):
    FE = cv.feats(Uo); P = FE.Pd; TS = FE.tscore(); k0 = Uo.k0
    def post(Uo_, FE_, W, c):
        W = np.ascontiguousarray(np.nan_to_num(W))
        if form in ("gt", "gtr"):
            return tilt_trade_days(W, P, Fr, k0, BAND, form == "gtr")
        if form == "gtb":
            return tilt_trade_days(W, P, Fr, k0, float(c[4]))
        if form == "ge":
            return entry_gate(W, Fr, k0)
        return exit_hint(W, Fr, TS, k0)
    return cv.Arm(form, post=post)


def base_turn(Uo, u, c, cf):
    for f in (f"{slib.D}/arm/{u}_{c}_base_turn.npy", f"{OUT}/turn/{u}_{c}_base_turn.npy"):
        if os.path.exists(f):
            return np.load(f)
    _, T, _ = slib.run_arm(Uo, cf, c, True, False, wfun=cv.make_wfun(Uo, cv.Arm("base")))
    os.makedirs(f"{OUT}/turn", exist_ok=True); np.save(f"{OUT}/turn/{u}_{c}_base_turn.npy.tmp.npy", T)
    os.replace(f"{OUT}/turn/{u}_{c}_base_turn.npy.tmp.npy", f"{OUT}/turn/{u}_{c}_base_turn.npy")
    return T


def run(path, form, nplac):
    z = np.load(path)
    al.GCAL = slib.Cal(RB.restrict(slib.U("M11")))
    cf = slib.arm_cfgs("base")
    res = {c: [] for c in al.COSTS}; plac = {c: [[] for _ in range(nplac)] for c in al.COSTS}; turn = {c: [] for c in al.COSTS}; m11 = {}
    import pickle
    for u in UNIS:
        ck = f"{OUT}/cache/{form}_{u}_{nplac}.pkl"
        if os.path.exists(ck):
            r_, p_, t_ = pickle.load(open(ck, "rb"))
            for c in al.COSTS:
                res[c].append(r_[c]); turn[c].append(t_[c])
                for k in range(nplac): plac[c][k].append(p_[c][k])
            print(form, u, "cached", flush=True)
            continue
        Uo = RB.restrict(slib.U(u)); cal = slib.Cal(Uo)
        nx = np.array([s not in slib.NOX for s in Uo.sy])
        F = RB.fmat(Uo, z, "mkt+r24")
        wf = cv.make_wfun(Uo, make_arm(form, Uo, frac(F, nx)))
        for c in al.COSTS:
            E, T, _ = slib.run_arm(Uo, cf, c, True, False, wfun=wf); E0 = al.off_E(Uo, c, "base")
            T0 = base_turn(Uo, u, c, cf)
            turn[c].append(float(np.median((T - T0) / np.maximum(T0, 1e-9))))
            res[c].append(al.summarize(E, E0, cal))
            if u == "M11":
                m11[c] = (E, wf)
        for k in range(nplac):
            rng = np.random.default_rng([20261118, k]); Fp = F.copy()
            for d in range(len(Fp)):
                m = np.flatnonzero(nx & np.isfinite(Fp[d])); Fp[d, m] = Fp[d, rng.permutation(m)]
            wfp = cv.make_wfun(Uo, make_arm(form, Uo, frac(Fp, nx)))
            for c in al.COSTS:
                E = al.run_cfgs(Uo, None, c, cf, wfun=wfp); E0 = al.off_E(Uo, c, "base")
                plac[c][k].append(al.summarize(E, E0, cal))
        os.makedirs(f"{OUT}/cache", exist_ok=True)
        pickle.dump(({c: res[c][-1] for c in al.COSTS}, {c: [plac[c][k][-1] for k in range(nplac)] for c in al.COSTS}, {c: turn[c][-1] for c in al.COSTS}), open(ck + ".tmp", "wb"))
        os.replace(ck + ".tmp", ck)
        print(form, u, "done", flush=True)
    rows_u = {c: [float(x) for x in al.pooled(res[c], boot=False, sign=False)["mds_u"]] for c in al.COSTS}
    sp = importlib.util.spec_from_file_location("algoruin", os.path.join(RB.HARN, "ruin.py")); ar = importlib.util.module_from_spec(sp); sp.loader.exec_module(ar)
    Uo = RB.restrict(slib.U("M11")); cal = slib.Cal(Uo); pts = slib.ruin_points(); w0 = cv.make_wfun(Uo, cv.Arm("base"))
    if not m11:
        zr = np.load(os.environ["RUIN_BOTPRED"]); nx = np.array([s not in slib.NOX for s in Uo.sy])
        wr = cv.make_wfun(Uo, make_arm(form, Uo, frac(RB.fmat(Uo, zr, "mkt+r24"), nx)))
        for c in al.COSTS:
            m11[c] = (al.run_cfgs(Uo, None, c, cf, wfun=wr), wr)
    out = {}
    for c in al.COSTS:
        r = al.pooled(res[c]); pm = [al.pooled(p, boot=False, sign=False)["mds"] for p in plac[c]]
        r["mds_u"] = rows_u[c]; r["nu_pos"] = int(sum(x > 0 for x in rows_u[c])); r["placebo_mds"] = pm; r["class"] = al.klass(r, max(pm) if pm else None, min(pm) if pm else None)
        r["turnover_change_median_by_universe"] = turn[c]
        E1, w1 = m11[c]; E0 = al.off_E(Uo, c, "base")
        a, b = ar.view(E0, w0, cal, Uo, pts, cf), ar.view(E1, w1, cal, Uo, pts, cf)
        wa = [max(a[p][s]["dd60"] for s in ("S1", "S2", "S3", "S4", "S5")) for p in pts]; wb = [max(b[p][s]["dd60"] for s in ("S1", "S2", "S3", "S4", "S5")) for p in pts]
        r["ruin"] = dict(off=100 * float(np.median(wa)), on=100 * float(np.median(wb)), down=int(sum(y < x for x, y in zip(wa, wb))), up=int(sum(y > x for x, y in zip(wa, wb))))
        out[c] = {k: v for k, v in r.items() if k != "bp_u"}
    return out


if __name__ == "__main__":
    path, form = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.makedirs(OUT, exist_ok=True)
    o = run(path, form, n)
    json.dump(o, open(os.path.join(OUT, f"gate_{form}.json"), "w"), indent=1)
    print(json.dumps({c: {k: o[c][k] for k in ("mds", "mds_lo", "mds_hi", "share", "bp", "years", "class", "ruin")} for c in o}, indent=1))
