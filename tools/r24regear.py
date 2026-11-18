"""PREREG-REGEAR: the confirmed trade-day tilt `gt` at a constant risk-budget multiplier k (pv and cap x k, before the tilt), against the
untilted base; pure-gearing control (base at k, no tilt). Gearing calibration, not a significance test.

  env: source <bgm-signal-wt>/tools/algos/env.sh; SHIFT_D per universe set; CORRVOL_D (scratch); REGEAR_OUT (results); REGEAR_TMP (resume cache)
       GATE_UNIS=CONF for R9-R24
  python tools/r24regear.py run  [U,U,...]     cells -> REGEAR_TMP/cell/<u>_<cost>_<arm>.npz (+ k 1 identity checks)
  python tools/r24regear.py live               the two live configurations on M11 -> REGEAR_OUT/live.json
  python tools/r24regear.py ana                -> REGEAR_OUT/regear.json, regear.md
"""
import importlib.util, json, math, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r24gate as G
import r24bots as RB
import al, cv, slib

OUT = os.environ["REGEAR_OUT"]; TMP = os.environ["REGEAR_TMP"]
AG = os.environ.get("REGEAR_AGENT", "/nvme0n1-disk/code/bitbankmojotime-agent-wt/runs-agent/r24")
KS = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30); KB = (0.85, 0.90, 0.95) + KS
KUC = (.20, 1.5, .06, 96, .05, 2, 10)
LEAD = list(al.POOL); CONF = [f"R{i}" for i in range(9, 25)]
NB = 2000; MET = ("cagr", "sharpe", "vol", "wm", "dd21", "dd60", "mdd")
SC = ("S1", "S2", "S3", "S4", "S5")


def kc(c, k):
    """configuration with the risk budget x k: pv and cap (the bear rows and the sim's gross cap follow from them)."""
    return c if k == 1.0 else (round(c[0] * k, 6), round(c[1] * k, 6)) + tuple(c[2:])


def aid(kind, k): return f"{kind}{k:.2f}"
def cellf(u, c, a): return f"{TMP}/cell/{u}_{c}_{a}.npz"
def fc_file(u): return f"{AG}/botpred_conf.npz" if u in CONF else f"{AG}/botpred.npz"
def first(fs):
    for f in fs:
        if os.path.exists(f): return f
    raise FileNotFoundError(fs)
def base_E(u, c): return np.load(first([f"{slib.D}/arm/{u}_{c}_base.npy", f"{AG}/gate/off/{u}_{c}_base.npy", f"{AG}/confirm/off/{u}_{c}_base.npy"]))
def base_T(u, c): return np.load(first([f"{slib.D}/arm/{u}_{c}_base_turn.npy", f"{AG}/gate/turn/{u}_{c}_base_turn.npy", f"{AG}/confirm/turn/{u}_{c}_base_turn.npy"]))


def gt_arm(Uo, Fr):
    """r24gate.make_arm('gt') without the unused trend score: tilt_trade_days(W, P, Fr, k0, 0.05, both=False) on the live book."""
    P = cv.feats(Uo).Pd; k0 = Uo.k0
    return cv.Arm("gt", post=lambda Uo_, FE_, W, c: G.tilt_trade_days(np.ascontiguousarray(np.nan_to_num(W)), P, Fr, k0, G.BAND, False))


def metrics(E, cal):
    """per configuration on the evaluation window: CAGR %, Sharpe, annual vol %, worst calendar month %, worst loss over any holding period of
    up to 21 / 60 days % (daily closes), max drawdown %; and the daily returns."""
    R = slib.rets(E, cal); f = slib.full(R, cal); X = np.log1p(R); le = np.concatenate([np.zeros((len(X), 1)), np.cumsum(X, 1)], 1)
    d = np.zeros(len(X)); dd = {}
    for l in range(1, 61):
        np.maximum(d, (le[:, :-l] - le[:, l:]).max(1), out=d)
        if l in (21, 60): dd[l] = 100 * -np.expm1(-d)
    return dict(cagr=f["cagr"], sharpe=f["sharpe"], vol=100 * R.std(1, ddof=1) * math.sqrt(365), wm=f["wm"], dd21=dd[21], dd60=dd[60], mdd=f["mdd"]), R


def msums(R, cal):
    """monthly sums of r, r^2, log(1 + r) on the global month grid (al.summarize's padding) -> [nm, N] each."""
    Gc = al.gcal(); ix = np.searchsorted(Gc.ms, cal.ms); assert (Gc.ms[ix] == cal.ms).all()
    def pad(X): Y = np.zeros((len(X), Gc.nm)); Y[:, ix] = X; return Y.T.copy()
    npm = np.zeros(Gc.nm); npm[ix] = cal.npm
    return pad(al.msum(R, cal)), pad(al.msum(R * R, cal)), pad(al.msum(np.log1p(R), cal)), npm


def draws(): nm = al.gcal().nm; return np.random.default_rng(7).multinomial(nm, np.ones(nm) / nm, size=NB).astype(float)
def boot_parts(S, Q, L, npm, cnt):
    Md = (cnt @ npm)[:, None]; return al._sh((cnt @ S) / Md, (cnt @ Q) / Md, Md), 100 * np.expm1(365.25 * (cnt @ L) / Md)


def gross(wf, cfgs, k0):
    g = {}; out = np.empty(len(cfgs))
    for i, c in enumerate(cfgs):
        wk = slib.wkey(c)
        if wk not in g: g[wk] = float(np.abs(wf(c)).sum(1)[k0:].mean())
        out[i] = g[wk]
    return out


def ruin_mod():
    sp = importlib.util.spec_from_file_location("algoruin", os.path.join(RB.HARN, "ruin.py")); ar = importlib.util.module_from_spec(sp); sp.loader.exec_module(ar); return ar


def run(us):
    al.GCAL = slib.Cal(RB.restrict(slib.U("M11"))); cf = slib.arm_cfgs("base"); cnt = draws(); ar = None; pts = slib.ruin_points()
    os.makedirs(f"{TMP}/cell", exist_ok=True); os.makedirs(f"{TMP}/k1", exist_ok=True)
    for u in us:
        Uo = RB.restrict(slib.U(u)); cal = slib.Cal(Uo); z = np.load(fc_file(u)); nx = np.array([s not in slib.NOX for s in Uo.sy])
        Fr = G.frac(RB.fmat(Uo, z, "mkt+r24"), nx); arms = {"gt": gt_arm(Uo, Fr), "b": cv.Arm("base")}; b0 = {}
        def base(c):
            if c not in b0:
                E0 = base_E(u, c); m0, R0 = metrics(E0, cal); S, Q, L, npm = msums(R0, cal); b0[c] = (E0, m0, boot_parts(S, Q, L, npm, cnt), base_T(u, c))
            return b0[c]
        for kind, ks in (("gt", KS), ("b", KB)):
            for k in ks:
                a = aid(kind, k)
                if os.environ.get("REGEAR_SMOKE") and k != 1.0: continue
                if all(os.path.exists(cellf(u, c, a)) for c in al.COSTS): continue
                wf = cv.make_wfun(Uo, arms[kind]); cfk = [kc(c, k) for c in cf]
                for c in al.COSTS:
                    if os.path.exists(cellf(u, c, a)): continue
                    E0, m0, (bS0, bC0), T0 = base(c)
                    if a == "b1.00": E, T = E0, T0
                    else: E, T, _ = slib.run_arm(Uo, cfk, c, True, False, wfun=wf)
                    m, R = metrics(E, cal); S, Q, L, npm = msums(R, cal); bS, bC = boot_parts(S, Q, L, npm, cnt)
                    o = dict(m, turn=T, gross=gross(wf, cfk, Uo.k0), bS=(bS - bS0).mean(1), bC=(bC - bC0).mean(1))
                    if a == "gt1.00":   # identity with the stored gt run: summaries (R9-R24 resume cache; lead compared in ana via the pooled json)
                        s = al.summarize(E, E0, cal); tm = float(np.median((T - T0) / np.maximum(T0, 1e-9))); ck = f"{AG}/confirm/cache/gt_{u}_20.pkl"; same = -1
                        if os.path.exists(ck):
                            r_, _, t_ = pickle.load(open(ck, "rb")); same = int(all(np.array_equal(np.asarray(s[q]), np.asarray(r_[c][q])) for q in s) and tm == t_[c])
                        np.savez(f"{TMP}/k1/{u}_{c}.npz", same=same, turn=tm, **s); print("k1", u, c, "identical" if same == 1 else ("DIFFERENT" if same == 0 else "saved"), flush=True)
                    if u == "M11":
                        ar = ar or ruin_mod(); v = ar.view(E, wf, cal, Uo, pts, cfk)
                        o.update(ruin_w=np.array([max(v[p][s]["dd60"] for s in SC) for p in pts]), ruin_s0=np.array([v[p]["S0"]["dd60"] for p in pts]), ruin_cagr=np.array([v[p]["S0"]["med"] for p in pts]))
                    np.savez(cellf(u, c, a) + ".tmp.npz", **o); os.replace(cellf(u, c, a) + ".tmp.npz", cellf(u, c, a))
                print(u, a, "done", flush=True)


def live():
    """the two live configurations on M11: Binance (pv .25 cap 1.875 cool 72) and KuCoin (pv .20 cap 1.5 cool 96) settings, MA200 bear rows."""
    al.GCAL = slib.Cal(RB.restrict(slib.U("M11"))); cnt = draws(); ar = ruin_mod(); Uo = RB.restrict(slib.U("M11")); cal = slib.Cal(Uo)
    z = np.load(fc_file("M11")); nx = np.array([s not in slib.NOX for s in Uo.sy]); Fr = G.frac(RB.fmat(Uo, z, "mkt+r24"), nx)
    arms = {"gt": gt_arm(Uo, Fr), "b": cv.Arm("base")}; cf = [slib.LIVE, KUC]; names = ["binance", "kucoin"]; out = {}; il = slib.sample().index(slib.cidx(slib.LIVE)); raw = {}
    for c in al.COSTS:
        for kind, ks in (("b", KB), ("gt", KS)):
            for k in ks:
                wf = cv.make_wfun(Uo, arms[kind]); cfk = [kc(x, k) for x in cf]; E, T, _ = slib.run_arm(Uo, cfk, c, True, False, wfun=wf)
                m, R = metrics(E, cal); S, Q, L, npm = msums(R, cal); bS, bC = boot_parts(S, Q, L, npm, cnt); v = ar.view(E, wf, cal, Uo, [0, 1], cfk); g = gross(wf, cfk, Uo.k0)
                raw[(c, kind, k)] = (m, T, g, bS, bC, v, E[:, -1] / E[:, Uo.k0])
        ok = bool(np.array_equal(base_E("M11", c)[il], slib.run_arm(Uo, [slib.LIVE], c, True, False, wfun=cv.make_wfun(Uo, cv.Arm("base")))[0][0]))
        m0, T0, g0, bS0, bC0, v0, x0 = raw[(c, "b", 1.0)]
        for i, nm in enumerate(names):
            rows = {}
            for (c_, kind, k), (m, T, g, bS, bC, v, x) in raw.items():
                if c_ != c: continue
                r = {q: float(m[q][i]) for q in MET}; r.update(gross=float(g[i]), turn=float(T[i]), turn_rel=float(T[i] / T0[i] - 1), mult=float(x[i]),
                    ruin_w=100 * max(v[i][s]["dd60"] for s in SC), ruin_s0=100 * v[i]["S0"]["dd60"], ruin_cagr=100 * v[i]["S0"]["med"])
                for q in MET: r["d_" + q] = float(m[q][i] - m0[q][i])
                dS, dC = bS[:, i] - bS0[:, i], bC[:, i] - bC0[:, i]
                r.update(dS_lo=float(np.percentile(dS, 2.5)), dS_hi=float(np.percentile(dS, 97.5)), dC_lo=float(np.percentile(dC, 2.5)), dC_hi=float(np.percentile(dC, 97.5)))
                rows[aid(kind, k)] = r
            out[f"{nm}|{c}"] = dict(cfg=list(cf[i]), base_identical_to_stored=ok if i == 0 else None, arms=rows)
        print("live", c, "stored base identical:", ok, flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(f"{OUT}/live.json", "w"), indent=1)


# ---------------------------------------------------------------- analysis
def interp_cross(xs, ys, y):
    """smallest x on the piecewise-linear path (xs ascending) at which ys reaches y; None outside the path's range."""
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if (a - y) * (b - y) <= 0 and a != b: return xs[i] + (xs[i + 1] - xs[i]) * (y - a) / (b - a)
        if a == y: return xs[i]
    return xs[-1] if ys[-1] == y else None


def interp_at(xs, ys, x): return float(np.interp(x, xs, ys))


def ana():
    al.GCAL = slib.Cal(RB.restrict(slib.U("M11"))); US = LEAD + CONF; out = {"k_grid": KS, "k_control": KB, "universes": US}
    live_ = json.load(open(f"{OUT}/live.json"))
    # ---- identity at k 1
    idn = {}
    for c in al.COSTS:
        k1 = {u: dict(np.load(f"{TMP}/k1/{u}_{c}.npz")) for u in US}; row = dict(conf_cache_identical=int(sum(int(k1[u]["same"]) == 1 for u in CONF)), conf_n=len(CONF))
        for nm, us, f in (("lead", LEAD, f"{AG}/gate/gate_gt.json"), ("conf", CONF, f"{AG}/confirm/gate_gt.json")):
            r = al.pooled([k1[u] for u in us]); st = json.load(open(f))[c]; r["turnover_change_median_by_universe"] = [float(k1[u]["turn"]) for u in us]
            keys = [q for q in ("mds", "mds_lo", "mds_hi", "share", "share_neg", "bp", "bp_lo", "bp_hi", "ydS", "med", "same", "sign", "mds_u", "turnover_change_median_by_universe") if q in st]
            row[nm + "_json_identical"] = bool(all(r[q] == st[q] for q in keys)); row[nm + "_keys"] = len(keys); row[nm + "_mds"] = r["mds"]
        A, B = dict(np.load(cellf("M11", c, "gt1.00"))), dict(np.load(cellf("M11", c, "b1.00"))); st = json.load(open(f"{AG}/gate/gate_gt.json"))[c]["ruin"]
        row["ruin_identical"] = bool(dict(off=100 * float(np.median(B["ruin_w"])), on=100 * float(np.median(A["ruin_w"])), down=int((A["ruin_w"] < B["ruin_w"]).sum()), up=int((A["ruin_w"] > B["ruin_w"]).sum())) == st)
        idn[c] = row
    out["identity_k1"] = idn
    # ---- pooled table
    res = {}
    for c in al.COSTS:
        B = {u: dict(np.load(cellf(u, c, "b1.00"))) for u in US}; rows = {}
        for a in [aid("gt", k) for k in KS] + [aid("b", k) for k in KB]:
            A = {u: dict(np.load(cellf(u, c, a))) for u in US}; r = {}
            for sub, us in (("", US), ("lead_", LEAD), ("conf_", CONF)):
                cat = lambda X, q: np.concatenate([X[u][q] for u in us])
                for q in MET + ("gross",):
                    x, b = cat(A, q), cat(B, q); r[sub + q] = float(np.median(x)); r[sub + "d_" + q] = float(np.median(x - b))
                r[sub + "vol_ratio"] = float(np.median(cat(A, "vol") / cat(B, "vol"))); r[sub + "gross_ratio"] = float(np.median(cat(A, "gross") / np.maximum(cat(B, "gross"), 1e-12)))
                r[sub + "share_cagr_up_wm_notup"] = float(((cat(A, "cagr") > cat(B, "cagr")) & (cat(A, "wm") <= cat(B, "wm"))).mean())
                r[sub + "share_cagr_up"] = float((cat(A, "cagr") > cat(B, "cagr")).mean()); r[sub + "share_sharpe_up"] = float((cat(A, "sharpe") > cat(B, "sharpe")).mean())
                r[sub + "turn"] = float(np.median([np.median((A[u]["turn"] - B[u]["turn"]) / np.maximum(B[u]["turn"], 1e-9)) for u in us]))
                dS = np.mean([A[u]["bS"] for u in us], 0); dC = np.mean([A[u]["bC"] for u in us], 0)
                r[sub + "mdS"] = float(np.mean([np.nan_to_num(A[u]["sharpe"] - B[u]["sharpe"]).mean() for u in us])); r[sub + "mdC"] = float(np.mean([(A[u]["cagr"] - B[u]["cagr"]).mean() for u in us]))
                r[sub + "mdS_ci"] = [float(np.percentile(dS, 2.5)), float(np.percentile(dS, 97.5))]; r[sub + "mdC_ci"] = [float(np.percentile(dC, 2.5)), float(np.percentile(dC, 97.5))]
                r[sub + "nu_cagr_up"] = int(sum(np.median(A[u]["cagr"] - B[u]["cagr"]) > 0 for u in us)); r[sub + "nu_wm_notup"] = int(sum(np.median(A[u]["wm"] - B[u]["wm"]) <= 0 for u in us))
            rw, bw = A["M11"]["ruin_w"], B["M11"]["ruin_w"]
            r["ruin"] = dict(off=100 * float(np.median(bw)), on=100 * float(np.median(rw)), down=int((rw < bw).sum()), up=int((rw > bw).sum()), s0_off=100 * float(np.median(B["M11"]["ruin_s0"])), s0_on=100 * float(np.median(A["M11"]["ruin_s0"])))
            rows[a] = r
        res[c] = rows
    out["pooled"] = res; out["live"] = live_
    # ---- pick rule
    def ok_pool(c, k): r = res[c][aid("gt", k)]; return r["d_wm"] <= 0 and r["d_dd60"] <= 0
    def ok_live(c, k): return all(live_[f"{nm}|{c}"]["arms"][aid("gt", k)]["d_" + q] <= 0 for nm in ("binance", "kucoin") for q in ("wm", "dd60"))
    def largest(f):
        ks = None
        for k in KS:
            if not all(f(c, k) for c in al.COSTS): break
            ks = k
        return ks
    pick = dict(k_star=largest(lambda c, k: ok_pool(c, k) and ok_live(c, k)), k_pool=largest(ok_pool), k_live=largest(ok_live),
                k_vol_grid=largest(lambda c, k: res[c][aid("gt", k)]["vol_ratio"] <= 1.0), conditions={c: {aid("gt", k): dict(pool=bool(ok_pool(c, k)), live=bool(ok_live(c, k))) for k in KS} for c in al.COSTS})
    pick["k_vol_interp"] = {c: interp_cross(KS, [res[c][aid("gt", k)]["vol_ratio"] for k in KS], 1.0) for c in al.COSTS}
    pick["k_vol_interp_live"] = {f"{nm}|{c}": interp_cross(KS, [live_[f"{nm}|{c}"]["arms"][aid("gt", k)]["vol"] / live_[f"{nm}|{c}"]["arms"]["b1.00"]["vol"] for k in KS], 1.0) for nm in ("binance", "kucoin") for c in al.COSTS}
    out["pick"] = pick
    # ---- tilt at equal drawdown vs pure gearing (frontiers of the paired medians vs base k 1; live: the configuration's own values)
    eq = {}
    for c in al.COSTS:
        for k in KS:
            e = {}
            for q in ("wm", "dd60", "mdd", "vol"):
                kb = interp_cross(KB, [res[c][aid("b", x)]["d_" + q] for x in KB], res[c][aid("gt", k)]["d_" + q])
                e[q] = dict(k_base=kb, d_cagr_tilt=res[c][aid("gt", k)]["d_cagr"], d_cagr_gear=None if kb is None else interp_at(KB, [res[c][aid("b", x)]["d_cagr"] for x in KB], kb))
                e[q]["tilt_minus_gear"] = None if kb is None else e[q]["d_cagr_tilt"] - e[q]["d_cagr_gear"]
            eq[f"pooled|{c}|{aid('gt', k)}"] = e
            for nm in ("binance", "kucoin"):
                L = live_[f"{nm}|{c}"]["arms"]; e = {}
                for q in ("wm", "dd60", "mdd", "vol"):
                    kb = interp_cross(KB, [L[aid("b", x)][q] for x in KB], L[aid("gt", k)][q])
                    e[q] = dict(k_base=kb, cagr_tilt=L[aid("gt", k)]["cagr"], cagr_gear=None if kb is None else interp_at(KB, [L[aid("b", x)]["cagr"] for x in KB], kb))
                    e[q]["tilt_minus_gear"] = None if kb is None else e[q]["cagr_tilt"] - e[q]["cagr_gear"]
                eq[f"{nm}|{c}|{aid('gt', k)}"] = e
    out["equal_drawdown"] = eq
    json.dump(out, open(f"{OUT}/regear.json", "w"), indent=1)
    # ---- markdown
    L = [f"k* = {pick['k_star']} (pooled-only {pick['k_pool']}, live-only {pick['k_live']}, vol-matched grid {pick['k_vol_grid']}, interpolated {pick['k_vol_interp']})", ""]
    L.append("identity k 1: " + json.dumps(idn)); L.append("")
    for c in al.COSTS:
        L += [f"### pooled 30 universes x 3,003, {c} (medians over cells; d = paired vs base k 1)", "",
              "| arm | CAGR (d) | mean d CAGR [95 %] | Sharpe (d) | mean d Sharpe [95 %] | gross | vol (ratio) | worst month (d) | dd21 (d) | dd60 (d) | maxDD (d) | turnover | ruin P(DD>60%) | CAGR up & wm not up | rule pool / live |", "|" + "---|" * 15]
        for a in ["b1.00"] + [aid("gt", k) for k in KS] + [aid("b", k) for k in KB if k != 1.0]:
            r = res[c][a]; cd = pick["conditions"][c].get(a)
            L.append(f"| {a} | {r['cagr']:.1f} ({r['d_cagr']:+.1f}) | {r['mdC']:+.2f} [{r['mdC_ci'][0]:+.2f}, {r['mdC_ci'][1]:+.2f}] | {r['sharpe']:.3f} ({r['d_sharpe']:+.3f}) | {r['mdS']:+.3f} [{r['mdS_ci'][0]:+.3f}, {r['mdS_ci'][1]:+.3f}] | {r['gross']:.3f} | "
                     f"{r['vol']:.1f} ({r['vol_ratio']:.3f}) | {r['wm']:.1f} ({r['d_wm']:+.2f}) | {r['dd21']:.1f} ({r['d_dd21']:+.2f}) | {r['dd60']:.1f} ({r['d_dd60']:+.2f}) | {r['mdd']:.1f} ({r['d_mdd']:+.2f}) | {100 * r['turn']:+.1f} % | "
                     f"{r['ruin']['off']:.1f} -> {r['ruin']['on']:.1f} | {100 * r['share_cagr_up_wm_notup']:.0f} % | " + ("" if cd is None else f"{'ok' if cd['pool'] else 'no'} / {'ok' if cd['live'] else 'no'}") + " |")
        L.append("")
    for nm in ("binance", "kucoin"):
        for c in al.COSTS:
            V = live_[f"{nm}|{c}"]; L += [f"### live configuration {nm} {V['cfg']} on M11, {c}", "",
                "| arm | CAGR (d) [95 %] | Sharpe (d) [95 %] | gross | vol | worst month (d) | dd21 (d) | dd60 (d) | maxDD (d) | turnover | x equity 2023-01..2026-10 | ruin worst S1-S5 / S0 P(DD>60%) |", "|" + "---|" * 12]
            for a in ["b1.00"] + [aid("gt", k) for k in KS] + [aid("b", k) for k in KB if k != 1.0]:
                r = V["arms"][a]
                L.append(f"| {a} | {r['cagr']:.1f} ({r['d_cagr']:+.1f}) [{r['dC_lo']:+.1f}, {r['dC_hi']:+.1f}] | {r['sharpe']:.3f} ({r['d_sharpe']:+.3f}) [{r['dS_lo']:+.3f}, {r['dS_hi']:+.3f}] | {r['gross']:.3f} | {r['vol']:.1f} | "
                         f"{r['wm']:.1f} ({r['d_wm']:+.2f}) | {r['dd21']:.1f} ({r['d_dd21']:+.2f}) | {r['dd60']:.1f} ({r['d_dd60']:+.2f}) | {r['mdd']:.1f} ({r['d_mdd']:+.2f}) | {100 * r['turn_rel']:+.1f} % | {r['mult']:.2f} | {r['ruin_w']:.1f} / {r['ruin_s0']:.1f} |")
            L.append("")
    L += ["### tilt at equal drawdown vs pure gearing (CAGR points; k_base = pure-gearing k with the same risk measure)", "", "| line | arm | worst month: k_base, tilt - gear | dd60: k_base, tilt - gear | maxDD: k_base, tilt - gear | vol: k_base, tilt - gear |", "|---|---|---|---|---|---|"]
    fm = lambda e: "outside control grid" if e["k_base"] is None else f"{e['k_base']:.3f}, {e['tilt_minus_gear']:+.2f}"
    for key, e in eq.items():
        nm, c, a = key.split("|"); L.append(f"| {nm} {c} | {a} | {fm(e['wm'])} | {fm(e['dd60'])} | {fm(e['mdd'])} | {fm(e['vol'])} |")
    open(f"{OUT}/regear.md", "w").write("\n".join(L) + "\n"); print("\n".join(L[:3]))


if __name__ == "__main__":
    cmd = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    if cmd == "run": run(sys.argv[2].split(",") if len(sys.argv) > 2 else (CONF if os.environ.get("GATE_UNIS") == "CONF" else LEAD))
    elif cmd == "live": live()
    else: ana()
