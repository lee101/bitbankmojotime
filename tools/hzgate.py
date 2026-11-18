"""PREREG-HORIZON steps 1-2: long-horizon forecast forms on the ALGOS pooled harness, paired against `base` and against `gt24`.

  env: source <bgm-signal-wt>/tools/algos/env.sh; CORRVOL_D=<out dir>; SHIFT_TH; HZ_UNIS=POOL (default) | NEW (R25-R40 in our own SHIFT_D)
       HZ_X=<other horizon> adds the descriptive arm gtX
  python tools/hzgate.py run    <pred_B.npz> <H> <forms> <n_placebo> <u,u,..|all>     per-universe caches (resumable)
  python tools/hzgate.py ruin   <pred_B.npz> <H> <forms> 0 <tag>                      ruin view on M11 (forms + gt24) -> <out>/hz_ruin_<tag>.json
  python tools/hzgate.py report <pred_B.npz> <H> <forms> <n_placebo> <tag>            pooled tables -> <out>/hz_<tag>.json
Forms: gtH gtB dtH dbH (step 1), gtX (descriptive), gtHr gtBr (risk-matched, with gt24r as the extra reference).
"""
import importlib.util, json, os, pickle, sys
import numpy as np
import numba
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r24bots as RB
import r24gate as G
import al, cv, slib

OUT = os.environ["CORRVOL_D"]
BAND, DELTA, PSEED = 0.05, 0.25, 20261118
NEWSETS = ["ATOM AVAX COMP DASH ONE SAND UNI", "AAVE DOT ICX IOST UNI ZEC ZRX", "AVAX DASH ENJ IOTA ONE QTUM SOL", "BAND ICX MANA ONE SAND SOL ZEC",
           "AAVE AVAX IOTA MANA ONT UNI ZEN", "BAT COMP ENJ ONE QTUM ZEN ZRX", "ATOM AVAX BAND BAT DOT IOTA ZEC", "AAVE BAT DOGE SOL TFUEL VET ZEC",
           "COMP DOT IOTA ONT SAND ZEC ZEN", "ATOM DOT NEO ONE UNI ZEC ZIL", "AVAX COMP ICX IOTA SAND ZEN ZIL", "ATOM BAT DOT IOTA ONE UNI ZEN",
           "AVAX BAND CHZ COMP ENJ IOTA ZEC", "AAVE CHZ IOTA ONE QTUM SOL ZEN", "BAT CHZ DOT IOST NEO QTUM UNI", "AVAX ONE ONT QTUM SNX UNI XTZ"]
NEW = []
for _i, _s in enumerate(NEWSETS):
    _x = [f"{c}USDT" for c in _s.split()]; slib.UNI[f"R{_i + 25}"] = (["BTCUSDT", "ETHUSDT"] + _x + ["PAXGUSDT"], slib.PX + tuple(_x), True, 2020); NEW.append(f"R{_i + 25}")
UNIS = [u for u in NEW if os.path.exists(f"{slib.D}/u/{u}/k20/SEL.deq")] if os.environ.get("HZ_UNIS") == "NEW" else list(al.POOL)
X = os.environ.get("HZ_X")
KIND = {"gt24": ("gt", "24"), "gt24r": ("gtr", "24"), "gtH": ("gt", "H"), "gtHr": ("gtr", "H"), "gtB": ("gt", "B"), "gtBr": ("gtr", "B"), "dtH": ("dt", "H"), "dbH": ("db", "H"), "gtX": ("gt", "X")}


@numba.njit(cache=True)
def band_mult(W, Fr, k0, delta):
    """multiplier 0.5 + f set when the base book starts holding the coin, reset only when it moved by more than delta; 1 when not held."""
    K, n = W.shape
    M = np.ones((K, n))
    for j in range(n):
        held = False; m = 1.0
        for d in range(k0, K):
            if W[d, j] <= 0.0:
                held = False
                continue
            t = 0.5 + Fr[d, j] if Fr[d, j] == Fr[d, j] else 1.0
            if (not held) or abs(t - m) > delta:
                m = t; held = True
            M[d, j] = m
    return M


def make_arm(form, Uo, fs):
    kind, key = KIND[form]; f = fs[key]; P = cv.feats(Uo).Pd; k0 = Uo.k0
    def post(Uo_, FE_, W, c):
        W = np.ascontiguousarray(np.nan_to_num(W))
        if kind in ("gt", "gtr"):
            return G.tilt_trade_days(W, P, f, k0, BAND, kind == "gtr")
        if kind == "dt":
            return al.mult_post(W, np.where(np.isfinite(f), 0.5 + f, 1.0))
        return al.mult_post(W, band_mult(W, f, k0, DELTA))
    return cv.Arm(form, post=post)


def cache_file(u, H, forms, nplac):
    return f"{OUT}/cache/{u}_{H}_{'-'.join(forms)}_{nplac}.pkl"


def mds1(E, E0, cal):
    return float(al.pooled([al.summarize(E, E0, cal)], boot=False, sign=False)["mds"])


_AR, _RV = [], {}
def ruin_worst(key, E, w, cal, Uo, cf):
    """worst-of-S1..S5 dd60 per ruin point of one arm on M11 (cached per key)."""
    if key not in _RV:
        if not _AR:
            sp = importlib.util.spec_from_file_location("algoruin", os.path.join(RB.HARN, "ruin.py")); ar = importlib.util.module_from_spec(sp); sp.loader.exec_module(ar); _AR.append(ar)
        pts = slib.ruin_points(); a = _AR[0].view(E, w, cal, Uo, pts, cf)
        _RV[key] = [max(a[p][s]["dd60"] for s in ("S1", "S2", "S3", "S4", "S5")) for p in pts]
    return _RV[key]


def ruin_view(Uo, cal, cf, E0, w0, E1, w1, c, name):
    wa, wb = ruin_worst(("base", c), E0, w0, cal, Uo, cf), ruin_worst((name, c), E1, w1, cal, Uo, cf)
    return dict(off=100 * float(np.median(wa)), on=100 * float(np.median(wb)), down=int(sum(y < x for x, y in zip(wa, wb))), up=int(sum(y > x for x, y in zip(wa, wb))))


def run_u(z, H, forms, nplac, u, ruin=False):
    ck = cache_file(u, H, forms, nplac)
    if os.path.exists(ck) and not ruin:
        print(u, "cached", flush=True); return
    Uo = RB.restrict(slib.U(u)); cal = slib.Cal(Uo); cf = slib.arm_cfgs("base")
    nx = np.array([s not in slib.NOX for s in Uo.sy])
    F24, FH = RB.fmat(Uo, z, "mkt+r24"), RB.fmat(Uo, z, f"mkt+r{H}")
    f24 = G.frac(F24, nx)
    def fset(FHx):
        fH = G.frac(FHx, nx); return {"24": f24, "H": fH, "B": G.frac(f24 + fH, nx)}
    fs = fset(FH)
    if X: fs["X"] = G.frac(RB.fmat(Uo, z, f"mkt+r{X}"), nx)
    rm = any(f.endswith("r") for f in forms)
    refs = ["gt24"] + (["gt24r"] if rm else [])
    wf = {"base": cv.make_wfun(Uo, cv.Arm("base")), **{r: cv.make_wfun(Uo, make_arm(r, Uo, fs)) for r in refs + forms}}
    E, T, out = {}, {}, dict(cov=float(np.isfinite(F24[Uo.k0:, nx]).mean()), forms={}, plac={}, ruin={})
    dturn = lambda a, b: float(np.median((a - b) / np.maximum(b, 1e-9)))
    for c in al.COSTS:
        for r in ["base"] + refs:
            E[r, c], T[r, c], _ = slib.run_arm(Uo, cf, c, True, False, wfun=wf[r])
        st = al.off_E(Uo, c, "base") if os.path.exists(f"{slib.D}/arm/{u}_{c}_base.npy") else None
        out.setdefault("base_vs_stored", {})[c] = None if st is None else float(np.max(np.abs(st / E["base", c] - 1)))
        for r in refs:
            out["forms"].setdefault(r, {})[c] = dict(b=al.summarize(E[r, c], E["base", c], cal), tb=dturn(T[r, c], T["base", c]))
        if rm:
            out["forms"]["gt24r"][c]["g"] = al.summarize(E["gt24r", c], E["gt24", c], cal)
        for f in forms:
            Ef, Tf, _ = slib.run_arm(Uo, cf, c, True, False, wfun=wf[f])
            ref = "gt24r" if f.endswith("r") else "gt24"
            out["forms"].setdefault(f, {})[c] = dict(b=al.summarize(Ef, E["base", c], cal), g=al.summarize(Ef, E[ref, c], cal), tb=dturn(Tf, T["base", c]), tg=dturn(Tf, T[ref, c]))
            if ruin:
                out["ruin"].setdefault(f, {})[c] = ruin_view(Uo, cal, cf, E["base", c], wf["base"], Ef, wf[f], c, f)
        if ruin:
            for r in refs:
                out["ruin"].setdefault(r, {})[c] = ruin_view(Uo, cal, cf, E["base", c], wf["base"], E[r, c], wf[r], c, r)
            print(u, "ruin", c, flush=True)
    if ruin:
        return out["ruin"]
    pf = [f for f in forms if f != "gtX" and not (f.endswith("r") and os.environ.get("HZ_PLAC_R") == "0")]
    for k in range(nplac):
        rng = np.random.default_rng([PSEED, 5, k]); Fp = FH.copy()
        for d in range(len(Fp)):
            m = np.flatnonzero(nx & np.isfinite(Fp[d])); Fp[d, m] = Fp[d, rng.permutation(m)]
        fp = fset(Fp)
        for f in pf:
            w = cv.make_wfun(Uo, make_arm(f, Uo, fp)); ref = "gt24r" if f.endswith("r") else "gt24"
            for c in al.COSTS:
                Ep = al.run_cfgs(Uo, None, c, cf, wfun=w)
                out["plac"].setdefault(f, {}).setdefault(c, []).append((mds1(Ep, E["base", c], cal), mds1(Ep, E[ref, c], cal)))
        print(u, "placebo", k, flush=True)
    os.makedirs(f"{OUT}/cache", exist_ok=True)
    pickle.dump(out, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
    print(u, "done", flush=True)


def stat(S, pl=None, k=0):
    r = al.pooled(S)
    r["nu_pos"] = int(sum(x > 0 for x in r["mds_u"])); r["se"] = (r["mds_hi"] - r["mds_lo"]) / 3.92
    if pl is not None:
        pm = [float(np.mean([p[i][k] for p in pl])) for i in range(len(pl[0]))] if pl and pl[0] else []
        r["placebo"] = pm; r["class"] = al.klass(r, max(pm) if pm else None, min(pm) if pm else None)
    else:
        r["class"] = al.klass(r)
    return {x: v for x, v in r.items() if x != "bp_u"}


def report(H, forms, nplac, tag):
    us = [u for u in UNIS if os.path.exists(cache_file(u, H, forms, nplac))]
    C = {u: pickle.load(open(cache_file(u, H, forms, nplac), "rb")) for u in us}
    res = dict(universes=us, H=H, n_placebo=nplac, coverage={u: C[u]["cov"] for u in us}, base_vs_stored={u: C[u].get("base_vs_stored") for u in us}, forms={}, ruin={})
    for f in sorted(os.listdir(OUT)):
        if f.startswith("hz_ruin_") and f.endswith(".json"):
            res["ruin"].update(json.load(open(os.path.join(OUT, f))))
    for f in C[us[0]]["forms"]:
        res["forms"][f] = {}
        for c in al.COSTS:
            pl = [C[u]["plac"][f][c] for u in us] if f in C[us[0]]["plac"] else None
            e = dict(vs_base=stat([C[u]["forms"][f][c]["b"] for u in us], pl, 0), turn_vs_base=float(np.median([C[u]["forms"][f][c]["tb"] for u in us])))
            if "g" in C[us[0]]["forms"][f][c]:
                e["vs_gt24"] = stat([C[u]["forms"][f][c]["g"] for u in us], pl, 1)
                if "tg" in C[us[0]]["forms"][f][c]:
                    e["turn_vs_gt24"] = float(np.median([C[u]["forms"][f][c]["tg"] for u in us]))
            res["forms"][f][c] = e
    elig = {}
    for f in forms:
        if f == "gtX" or f.endswith("r"):
            continue
        g = {c: res["forms"][f][c]["vs_gt24"] for c in al.COSTS}
        elig[f] = dict(eligible=bool(all(g[c]["mds_lo"] > 0 and g[c]["placebo"] and g[c]["mds"] > max(g[c]["placebo"]) and g[c]["share"] >= 0.8 for c in al.COSTS)),
                       c7=g["c7"]["mds"], interval_above_zero=[bool(g[c]["mds_lo"] > 0) for c in al.COSTS], above_placebo=[bool(g[c]["placebo"] and g[c]["mds"] > max(g[c]["placebo"])) for c in al.COSTS],
                       share=[g[c]["share"] for c in al.COSTS])
    ok = [f for f in elig if elig[f]["eligible"]]
    res["pick"] = dict(rule=elig, picked=max(ok, key=lambda f: elig[f]["c7"]) if ok else None)
    json.dump(res, open(os.path.join(OUT, f"hz_{tag}.json"), "w"), indent=1)
    f3 = lambda r: f"{r['mds']:+.3f} [{r['mds_lo']:+.3f}, {r['mds_hi']:+.3f}]"
    print("| form | cost | vs base: d Sharpe [95 %] | share | d bp/day | years (per-year d Sharpe) | universes + | placebo min .. max | class | turnover | vs gt24: d Sharpe [95 %] | share | universes + | placebo min .. max | turnover vs gt24 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f, e in res["forms"].items():
        for c in al.COSTS:
            b = e[c]["vs_base"]; g = e[c].get("vs_gt24")
            pb = f"{min(b['placebo']):+.3f} .. {max(b['placebo']):+.3f}" if b.get("placebo") else ""
            row = f"| `{f}` | {c} | {f3(b)} | {100 * b['share']:.0f} % | {b['bp']:+.2f} | {b['years']}/{len(b['ydS'])} ({' '.join(f'{x:+.2f}' for x in b['ydS'])}) | {b['nu_pos']}/{b['nu']} | {pb} | {b['class']} | {100 * e[c]['turn_vs_base']:+.1f} % | "
            if g:
                pg = f"{min(g['placebo']):+.3f} .. {max(g['placebo']):+.3f}" if g.get("placebo") else ""
                row += f"{f3(g)} | {100 * g['share']:.0f} % | {g['nu_pos']}/{g['nu']} | {pg} | {100 * e[c].get('turn_vs_gt24', float('nan')):+.1f} % |"
            else:
                row += " | | | | |"
            print(row)
    print("ruin", json.dumps(res["ruin"])); print("pick", json.dumps(res["pick"]))


if __name__ == "__main__":
    cmd, path, H, forms, nplac, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(","), int(sys.argv[5]), sys.argv[6]
    os.makedirs(OUT, exist_ok=True)
    al.GCAL = slib.Cal(RB.restrict(slib.U("M11")))
    if cmd == "run":
        z = np.load(path)
        for u in (UNIS if last == "all" else last.split(",")):
            run_u(z, H, forms, nplac, u)
        print("HZGATE_RUN_DONE", last, flush=True)
    elif cmd == "ruin":
        z = np.load(path)
        r = run_u(z, H, forms, 0, "M11", ruin=True)
        json.dump(r, open(os.path.join(OUT, f"hz_ruin_{last}.json"), "w"), indent=1); print(json.dumps(r))
    else:
        report(H, forms, nplac, last)
