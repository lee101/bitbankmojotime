"""PREREG-REGEAR: the confirmed trade-day tilt `gt` at a constant risk-budget multiplier k (pv and cap x k, before the tilt), against the
untilted base; pure-gearing control (base at k, no tilt). Gearing calibration, not a significance test.

  env: source <bgm-signal-wt>/tools/algos/env.sh; SHIFT_D per universe set; CORRVOL_D (scratch); REGEAR_OUT (results); REGEAR_TMP (resume cache)
       GATE_UNIS=CONF for R9-R24
  python tools/r24regear.py run  [U,U,...]     cells -> REGEAR_TMP/cell/<u>_<cost>_<arm>.npz (+ k 1 identity checks)
  python tools/r24regear.py live               the two live configurations on M11 -> REGEAR_OUT/live.json
  python tools/r24regear.py ana                -> REGEAR_OUT/regear.json, regear.md

PREREG-REGEAR2 (confirmation of k 1.20 on new universes R41-R56; grids from REGEAR_KS / REGEAR_KB; own SHIFT_D with links to arm / bv /
hetzner / u/M11):
  python tools/r24regear.py draw               the draw procedure: reproduces R9-R24 and R25-R40, prints R41-R56
  python tools/r24regear.py fcmerge            stored predictions of the frozen checkpoints for the R41-R56 coins -> REGEAR_OUT/botpred_r41.npz
  python tools/r24regear.py buildcheck         prep.build of R9 into the own SHIFT_D == the stored R9 files
  python tools/r24regear.py build              prep.build (bgm-signal-wt, unchanged) of R41-R56 into the own SHIFT_D
  python tools/r24regear.py run [U,..]         as above; the base arm of a new universe is simulated here; ruin view on 27 points per universe
  python tools/r24regear.py ana2               -> REGEAR_OUT/regear2.json, regear2.md
"""
import importlib.util, json, math, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r24gate as G
import r24bots as RB
import al, cv, slib

OUT = os.environ["REGEAR_OUT"]; TMP = os.environ["REGEAR_TMP"]
AG = os.environ.get("REGEAR_AGENT", "/nvme0n1-disk/code/bitbankmojotime-agent-wt/runs-agent/r24")
def _grid(name, d): return tuple(float(x) for x in os.environ[name].split(",")) if os.environ.get(name) else d
KS = _grid("REGEAR_KS", (1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30)); KB = _grid("REGEAR_KB", (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30))
KUC = (.20, 1.5, .06, 96, .05, 2, 10)
LEAD = list(al.POOL); CONF = [f"R{i}" for i in range(9, 25)]
NB = 2000; MET = ("cagr", "sharpe", "vol", "wm", "dd21", "dd60", "mdd")
SC = ("S1", "S2", "S3", "S4", "S5")

# ---- PREREG-REGEAR2: R41-R56, drawn by draw() with rng [20261119, 41, i] (the R9-R24 procedure), none equal to an R1-R40 set
NEWSEED = (20261119, 41)
NEWSETS = ["AVAX COMP DASH ONE QTUM SOL XTZ", "DOT IOTA QTUM TFUEL UNI ZEC ZEN", "AVAX COMP DOGE DOT ONE SNX SOL", "COMP IOTA RVN SOL ZEC ZEN ZRX",
           "AAVE AVAX QTUM SNX SOL ZEC ZIL", "BAT DASH DOT MANA ONE SAND ZIL", "AAVE BCH DASH DOT QTUM SOL ZEN", "BAT DOGE DOT IOST SAND SOL ZEC",
           "ALGO DASH IOTA MANA QTUM UNI ZIL", "AAVE ALGO AVAX BAT COMP DOT ZIL", "BAND IOTA NEO SNX SOL TFUEL ZEC", "COMP IOTA MANA ONE TFUEL VET ZEC",
           "AAVE AVAX DOT MANA UNI ZIL ZRX", "AVAX BAND BCH COMP IOTA ONE SNX", "ATOM AVAX BCH COMP SNX ZEC ZEN", "COMP DOT MANA ONE SNX SOL VET"]
R9SETS = ["AVAX COMP IOTA NEO ONE QTUM UNI", "AVAX DOT MANA ONE RVN UNI ZEC", "AVAX BAT NEO ONE SNX SOL UNI", "AVAX IOTA MANA NEO SAND SOL VET",
          "AAVE ALGO BAT COMP DOT QTUM SAND", "AVAX BAND DOGE IOTA SNX SOL ZEC", "BAT DOT RVN SAND UNI ZEC ZEN", "AVAX BAT COMP ENJ MANA QTUM ZEC",
          "COMP ENJ HBAR ONE SOL XTZ ZEN", "AAVE BAND COMP DOT ONE ZEN ZIL", "BAND HBAR IOTA ONE QTUM SNX SOL", "AVAX NEO SOL TFUEL UNI XTZ ZEC",
          "ATOM BAT DOT ONE QTUM TFUEL UNI", "AAVE DOGE ONE SAND VET ZEN ZRX", "BAT ENJ NEO UNI XTZ ZEC ZEN", "AAVE ALGO AVAX ENJ MANA SNX ZEN"]          # bgm-signal-wt tools/signal/conf.py
R25SETS = ["ATOM AVAX COMP DASH ONE SAND UNI", "AAVE DOT ICX IOST UNI ZEC ZRX", "AVAX DASH ENJ IOTA ONE QTUM SOL", "BAND ICX MANA ONE SAND SOL ZEC",
           "AAVE AVAX IOTA MANA ONT UNI ZEN", "BAT COMP ENJ ONE QTUM ZEN ZRX", "ATOM AVAX BAND BAT DOT IOTA ZEC", "AAVE BAT DOGE SOL TFUEL VET ZEC",
           "COMP DOT IOTA ONT SAND ZEC ZEN", "ATOM DOT NEO ONE UNI ZEC ZIL", "AVAX COMP ICX IOTA SAND ZEN ZIL", "ATOM BAT DOT IOTA ONE UNI ZEN",
           "AVAX BAND CHZ COMP ENJ IOTA ZEC", "AAVE CHZ IOTA ONE QTUM SOL ZEN", "BAT CHZ DOT IOST NEO QTUM UNI", "AVAX ONE ONT QTUM SNX UNI XTZ"]        # agent/horizon PREREG-HORIZON (reserved)
A15 = "AAVE AVAX BAT COMP DOT IOTA MANA ONE QTUM SAND SNX SOL UNI ZEC ZEN".split()
def _reg(u, s): x = [f"{c}USDT" for c in s.split()]; slib.UNI[u] = (["BTCUSDT", "ETHUSDT"] + x + ["PAXGUSDT"], slib.PX + tuple(x), True, 2020)
NEW = [f"R{i + 41}" for i in range(len(NEWSETS))]
for _u, _s in zip(NEW, NEWSETS): _reg(_u, _s)
if os.environ.get("REGEAR_TESTSET"):   # code-path test on a stored universe, e.g. "R9": built and simulated like a new one, compared with the stored files
    NEW = os.environ["REGEAR_TESTSET"].split(",")
    for _u in NEW: _reg(_u, R9SETS[int(_u[1:]) - 9])
SIG_U = "/sdb-disk/tradingbots/binance/signal/shift/u"


def kc(c, k):
    """configuration with the risk budget x k: pv and cap (the bear rows and the sim's gross cap follow from them)."""
    return c if k == 1.0 else (round(c[0] * k, 6), round(c[1] * k, 6)) + tuple(c[2:])


def aid(kind, k): return f"{kind}{k:.2f}"
def cellf(u, c, a): return f"{TMP}/cell/{u}_{c}_{a}.npz"
def fc_file(u):
    if u in NEW and not os.environ.get("REGEAR_TESTSET"): return f"{OUT}/botpred_r41.npz"
    return f"{AG}/botpred_conf.npz" if u in CONF else f"{AG}/botpred.npz"
def first(fs):
    for f in fs:
        if os.path.exists(f): return f
    raise FileNotFoundError(fs)
def base_E(u, c): return np.load(first([f"{slib.D}/arm/{u}_{c}_base.npy", f"{AG}/gate/off/{u}_{c}_base.npy", f"{AG}/confirm/off/{u}_{c}_base.npy"]))
def base_T(u, c): return np.load(first([f"{slib.D}/arm/{u}_{c}_base_turn.npy", f"{AG}/gate/turn/{u}_{c}_base_turn.npy", f"{AG}/confirm/turn/{u}_{c}_base_turn.npy"]))
def base_new(Uo, u, c, cf):
    """base arm of a new universe: simulated here with the same binary as the forms (paired), cached in REGEAR_TMP."""
    f = f"{TMP}/base/{u}_{c}.npy"; ft = f"{TMP}/base/{u}_{c}_turn.npy"
    if not os.path.exists(f):
        E, T, _ = slib.run_arm(Uo, cf, c, True, False, wfun=cv.make_wfun(Uo, cv.Arm("base"))); os.makedirs(f"{TMP}/base", exist_ok=True)
        np.save(ft + ".tmp.npy", T); os.replace(ft + ".tmp.npy", ft); np.save(f + ".tmp.npy", E); os.replace(f + ".tmp.npy", f)
        if os.environ.get("REGEAR_TESTSET"): print("base", u, c, "identical to stored" if np.array_equal(E, base_E(u, c)) and np.array_equal(T, base_T(u, c)) else "DIFFERENT from stored", flush=True)
    return np.load(f), np.load(ft)


# ---------------------------------------------------------------- PREREG-REGEAR2: universes and forecast file
def draw_sets(seed, tag, taken=()):
    """the R9-R24 procedure: 4 by rng.choice without replacement from the 15 archive coins never used in R1-R8 (alphabetical), then 3 from
    the remaining non-M11 archive coins in order of first archive bar; rng [seed, tag, i]; a set equal to a taken one is redrawn."""
    import csv
    def first_ts(f):
        with open(f) as h: r = csv.reader(h); hd = next(r); return int(next(r)[hd.index("ts")])
    m11 = [x[:-4] for x in slib.M11]; arch = sorted(f[:-8] for f in os.listdir(f"{slib.D}/bv") if f.endswith("USDT.csv")); pool = [c for c in arch if c not in m11]
    ft = {c: first_ts(f"{slib.D}/bv/{c}USDT.csv") for c in pool}; pool = sorted(pool, key=lambda c: ft[c]); taken = set(taken); out = []
    for i in range(16):
        r = np.random.default_rng([seed, tag, i])
        while True:
            a = [str(x) for x in r.choice(A15, 4, replace=False)]; b = [str(x) for x in r.choice([c for c in pool if c not in a], 3, replace=False)]; s = " ".join(sorted(a + b))
            if s not in taken: break
        taken.add(s); out.append(s)
    return out
def draw():
    print("R9-R24 reproduced", sum(x == y for x, y in zip(draw_sets(20261115, 9), R9SETS)), "/ 16; R25-R40 reproduced", sum(x == y for x, y in zip(draw_sets(20261118, 25), R25SETS)), "/ 16")
    new = draw_sets(*NEWSEED, taken=slib.RSETS + R9SETS + R25SETS); print("R41-R56", new); assert new == NEWSETS and not os.environ.get("REGEAR_TESTSET")
def fcmerge():
    """every non-PAXG archive coin already has stored predictions of the frozen mkt+r24 checkpoints: botpred.npz (36 coins) first, botpred_conf.npz
    for the coins only there. Prediction only; nothing is trained or predicted anew."""
    a, b = np.load(f"{AG}/botpred.npz"), np.load(f"{AG}/botpred_conf.npz"); ks = ["mkt+r24_s0", "mkt+r24_s1"]; sa = set(a["sym"].tolist()); mb = ~np.isin(b["sym"], sorted(sa))
    need = sorted({f"{c}USDT" for s in NEWSETS for c in s.split()} | {"BTCUSDT", "ETHUSDT"}); have = sa | set(b["sym"][mb].tolist()); dmax = 0.0
    for sy in sorted(sa & set(b["sym"].tolist())):
        x, y = a["sym"] == sy, b["sym"] == sy; assert np.array_equal(a["t_abs"][x], b["t_abs"][y]); dmax = max([dmax] + [float(np.abs(a[k][x] - b[k][y]).max()) for k in ks])
    o = {k: np.concatenate([a[k], b[k][mb]]) for k in ["t_abs", "sym"] + ks}; np.savez(f"{OUT}/botpred_r41.npz", **o)
    print("fcmerge coins", len(have), "needed", len(need), "missing", [x for x in need if x not in have], "max abs difference on the", len(sa & set(b["sym"].tolist())), "common coins", dmax, "rows", len(o["sym"]))
def _prep():
    assert os.path.realpath(slib.D).startswith(os.path.realpath(os.environ["REGEAR_OWN"])), "build only into the own SHIFT_D"
    sys.path.insert(0, os.path.dirname(os.path.abspath(slib.__file__))); import prep; return prep
def build():
    prep = _prep()
    for u in NEW:
        if os.path.exists(f"{slib.D}/u/{u}/k20/SEL.deq"): continue
        try: prep.build(u)
        except Exception as e: print("BUILD FAIL", u, repr(e)[:300], flush=True)
    print("BUILD_DONE", [u for u in NEW if os.path.exists(f"{slib.D}/u/{u}/k20/SEL.deq")], flush=True)
def buildcheck():
    """our build path (own SHIFT_D, prep.py unchanged, same binary) on R9 against the stored R9 files."""
    import filecmp
    prep = _prep(); _reg("R9", R9SETS[0]); prep.build("R9")
    fs = ["panel.bin", "L.f64", "grid.cells"] + [f"k{t}/{x}" for t in (5, 10, 20) for x in ("SEL10.wsel", "SEL.deq")]
    print("buildcheck R9", {f: filecmp.cmp(f"{slib.D}/u/R9/{f}", f"{SIG_U}/R9/{f}", shallow=False) for f in fs}, flush=True)


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
    pts2 = [p for p in pts if p >= 96 or p % 4 == 0]          # new universes: every 4th of the 96 sample points + the anchors (27)
    os.makedirs(f"{TMP}/cell", exist_ok=True); os.makedirs(f"{TMP}/k1", exist_ok=True)
    for u in us:
        Uo = RB.restrict(slib.U(u)); cal = slib.Cal(Uo); z = np.load(fc_file(u)); nx = np.array([s not in slib.NOX for s in Uo.sy])
        Fr = G.frac(RB.fmat(Uo, z, "mkt+r24"), nx); arms = {"gt": gt_arm(Uo, Fr), "b": cv.Arm("base")}; b0 = {}
        def base(c):
            if c not in b0:
                E0, T0 = base_new(Uo, u, c, cf) if u in NEW else (base_E(u, c), base_T(u, c))
                m0, R0 = metrics(E0, cal); S, Q, L, npm = msums(R0, cal); b0[c] = (E0, m0, boot_parts(S, Q, L, npm, cnt), T0)
            return b0[c]
        order = [(kind, k) for kind, ks in (("gt", KS), ("b", KB)) for k in ks]
        for kind, k in (order[::-1] if os.environ.get("REGEAR_REV") else order):   # REGEAR_REV: a helper process working from the other end
            if True:
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
                    rp = pts if u == "M11" else (pts2 if u in NEW and (kind == "gt" or a == "b1.00") else None)
                    if rp is not None:
                        ar = ar or ruin_mod(); v = ar.view(E, wf, cal, Uo, rp, cfk)
                        o.update(ruin_w=np.array([max(v[p][s]["dd60"] for s in SC) for p in rp]), ruin_s0=np.array([v[p]["S0"]["dd60"] for p in rp]), ruin_cagr=np.array([v[p]["S0"]["med"] for p in rp]))
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


def gross_view():
    """descriptive (feasibility): row gross of the target book of the two live configurations per arm: mean, max, days above 1 (borrow) and above 2
    (the Binance executor's default --min-margin-level 2.0 blocks increases there), on the evaluation window and on the whole history."""
    Uo = RB.restrict(slib.U("M11")); z = np.load(fc_file("M11")); nx = np.array([s not in slib.NOX for s in Uo.sy]); Fr = G.frac(RB.fmat(Uo, z, "mkt+r24"), nx)
    arms = {"gt": gt_arm(Uo, Fr), "b": cv.Arm("base")}; out = {}
    for nm, c in (("binance", slib.LIVE), ("kucoin", KUC)):
        for kind, ks in (("b", KB), ("gt", KS)):
            wf = cv.make_wfun(Uo, arms[kind])
            for k in ks:
                g = np.abs(wf(kc(c, k))).sum(1); r = {}
                for lab, x in (("win", g[Uo.k0:]), ("all", g)):
                    r[lab] = dict(mean=float(x.mean()), p99=float(np.percentile(x, 99)), max=float(x.max()), days=int(len(x)), gt1=int((x > 1.0 + 1e-9).sum()), gt2=int((x > 2.0 + 1e-9).sum()), gt25=int((x > 2.5 + 1e-9).sum()))
                    e = x[x > 2.0 + 1e-9] - 2.0; up = x > 2.0 + 1e-9; r[lab].update(ex2_mean=float(e.mean()) if len(e) else 0.0, ex2_max=float(e.max()) if len(e) else 0.0, ex2_spells=int((up[1:] & ~up[:-1]).sum() + int(up[0])))
                out[f"{nm}|{aid(kind, k)}"] = r
    json.dump(out, open(f"{OUT}/gross.json", "w"), indent=1)
    for k_, v in out.items(): print(k_, {q: (round(x, 3) if isinstance(x, float) else x) for q, x in v["win"].items()}, "| all max", round(v["all"]["max"], 3), "gt2", v["all"]["gt2"])


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


def ana2():
    """PREREG-REGEAR2: confirmation of k 1.20 on the new universes (pooled over those that built)."""
    al.GCAL = slib.Cal(RB.restrict(slib.U("M11"))); US = [u for u in NEW if os.path.exists(f"{slib.D}/u/{u}/k20/SEL.deq")]; need = math.ceil(0.625 * len(US))
    out = dict(universes=US, sets={u: " ".join(x[:-4] for x in slib.UNI[u][0][2:-1]) for u in US}, k_grid=KS, k_control=KB, universes_needed=need); res = {}; fro = {}
    garms = [aid("gt", k) for k in KS]; barms = [aid("b", k) for k in KB if k != 1.0]
    for c in al.COSTS:
        B = {u: dict(np.load(cellf(u, c, "b1.00"))) for u in US}; rows = {}; cat = lambda X, q: np.concatenate([X[u][q] for u in US])
        for a in garms + barms:
            A = {u: dict(np.load(cellf(u, c, a))) for u in US}; r = {}
            for q in MET + ("gross",):
                x, b = cat(A, q), cat(B, q); r[q] = float(np.median(x)); r["b_" + q] = float(np.median(b)); r["d_" + q] = float(np.median(x - b))
            r["vol_ratio"] = float(np.median(cat(A, "vol") / cat(B, "vol"))); r["share_cagr_up_wm_notup"] = float(((cat(A, "cagr") > cat(B, "cagr")) & (cat(A, "wm") <= cat(B, "wm"))).mean())
            r["share_cagr_up"] = float((cat(A, "cagr") > cat(B, "cagr")).mean()); r["share_sharpe_up"] = float((cat(A, "sharpe") > cat(B, "sharpe")).mean())
            r["turn"] = float(np.median([np.median((A[u]["turn"] - B[u]["turn"]) / np.maximum(B[u]["turn"], 1e-9)) for u in US]))
            dS = np.mean([A[u]["bS"] for u in US], 0); dC = np.mean([A[u]["bC"] for u in US], 0)
            r["mdS"] = float(np.mean([np.nan_to_num(A[u]["sharpe"] - B[u]["sharpe"]).mean() for u in US])); r["mdC"] = float(np.mean([(A[u]["cagr"] - B[u]["cagr"]).mean() for u in US]))
            r["mdS_ci"] = [float(np.percentile(dS, 2.5)), float(np.percentile(dS, 97.5))]; r["mdC_ci"] = [float(np.percentile(dC, 2.5)), float(np.percentile(dC, 97.5))]
            mu = {q: np.array([np.median(A[u][q] - B[u][q]) for u in US]) for q in ("cagr", "wm", "dd60", "mdd", "sharpe")}
            r["u_d_cagr"] = [float(x) for x in mu["cagr"]]; r["u_d_wm"] = [float(x) for x in mu["wm"]]
            r["nu_joint"] = int(((mu["cagr"] > 0) & (mu["wm"] <= 0)).sum()); r["nu_cagr_up"] = int((mu["cagr"] > 0).sum()); r["nu_sharpe_up"] = int((mu["sharpe"] > 0).sum())
            for q in ("wm", "dd60", "mdd"): r["nu_" + q + "_notup"] = int((mu[q] <= 0).sum())
            if "ruin_w" in A[US[0]]:
                rw, bw = cat(A, "ruin_w"), cat(B, "ruin_w")
                r["ruin"] = dict(off=100 * float(np.median(bw)), on=100 * float(np.median(rw)), mean_off=100 * float(bw.mean()), mean_on=100 * float(rw.mean()), down=int((rw < bw).sum()), up=int((rw > bw).sum()), cells=int(len(rw)),
                                 nu_notup=int(sum(np.median(A[u]["ruin_w"]) <= np.median(B[u]["ruin_w"]) for u in US)))
            rows[a] = r
        res[c] = rows
        k1 = [dict(np.load(f"{TMP}/k1/{u}_{c}.npz")) for u in US]; r = al.pooled(k1); r["class_sign_placebo_only"] = al.klass(r); r["nu_pos"] = int(sum(x > 0 for x in r["mds_u"]))
        r["turnover_change_median_by_universe"] = [float(x["turn"]) for x in k1]; fro[c] = {q: v for q, v in r.items() if q != "bp_u"}
    out["pooled"] = res; out["frozen_gt_k1"] = fro
    # ---- rule
    def risk(c, k): r = res[c][aid("gt", k)]; return bool(r["d_wm"] <= 0 and r["d_dd60"] <= 0 and r["d_mdd"] <= 0 and r["ruin"]["on"] <= r["ruin"]["off"])
    def gain(c, k): r = res[c][aid("gt", k)]; return bool(r["mdC"] > 0 and r["nu_joint"] >= need)
    cond = {c: {aid("gt", k): dict(risk=risk(c, k), gain=gain(c, k)) for k in KS} for c in al.COSTS}; ok = lambda k, f: k in KS and all(f(c, k) for c in al.COSTS)
    verdict = "k 1.20 confirmed" if ok(1.2, risk) and ok(1.2, gain) else ("keep 1.10" if ok(1.1, risk) else "lower")
    out["rule"] = dict(conditions=cond, verdict=verdict, same_footing={f"{k:.2f}": bool(ok(k, risk) and ok(k, gain)) for k in KS})
    # ---- tilt at equal drawdown vs pure gearing (paired-median paths vs base k 1)
    eq = {}
    for c in al.COSTS:
        path = lambda q: [0.0 if x == 1.0 else res[c][aid("b", x)]["d_" + q] for x in KB]
        for k in KS:
            e = {}
            for q in ("wm", "dd60", "mdd", "vol"):
                kb = interp_cross(KB, path(q), res[c][aid("gt", k)]["d_" + q]); e[q] = dict(k_base=kb, d_cagr_tilt=res[c][aid("gt", k)]["d_cagr"], d_cagr_gear=None if kb is None else interp_at(KB, path("cagr"), kb))
                e[q]["tilt_minus_gear"] = None if kb is None else e[q]["d_cagr_tilt"] - e[q]["d_cagr_gear"]
            eq[f"{c}|{aid('gt', k)}"] = e
    out["equal_drawdown"] = eq; json.dump(out, open(f"{OUT}/regear2.json", "w"), indent=1)
    L = [f"verdict: {verdict}; universes {len(US)} (needed {need}); same footing {out['rule']['same_footing']}", ""]
    for c in al.COSTS:
        f = fro[c]; L += [f"frozen gt k 1, {c}: mean d Sharpe {f['mds']:+.3f} [{f['mds_lo']:+.3f}, {f['mds_hi']:+.3f}], share {100 * f['share']:.0f} %, universes + {f['nu_pos']}/{len(US)}, years {f['years']}/{f['years'] + f['years_neg']}, sign p95 {f['sign']['p95']:+.3f}, class (sign placebo only) {f['class_sign_placebo_only']}", ""]
        L += [f"### R41-R56 pooled, {c} (medians over cells; d = paired vs base k 1)", "",
              "| arm | CAGR (d) | mean d CAGR [95 %] | mean d Sharpe [95 %] | gross | vol ratio | worst month (d) | dd21 (d) | dd60 (d) | maxDD (d) | turnover | ruin median (mean), base -> arm | universes: CAGR up & wm not up / wm / dd60 / maxDD / ruin not up | cells CAGR up & wm not up | rule risk / gain |", "|" + "---|" * 15]
        b = res[c][garms[0]]; L.append(f"| b1.00 | {b['b_cagr']:.1f} | | | {b['b_gross']:.3f} | 1.000 | {b['b_wm']:.1f} | {b['b_dd21']:.1f} | {b['b_dd60']:.1f} | {b['b_mdd']:.1f} | | {b['ruin']['off']:.1f} ({b['ruin']['mean_off']:.1f}) | | | |")
        for a in garms + barms:
            r = res[c][a]; cd = cond[c].get(a); ru = r.get("ruin")
            L.append(f"| {a} | {r['cagr']:.1f} ({r['d_cagr']:+.1f}) | {r['mdC']:+.2f} [{r['mdC_ci'][0]:+.2f}, {r['mdC_ci'][1]:+.2f}] | {r['mdS']:+.3f} [{r['mdS_ci'][0]:+.3f}, {r['mdS_ci'][1]:+.3f}] | {r['gross']:.3f} | {r['vol_ratio']:.3f} | "
                     f"{r['wm']:.1f} ({r['d_wm']:+.2f}) | {r['dd21']:.1f} ({r['d_dd21']:+.2f}) | {r['dd60']:.1f} ({r['d_dd60']:+.2f}) | {r['mdd']:.1f} ({r['d_mdd']:+.2f}) | {100 * r['turn']:+.1f} % | "
                     + ("" if ru is None else f"{ru['off']:.1f} -> {ru['on']:.1f} ({ru['mean_off']:.1f} -> {ru['mean_on']:.1f})") + f" | {r['nu_joint']} / {r['nu_wm_notup']} / {r['nu_dd60_notup']} / {r['nu_mdd_notup']} / " + ("" if ru is None else str(ru['nu_notup']))
                     + f" | {100 * r['share_cagr_up_wm_notup']:.0f} % | " + ("" if cd is None else f"{'pass' if cd['risk'] else 'fail'} / {'pass' if cd['gain'] else 'fail'}") + " |")
        L.append("")
    L += ["### tilt minus pure gearing at equal risk (CAGR points; k' = pure-gearing k with the same median risk measure)", "", "| cost | arm | worst month: k', tilt - gear | dd60 | maxDD | vol |", "|---|---|---|---|---|---|"]
    fm = lambda e: "outside control grid" if e["k_base"] is None else f"{e['k_base']:.3f}, {e['tilt_minus_gear']:+.2f}"
    for key, e in eq.items(): c, a = key.split("|"); L.append(f"| {c} | {a} | {fm(e['wm'])} | {fm(e['dd60'])} | {fm(e['mdd'])} | {fm(e['vol'])} |")
    open(f"{OUT}/regear2.md", "w").write("\n".join(L) + "\n"); print("\n".join(L[:6]))


if __name__ == "__main__":
    cmd = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    if cmd == "run": run(sys.argv[2].split(",") if len(sys.argv) > 2 else (CONF if os.environ.get("GATE_UNIS") == "CONF" else LEAD))
    elif cmd == "live": live()
    elif cmd == "gross": gross_view()
    elif cmd in ("draw", "fcmerge", "build", "buildcheck", "ana2"): globals()[cmd]()
    else: ana()
