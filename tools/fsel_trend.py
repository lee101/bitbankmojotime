"""PREREG-FSEL (harness side): the live book, the engine check, and form 2 (the trend book restricted to the forecast top-K).

  env: tools/fsel_harness.sh (bgm-univ-wt tools, our own SHIFT_D with read-only links, CORRVOL_D, SHIFT_TH)
  python3 tools/fsel_trend.py live                         M11 + gt24 at k 1.10, live Binance configuration -> $BMT_FSEL_OUT/live.npz
  python3 tools/fsel_trend.py engine <lead|full>           numpy engine vs bgm trend-sim on one weight file -> engine_<seg>.json
  python3 tools/fsel_trend.py arms <lead|full> <list>      form-2 arms; list = signals | needed | pick:<N>:<K> | ids joined by ','; optional i/n shard
  python3 tools/fsel_trend.py merge <lead|full>            -> trend_<seg>.npz
"""
import hashlib, json, os, shutil, sys, time, uuid
import numpy as np
UT = "/nvme0n1-disk/code/bgm-univ-wt/tools"
sys.path.insert(0, os.path.join(UT, "univ"))
import uv, slib, cv, al, cands
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r24bots as RB
import r24gate as G
import fsel as F

OUT = F.OUT
slib.COSTS["c20"] = "fee=15e-4 slip=5e-4"
COSTS = ("c7", "c10", "c20")
K110 = (round(slib.LIVE[0] * 1.10, 6), round(slib.LIVE[1] * 1.10, 6)) + tuple(slib.LIVE[2:])
XD = lambda N: f"{F.U}/shift/u/X{N}"


def live():
    Uo = RB.restrict(slib.U("M11")); z = np.load(os.environ["FSEL_BOTPRED"]); nx = np.array([s not in slib.NOX for s in Uo.sy])
    Fr = G.frac(RB.fmat(Uo, z, "mkt+r24"), nx); P = cv.feats(Uo).Pd; kl = int(np.searchsorted(Uo.td, F.TL)); k0 = int(np.searchsorted(Uo.td, F.T0)); out = dict(td=Uo.td)
    def arm(cut=None):
        def post(Uo_, FE_, W, c):
            W = G.tilt_trade_days(np.ascontiguousarray(np.nan_to_num(W)), P, Fr, Uo.k0, 0.05, False)
            if cut is not None: W = W.copy(); W[cut:] = 0.0
            return W
        return cv.Arm("gt", post=post)
    wf = cv.make_wfun(Uo, arm()); g = np.abs(wf(K110)).sum(1); out["gross"] = g
    for c in COSTS:
        E, T, _ = slib.run_arm(Uo, [K110], c, True, False, wfun=wf); out[f"E_{c}"] = E[0]
        Ta = slib.run_arm(Uo, [K110], c, True, False, wfun=cv.make_wfun(Uo, arm(k0)))[1][0] - g[k0 - 1]
        Tb = slib.run_arm(Uo, [K110], c, True, False, wfun=cv.make_wfun(Uo, arm(kl)))[1][0] - g[kl - 1]
        out[f"turn_lead_{c}"] = (Tb - Ta) / ((kl - k0) / 365.25); out[f"turn_conf_{c}"] = (T[0] - Tb) / ((int(np.searchsorted(Uo.td, F.T1)) - kl) / 365.25)
        print("live", c, "turnover / yr lead", round(float(out[f"turn_lead_{c}"]), 2), flush=True)
    out["gross_lead"] = float(g[k0:kl].mean()); out["gross_conf"] = float(g[kl:int(np.searchsorted(Uo.td, F.T1))].mean())
    np.savez(f"{OUT}/live.npz", **out); print("FSEL_LIVE_DONE", flush=True)


def engine(seg):
    """form 1 (N 50, K 10, f168) target weights through bgm trend-sim (no stop, band 0, NaN rows = no decision) vs the numpy engine."""
    W = np.load(f"{OUT}/engine_w_{seg}.npy"); Uo = uv.UV("X50"); Wf = np.zeros((Uo.K, Uo.n)); Wf[:len(W)] = W
    d = f"/dev/shm/fsel_{uuid.uuid4().hex[:8]}"; os.makedirs(d); res = {}
    try:
        Wf.astype("<f8").tofile(f"{d}/w.w")
        st, _ = slib.trend_sim(Uo, [f"{c} wfile={d}/w.w stop=0 smode=3 cool=72 gx=1 band=0 g={F.CAP} {slib.COSTS[c]} apr={F.APR}" for c in COSTS], d)
        z = np.load(f"{OUT}/books_{seg}.npz"); td = z["td"].astype(np.int64); k = np.searchsorted(Uo.td, td); assert (Uo.td[k] == td).all(); x = z["sel|50|10|f168"]
        for c in COSTS:
            E = np.fromfile(f"{d}/{c}.deq", "<f8"); a = E[k[1:]] / E[k[:-1]] - 1.0; b = F.net(x[0], x[1], c)
            res[c] = dict(corr=float(np.corrcoef(a, b)[0, 1]), max_abs_daily_diff=float(np.abs(a - b).max()), ann_return_diff=float((a.mean() - b.mean()) * 365), sim_turnover=float(st[c][0]), numpy_turnover=float(x[1].sum()),
                          sharpe_sim=float(a.mean() / a.std(ddof=1) * np.sqrt(365)), sharpe_numpy=float(b.mean() / b.std(ddof=1) * np.sqrt(365)))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    json.dump(res, open(f"{OUT}/engine_{seg}.json", "w"), indent=1); print(json.dumps(res), "FSEL_ENGINE_DONE", flush=True)


class FV(uv.UV):
    """bear rows from the N-member index, eligibility = the restricted mask."""
    def bear(self, L):
        e = self.elig; self.elig = self.memb
        try: return uv.UV.bear(self, L)
        finally: self.elig = e


_P, _V, _SH = {}, {}, {}
def pan(N, seg):
    if (N, seg) not in _P:
        p = F.Pan(N, seg); p.forecasts(); _P[(N, seg)] = p
    return _P[(N, seg)]


def run_one(bid, seg):
    f = f"{OUT}/trend_{seg}/{bid.replace('|', '_').replace(':', '-')}.npz"
    if os.path.exists(f): return
    t0 = time.time(); form, N, K, nm = bid.split("|"); N, K = int(N), int(K); p = pan(N, seg); memb = np.load(f"{XD(N)}/elig.npy")
    E = np.zeros(memb.shape, bool); E[:p.k0] = memb[:p.k0]
    if form == "tsel":
        if (N, seg, nm) not in _V: _V[(N, seg, nm)] = F.ranks(p, nm)
        H = F.held_daily(p, F.held_sets(p, _V[(N, seg, nm)], K)); E[p.k0:p.k1] = H[p.k0:p.k1] & memb[p.k0:p.k1]
    aid = "f" + hashlib.md5(f"{bid}|{seg}|{os.getpid()}".encode()).hexdigest()[:12]; d = f"{slib.D}/u/{aid}"; os.makedirs(d, exist_ok=True)
    try:
        for x in ("panel.bin", "L.f64", "grid.cells", "cols.json"):
            if not os.path.lexists(f"{d}/{x}"): os.symlink(f"{XD(N)}/{x}", f"{d}/{x}")
        np.save(f"{d}/elig.npy", E); Uo = FV(aid); Uo.memb = memb
        if N in _SH: uv._SIG[aid], cv._FC[aid] = _SH[N]
        wf = cv.make_wfun(Uo, uv.UVArm(aid)); out = dict(td=Uo.td, held=float(E[p.k0:p.k1].sum(1).mean()))
        for c in COSTS:
            Ec, T, _ = slib.run_arm(Uo, [slib.LIVE], c, True, False, wfun=wf); out[f"E_{c}"] = Ec[0]; out[f"T_{c}"] = float(T[0])
        if N not in _SH: _SH[N] = (uv._SIG[aid], cv._FC[aid])
        out["gross"] = float(np.abs(wf(slib.LIVE)).sum(1)[p.k0:p.k1].mean())
        os.makedirs(os.path.dirname(f), exist_ok=True); np.savez(f + ".tmp.npz", **out); os.replace(f + ".tmp.npz", f)
    finally:
        cands.cleanup(Uo) if "Uo" in dir() else None
        uv._SIG.pop(aid, None); cv._FC.pop(aid, None); shutil.rmtree(d, ignore_errors=True)
    print("arm", bid, f"{time.time() - t0:.0f}s", flush=True)


def quiet():
    """no new arm inside the host's quiet window (FSEL_QUIET=HH:MM-HH:MM UTC, default 20:43-20:49)."""
    lo, hi = os.environ.get("FSEL_QUIET", "20:43-20:49").split("-")
    while lo <= time.strftime("%H:%M", time.gmtime()) < hi: time.sleep(10)


def arm_list(seg, spec):
    if spec == "signals":
        return [f"tpre|{N}|0|x" for N in F.NS] + [f"tsel|{N}|{K}|{s}" for N in F.NS for K in F.KS for s in F.SIG]
    if spec == "needed":
        need = json.load(open(f"{OUT}/report_lead.json"))["trend_placebo_needed"]
        return [f"{k}|{fam}:{i}" for k in need for fam in F.FAM for i in range(F.NPL)]
    if spec.startswith("pick:"):
        _, N, K = spec.split(":"); return [f"tpre|{N}|0|x"] + [f"tsel|{N}|{K}|{s}" for s in F.SIG] + [f"tsel|{N}|{K}|{fam}:{i}" for fam in F.FAM for i in range(F.NPL)]
    return spec.split(",")


def merge(seg):
    out = {}
    d = f"{OUT}/trend_{seg}"
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        z = np.load(f"{d}/{f}"); b = f[:-4].split("_"); bid = "/".join(b[:3] + ["_".join(b[3:]).replace("-", ":")]); out["td"] = z["td"]
        for c in COSTS: out[f"E|{bid}|{c}"] = z[f"E_{c}"]; out[f"T|{bid}|{c}"] = z[f"T_{c}"]
        out[f"G|{bid}"] = z["gross"]
    np.savez(f"{OUT}/trend_{seg}.npz", **out); print("merged", len(out), "FSEL_MERGE_DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    if cmd == "live": live()
    elif cmd == "engine": engine(sys.argv[2])
    elif cmd == "merge": merge(sys.argv[2])
    else:
        ids = arm_list(sys.argv[2], sys.argv[3]); sh = sys.argv[4] if len(sys.argv) > 4 else "0/1"; i, n = (int(x) for x in sh.rstrip("r").split("/"))
        for b in (ids[i::n][::-1] if sh.endswith("r") else ids[i::n]):
            quiet(); run_one(b, sys.argv[2])
        print("FSEL_ARMS_DONE", sys.argv[3], i, n, flush=True)
