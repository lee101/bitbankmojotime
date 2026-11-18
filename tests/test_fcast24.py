"""tools/fcast24.py: kline refresh against a local mock venue (politeness, resume, atomic files), the file contract, the input arithmetic
against train/bank.py and the Mojo engine against PyTorch on real (day, coin) rows.

  pytest tests/test_fcast24.py                       unit tests always; the parity tests need the research cache + rank24 checkpoints
  python tests/test_fcast24.py parity [n_days]       prints the parity table (BMT_DATA, BMT_R24_CKPT, BMT_BOTPRED)"""
import http.server, json, math, os, struct, subprocess, sys, threading, time, urllib.parse
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT, os.path.join(ROOT, "bench")]
import fcast24 as F

BMT2 = os.environ.get("BMT2", os.path.join(ROOT, "build", "bmt2"))
CACHE = os.environ.get("BMT_DATA", "/nvme0n1-disk/code/bitbankmojotime/data/cache")
CKPT = os.environ.get("BMT_R24_CKPT", "/nvme0n1-disk/code/bitbankmojotime-agent-wt/runs-agent/r24/ckpt")
BOTPRED = os.environ.get("BMT_BOTPRED", "/nvme0n1-disk/code/bitbankmojotime-agent-wt/runs-agent/r24/botpred.npz")
os.environ.setdefault("BMT_DATA", CACHE)
H = 3600


# ---------------------------------------------------------------- mock venue (KuCoin-shaped)
class Venue:
    def __init__(s, syms, t_end, fail429=0, delay=0.03, gaps=(), ended=None, started=None):
        s.syms, s.t_end, s.fail429, s.delay = list(syms), t_end, fail429, delay
        s.gaps, s.ended, s.started = set(gaps), ended or {}, started or {}
        s.lock = threading.Lock(); s.live = s.peak = s.calls = s.n429 = 0; s.asked = []; s.down = set()

    def price(s, x, k):
        j = s.syms.index(x)
        return (5.0 + j) * math.exp(0.02 * math.sin(k / 37.0 + j) + 0.0004 * ((k * 7919 + j * 104729) % 97 - 48))

    def has(s, x, k):
        if k > s.ended.get(x, s.t_end) or k < s.started.get(x, 0) or k > s.t_end:
            return False
        return not (x in s.gaps and k % 5 in (1, 2))

    def bars(s, x, a, b):
        out = []
        for k in range(a // H, b // H + 1):
            if s.has(x, k):
                c = s.price(x, k); o = s.price(x, k - 1)
                out.append([str(k * H), repr(o), repr(c), repr(max(o, c) * 1.002), repr(min(o, c) * 0.997), "10", "1000"])
        return out[::-1][:1500]

    def serve(s):
        v = s

        class Hd(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query))
                with v.lock:
                    v.calls += 1; v.live += 1; v.peak = max(v.peak, v.live)
                try:
                    time.sleep(v.delay)
                    if u.path == "/api/v2/symbols":
                        body = dict(code="200000", data=[dict(symbol=x[:-4] + "-USDT", quoteCurrency="USDT", enableTrading=True) for x in v.syms])
                    else:
                        x = q["symbol"].replace("-", "")
                        with v.lock:
                            v.asked.append((x, int(q["startAt"]) // H, int(q["endAt"]) // H))
                            lim = v.fail429 > 0
                            v.fail429 -= lim; v.n429 += lim
                        if lim:
                            self.send_response(429); self.send_header("Retry-After", "0.2"); self.end_headers(); return
                        if x in v.down:
                            self.send_response(400); self.end_headers(); return
                        body = dict(code="200000", data=v.bars(x, int(q["startAt"]), int(q["endAt"])))
                    b = json.dumps(body).encode()
                    self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
                finally:
                    with v.lock:
                        v.live -= 1
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hd)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"


SY = [f"C{i:02d}USDT" for i in range(24)]
T_END = 479999                                                       # T_END + 1 is 00:00 UTC


def refresh(v, base, cache, origin, span=F.L + F.PAD + 1, syms=SY, tries=7):
    http_ = F.Http(conc=8, rps=400, tries=tries)                     # asks for 8 workers: the client must cap at 4
    return F.refresh("kucoin", syms, cache, origin, span, http_, base, now=(origin + 1) * H), http_


def test_refresh_polite_resumable_atomic(tmp_path):
    v = Venue(SY, T_END, fail429=9, gaps=("C03USDT",), ended={"C05USDT": T_END - 100}, started={"C07USDT": T_END - 200})
    srv, base = v.serve(); cache = str(tmp_path / "k")
    v.down = {"C11USDT", "C12USDT"}                                  # two symbols fail hard: the run reports them, the rest is kept
    st, h = refresh(v, base, cache, T_END, syms=SY + ["NOPEUSDT"])
    assert v.peak <= 4 and h.peak <= 4 and h.conc == 4               # never more than 4 in flight
    assert v.n429 == 9 and h.limited == 9 and st["http429"] == 9     # every 429 was retried after backing off
    assert st["listed"] == 24 and st["fetched"] == 22 and st["failed"] == 2 and st["skipped"] == 0
    assert not [f for f in os.listdir(cache) if ".tmp" in f] and len(os.listdir(cache)) == 22
    frm, to, rows = F.cache_read(cache, "C00USDT")
    assert (frm, to) == (T_END - F.L - F.PAD, T_END) and len(rows) == F.L + F.PAD + 1 and rows[T_END][3] == v.price("C00USDT", T_END)
    assert max(F.cache_read(cache, "C05USDT")[2]) == T_END - 100 and min(F.cache_read(cache, "C07USDT")[2]) == T_END - 200
    assert all(k % 5 not in (1, 2) for k in F.cache_read(cache, "C03USDT")[2])
    n0 = len(v.asked); v.down = set()                                 # resume: only the two missing symbols are fetched
    st2, _ = refresh(v, base, cache, T_END)
    assert st2["skipped"] == 22 and st2["fetched"] == 2 and st2["failed"] == 0 and {x for x, _, _ in v.asked[n0:]} == {"C11USDT", "C12USDT"}
    n1 = len(v.asked); v.t_end = T_END + 24                           # next day: only the new bars are asked for
    st3, _ = refresh(v, base, cache, T_END + 24)
    assert st3["fetched"] == 24 and all(a == T_END + 1 for _, a, _ in v.asked[n1:]) and len(v.asked) - n1 == 24
    assert F.cache_read(cache, "C00USDT")[:2] == (T_END - F.L - F.PAD, T_END + 24)
    before = {f: open(os.path.join(cache, f)).read() for f in os.listdir(cache)}
    st4, _ = refresh(v, base, cache, T_END + 24)                      # nothing to do: no kline request, files untouched
    assert st4["skipped"] == 24 and st4["http"] == 1 and before == {f: open(os.path.join(cache, f)).read() for f in os.listdir(cache)}
    srv.shutdown()


def test_refresh_never_stores_an_open_bar(tmp_path):
    v = Venue(SY[:3], T_END + 5); srv, base = v.serve(); cache = str(tmp_path / "k")
    F.refresh("kucoin", SY[:3], cache, T_END + 5, 40, F.Http(rps=400), base, now=(T_END + 3) * H + 10)      # the clock says bars > T_END + 2 are not closed
    assert max(F.cache_read(cache, SY[0])[2]) == T_END + 2
    srv.shutdown()


def test_refresh_gives_up_without_touching_files(tmp_path):
    v = Venue(SY[:2], T_END, fail429=10 ** 6); srv, base = v.serve(); cache = str(tmp_path / "k")
    st, h = F.refresh("kucoin", SY[:2], cache, T_END, 40, F.Http(rps=400, tries=2), base, now=(T_END + 1) * H), None
    assert st["failed"] == 2 and st["fetched"] == 0 and os.listdir(cache) == []
    srv.shutdown()


# ---------------------------------------------------------------- inputs: pure-Python arithmetic == numpy float32 / float16
def synth_data(v, syms, t):
    return {x: {k: (v.price(x, k - 1), max(v.price(x, k - 1), v.price(x, k)) * 1.002, min(v.price(x, k - 1), v.price(x, k)) * 0.997, v.price(x, k), 1.0)
                for k in range(t - F.L - F.PAD, t + 1) if v.has(x, k)} for x in syms}


def np_inputs(data, pool, coins, t, ffill_h=48):
    """independent numpy version of F.build_rows (vectorised, the bank's formulas)."""
    import numpy as np
    n = F.L + F.PAD + 1; h0 = t - n + 1
    def lp_of(x, pool_rule):
        rows = data.get(x) or {}
        real = np.array([h0 + i in rows for i in range(n)])
        if not real.any():
            return None, real
        c = np.array([math.log(rows[h0 + i][3]) if real[i] else np.nan for i in range(n)])
        idx = np.maximum.accumulate(np.where(real, np.arange(n), -1))
        lp = np.where(idx >= 0, c[np.maximum(idx, 0)], np.nan)
        last = np.flatnonzero(real)[-1]
        if pool_rule and n - 1 - last > ffill_h:
            lp[last + 1:] = np.nan
        return lp, real
    LP = np.array([lp for lp in (lp_of(x, True)[0] for x in pool) if lp is not None])
    live = np.isfinite(LP)
    r = np.nan_to_num(np.diff(LP, axis=1, prepend=np.nan))
    mkt = (np.where(live, r, 0).sum(0) / np.maximum(live.sum(0), 1))[-F.L:]
    rk = LP - np.roll(LP, 1, 1); rk[:, 0] = np.nan
    with np.errstate(all="ignore"):
        disp, brd = np.nan_to_num(np.nanstd(rk, 0))[-F.L:], np.nan_to_num(np.nanmean(np.where(np.isfinite(rk), rk > 0, np.nan), 0) - 0.5)[-F.L:]
    out = {}
    for x in coins:
        lp, real = lp_of(x, False)
        if lp is None or not real[-1] or not np.isfinite(lp[F.PAD]):
            continue
        rows = data[x]
        up = np.array([np.float32(math.log(rows[h0 + i][1]) - math.log(rows[h0 + i][3])) if real[i] else 0.0 for i in range(n)], np.float64)
        dn = np.array([np.float32(math.log(rows[h0 + i][3]) - math.log(rows[h0 + i][2])) if real[i] else 0.0 for i in range(n)], np.float64)
        rr = (lp[1:] - lp[:-1]).astype(np.float32)[-F.L:]
        hl = ((lp + up) - (lp - dn)).astype(np.float32).astype(np.float16).astype(np.float32)[-F.L:]
        gk = np.sqrt(np.maximum(np.float32(0.5) * hl * hl - np.float32(0.3862944) * rr * rr, 0))
        out[x] = dict(r=rr, hl=hl, gk=gk, mkt=mkt.astype(np.float32), disp=disp.astype(np.float32), brd=brd.astype(np.float32))
    return out


def test_inputs_match_numpy_bit_for_bit():
    np = pytest.importorskip("numpy")
    v = Venue(SY, T_END, gaps=("C03USDT", "C09USDT"), ended={"C05USDT": T_END - 100, "C06USDT": T_END - 20}, started={"C07USDT": T_END - 200, "C08USDT": T_END - 540})
    data = synth_data(v, SY, T_END)
    rows, cal, info = F.build_rows(data, SY[4:], SY[:10], T_END)
    ref = np_inputs(data, SY[4:], SY[:10], T_END)
    assert sorted(rows) == sorted(ref) and len(rows) == 7 and info["skipped"] == {"C05USDT": "no_bar_at_origin", "C06USDT": "no_bar_at_origin", "C07USDT": "short_history"}
    for x in rows:
        for ch in ("r", "hl", "gk", "brd"):
            assert np.array_equal(np.array(rows[x][ch], np.float32), ref[x][ch]), (x, ch)
        for ch in ("mkt", "disp"):                                   # sums: exactly rounded here, pairwise in numpy -> last float32 bit at most
            a, b = np.array(rows[x][ch], np.float32), ref[x][ch]
            assert np.allclose(a, b, rtol=3e-7, atol=0) and (a == b).mean() > 0.99, (x, ch)
    assert info["pool_live"] == 19 and info["pool_min"] == 19 and cal == [(T_END + 1) % 24, ((T_END + 1) // 24 + 4) % 7, 0]
    assert max(abs(z) for z in rows["C03USDT"]["hl"]) > 0 and 0.0 in rows["C03USDT"]["hl"] and 0.0 in rows["C03USDT"]["r"]      # gaps: flat bars


# ---------------------------------------------------------------- file contract
def test_file_roundtrip_merge_and_strict_check(tmp_path):
    f = str(tmp_path / "f.tsv")
    F.write_file(f, "m~abcd1234", 2, {(86400 * 3, "ETH_USDT"): -1.5e-3, (86400 * 2, "BTC_USDT"): 2.5e-3, (86400 * 3, "BTC_USDT"): 1e-9}, created=7)
    assert open(f).read() == "# bmt-fcast24 v1 model=m~abcd1234 seeds=2 horizon_h=24 created=7\n172800\tBTC_USDT\t2.50000000e-03\n259200\tBTC_USDT\t1.00000000e-09\n259200\tETH_USDT\t-1.50000000e-03\n"
    hd, rows = F.read_file(f)
    assert hd["model"] == "m~abcd1234" and len(rows) == 3 and not [x for x in os.listdir(tmp_path) if "tmp" in x]
    good = open(f).read()
    for bad in (good[:-5], good.replace("v1", "v2"), good + "259200\tBTC_USDT\t1\n", good + "259201\tXRP_USDT\t1\n", good + "86400\tXRP_USDT\t1\n", good + "345600\tXRP_USDT\tnan\n",
                good + "345600\tXRP_USDT\n", good.replace("horizon_h=24", "horizon_h=1"), ""):
        open(f, "w").write(bad)
        with pytest.raises(Exception):
            F.read_file(f)


class A:                                                             # arguments of cmd_write
    def __init__(s, **k):
        s.name, s.bmt2, s.threads, s.ffill_h, s.min_pool, s.max_catchup = "bmt2-r24-test", BMT2, 2, 48, 10, 45
        s.__dict__.update(k)


def rand_weights(tmp_path, seeds=(0, 1)):
    torch = pytest.importorskip("torch")
    from train.model2 import BMT2 as M
    from train.model import SIZES
    from train import export2
    out = []
    for sd in seeds:
        torch.manual_seed(sd)
        m = M(fine=["r", "hl", "gk", "mkt", "disp", "brd"], cal=True, **SIZES["t"])
        for n_, p in m.named_parameters():
            torch.nn.init.normal_(p, mean=1.0, std=0.1) if n_.endswith(("n1.weight", "n2.weight", "nf.weight")) else torch.nn.init.normal_(p, std=0.05 if p.dim() > 1 else 0.3)
        d = str(tmp_path / f"w{sd}"); export2.export(m, d)
        torch.save(dict(sd=m.state_dict(), cfg={k: v for k, v in m.cfg.items() if k != "horizons"}), d + ".pt")
        out.append(os.path.join(d, "bmt2.bin"))
    return out


need_bin = pytest.mark.skipif(not os.path.exists(BMT2), reason="build/bmt2 not built")


@need_bin
def test_run_end_to_end_against_mock_venue(tmp_path, monkeypatch):
    """refresh -> inputs -> Mojo engine -> file; catch-up of missed days, idempotent re-run, gates leave the file untouched, model pin."""
    np = pytest.importorskip("numpy")
    W = rand_weights(tmp_path)
    day = (T_END + 1) * H                                            # T_END + 1 is 00:00 UTC
    assert day % 86400 == 0
    v = Venue(SY, T_END + 48, gaps=("C05USDT",)); srv, base = v.serve(); cache = str(tmp_path / "k"); out = str(tmp_path / "fcast24.tsv")   # C05: no bar at the third origin
    a = A(weights=W, cache=cache, out=out, pool=SY[4:], coins=SY[:6], require=SY[:4])
    span = F.L + F.PAD + 1 + 48
    F.refresh("kucoin", SY, cache, T_END + 48, span, F.Http(rps=400), base, now=day + 2 * 86400 + 60)
    assert F.cmd_write(a, now=day + 60) == 0                         # first day
    hd, rows = F.read_file(out)
    assert hd["model"] == "bmt2-r24-test~" + F.model_hash(W) and hd["seeds"] == "2" and sorted(rows) == [(day, F.pair_of(x)) for x in sorted(SY[:6])]
    first = open(out).read()
    assert F.cmd_write(a, now=day + 3600) == 0 and open(out).read() == first                       # same day again: nothing changes
    a.also = [str(tmp_path / "copy" ) + ".tsv"]
    assert F.cmd_write(a, now=day + 2 * 86400 + 60) == 0             # two days later: both missing days are written, the old rows are kept
    hd, rows2 = F.read_file(out)
    assert sorted({t for t, _ in rows2}) == [day, day + 86400, day + 2 * 86400] and all(rows2[k] == rows[k] for k in rows)
    assert open(a.also[0]).read() == open(out).read()                # the second reader's copy
    assert (day + 86400, "C05_USDT") in rows2 and (day + 2 * 86400, "C05_USDT") not in rows2 and len(rows2) == 17
    data = F.load_cache(cache, SY)                                    # the rows are the reference engine's (PyTorch, same inputs), seed mean in float32
    r, cal, _ = F.build_rows(data, a.pool, a.coins, (day + 86400) // H - 1)
    xs = sorted(r); per = [F.torch_medians(w.replace("/bmt2.bin", ".pt"), [r[x] for x in xs], [cal] * len(xs))[0] for w in W]
    want = np.mean(np.array(per, np.float32), 0)
    got = np.array([rows2[(day + 86400, F.pair_of(x))] for x in xs])
    sc = np.array([math.sqrt(sum(z * z for z in r[x]["r"]) / F.L * 24) for x in xs])
    assert (np.abs(got - want) / sc).max() < 2e-5 and np.abs(want).max() > 0
    keep = open(out).read()
    assert F.cmd_write(A(weights=W, cache=cache, out=out, pool=SY[4:], coins=SY[:6], require=SY[:4], min_pool=999), [day + 2 * 86400]) == 3 and open(out).read() == keep      # thin pool: untouched
    assert F.cmd_write(A(weights=W, cache=cache, out=out, pool=SY[4:], coins=SY[:6], require=SY[:4] + ["ZZZUSDT"]), [day + 2 * 86400]) == 3 and open(out).read() == keep    # a required coin missing
    assert F.cmd_write(A(weights=W[::-1], cache=cache, out=out, pool=SY[4:], coins=SY[:6], require=SY[:4]), [day + 2 * 86400]) == 2 and open(out).read() == keep           # other weights: refuse
    srv.shutdown()


@need_bin
def test_run_retries_until_the_origin_bar_is_served(tmp_path, monkeypatch):
    """a run started seconds after 00:00: the venue serves the 23:00 bar late; --retry-s re-asks instead of leaving the day frozen for an hour."""
    pytest.importorskip("numpy")
    W = rand_weights(tmp_path)
    day = (T_END + 1) * H
    v = Venue(SY, T_END - 1); srv, base = v.serve(); out = str(tmp_path / "f.tsv")       # the venue is one bar behind
    threading.Timer(4.0, lambda: setattr(v, "t_end", T_END)).start()                       # ... and catches up 4 s later
    monkeypatch.setattr(F.time, "time", lambda t=time.time, off=day + 3 - time.time(): t() + off)          # wall clock = 00:00:03 of the day
    common = ["--venue", "kucoin", "--base-url", base, "--rps", "400", "--cache", str(tmp_path / "k"), "--weights", ",".join(W), "--bmt2", BMT2, "--out", out,
              "--universe", str(tmp_path / "u.json"), "--coins", ",".join(x[:-4] for x in SY[:6]), "--require", ",".join(x[:-4] for x in SY[:4]), "--min-pool", "10", "--name", "t"]
    json.dump(dict(train=SY[4:]), open(tmp_path / "u.json", "w"))
    assert F.main(["run"] + common) == 3 and not os.path.exists(out)                       # control: without the retry the day is not written
    assert F.main(["run", "--retry-s", "30"] + common) == 0
    hd, rows = F.read_file(out)
    assert sorted({t for t, _ in rows}) == [day] and len(rows) == 6
    n = v.calls
    assert F.main(["run"] + common) == 0 and v.calls == n                                  # up to date: no request at all
    srv.shutdown()


# ---------------------------------------------------------------- real rows: inputs vs train/bank.py, Mojo vs PyTorch, vs the research predictions
def have_research():
    return os.path.exists(os.path.join(CACHE, "hourly", "BTCUSDT.npz")) and os.path.exists(os.path.join(CKPT, "mkt+r24_s0_F4.pt")) and os.path.exists(BOTPRED)


def day_data(np, Z, syms, t):
    out = {}
    for x in syms:
        z = Z.get(x)
        if z is None:
            continue
        a, b = np.searchsorted(z["t"], t - F.L - F.PAD), np.searchsorted(z["t"], t, side="right")
        out[x] = {int(k): (float(o), float(h), float(l), float(c), 0.0) for k, o, h, l, c in zip(z["t"][a:b], z["o"][a:b], z["h"][a:b], z["l"][a:b], z["c"][a:b])}
    return out


def parity(n_days=14, tmp=None, verbose=True):
    """-> dict of parity numbers on (day, coin) rows of 2026 (fold F4): every coin of the research forecast file on n_days days."""
    import numpy as np, torch, tempfile
    from bmtbench import data as D
    from train.bank import Bank, FINE
    from train.model2 import BMT2 as M
    torch.set_num_threads(4)
    tmp = tmp or tempfile.mkdtemp(prefix="fc24_")
    u = D.universe()
    bank = Bank(u["pool"], u["train"], "cpu")
    Z = {x: np.load(os.path.join(CACHE, "hourly", x + ".npz")) for x in bank.syms}
    Z = {x: {k: z[k] for k in ("t", "o", "h", "l", "c")} for x, z in Z.items()}
    bp = np.load(BOTPRED)
    days = [int(d) for d in np.linspace(D.eh(2026, 1, 12) // 24, D.eh(2026, 10, 30) // 24, n_days).astype(int) * 86400]
    last = {x: int(bank.last[bank.ix[x]] + bank.t0) for x in bank.syms}
    res = dict(rows=0, days=len(days))
    rows, cals, keys, ref_fine, onl = [], [], [], [], []
    for d in days:
        t = d // H - 1
        coins = sorted(set(str(s) for s in bp["sym"][bp["t_abs"] == t]))
        data = day_data(np, Z, bank.syms, t)
        r, cal, info = F.build_rows(data, u["train"], coins, t, live_end=last)          # the bank's liveness (it knows each symbol's last archived bar)
        r2, _, info2 = F.build_rows(data, u["train"], coins, t)                         # the online rule (48 h forward fill)
        assert sorted(r) == coins, (d, set(coins) ^ set(r), info["skipped"])
        g = bank.gather(torch.tensor([bank.ix[x] for x in coins]), torch.tensor([t - bank.t0] * len(coins)), 1)
        assert g["cal"][0].tolist() == cal
        for i, x in enumerate(coins):
            rows.append(r[x]); cals.append(cal); keys.append((t, x)); ref_fine.append(g["fine"][i].numpy()); onl.append(r2[x])
    N = len(rows); res["rows"] = N
    ref = np.stack(ref_fine)
    for ch in F.BUILT:
        a = np.array([r[ch] for r in rows], np.float32); b = ref[:, FINE.index(ch)]
        res[f"in_{ch}_maxabs"] = float(np.abs(a - b).max()); res[f"in_{ch}_equal"] = float((a == b).mean())
        res[f"online_{ch}_maxabs"] = float(np.abs(np.array([r[ch] for r in onl], np.float32) - b).max())
    sc = np.sqrt((ref[:, 0].astype(np.float64) ** 2).mean(1)) * math.sqrt(24)
    med = {}
    for sd in (0, 1):
        ck = os.path.join(CKPT, f"mkt+r24_s{sd}_F4.pt"); wb = os.path.join(tmp, f"s{sd}.bin")
        F.main(["export", ck, wb])
        c = torch.load(ck, map_location="cpu", weights_only=False)
        m = M(horizons=F.HZ_, **c["cfg"]); m.load_state_dict(c["sd"]); m.eval()
        with torch.no_grad():
            bank_q = m.quantiles(dict(fine=torch.tensor(ref), cal=torch.tensor(np.array(cals))))[:, 3, 10].numpy()      # research inputs, PyTorch CPU
        mine_t = np.array(F.torch_medians(ck, rows, cals)[0], np.float32)                                               # writer inputs, PyTorch CPU
        mine_m = np.array(F.mojo_medians(BMT2, wb, rows, cals, 2), np.float32)                                          # writer inputs, Mojo
        onl_m = np.array(F.mojo_medians(BMT2, wb, onl, cals, 2), np.float32)
        gpu = np.array([bp[f"mkt+r24_s{sd}"][(bp["t_abs"] == t) & (bp["sym"] == x)][0] for t, x in keys])                # research file (GPU, TF32)
        res[f"s{sd}_mojo_vs_torch_scale"] = float((np.abs(mine_m - mine_t) / sc).max())
        res[f"s{sd}_mojo_vs_torch_abs"] = float(np.abs(mine_m - mine_t).max())
        res[f"s{sd}_writer_vs_bank_inputs_scale"] = float((np.abs(mine_t - bank_q) / sc).max())
        res[f"s{sd}_mojo_vs_research_file_scale"] = float((np.abs(mine_m - gpu) / sc).max())
        res[f"s{sd}_online_rule_scale"] = float((np.abs(onl_m - mine_m) / sc).max())
        med[sd] = (mine_m, gpu, onl_m)
    ens = (med[0][0] + med[1][0]) / 2; ens_gpu = (med[0][1] + med[1][1]) / 2; ens_onl = (med[0][2] + med[1][2]) / 2
    rk = lambda x: np.argsort(np.argsort(x))
    M10 = [c + "USDT" for c in F.M10]
    flips = flips_onl = cells = 0
    rho = []
    for d in days:
        t = d // H - 1
        i = np.array([j for j, (tt, x) in enumerate(keys) if tt == t and x in M10])
        flips += int((rk(ens[i]) != rk(ens_gpu[i])).sum()); flips_onl += int((rk(ens_onl[i]) != rk(ens[i])).sum()); cells += len(i)
        j = np.array([q for q, (tt, _) in enumerate(keys) if tt == t])
        rho.append(float(np.corrcoef(rk(ens[j]), rk(ens_gpu[j]))[0, 1]))
    res.update(m10_cells=cells, m10_rank_changes_vs_research_file=flips, m10_rank_changes_online_rule=flips_onl, min_day_rank_corr_vs_research_file=min(rho),
               ens_maxabs_vs_research_file=float(np.abs(ens - ens_gpu).max()), median_abs_forecast=float(np.median(np.abs(ens_gpu))))
    if verbose:
        for k, v_ in res.items():
            print(f"{k:42s} {v_:.3e}" if isinstance(v_, float) else f"{k:42s} {v_}")
    return res


@need_bin
@pytest.mark.skipif(not have_research(), reason="needs the research cache, rank24 checkpoints and botpred")
def test_parity_on_real_rows(tmp_path):
    r = parity(8, str(tmp_path), verbose=False)
    assert r["rows"] >= 200
    for ch in ("r", "hl", "gk", "brd"):
        assert r[f"in_{ch}_maxabs"] == 0.0, ch                        # the bank's inputs, bit for bit
    for ch in ("mkt", "disp"):                                        # exact sums here, pairwise sums in numpy: a last float32 bit at most
        assert r[f"in_{ch}_maxabs"] < 1e-9 and r[f"in_{ch}_equal"] > 0.999, ch
    for sd in (0, 1):
        assert r[f"s{sd}_mojo_vs_torch_scale"] < 2e-5                 # Mojo engine == PyTorch on the same rows
        assert r[f"s{sd}_writer_vs_bank_inputs_scale"] < 2e-5         # the writer's inputs give the research model's answer
        assert r[f"s{sd}_mojo_vs_research_file_scale"] < 5e-3         # the research file was written on a GPU with TF32 matmuls
    assert r["min_day_rank_corr_vs_research_file"] > 0.99


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "parity":
        parity(int(sys.argv[2]) if len(sys.argv) > 2 else 14)
