#!/usr/bin/env python3
"""Daily 24 h forecast file (`bmt-fcast24 v1`) for the trend producers' --fcast-tilt flag. Standard library only on the production path.

  fcast24.py run     --venue kucoin --universe data/universe.json --cache DIR --weights s0.bin,s1.bin --bmt2 build/bmt2 --out FILE [--also F2]
                     refresh the pool's hourly klines, then write every missing day up to the latest 00:00 UTC (atomic replace).
                     Safe to schedule hourly: once the day's row exists it exits without any network call.
  fcast24.py refresh --venue V --universe U --cache DIR [--origin TS] [--days N]      klines only (resumable, <= 4 concurrent, backs off on 429)
  fcast24.py write   --cache DIR/venue ... --out FILE [--days TS,TS]                  no network
  fcast24.py seed    --botpred NPZ --weights ... --out FILE                           history from research predictions (needs numpy)
  fcast24.py check   FILE                                                             strict validation (the producer's reader rules)
  fcast24.py export  CKPT.pt OUT.bin                                                  weights for the Mojo engine (needs torch)

File: `# bmt-fcast24 v1 model=<name>~<hash8> seeds=<n> horizon_h=24 created=<epoch s>` then `<decision ts, 00:00 UTC>\\t<PAIR>\\t<median %.8e>`,
sorted by (time, pair). A row's inputs are the 512 hourly bars closing at or before its decision ts. hash8 = sha256 over the exported weight
files in seed order, so any weight set can be dropped in; the file then starts a new history (seed) under the new hash.

Inputs (train/bank.py, same arithmetic): per coin r, hl (float16 like the bank), gk; over the model's market pool mkt (mean return of the
symbols alive at the bar), disp (cross-sectional std), brd (share up - 0.5); calendar of the next bar. Online liveness: a pool symbol counts
up to the origin while its last real bar is at most --ffill-h hours old (gaps forward-filled, like the bank inside a symbol's life).
This never runs in the trading path: the producer only reads the file, and freezes its multipliers when the file is late."""
import argparse, array, concurrent.futures as cf, hashlib, json, math, os, random, struct, subprocess, sys, threading, time
import urllib.error, urllib.parse, urllib.request

H, L, PAD = 3600, 512, 48
FINE = ("r", "hl", "gk", "v", "tb", "rv", "btc", "eth", "mkt", "disp", "brd", "rv5", "rsn", "peer", "bvol")      # train/bank.py
BUILT = ("r", "hl", "gk", "mkt", "disp", "brd")
MAGIC = 0x32544D42
HEAD = "# bmt-fcast24 v1 "
UA = "Mozilla/5.0 (compatible; kline-refresh/1.0)"
M10 = "BTC ETH BNB XRP LTC ADA TRX LINK XLM ETC".split()
COINS = sorted(set(M10 + ("ALGO ATOM AVAX BAND BAT BCH CHZ DASH DOGE DOT ENJ HBAR ICX IOST IOTA KAVA NEO ONT RVN SOL TFUEL VET XTZ ZEC ZIL ZRX "
                          "AAVE COMP MANA ONE QTUM SAND SNX UNI ZEN").split()))


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- venues (public REST, no key)
class Http:
    """<= conc requests in flight, paced to rps; 429 / 5xx / venue rate codes push every worker back (Retry-After or exponential)."""

    def __init__(s, conc=4, rps=8.0, timeout=20, tries=7):
        s.conc = max(1, min(4, conc)); s.sem = threading.Semaphore(s.conc); s.lock = threading.Lock()
        s.gap, s.timeout, s.tries, s.next = 1.0 / rps, timeout, tries, 0.0
        s.calls = s.limited = s.errors = 0; s.live = s.peak = 0

    def _slot(s):
        with s.lock:
            now = time.monotonic(); t = max(now, s.next); s.next = t + s.gap
        if t > now:
            time.sleep(t - now)

    def back_off(s, i, hint=None):
        w = min(120.0, float(hint)) if hint else min(60.0, 2.0 ** (i + 1)) * (0.75 + 0.5 * random.random())
        with s.lock:
            s.next = max(s.next, time.monotonic() + w)

    def get(s, url):
        err = None
        for i in range(s.tries):
            s._slot()
            with s.sem:
                with s.lock:
                    s.calls += 1; s.live += 1; s.peak = max(s.peak, s.live)
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"}), timeout=s.timeout) as r:
                        body = r.read()
                    j = json.loads(body)
                    if isinstance(j, dict) and str(j.get("code", "")) in ("429000", "-1003"):
                        raise urllib.error.HTTPError(url, 429, "venue rate code", None, None)
                    return j
                except urllib.error.HTTPError as e:
                    err = f"HTTP {e.code}"
                    if e.code in (429, 418) or e.code >= 500:
                        with s.lock:
                            s.limited += e.code in (429, 418)
                        s.back_off(i, e.headers.get("Retry-After") if e.headers else None)
                        continue
                    raise RuntimeError(f"{err} {url}")
                except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                    err = type(e).__name__
                    with s.lock:
                        s.errors += 1
                    s.back_off(min(i, 2))
                finally:
                    with s.lock:
                        s.live -= 1
        raise RuntimeError(f"{err} after {s.tries} tries {url}")


class KuCoin:
    name, base, per = "kucoin", "https://api.kucoin.com", 1500

    def sym(s, x):
        return x[:-4] + "-USDT"

    def listed(s, http, base):
        j = http.get(f"{base}/api/v2/symbols")
        return {d["symbol"].replace("-", "") for d in j["data"] if d.get("quoteCurrency") == "USDT" and d.get("enableTrading", True)}

    def klines(s, http, base, x, h0, h1):
        """closed hourly bars with h0 <= open hour <= h1 -> [(hour, o, h, l, c, quote volume)]; hours without trades are absent."""
        j = http.get(f"{base}/api/v1/market/candles?type=1hour&symbol={s.sym(x)}&startAt={h0 * H}&endAt={(h1 + 1) * H}")
        if str(j.get("code")) != "200000":
            raise RuntimeError(f"kucoin {x} code {j.get('code')}")
        return [(int(r[0]) // H, float(r[1]), float(r[3]), float(r[4]), float(r[2]), float(r[6])) for r in j.get("data") or []]


class Binance:
    name, base, per = "binance", "https://api.binance.com", 1000

    def sym(s, x):
        return x

    def listed(s, http, base):
        j = http.get(f"{base}/api/v3/exchangeInfo")
        return {d["symbol"] for d in j["symbols"] if d.get("quoteAsset") == "USDT" and d.get("status") == "TRADING"}

    def klines(s, http, base, x, h0, h1):
        j = http.get(f"{base}/api/v3/klines?symbol={x}&interval=1h&startTime={h0 * H * 1000}&endTime={(h1 + 1) * H * 1000 - 1}&limit=1000")
        return [(int(r[0]) // (H * 1000), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])) for r in j]


VENUES = {"kucoin": KuCoin, "binance": Binance}


# ---------------------------------------------------------------- kline cache: one small file per symbol, atomic, resumable
def cpath(cache, x):
    return os.path.join(cache, x + ".csv")


KEEP_H = 2400


def cache_read(cache, x):
    """-> (from, to, {hour: (o, h, l, c, qv)}): the file holds every real bar the venue had in [from, to] (hours without trades are absent)."""
    try:
        Ls = open(cpath(cache, x)).read().split("\n")
    except OSError:
        return 0, -1, {}
    frm, to, rows = 0, -1, {}
    for l in Ls:
        if l.startswith("#range="):
            frm, to = (int(v) for v in l[7:].split(","))
        elif l:
            p = l.split(",")
            if len(p) == 6:
                rows[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]))
    return frm, to, rows


def atomic_write(path, text):
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def cache_write(cache, x, frm, to, rows):
    frm = max(frm, to - KEEP_H + 1)
    atomic_write(cpath(cache, x), f"#range={frm},{to}\n" + "".join("%d,%r,%r,%r,%r,%r\n" % ((k,) + rows[k]) for k in sorted(rows) if frm <= k <= to))


def refresh(venue, syms, cache, origin_h, span, http, base=None, now=None):
    """bring every listed symbol's bars up to origin_h (>= span bars back). Symbols already checked for origin_h are skipped, so a rerun
    after an interruption only fetches the rest. -> stats dict."""
    v = VENUES[venue](); base = (base or os.environ.get("BMT_FCAST_BASE") or v.base).rstrip("/")
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    listed = v.listed(http, base)
    st = dict(venue=venue, origin=origin_h * H, symbols=len(syms), listed=0, fetched=0, skipped=0, failed=0, bars=0)
    todo, lo = [], origin_h - span + 1
    for x in syms:
        if x not in listed:
            continue
        st["listed"] += 1
        frm, to, _ = cache_read(cache, x)
        if to >= origin_h and frm <= lo:
            st["skipped"] += 1
        else:
            todo.append(x)
    lock = threading.Lock()

    def one(x):
        frm, to, rows = cache_read(cache, x)
        if not (frm <= lo and to >= lo - 1):                         # no usable overlap: start over for this symbol
            frm, to, rows = lo, lo - 1, {}
        h = to + 1
        while h <= origin_h:
            e = min(origin_h, h + v.per - 2)
            for k, o, hi, lw, c, qv in v.klines(http, base, x, h, e):
                if h <= k <= e and c > 0 and hi > 0 and lw > 0 and (k + 1) * H <= (now or time.time()):
                    rows[k] = (o, hi, lw, c, qv)
            h = e + 1
        cache_write(cache, x, frm, origin_h, rows)
        with lock:
            st["fetched"] += 1; st["bars"] += len(rows)

    with cf.ThreadPoolExecutor(http.conc) as ex:
        fs = {ex.submit(one, x): x for x in todo}
        for f in cf.as_completed(fs):
            if f.exception() is not None:
                st["failed"] += 1
                log(f"WARN FCAST24_REFRESH symbol={fs[f]} error={str(f.exception())[:120].replace(' ', '_')}")
    st.update(http=http.calls, http429=http.limited, neterr=http.errors, peak_conc=http.peak, elapsed_s=round(time.time() - t0, 1))
    return st


# ---------------------------------------------------------------- inputs (train/bank.py arithmetic on the fetched window)
class Pool:
    """market series over the pool for origin hour t: mkt, disp, brd on the L bars ending at t."""

    def __init__(s, data, pool, t, ffill_h=48, live_end=None):
        s.t, s.n = t, L + PAD + 1
        s.h0 = t - s.n + 1
        s.lp = {}
        for x in pool:
            lp = s.series(data.get(x), ffill_h, (live_end or {}).get(x))
            if lp is not None:
                s.lp[x] = lp
        n = s.n
        s.mkt, s.disp, s.brd, s.alive = [0.0] * L, [0.0] * L, [0.0] * L, [0] * L
        cols = list(s.lp.values())
        for q in range(L):
            i = n - L + q
            cnt, sm, rk = 0, [], []
            for lp in cols:
                a = lp[i]
                if a is None:
                    continue
                cnt += 1
                b = lp[i - 1]
                if b is not None:
                    sm.append(a - b); rk.append(a - b)
            s.alive[q] = cnt
            s.mkt[q] = math.fsum(sm) / max(cnt, 1)
            if rk:
                mu = math.fsum(rk) / len(rk)
                s.disp[q] = math.sqrt(math.fsum((v - mu) * (v - mu) for v in rk) / len(rk))
                s.brd[q] = sum(v > 0 for v in rk) / len(rk) - 0.5

    def series(s, rows, ffill_h, end=None):
        """forward-filled log close on [h0, t] inside the symbol's life (None outside), or None when it has no bar in the window."""
        if not rows:
            return None
        ks = [k for k in rows if s.h0 <= k <= s.t]
        if not ks:
            return None
        first, last = min(ks) - s.h0, max(ks) - s.h0
        stop = s.n - 1 if s.n - 1 - last <= ffill_h else last
        if end is not None:
            stop = min(s.n - 1, end - s.h0)
        lp, cur = [None] * s.n, None
        for i in range(first, stop + 1):
            r = rows.get(s.h0 + i)
            if r is not None:
                cur = math.log(r[3])
            lp[i] = cur
        return lp


def coin_inputs(rows, t):
    """-> (r, hl, gk) float32-exact lists over the L bars ending at t, or (None, reason). Needs a real bar at t and a close at or before t - L."""
    if not rows or t not in rows:
        return None, "no_bar_at_origin"
    h0 = t - L - PAD
    lp, cur, seen = [], None, []
    for k in range(h0, t + 1):
        r = rows.get(k)
        if r is not None:
            cur = math.log(r[3])
        lp.append(cur); seen.append(r)
    if lp[PAD] is None:
        return None, "short_history"
    r = array.array("f", [lp[i] - lp[i - 1] for i in range(PAD + 1, PAD + L + 1)])
    hl = []
    for i in range(PAD + 1, PAD + L + 1):
        b = seen[i]
        # float64 -> float32 -> float16, the two roundings torch.tensor(float64, dtype=float16) does in the bank
        hl.append(f16(f32((lp[i] + f32(math.log(b[1]) - math.log(b[3]))) - (lp[i] - f32(math.log(b[3]) - math.log(b[2]))))) if b is not None else 0.0)
    c = f32(0.3862944)
    gk = array.array("f", [math.sqrt(max(f32(f32(0.5 * a * a) - f32(f32(c * x) * x)), 0.0)) for a, x in zip(hl, r)])
    return (list(r), hl, list(gk)), ""


def cal_of(t):
    return [(t + 1) % 24, ((t + 1) // 24 + 4) % 7, 0]


def build_rows(data, pool, coins, t, ffill_h=48, live_end=None):
    """-> (rows {coin: {channel: [L floats]}}, cal, info). One Pool per origin."""
    P = Pool(data, pool, t, ffill_h, live_end)
    chan = dict(mkt=list(array.array("f", P.mkt)), disp=list(array.array("f", P.disp)), brd=list(array.array("f", P.brd)))
    out, skip = {}, {}
    for x in coins:
        v, why = coin_inputs(data.get(x), t)
        if v is None:
            skip[x] = why
            continue
        out[x] = dict(r=v[0], hl=v[1], gk=v[2], **chan)
    return out, cal_of(t), dict(pool_live=P.alive[-1], pool_min=min(P.alive), skipped=skip)


# ---------------------------------------------------------------- model: exported weights (train/export2.py) through the Mojo kernel
def bin_header(path):
    h = struct.unpack("<64i", open(path, "rb").read(256))
    if h[0] != MAGIC:
        raise RuntimeError(f"{path}: not a bmt2.bin")
    nf = h[10]
    chans = [FINE[i] for i in h[44:44 + nf]]
    if h[1] != L or h[11] or h[12] or h[13] or not h[9]:
        raise RuntimeError(f"{path}: needs ctx {L}, calendar, no coarse / scalar inputs")
    bad = [c for c in chans if c not in BUILT]
    if bad:
        raise RuntimeError(f"{path}: inputs {bad} are not built by this writer")
    hz = list(h[14:14 + h[8]])
    if 24 not in hz:
        raise RuntimeError(f"{path}: no 24 h horizon")
    return dict(chans=chans, hz=hz, H=h[8])


def model_hash(weights):
    h = hashlib.sha256()
    for w in weights:
        h.update(hashlib.sha256(open(w, "rb").read()).hexdigest().encode())
    return h.hexdigest()[:8]


def block_of(rows, cals, chans):
    """serve input block: fine[N, n_fine, L] then cal[N, 3], float32."""
    a = array.array("f")
    for r in rows:
        for c in chans:
            a.extend(r[c])
    for c in cals:
        a.extend([float(v) for v in c])
    return a


def mojo_medians(bmt2, weight, rows, cals, threads=1):
    """24 h median (log return) per row from `bmt2 serve`."""
    hd = bin_header(weight)
    blk = block_of(rows, cals, hd["chans"])
    n = len(rows)
    p = subprocess.run([bmt2, "serve", weight, str(threads)], input=struct.pack("<i", n) + blk.tobytes() + struct.pack("<i", 0), capture_output=True, timeout=600)
    if p.returncode != 0 or len(p.stdout) != 4 + n * hd["H"] * 21 * 4 or struct.unpack("<i", p.stdout[:4])[0] != n:
        raise RuntimeError(f"bmt2 serve failed rc={p.returncode} out={len(p.stdout)} {p.stderr[-200:]!r}")
    q = array.array("f"); q.frombytes(p.stdout[4:])
    hi = hd["hz"].index(24)
    return [q[i * hd["H"] * 21 + hi * 21 + 10] for i in range(n)]


def torch_medians(ckpt, rows, cals):
    """reference engine (PyTorch, CPU): the same rows through train/model2.py. ckpt: a rank24 checkpoint (.pt)."""
    import numpy as np, torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train.model2 import BMT2
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = BMT2(horizons=HZ_, **c["cfg"]); m.load_state_dict(c["sd"]); m.eval()
    fine = np.zeros((len(rows), len(FINE), L), np.float32)
    for i, r in enumerate(rows):
        for k, v in r.items():
            fine[i, FINE.index(k)] = v
    with torch.no_grad():
        q = m.quantiles(dict(fine=torch.tensor(fine), cal=torch.tensor(np.array(cals, np.int64))))
    return q[:, HZ_.index(24), 10].numpy().tolist(), q.numpy()


HZ_ = (1, 5, 6, 24)


def medians(a, rows, cals):
    """seed mean in float32 (np.mean of float32 arrays, as tools/r24bots.py:fmat)."""
    per = [mojo_medians(a.bmt2, w, rows, cals, a.threads) for w in a.weights]
    out = []
    for i in range(len(rows)):
        s = 0.0
        for p in per:
            s = f32(s + p[i])
        out.append(f32(s / len(per)))
    return out


# ---------------------------------------------------------------- the file
def read_file(path):
    """strict (the producer's rules) -> (header dict, {(ts, pair): value})."""
    b = open(path, "rb").read()
    if not b or not b.endswith(b"\n"):
        raise RuntimeError("empty or torn file")
    Ls = b.decode().split("\n")
    if not Ls[0].startswith(HEAD):
        raise RuntimeError("not a bmt-fcast24 v1 header")
    hd = dict(t.split("=", 1) for t in Ls[0][len(HEAD):].split() if "=" in t)
    if "~" not in hd.get("model", "") or hd.get("horizon_h") != "24":
        raise RuntimeError("header needs model=<name>~<hash> horizon_h=24")
    rows, prev = {}, -1
    for i, l in enumerate(Ls[1:-1], 2):
        if not l or l.startswith("#"):
            continue
        p = l.split("\t")
        if len(p) != 3:
            raise RuntimeError(f"line {i}: need 3 columns")
        t, v = int(p[0]), float(p[2])
        if t % 86400 or t < prev or not math.isfinite(v) or (t, p[1]) in rows:
            raise RuntimeError(f"line {i}: bad row")
        prev = t; rows[(t, p[1])] = v
    return hd, rows


def write_file(path, model, seeds, rows, created=None):
    body = "".join(f"{t}\t{p}\t{v:.8e}\n" for (t, p), v in sorted(rows.items()))
    atomic_write(path, f"{HEAD}model={model} seeds={seeds} horizon_h=24 created={int(time.time()) if created is None else created}\n" + body)


def pair_of(x):
    return x[:-4] + "_USDT"


def load_cache(cache, syms):
    return {x: cache_read(cache, x)[2] for x in syms}


def forecast_days(a, data, days):
    """-> ({(ts, pair): median}, per-day info) for decision days (00:00 UTC epoch s)."""
    rows, cals, keys, info = [], [], [], {}
    for d in days:
        t = d // H - 1
        r, cal, inf = build_rows(data, a.pool, a.coins, t, a.ffill_h)
        info[d] = inf; inf["coins"] = len(r)
        for x in sorted(r):
            rows.append(r[x]); cals.append(cal); keys.append((d, pair_of(x)))
    med = medians(a, rows, cals) if rows else []
    return dict(zip(keys, med)), info


def gate(a, d, vals, inf):
    """the day's rows are written only when the inputs look like the model's: enough of the pool alive, the required coins present, sane values."""
    need = [pair_of(x) for x in a.require]
    miss = [p for p in need if (d, p) not in vals]
    if inf["pool_min"] < a.min_pool:
        return f"pool_alive={inf['pool_min']}<{a.min_pool}"
    if miss:
        return "missing=" + ",".join(miss)
    if any(not math.isfinite(v) or abs(v) > 1 for (t, _), v in vals.items() if t == d):
        return "bad_value"
    return ""


def cmd_write(a, days=None, now=None):
    t0 = time.time()
    model = f"{a.name}~{model_hash(a.weights)}"
    hd, rows = read_file(a.out) if os.path.exists(a.out) else ({"model": model}, {})
    if hd["model"] != model:
        log(f"FCAST24 status=failed reason=model_mismatch file={hd['model']} weights={model} action=file_untouched (seed a new history for new weights)")
        return 2
    now = now or int(time.time())
    last = max((t for t, _ in rows), default=None)
    today = now - now % 86400
    if days is None:
        first = today if last is None else last + 86400
        days = list(range(max(first, today - 86400 * (a.max_catchup - 1)), today + 1, 86400))
    if not days:
        log(f"FCAST24 status=ok day={last} rows=0 note=up_to_date model={model.rsplit('~', 1)[1]}")
        return 0
    data = load_cache(a.cache, sorted(set(a.pool) | set(a.coins)))
    vals, info = forecast_days(a, data, days)
    wrote, bad = [], []
    for d in days:
        why = gate(a, d, vals, info[d])
        if why:
            bad.append(f"{d}:{why}")
            continue
        for k in [k for k in rows if k[0] == d]:
            del rows[k]
        rows.update({k: v for k, v in vals.items() if k[0] == d}); wrote.append(d)
    if wrote:
        write_file(a.out, model, len(a.weights), rows)
        copy_also(a)
    ok = bool(wrote) and wrote[-1] == days[-1]
    log(f"FCAST24 status={'ok' if ok else 'failed'} day={days[-1]} days_written={len(wrote)} rows={sum(info[d]['coins'] for d in wrote)} "
        f"pool_alive={info[days[-1]]['pool_live']} skipped={','.join(sorted(info[days[-1]]['skipped'])) or '-'} model={model.rsplit('~', 1)[1]} engine=mojo "
        f"elapsed_s={time.time() - t0:.1f}" + (f" gated={';'.join(bad)}" if bad else ""))
    return 0 if ok else 3


def copy_also(a):
    """atomic copies of the file for readers in other state directories (each producer reads its own path)."""
    for dst in getattr(a, "also", None) or []:
        try:
            atomic_write(dst, open(a.out).read())
        except OSError as e:
            log(f"WARN FCAST24 copy_failed file={dst} error={type(e).__name__}")


def cmd_refresh(a, now=None):
    now = now or int(time.time())
    origin = (a.origin or (now - now % 86400)) // H - 1
    span = L + PAD + 1 + 24 * max(0, a.days - 1)
    st = refresh(a.venue, sorted(set(a.pool) | set(a.coins)), a.cache, origin, span, Http(a.conc, a.rps), a.base_url, now)
    log("FCAST24_REFRESH " + " ".join(f"{k}={v}" for k, v in st.items()))
    return st


def cmd_run(a):
    now = int(time.time())
    last = None
    if os.path.exists(a.out):
        last = max((t for t, _ in read_file(a.out)[1]), default=None)
    today = now - now % 86400
    if last is not None and last >= today:                           # hourly re-runs after the day's row exists: no network at all
        copy_also(a)
        log(f"FCAST24 status=ok day={last} days_written=0 note=up_to_date")
        return 0
    a.days = 1 if last is None else max(1, min(a.max_catchup, (today - last) // 86400))
    t_end = time.time() + a.retry_s
    while True:
        st = cmd_refresh(a, now)
        if st["listed"] == 0:
            log("FCAST24 status=failed reason=no_symbols_listed action=file_untouched")
            return 3
        rc = cmd_write(a, now=now)
        if rc != 3 or time.time() + 5 > t_end:
            return rc
        # started right after the hour: the venue may not serve the origin bar yet. Ask again for every symbol without it.
        origin = today // H - 1
        for x in sorted(set(a.pool) | set(a.coins)):
            frm, to, rows = cache_read(a.cache, x)
            if to >= origin and origin not in rows and rows:
                cache_write(a.cache, x, frm, origin - 1, rows)
        time.sleep(5)


def cmd_seed(a):
    import numpy as np
    z = np.load(a.botpred)
    ks = [k for k in z.files if k == a.key or k.startswith(a.key + "_s")]
    v = np.mean([z[k] for k in ks], 0)
    ts = (z["t_abs"].astype(np.int64) + 1) * H
    rows = {(int(t), pair_of(str(x))): float(y) for t, x, y in zip(ts, z["sym"], v)}
    model = f"{a.name}~{model_hash(a.weights)}"
    write_file(a.out, model, len(a.weights), rows)
    log(f"FCAST24_SEED rows={len(rows)} first={min(ts)} last={max(ts)} model={model}")
    return 0


def cmd_export(a):
    import torch
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [root, os.path.join(root, "bench")]
    from train.model2 import BMT2
    from train import export2
    import tempfile, shutil
    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = BMT2(horizons=HZ_, **c["cfg"]); m.load_state_dict(c["sd"])
    with tempfile.TemporaryDirectory() as d:
        export2.export(m, d)
        os.makedirs(os.path.dirname(os.path.abspath(a.bin)), exist_ok=True)
        shutil.copyfile(os.path.join(d, "bmt2.bin"), a.bin + ".tmp"); os.replace(a.bin + ".tmp", a.bin)
        if a.selftest:
            os.makedirs(a.selftest, exist_ok=True)
            for f in ("bmt2.bin", "selftest_in.f32", "selftest_ref.f32"):
                shutil.copyfile(os.path.join(d, f), os.path.join(a.selftest, f))
    log(f"FCAST24_EXPORT {a.bin} sha256={hashlib.sha256(open(a.bin, 'rb').read()).hexdigest()[:16]} inputs={bin_header(a.bin)['chans']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, net=False, model=False):
        p.add_argument("--universe", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "universe.json"), help="json with the model's market pool under --pool-key")
        p.add_argument("--pool-key", default="train")
        p.add_argument("--coins", default=",".join(COINS), help="bases to forecast (rows for coins the venue lacks are skipped)")
        p.add_argument("--cache", required=True, help="kline cache directory of one venue")
        if net:
            p.add_argument("--venue", default="kucoin", choices=sorted(VENUES)); p.add_argument("--base-url", default=None)
            p.add_argument("--conc", type=int, default=4); p.add_argument("--rps", type=float, default=8.0)
            p.add_argument("--origin", type=int, default=0, help="decision ts (00:00 UTC) to refresh for; default the latest"); p.add_argument("--days", type=int, default=1)
        if model:
            p.add_argument("--weights", required=True, help="exported weights, one per seed, comma separated (fcast24.py export)")
            p.add_argument("--bmt2", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "bmt2"))
            p.add_argument("--name", default="bmt2-r24-mkt+r24"); p.add_argument("--out", required=True); p.add_argument("--threads", type=int, default=2)
            p.add_argument("--ffill-h", type=int, default=48); p.add_argument("--min-pool", type=int, default=100)
            p.add_argument("--require", default=",".join(M10), help="bases that must have a forecast for a day to be written")
            p.add_argument("--max-catchup", type=int, default=45)
            p.add_argument("--also", default="", help="extra paths that get an atomic copy of the file after each write")
            p.add_argument("--retry-s", type=float, default=0, help="run: keep retrying this long while the day's row fails its gates (origin bar not served yet)")

    common(sub.add_parser("refresh"), net=True)
    p = sub.add_parser("write"); common(p, model=True); p.add_argument("--day-list", default="", help="decision ts list (default: every missing day up to today)")
    common(sub.add_parser("run"), net=True, model=True)
    p = sub.add_parser("seed"); p.add_argument("--botpred", required=True); p.add_argument("--key", default="mkt+r24"); p.add_argument("--weights", required=True)
    p.add_argument("--name", default="bmt2-r24-mkt+r24"); p.add_argument("--out", required=True)
    p = sub.add_parser("check"); p.add_argument("file")
    p = sub.add_parser("export"); p.add_argument("ckpt"); p.add_argument("bin"); p.add_argument("--selftest", default="")
    a = ap.parse_args(argv)
    if hasattr(a, "weights"):
        a.weights = a.weights.split(",")
    if hasattr(a, "universe"):
        a.pool = json.load(open(a.universe))[a.pool_key]
        a.coins = [c + "USDT" for c in a.coins.split(",") if c]
    if hasattr(a, "require"):
        a.require = [c + "USDT" for c in a.require.split(",") if c]
        a.also = [x for x in a.also.split(",") if x]
    if a.cmd == "refresh":
        st = cmd_refresh(a)
        return 0 if st["failed"] == 0 else 3
    if a.cmd == "write":
        return cmd_write(a, [int(x) for x in a.day_list.split(",")] if a.day_list else None)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "seed":
        return cmd_seed(a)
    if a.cmd == "export":
        return cmd_export(a)
    hd, rows = read_file(a.file)
    ds = sorted({t for t, _ in rows})
    log(f"FCAST24_CHECK ok model={hd['model']} rows={len(rows)} days={len(ds)} first={ds[0] if ds else '-'} last={ds[-1] if ds else '-'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"FCAST24 status=failed reason={type(e).__name__} error={str(e)[:200].replace(' ', '_')} action=file_untouched")
        sys.exit(1)
