"""PREREG-LIC2: KuCoin public REST 1 h klines (USDT pairs) -> hourly cache in the bench format + universe.json. Private, never published.

  python tools/kucoin_cache.py fetch <dir>             resumable: <dir>/dl/<SYMBOL>.npz raw bars + <dir>/dl/state.json
  python tools/kucoin_cache.py build <dir>             <dir>/cache/{hourly,index_1h*.json,pairs.json}, <dir>/universe.json
  python tools/kucoin_cache.py okx <dir> BASE[,BASE]   OKX history-candles 1H for the named bases -> <dir>/okx/hourly/<BASE>USDT.npz
"""
import gzip, http.client, json, os, re, sys, threading, time, urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D

KC, OKX = "https://api.kucoin.com", "https://www.okx.com"
UA = {"User-Agent": "kline-research/1.0", "Accept-Encoding": "gzip"}
RATE, WORKERS, WIN = float(os.environ.get("KC_RATE", 5)), int(os.environ.get("KC_WORKERS", 4)), 1400 * 3600
STABLE = {"USDT", "USDC", "DAI", "PYUSD", "TUSD", "USDP", "EURT", "USDE", "RLUSD", "USDG", "USD1", "USDS", "FDUSD", "USDQ", "USDR", "GUSD",
          "BUSD", "UST", "PAXG", "XAUT"}
FIAT = {"EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "USD", "AED", "NZD", "SGD", "HKD", "PLN", "CZK", "TRY", "BRL", "MXN", "ZAR"}
BINX = set("USDC BUSD TUSD USDP PAX DAI FDUSD EUR GBP AUD UST USTC SUSD USDS USDSB AEUR EURI BFUSD USD1 XUSD PAXG WBTC BETH WBETH "
           "BTTC VEN BCC BCHABC BCHSV USDE BKRW IDRT BIDR TRY BRL RUB UAH NGN ZAR BVND DAI1 EURS LUSD FRAX PYUSD RLUSD".split())
LEV = re.compile(r"(\d+[LS]|UP|DOWN|BULL|BEAR)$")
CONF = "AAVE ALGO ATOM AVAX BAND BAT BTC COMP DOGE DOT ENJ ETH HBAR IOTA MANA NEO ONE QTUM RVN SAND SNX SOL TFUEL UNI VET XTZ ZEC ZEN ZIL ZRX".split()
ALGO = "ADA ALGO ATOM AVAX BAND BAT BCH BNB BTC CHZ DASH DOGE DOT ENJ ETC ETH HBAR ICX IOST IOTA KAVA LINK LTC NEO ONT RVN SOL TFUEL TRX VET XLM XRP XTZ ZEC ZIL ZRX".split()
HARN = sorted(set(CONF) | set(ALGO))
E22, E23 = D.eh(2022) * 3600, D.eh(2023) * 3600

_lk, _next, _hold, _nreq, _tl = threading.Lock(), [0.0], [0.0], [0], threading.local()


def get(url, tries=8):
    """one polite GET: global rate limit, shared back-off on 429. Returns the parsed JSON (or None on a 4xx other than 429)."""
    for i in range(tries):
        with _lk:
            t = max(_next[0], _hold[0], time.time())
            _next[0] = t + 1.0 / RATE
            _nreq[0] += 1
        time.sleep(max(0.0, t - time.time()))
        try:
            u = urllib.parse.urlsplit(url)
            cs = _tl.__dict__.setdefault("c", {})
            c = cs.get(u.netloc) or cs.setdefault(u.netloc, http.client.HTTPSConnection(u.netloc, timeout=40))   # one kept-alive connection per worker
            try:
                c.request("GET", u.path + "?" + u.query, headers=UA)
                r = c.getresponse(); b = r.read()
            except Exception:
                c.close(); cs.pop(u.netloc, None); raise
            if r.getheader("Content-Encoding") == "gzip":
                b = gzip.decompress(b)
            if r.status >= 400:
                raise urllib.error.HTTPError(url, r.status, "http", None, None)
            j = json.loads(b)
            if str(j.get("code")) in ("429000", "50011"):
                raise urllib.error.HTTPError(url, 429, "rate", None, None)
            return j
        except urllib.error.HTTPError as e:
            if e.code == 429:
                with _lk:
                    _hold[0] = time.time() + min(60, 5 * 2 ** i)
                print("429 back-off", min(60, 5 * 2 ** i), flush=True)
                continue
            if 400 <= e.code < 500:
                return None
            time.sleep(2 + 2 * i)
        except Exception:
            time.sleep(2 + 2 * i)
    raise RuntimeError("fetch failed: " + url)


def kc(sym, typ, a, b):
    j = get(f"{KC}/api/v1/market/candles?type={typ}&symbol={sym}&startAt={int(a)}&endAt={int(b)}")
    if j is None or str(j.get("code")) != "200000":
        return None
    d = j.get("data") or []
    assert typ != "1hour" or len(d) < 1500, (sym, a, b, len(d))
    return np.array(d, dtype=np.float64).reshape(-1, 7)                # time s (bar open), open, close, high, low, base volume, quote turnover


def usable(base, market=""):
    return bool(re.fullmatch(r"[A-Z0-9]+", base)) and base not in STABLE | FIAT | BINX and market != "Stocks" and not (LEV.search(base) and len(base) > 4)


def fill22(th):
    """share of the hours from the first bar to 2022-12-31 (last 365 days of that span at most) that have a bar; th = epoch hours, sorted."""
    e = D.eh(2023)
    if not len(th) or th[0] >= e:
        return 0.0
    lo = max(int(th[0]), e - 365 * 24)
    return float(((th >= lo) & (th < e)).sum() / max(e - lo, 1))


def fetch_one(sym, base, end, dl):
    fo = os.path.join(dl, sym + ".json")
    if os.path.exists(fo):
        return json.load(open(fo))
    wk = kc(sym, "1week", D.eh(2017) * 3600, end)                      # startAt=0 is read as "absent" (latest 100 rows only)
    meta = dict(symbol=sym, base=base, served=wk is not None and len(wk) > 0, full=False, bars=0)
    if meta["served"]:
        f = int(wk[:, 0].min()) - 8 * 86400
        f -= f % 3600
        meta["first_week"] = int(wk[:, 0].min())
        harn = base in HARN
        rows = []
        def span(a, b):
            for s in range(int(a), int(b), WIN):
                x = kc(sym, "1hour", s, min(s + WIN, b))
                if x is not None and len(x):
                    rows.append(x)
        if f < E23:
            span(max(f, E22), E23)
        x = np.concatenate(rows) if rows else np.zeros((0, 7))
        th = np.unique((x[:, 0] // 3600).astype(np.int64))
        meta["fill22_probe"] = round(fill22(th[th < D.eh(2023)]) if f >= E22 else float((th >= D.eh(2022)).sum() / (365 * 24)), 3)
        if harn or (f < E23 and meta["fill22_probe"] >= 0.45):       # 0.45: margin for the probe; the exact filter is applied in build
            if f < E22:
                span(f, E22)
            span(max(f, E23), end)
            meta["full"] = True
        x = np.concatenate(rows) if rows else np.zeros((0, 7))
        x = x[x[:, 0] < end]
        _, k = np.unique(x[:, 0], return_index=True)
        x = x[k]
        meta["bars"] = int(len(x))
        np.savez(os.path.join(dl, sym + ".tmp.npz"), x=x)
        os.replace(os.path.join(dl, sym + ".tmp.npz"), os.path.join(dl, sym + ".npz"))
    json.dump(meta, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    return meta


def cmd_fetch(out):
    dl = os.path.join(out, "dl"); os.makedirs(dl, exist_ok=True)
    sf = os.path.join(dl, "state.json")
    if os.path.exists(sf):
        st = json.load(open(sf))
    else:
        j = get(f"{KC}/api/v2/symbols")
        end = int(time.time()) // 3600 * 3600                          # only completed hourly bars: open time < this hour
        st = dict(end=end, at=time.strftime("%FT%TZ", time.gmtime()), symbols=j["data"])
        json.dump(st, open(sf, "w"))
    cur = {x["baseCurrency"]: x for x in st["symbols"] if x["quoteCurrency"] == "USDT"}
    todo = [(x["symbol"], b) for b, x in sorted(cur.items()) if usable(b, x.get("market", ""))]
    todo += [(f"{b}-USDT", b) for b in HARN if b not in cur]            # harness coins KuCoin no longer lists: ask the endpoint anyway
    print("usdt pairs", len(cur), "usable", len(todo), "end", st["end"], flush=True)
    t0, n = time.time(), 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for m in ex.map(lambda a: fetch_one(a[0], a[1], st["end"], dl), todo):
            n += 1
            if n % 25 == 0:
                print(n, len(todo), m["symbol"], m["bars"], f"req {_nreq[0]} {_nreq[0] / max(time.time() - t0, 1):.2f}/s", flush=True)
    print("FETCH_DONE", n, "requests", _nreq[0], round(time.time() - t0), "s", flush=True)


def save_sym(dst, sym, th, o, h, l, c, qv):
    ok = (c > 0) & (h >= l) & (l > 0)
    th, o, h, l, c, qv = th[ok], o[ok], h[ok], l[ok], c[ok], qv[ok]
    u, k = np.unique(th, return_index=True)
    qv = qv[k].astype(np.float32)
    np.savez(os.path.join(dst, sym + ".npz"), t=u, o=o[k], h=h[k], l=l[k], c=c[k], qv=qv, n=np.zeros(len(u), np.float32), tb=(qv * 0.5).astype(np.float32))
    return u


def months(th):
    return sorted(set(np.datetime_as_string(th.astype("datetime64[h]"), unit="M").tolist()))


def cmd_build(out):
    dl, ca = os.path.join(out, "dl"), os.path.join(out, "cache")
    os.makedirs(os.path.join(ca, "hourly"), exist_ok=True)
    idx, rep = {}, {}
    for f in sorted(os.listdir(dl)):
        if not f.endswith(".json") or f == "state.json":
            continue
        m = json.load(open(os.path.join(dl, f)))
        if not m["served"] or m["bars"] < 2:
            continue
        x = np.load(os.path.join(dl, m["symbol"] + ".npz"))["x"]
        sym = m["base"] + "USDT"
        th = save_sym(os.path.join(ca, "hourly"), sym, (x[:, 0] // 3600).astype(np.int64), x[:, 1], x[:, 3], x[:, 4], x[:, 2], x[:, 6])
        if len(th) < 2:
            continue
        mon = months(th)
        if m["full"]:
            idx[sym] = mon
        rep[sym] = dict(pair=m["symbol"], first=mon[0], last=mon[-1], bars=int(len(th)), fill22=round(fill22(th), 3), full=m["full"])
    json.dump(idx, open(os.path.join(ca, "index_1h_all.json"), "w"))
    json.dump(rep, open(os.path.join(ca, "pairs.json"), "w"), indent=0)
    keep = {s: m for s, m in idx.items() if m[0] <= "2022-12" and rep[s]["fill22"] >= 0.5}
    json.dump(keep, open(os.path.join(ca, "index_1h.json"), "w"))
    D.CACHE = ca
    u = D.build_universe()
    json.dump(u, open(os.path.join(out, "universe.json"), "w"), indent=0)
    yrs = {y: sum(1 for s in keep if keep[s][0] <= f"{y}-01") for y in range(2018, 2024)}
    print(json.dumps(dict(served=len(rep), full=len(idx), pool=len(u["pool"]), eval=len(u["eval"]), unseen=len(u["unseen"]), train=len(u["train"]), listed_by_jan=yrs,
                          dropped_fill=sorted(s for s, m in idx.items() if m[0] <= "2022-12" and rep[s]["fill22"] < 0.5),
                          harness_not_in_pool={b: rep.get(b + "USDT", "no bars") for b in HARN if b + "USDT" not in u["pool"]}), indent=0))


def cmd_okx(out, bases):
    dst = os.path.join(out, "okx", "hourly"); os.makedirs(dst, exist_ok=True)
    end = json.load(open(os.path.join(out, "dl", "state.json")))["end"]
    for b in bases:
        rows, after = [], end * 1000
        while True:
            j = get(f"{OKX}/api/v5/market/history-candles?instId={b}-USDT&bar=1H&limit=100&after={after}")
            d = (j or {}).get("data") or []
            if not d:
                break
            x = np.array([r[:8] for r in d], dtype=np.float64)          # ts ms, o, h, l, c, vol, volCcy, volCcyQuote
            rows.append(x); after = int(x[:, 0].min())
        if not rows:
            print("okx", b, "no bars", flush=True); continue
        x = np.concatenate(rows)
        th = save_sym(dst, b + "USDT", (x[:, 0] // 3600000).astype(np.int64), x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 7])
        print("okx", b, len(th), np.datetime64(int(th[0]), "h"), np.datetime64(int(th[-1]), "h"), "requests", _nreq[0], flush=True)
    print("OKX_DONE", flush=True)


if __name__ == "__main__":
    {"fetch": cmd_fetch, "build": cmd_build}[sys.argv[1]](sys.argv[2]) if sys.argv[1] != "okx" else cmd_okx(sys.argv[2], sys.argv[3].split(","))
