#!/usr/bin/env python3
"""Fetch public Binance spot klines from the data.binance.vision archive (no API key, no api.binance.com).

  list   : index every *USDT spot symbol and its monthly 1h archive months -> cache/index.json
  fetch  : download + sha256-verify monthly zips for the chosen symbols      -> cache/zips/<SYM>/
  build  : parse zips into one compact array per symbol                      -> cache/hourly/<SYM>.npz
           (t = open time in epoch hours, o/h/l/c, qv = quote volume, n = trade count)

Nothing downloaded is committed (data/cache is git-ignored). Archive timestamps are ms before 2025 and us from 2025 on.
"""
import argparse, concurrent.futures as cf, hashlib, io, json, os, re, sys, time, urllib.request, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BMT_DATA", os.path.join(HERE, "cache"))
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
WEB = "https://data.binance.vision"
UA = {"User-Agent": "bmt-fetch"}
# bases that are not free-floating crypto assets: stablecoins, fiat, pegged or wrapped duplicates
EXCLUDE = set("USDC BUSD TUSD USDP PAX DAI FDUSD EUR GBP AUD UST USTC SUSD USDS USDSB AEUR EURI BFUSD USD1 XUSD PAXG WBTC BETH WBETH "
              "BTTC VEN BCC BCHABC BCHSV USDE BKRW IDRT BIDR TRY BRL RUB UAH NGN ZAR BVND DAI1 EURS LUSD FRAX PYUSD RLUSD".split())
LEV = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1 + 2 * i)
        except Exception:
            time.sleep(1 + 2 * i)
    raise RuntimeError("fetch failed: " + url)


def s3_list(prefix, delim=True):
    out, marker = [], ""
    while True:
        x = get(f"{S3}?{'delimiter=/&' if delim else ''}prefix={prefix}&marker={marker}").decode()
        out += re.findall(r"<Prefix>([^<]+)</Prefix>" if delim else r"<Key>([^<]+)</Key>", x)
        if "<IsTruncated>true" not in x:
            return out
        m = re.search(r"<NextMarker>([^<]+)", x)
        marker = m.group(1) if m else out[-1]


def usable(sym):
    if not sym.endswith("USDT") or not re.fullmatch(r"[A-Z0-9]+", sym):
        return False
    b = sym[:-4]
    return b not in EXCLUDE and not (LEV.search(b) and len(b) > 4)


def cmd_list(a):
    syms = sorted({p.rstrip("/").rsplit("/", 1)[1] for p in s3_list("data/spot/monthly/klines/") if p != "data/spot/monthly/klines/"})
    syms = [s for s in syms if usable(s)]

    def months(s):
        ks = s3_list(f"data/spot/monthly/klines/{s}/{a.interval}/", delim=False)
        return s, sorted(re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) for k in ks if k.endswith(".zip"))
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        idx = dict(ex.map(months, syms))
    idx = {s: m for s, m in idx.items() if m}
    os.makedirs(CACHE, exist_ok=True)
    json.dump(idx, open(os.path.join(CACHE, f"index_{a.interval}.json"), "w"))
    print(len(idx), "symbols", sum(map(len, idx.values())), "month files")


def fetch_one(sym, interval, month):
    d = os.path.join(CACHE, "zips", interval, sym)
    f = os.path.join(d, f"{sym}-{interval}-{month}.zip")
    if os.path.exists(f):
        return 0
    url = f"{WEB}/data/spot/monthly/klines/{sym}/{interval}/{sym}-{interval}-{month}.zip"
    z = get(url)
    if z is None:
        return -1
    c = get(url + ".CHECKSUM")
    if c is not None and c.decode().split()[0] != hashlib.sha256(z).hexdigest():
        raise RuntimeError("checksum mismatch " + url)
    os.makedirs(d, exist_ok=True)
    with open(f + ".tmp", "wb") as o:
        o.write(z)
    os.replace(f + ".tmp", f)
    return len(z)


def cmd_fetch(a):
    idx = json.load(open(os.path.join(CACHE, f"index_{a.interval}.json")))
    syms = [l.split()[0] for l in open(a.symbols) if l.strip() and not l.startswith("#")] if a.symbols else sorted(idx)
    jobs = [(s, a.interval, m) for s in syms for m in idx.get(s, []) if a.first <= m <= a.last]
    t0, n, b = time.time(), 0, 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for r in ex.map(lambda j: fetch_one(*j), jobs):
            n += 1; b += max(r, 0)
            if n % 2000 == 0:
                print(f"{n}/{len(jobs)} {b / 1e6:.0f} MB {time.time() - t0:.0f}s", flush=True)
    print(f"done {n} files {b / 1e6:.0f} MB new in {time.time() - t0:.0f}s")


def parse_zip(path):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode()
    if raw[:9] == "open_time":
        raw = raw.split("\n", 1)[1]
    a = np.loadtxt(io.StringIO(raw), delimiter=",", usecols=(0, 1, 2, 3, 4, 7, 8), ndmin=2, dtype=np.float64)
    t = a[:, 0]
    t = np.where(t > 1e14, t / 1e6, t / 1e3)                        # us (2025+) or ms -> s
    return np.column_stack([t, a[:, 1:]])


def build_one(sym, interval):
    d = os.path.join(CACHE, "zips", interval, sym)
    if not os.path.isdir(d):
        return sym, 0
    a = np.concatenate([parse_zip(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".zip")])
    step = {"1h": 3600, "1d": 86400}[interval]
    t = np.round(a[:, 0] / step).astype(np.int64)
    _, i = np.unique(t, return_index=True)                          # sorted, de-duplicated
    a, t = a[i], t[i]
    ok = (a[:, 1:5] > 0).all(1) & np.isfinite(a).all(1)
    a, t = a[ok], t[ok]
    out = os.path.join(CACHE, {"1h": "hourly", "1d": "daily"}[interval])
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, sym + ".npz"), t=t, o=a[:, 1].astype(np.float64), h=a[:, 2], l=a[:, 3], c=a[:, 4],
                        qv=a[:, 5].astype(np.float32), n=a[:, 6].astype(np.float32))
    return sym, len(t)


def cmd_build(a):
    d = os.path.join(CACHE, "zips", a.interval)
    syms = sorted(os.listdir(d))
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(build_one, syms, [a.interval] * len(syms)))
    print(len(res), "symbols", sum(n for _, n in res), "bars")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["list", "fetch", "build"])
    p.add_argument("--interval", default="1h")
    p.add_argument("--symbols", help="file with one symbol per line (default: every indexed symbol)")
    p.add_argument("--first", default="2017-01")
    p.add_argument("--last", default="2026-10")
    p.add_argument("--jobs", type=int, default=12)
    a = p.parse_args()
    {"list": cmd_list, "fetch": cmd_fetch, "build": cmd_build}[a.cmd](a)
