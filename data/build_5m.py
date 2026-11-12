#!/usr/bin/env python3
"""Hourly realised measures from the 5-minute archive: cache/zips/5m/<SYM>/ -> cache/rv5/<SYM>.npz
(t = epoch hour, rv = sum of squared 5-minute log returns ending inside the hour, rsn = the part of rv from negative returns)."""
import concurrent.futures as cf, io, os, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BMT_DATA", os.path.join(HERE, "cache"))


def one(sym):
    d = os.path.join(CACHE, "zips", "5m", sym)
    if not os.path.isdir(d):
        return sym, 0
    T, C = [], []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".zip"):
            continue
        with zipfile.ZipFile(os.path.join(d, f)) as z:
            raw = z.read(z.namelist()[0]).decode()
        if raw[:9] == "open_time":
            raw = raw.split("\n", 1)[1]
        a = np.loadtxt(io.StringIO(raw), delimiter=",", usecols=(0, 4), ndmin=2, dtype=np.float64)
        t = np.where(a[:, 0] > 1e14, a[:, 0] / 1e6, a[:, 0] / 1e3)
        T.append(np.round(t / 300).astype(np.int64)); C.append(a[:, 1])
    t, c = np.concatenate(T), np.concatenate(C)
    _, i = np.unique(t, return_index=True)
    t, c = t[i], c[i]
    ok = c > 0
    t, c = t[ok], c[ok]
    r = np.diff(np.log(c))                                           # return of bar t[1:], gaps fold into the bar after the gap
    h = t[1:] // 12
    hours, inv = np.unique(h, return_inverse=True)
    rv, rsn = np.bincount(inv, r * r), np.bincount(inv, r * r * (r < 0))
    os.makedirs(os.path.join(CACHE, "rv5"), exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "rv5", sym + ".npz"), t=hours, rv=rv.astype(np.float32), rsn=rsn.astype(np.float32))
    return sym, len(hours)


if __name__ == "__main__":
    syms = sorted(os.listdir(os.path.join(CACHE, "zips", "5m")))
    with cf.ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        res = list(ex.map(one, syms))
    print(len(res), "symbols", sum(n for _, n in res), "hours")
