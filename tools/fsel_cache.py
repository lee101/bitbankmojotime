"""FSEL: kline cache for the wide universe = links to the forecaster's cache + the missing names from the UNIV hourly archive (same public
source), cut at the cache's last hour. usage: fsel_cache.py <out dir>   -> <out>/hourly/*.npz, <out>/names.json (panel names -> symbol)"""
import json, os, sys
import numpy as np, pandas as pd
C = "/nvme0n1-disk/code/bitbankmojotime/data/cache"; U = "/sdb-disk/tradingbots/binance/univ"
out = sys.argv[1]; os.makedirs(f"{out}/hourly", exist_ok=True)
have = {f[:-4] for f in os.listdir(f"{C}/hourly")}
tmax = int(np.load(f"{C}/hourly/BTCUSDT.npz")["t"][-1])
E = np.load(f"{U}/shift/u/X50/elig.npy"); cols = json.load(open(f"{U}/shift/u/X50/cols.json"))
f = f"{U}/shift/u/X50/panel.bin"; t = int(np.fromfile(f, "<i8", count=1, offset=8)[0]); ts = np.fromfile(f, "<i8", count=t, offset=24)
di = np.nonzero(ts % 86400 == 0)[0]; di = di[di > 0]; td = ts[di]
a, b = np.searchsorted(td, np.datetime64("2023-01-01", "s").astype(int)), np.searchsorted(td, np.datetime64("2026-11-01", "s").astype(int))
names = [c for j, c in enumerate(cols) if E[a:b, j].any()]
syms = sorted({c.split("~")[0] for c in names}); new = [s for s in syms if s not in have]
for s in have:
    if not os.path.lexists(f"{out}/hourly/{s}.npz"): os.symlink(f"{C}/hourly/{s}.npz", f"{out}/hourly/{s}.npz")
for s in new:
    d = pd.read_csv(f"{U}/h1/{s}.csv"); d = d[~d.ts.duplicated()].sort_values("ts"); tt = (d.ts.values // 3600).astype(np.int64); k = tt <= tmax
    z = np.zeros(int(k.sum()), np.float32)
    np.savez(f"{out}/hourly/{s}.tmp.npz", t=tt[k], o=d.open.values[k].astype(float), h=d.high.values[k].astype(float), l=d.low.values[k].astype(float), c=d.close.values[k].astype(float),
             qv=d.qv.values[k].astype(np.float32), n=z, tb=z)
    os.replace(f"{out}/hourly/{s}.tmp.npz", f"{out}/hourly/{s}.npz")
json.dump(dict(names=names, syms=syms, new=new, tmax=tmax), open(f"{out}/names.json", "w"))
print("names", len(names), "symbols", len(syms), "new from the UNIV archive", len(new), "cache last hour", tmax, flush=True)
