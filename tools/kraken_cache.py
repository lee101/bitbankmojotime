"""PREREG-LIC: Kraken OHLCVT archive (60-minute files) -> hourly cache in the bench format + universe.json. Private, never published.

  python tools/kraken_cache.py <dir with Kraken_OHLCVT_Full.zip.partNN> <out dir>
"""
import io, json, os, re, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D

STABLE = {"USDT", "USDC", "DAI", "PYUSD", "TUSD", "USDP", "EURT", "USDE", "RLUSD", "USDG", "USD1", "USDS", "FDUSD", "USDQ", "USDR", "GUSD",
          "BUSD", "UST", "PAXG", "XAUT"}
FIAT = {"EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "USD", "AED", "NZD", "SGD", "HKD", "PLN", "CZK", "TRY", "BRL", "MXN", "ZAR"}
REN = {"XBT": "BTC", "XDG": "DOGE"}
LEG = re.compile(r"^X([A-Z]{3,4})ZUSD$")
NAME = re.compile(r"^(?:.*/)?([A-Z0-9]+)_60\.csv$")


class Parts(io.RawIOBase):
    def __init__(s, paths):
        s.f = [open(p, "rb") for p in paths]
        s.n = [os.path.getsize(p) for p in paths]
        s.c = np.concatenate([[0], np.cumsum(s.n)])
        s.pos = 0

    def seekable(s):
        return True

    def readable(s):
        return True

    def tell(s):
        return s.pos

    def seek(s, off, wh=0):
        s.pos = off if wh == 0 else s.pos + off if wh == 1 else int(s.c[-1]) + off
        return s.pos

    def readinto(s, b):
        out, want = 0, len(b)
        while want > 0 and s.pos < s.c[-1]:
            i = int(np.searchsorted(s.c, s.pos, "right") - 1)
            s.f[i].seek(s.pos - int(s.c[i]))
            d = s.f[i].read(min(want, int(s.c[i + 1]) - s.pos))
            b[out:out + len(d)] = d
            out += len(d); want -= len(d); s.pos += len(d)
        return out


def base_of(pair):
    m = LEG.match(pair)
    if m:
        b = m.group(1)
    elif pair.endswith("USD"):
        b = pair[:-3]
    else:
        return None
    b = REN.get(b, b)
    return None if not b or b in STABLE or b in FIAT else b


def main(src, out):
    parts = sorted(os.path.join(src, x) for x in os.listdir(src) if re.match(r"Kraken_OHLCVT_Full\.zip\.part\d+$", x))
    z = zipfile.ZipFile(io.BufferedReader(Parts(parts), 1 << 20))
    os.makedirs(os.path.join(out, "hourly"), exist_ok=True)
    cands = {}
    for n in z.namelist():
        m = NAME.match(n)
        if m and (b := base_of(m.group(1))):
            cands.setdefault(b, []).append((m.group(1), n))
    idx, rep = {}, {}
    for b, lst in sorted(cands.items()):
        lst.sort(key=lambda x: (not LEG.match(x[0]), x[0]))
        pair, n = lst[0]
        a = np.loadtxt(io.TextIOWrapper(z.open(n)), delimiter=",", ndmin=2)
        if len(a) < 2:
            continue
        t = (a[:, 0] // 3600).astype(np.int64)
        o, h, l, c, v, nt = (a[:, k] for k in range(1, 7))
        ok = (c > 0) & (h >= l) & (l > 0)
        t, o, h, l, c, v, nt = t[ok], o[ok], h[ok], l[ok], c[ok], v[ok], nt[ok]
        u, k = np.unique(t, return_index=True)
        t, o, h, l, c, v, nt = u, o[k], h[k], l[k], c[k], v[k], nt[k]
        qv = (v * c).astype(np.float32)
        sym = b + "USDT"
        np.savez(os.path.join(out, "hourly", sym + ".npz"), t=t, o=o, h=h, l=l, c=c, qv=qv, n=nt.astype(np.float32), tb=(qv * 0.5).astype(np.float32))
        mon = sorted({f"{y:04d}-{m:02d}" for y, m in {(int(x[:4]), int(x[5:7])) for x in np.datetime_as_string(t.astype("datetime64[h]"), unit="M")}})
        idx[sym] = mon
        e = D.eh(2023)
        lo = max(int(t[0]), e - 365 * 24)
        fill = float(((t >= lo) & (t < e)).sum() / max(e - lo, 1)) if t[0] < e else 0.0
        rep[sym] = dict(pair=pair, first=mon[0], last=mon[-1], bars=int(len(t)), fill22=round(fill, 3), alts=[x[0] for x in lst[1:]])
    json.dump(idx, open(os.path.join(out, "index_1h_all.json"), "w"))
    json.dump(rep, open(os.path.join(out, "pairs.json"), "w"), indent=0)
    keep = {s: m for s, m in idx.items() if m[0] <= "2022-12" and rep[s]["fill22"] >= 0.5}
    json.dump(keep, open(os.path.join(out, "index_1h.json"), "w"))
    D.CACHE = out
    u = D.build_universe()
    json.dump(u, open(os.path.join(os.path.dirname(out.rstrip("/")), "universe.json"), "w"), indent=0)
    yrs = {y: sum(1 for s in keep if keep[s][0] <= f"{y}-01") for y in range(2018, 2024)}
    print(json.dumps(dict(usd_pairs=len(idx), pool=len(u["pool"]), eval=len(u["eval"]), unseen=len(u["unseen"]), train=len(u["train"]),
                          listed_by_jan=yrs, dropped_fill=sorted(s for s, m in idx.items() if m[0] <= "2022-12" and rep[s]["fill22"] < 0.5)), indent=0))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
