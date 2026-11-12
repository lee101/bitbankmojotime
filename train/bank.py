"""Feature bank: every raw model input for any (symbol, origin bar, bar size) by index arithmetic on resident GPU arrays.

All inputs of a row use bars at or before the origin t only (tests/test_bank.py poisons the future and checks). Bar sizes: 1 h and 24 h.
Raw inputs (the model normalises them, train/model2.py):
  fine   [N, C, 512]  r, hl (log high/low range), gk (Garman-Klass sigma), v (quote volume), tb (taker-buy share), rv (realised sigma
                      of the bar from hourly returns), btc, eth, mkt (other series' returns on the same bars), disp (cross-sectional
                      std of returns), brd (share of symbols up - 0.5)
  coarse [N, C, 256]  24 h bars ending at t: r, rv, hl, v
  scal   [N, 11]      har: RMS of hourly returns over 24 / 168 / 720 / 2160 / 6000 h (x sqrt(bar hours)); vov: std of log daily RV, 30 d;
                      har5: the same five windows from 5-minute realised variance
Optional (second search wave, symbols with a 5-minute archive): rv5 (realised sigma of the bar from 5-minute returns), rsn (share of
that variance from negative returns), peer (mean return of the 5 most correlated training symbols, chosen on the year before each
fold's training limit), bvol (BTC's bar range).
Market series (mkt, disp, brd) are computed from the training pool only, never from a held-out symbol.
"""
import os
import numpy as np, torch
from bmtbench import data as D

FINE = ("r", "hl", "gk", "v", "tb", "rv", "btc", "eth", "mkt", "disp", "brd", "rv5", "rsn", "peer", "bvol")
COARSE = ("r", "rv", "hl", "v")
HAR = (24, 168, 720, 2160, 6000)
L, LC, HIST = 512, 256, 6144 + 25                                    # bars of fine / coarse context; hourly history a row needs
KS = (1, 24)


def _roll(x, k, fn):
    if k == 1:
        return x
    w = fn(np.lib.stride_tricks.sliding_window_view(x, k, axis=1), -1)
    return np.concatenate([np.repeat(w[:, :1], k - 1, 1), w], 1)


class Bank:
    def __init__(s, syms, mkt_syms, dev, placebo_seed=None, use5m=False):
        raw = {}
        for x in syms:
            f = os.path.join(D.CACHE, "hourly", x + ".npz")
            if os.path.exists(f):
                z = np.load(f)
                if len(z["t"]) > 1:
                    raw[x] = z
        s.syms = [x for x in syms if x in raw]
        s.ix = {x: i for i, x in enumerate(s.syms)}
        s.t0 = int(min(z["t"][0] for z in raw.values()))
        T = int(max(z["t"][-1] for z in raw.values()) - s.t0 + 1)
        A = len(s.syms)
        logp, up, dn = np.full((A, T), np.nan), np.zeros((A, T), np.float32), np.zeros((A, T), np.float32)
        v, tbv, real = np.zeros((A, T), np.float32), np.zeros((A, T), np.float32), np.zeros((A, T), bool)
        for a, x in enumerate(s.syms):
            z = raw[x]
            i = (z["t"] - s.t0).astype(np.int64)
            real[a, i] = True
            last = np.maximum.accumulate(np.where(real[a], np.arange(T), -1))
            ok = last >= 0
            ok[i[-1] + 1:] = False
            lp = np.full(T, np.nan)
            lp[i] = np.log(z["c"])
            logp[a, ok] = lp[last[ok]]
            up[a, i] = np.log(z["h"]) - np.log(z["c"])
            dn[a, i] = np.log(z["c"]) - np.log(z["l"])
            v[a, i], tbv[a, i] = z["qv"], z["tb"]
        live = np.isfinite(logp)
        s.first = np.where(live.any(1), live.argmax(1), T)
        s.last = T - 1 - live[:, ::-1].argmax(1)
        r = np.nan_to_num(np.diff(logp, axis=1, prepend=np.nan))
        rv5, rsn5 = np.zeros((A, T), np.float32), np.zeros((A, T), np.float32)
        if use5m:
            for a, x in enumerate(s.syms):
                z = np.load(os.path.join(D.CACHE, "rv5", x + ".npz"))
                i = (z["t"] - s.t0).astype(np.int64)
                k = (i >= 0) & (i < T)
                rv5[a, i[k]], rsn5[a, i[k]] = z["rv"][k], z["rsn"][k]
            rv5, rsn5 = np.where(real, rv5, 0), np.where(real, rsn5, 0)
        if placebo_seed is not None:                                 # permute whole bars in time inside each calendar segment
            rng = np.random.default_rng(placebo_seed)
            cuts = [0] + [min(max(b - s.t0, 0), T) for _, b, _ in D.FOLDS] + [T]
            for a in range(A):
                for lo, hi in zip(cuts[:-1], cuts[1:]):
                    lo2, hi2 = max(lo, s.first[a] + 1), min(hi, s.last[a] + 1)
                    if hi2 - lo2 > 1:
                        p = lo2 + rng.permutation(hi2 - lo2)
                        for arr in (r, up, dn, v, tbv, rv5, rsn5):
                            arr[a, lo2:hi2] = arr[a, p]
                if live[a].any():
                    f, l = s.first[a], s.last[a]
                    logp[a, f:l + 1] = logp[a, f] + np.cumsum(r[a, f:l + 1])
        s.T, s.A, s.real = T, A, real
        base = np.nan_to_num(logp)
        lh, ll = base + up, base - dn
        cs2 = np.concatenate([np.zeros((A, 1)), np.cumsum(r * r, 1)], 1)
        mk = np.array([s.ix[x] for x in mkt_syms if x in s.ix])
        ml = live[mk]
        s.g = {"logp": torch.tensor(base, dtype=torch.float64, device=dev), "cs2": torch.tensor(cs2, dtype=torch.float64, device=dev)}
        s.use5m = use5m
        if use5m:
            z1 = np.zeros((A, 1))
            s.g["cs5"] = torch.tensor(np.concatenate([z1, np.cumsum(rv5.astype(np.float64), 1)], 1), dtype=torch.float64, device=dev)
            s.g["csn"] = torch.tensor(np.concatenate([z1, np.cumsum(rsn5.astype(np.float64), 1)], 1), dtype=torch.float64, device=dev)
            pe = np.zeros((len(D.FOLDS), A, 5), np.int64)             # peers per fold: most correlated training symbols before the training limit
            for fi, (_, b0, _) in enumerate(D.FOLDS):
                lim = b0 - 168 - 90 * 24 - 168 - s.t0
                w = r[:, max(lim - 8760, 0):lim]
                okc = live[:, max(lim - 8760, 0):lim].mean(1) > 0.9
                with np.errstate(all="ignore"):
                    cm = np.nan_to_num(np.corrcoef(w))
                cm[:, [i for i in range(A) if i not in set(mk.tolist()) or not okc[i]]] = -2
                np.fill_diagonal(cm, -2)
                pe[fi] = np.argsort(-cm, 1)[:, :5]
            s.peers = torch.tensor(pe, device=dev)
        mr = np.where(ml, r[mk], 0).sum(0) / np.maximum(ml.sum(0), 1)
        s.g["mkt"] = torch.tensor(np.concatenate([[0.0], np.cumsum(mr)]), dtype=torch.float64, device=dev)
        for k in KS:
            hl = _roll(lh, k, np.max) - _roll(ll, k, np.min)
            vk = _roll(v.astype(np.float64), k, np.sum)
            tk = np.where(vk > 0, _roll(tbv.astype(np.float64), k, np.sum) / np.maximum(vk, 1e-12), 0.5)
            rk = logp[mk] - np.roll(logp[mk], k, 1)
            rk[:, :k] = np.nan
            with np.errstate(all="ignore"):
                disp, brd = np.nan_to_num(np.nanstd(rk, 0)), np.nan_to_num(np.nanmean(np.where(np.isfinite(rk), rk > 0, np.nan), 0) - 0.5)
            s.g[f"hl{k}"] = torch.tensor(hl, dtype=torch.float16, device=dev)
            s.g[f"lv{k}"] = torch.tensor(np.log1p(vk), dtype=torch.float16, device=dev)
            s.g[f"tb{k}"] = torch.tensor(tk, dtype=torch.float16, device=dev)
            s.g[f"disp{k}"] = torch.tensor(disp, dtype=torch.float32, device=dev)
            s.g[f"brd{k}"] = torch.tensor(brd, dtype=torch.float32, device=dev)
        s.dev = dev
        s.off, s.offc = torch.arange(-L, 1, device=dev), torch.arange(-LC, 1, device=dev) * 24
        s.har = torch.tensor(HAR, device=dev)
        s.btc, s.eth = s.ix["BTCUSDT"], s.ix["ETHUSDT"]

    def gather(s, a, t, k, horizons=None, fold=0):
        """a, t: LongTensors (symbol index, hourly index of the origin bar); k: bar hours. -> dict(fine, coarse, scal, cal[, y])."""
        g = s.g
        idx = t[:, None] + k * s.off[None, :]
        A = a[:, None]
        lp = g["logp"][A, idx]
        r = (lp[:, 1:] - lp[:, :-1]).float()
        i1 = idx[:, 1:]
        c2 = g["cs2"][A, idx + 1]
        rv = (c2[:, 1:] - c2[:, :-1]).clamp_min(0).sqrt().float()
        hl = g[f"hl{k}"][A, i1].float()
        gk = (0.5 * hl * hl - 0.3862944 * r * r).clamp_min(0).sqrt()
        out = {}

        def other(j):
            x = g["logp"][j, idx]
            return (x[:, 1:] - x[:, :-1]).float()
        m = g["mkt"][idx + 1]
        z0 = torch.zeros_like(r)
        rv5, rsn, peer = z0, z0 + 0.5, z0
        if s.use5m:
            c5, cn = g["cs5"][A, idx + 1], g["csn"][A, idx + 1]
            d5 = (c5[:, 1:] - c5[:, :-1]).clamp_min(0)
            rv5 = d5.sqrt().float()
            rsn = torch.where(d5 > 0, (cn[:, 1:] - cn[:, :-1]) / d5.clamp_min(1e-30), torch.full_like(d5, 0.5)).float()
            pi = s.peers[fold][a] if isinstance(fold, int) else s.peers[fold, a]          # [N, 5]
            pl = g["logp"][pi[:, :, None], idx[:, None, :]]
            peer = (pl[:, :, 1:] - pl[:, :, :-1]).mean(1).float()
        out["fine"] = torch.stack([r, hl, gk, torch.expm1(g[f"lv{k}"][A, i1].float()), g[f"tb{k}"][A, i1].float(), rv, other(s.btc), other(s.eth),
                                   (m[:, 1:] - m[:, :-1]).float(), g[f"disp{k}"][i1], g[f"brd{k}"][i1], rv5, rsn, peer, g[f"hl{k}"][s.btc, i1].float()], 1)
        ic = t[:, None] + s.offc[None, :]
        lc = g["logp"][A, ic]
        cc = g["cs2"][A, ic + 1]
        out["coarse"] = torch.stack([(lc[:, 1:] - lc[:, :-1]).float(), (cc[:, 1:] - cc[:, :-1]).clamp_min(0).sqrt().float(),
                                     g["hl24"][A, ic[:, 1:]].float(), torch.expm1(g["lv24"][A, ic[:, 1:]].float())], 1)
        e = g["cs2"][a, t + 1]
        har = ((e[:, None] - g["cs2"][A, t[:, None] + 1 - s.har[None, :]]) / s.har[None, :]).clamp_min(0).sqrt().float() * (k ** 0.5)
        dj = t[:, None] + 1 - 24 * torch.arange(0, 31, device=s.dev)[None, :]
        d2 = g["cs2"][A, dj]
        vov = (d2[:, :-1] - d2[:, 1:]).clamp_min(1e-12).log().float().std(1, keepdim=True)
        har5 = torch.zeros_like(har)
        if s.use5m:
            e5 = g["cs5"][a, t + 1]
            har5 = ((e5[:, None] - g["cs5"][A, t[:, None] + 1 - s.har[None, :]]) / s.har[None, :]).clamp_min(0).sqrt().float() * (k ** 0.5)
        out["scal"] = torch.cat([har, vov, har5], 1)
        ta = t + s.t0
        out["cal"] = torch.stack([(ta + 1) % 24, ((ta + 1) // 24 + 4) % 7, torch.full_like(ta, KS.index(k))], 1)
        if horizons is not None:
            out["y"] = torch.stack([g["logp"][a, t + k * h] - lp[:, -1] for h in horizons], 1).float()
        return out


def flip(b):
    """mirror a batch: every return-like input changes sign, the taker-buy share and the breadth mirror."""
    f = b["fine"].clone()
    for n in ("r", "btc", "eth", "mkt", "brd", "peer"):
        f[:, FINE.index(n)] = -f[:, FINE.index(n)]
    for n in ("tb", "rsn"):
        f[:, FINE.index(n)] = 1 - f[:, FINE.index(n)]
    c = b["coarse"].clone()
    c[:, 0] = -c[:, 0]
    o = dict(b, fine=f, coarse=c)
    if "y" in b:
        o["y"] = -b["y"]
    return o
