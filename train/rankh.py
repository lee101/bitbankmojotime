"""Longer ranking horizons (agent/PREREG-HORIZON.md): mkt+r72, mkt+r168 = the mkt+r24 recipe with the correlation term at H hours.

  python -m train.rankh train --hs 72,168 --seeds 0,1     fold checkpoints + validation predictions -> $BMT_HZ_OUT
  python -m train.rankh pred  --hs 72,168                  scores of mkt+r24 (frozen) and the long models on the row sets V, T, B
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train import search as S
from train.bank import flip, L, HIST, KS
from train.model2 import BMT2
from train.rank24 import R24, BASE, STEPS, H24

OUT = os.environ.get("BMT_HZ_OUT", os.path.join(D.RUNS, "hz"))
R24CK = os.environ.get("BMT_R24_CKPT", os.path.join(D.RUNS, "r24", "ckpt"))
LAM, HS, HEV = 0.1, (72, 168), (24, 72, 168)
BOT_COINS = [c + "USDT" for c in ("AAVE ADA ALGO ATOM AVAX BAND BAT BCH BNB BTC CHZ COMP DASH DOGE DOT ENJ ETC ETH HBAR ICX IOST IOTA KAVA LINK LTC MANA NEO ONE ONT "
                                  "QTUM RVN SAND SNX SOL TFUEL TRX UNI VET XLM XRP XTZ ZEC ZEN ZIL ZRX").split()]


def arm_of(H):
    return f"mkt+r{H}"


def ckpt(H, seed, fn):
    return os.path.join(OUT, "ckpt", f"{arm_of(H)}_s{seed}_{fn}.pt")


class RH(R24):
    def sampler_h(s, fi, seed, H):
        """S.Runner.sampler with the 1 h-bar look-ahead raised to H hours: every hourly training label ends before the limit."""
        b = s.bank
        limit = D.FOLDS[fi][1] - S.PURGE - S.VAL - S.PURGE - b.t0
        keep = np.zeros(b.A, bool); keep[s.train_idx] = True
        los, ns, his = [], [], []
        for k in KS:
            look = max(max(S.HZ), H) if k == 1 else k * max(S.HZ)
            lo = b.first + max(k * L, HIST) + 25
            hi = np.minimum(b.last, limit - 1) - look
            n = np.where(keep, np.maximum(hi - lo + 1, 0), 0)
            assert not n.sum() or int((hi + look)[n > 0].max()) < limit
            los.append(torch.tensor(lo, device=s.dev)); ns.append(torch.tensor(n, dtype=torch.float32, device=s.dev))
            his.append(torch.tensor(np.where(n > 0, hi, -1), device=s.dev))
        g = torch.Generator(device=s.dev); g.manual_seed(1000 * fi + seed)
        return los, ns, g, int(keep.sum()), limit + b.t0, fi, his

    def with_h(s, p, a, t, k, H):
        """append the H-hour label (1 h bars only) and its loss weight."""
        m = len(a)
        if k == 1:
            lp = s.bank.g["logp"]
            yh, wh = (lp[a, t + H] - lp[a, t]).float(), torch.ones(m, device=s.dev)
        else:
            yh, wh = torch.zeros(m, device=s.dev), torch.zeros(m, device=s.dev)
        p["y"] = torch.cat([p["y"], yh[:, None]], 1); p["wh"] = wh
        return p

    def batch_h(s, smp, N, H):
        los, ns, g, _, _, fi, _ = smp
        ki = torch.multinomial(torch.tensor(S.KP, device=s.dev), N, replacement=True, generator=g)
        parts = []
        for j, k in enumerate(KS):
            m = int((ki == j).sum())
            if m:
                a = torch.multinomial(ns[j], m, replacement=True, generator=g)
                t = los[j][a] + (torch.rand(m, device=s.dev, generator=g) * ns[j][a]).long().clamp_max((ns[j][a] - 1).long())
                parts.append(s.with_h(s.bank.gather(a, t, k, S.HZ, fold=fi), a, t, k, H))
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def groups_h(s, smp, H, G=32, n_as=32):
        los, ns, g, _, _, fi, his = smp
        a0 = torch.multinomial(ns[0], G, replacement=True, generator=g)
        t = los[0][a0] + (torch.rand(G, device=s.dev, generator=g) * ns[0][a0]).long().clamp_max((ns[0][a0] - 1).long())
        V = (los[0][None, :] <= t[:, None]) & (t[:, None] <= his[0][None, :])
        a = torch.multinomial(V.float() + 1e-6, n_as, replacement=False, generator=g)
        af, tf = a.reshape(-1), t[:, None].expand(G, n_as).reshape(-1)
        b = s.with_h(s.bank.gather(af, tf, 1, S.HZ, fold=fi), af, tf, 1, H)
        return b, torch.arange(G, device=s.dev)[:, None].expand(G, n_as).reshape(-1), V.gather(1, a).reshape(-1)

    def train_h(s, H, fi, seed, bs=1024, lr=1e-3):
        cfg, hz, SL = S.make_cfg(BASE + ["mkt"]), S.HZ + (H,), len(S.HZ)
        t0, paused = time.time(), 0.0
        torch.manual_seed(1000 * fi + seed)
        smp = s.sampler_h(fi, seed, H)
        model = BMT2(horizons=hz, **cfg).to(s.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / STEPS))))
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for it in range(STEPS):
            if it % 100 == 0:
                paused += s.yield_gpu()
            model.train()
            b = s.batch_h(smp, bs, H)
            m = torch.rand(len(b["y"]), device=s.dev) < 0.5
            n0 = len(b["y"])
            bg, gid, gok = s.groups_h(smp, H)
            b = {k: torch.cat([b[k], bg[k]]) for k in b}
            m = torch.cat([m, (torch.rand(int(gid.max()) + 1, device=s.dev) < 0.5)[gid]])
            fb = flip(b)
            b = {k: torch.where(m.view(-1, *[1] * (v.dim() - 1)), fb[k], v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.dev == "cuda"):
                q, sc = model(b)
            q = q.float()
            ok = ((b["fine"][:, 0] != 0).float().mean(1) > 0.5) & torch.cat([torch.ones(n0, dtype=torch.bool, device=s.dev), gok])
            ys = (b["y"] / (sc * model.sqh[None, :])).clamp(-40, 40)
            d = ys[..., None] - q
            w = ok.float()[:, None] * torch.cat([torch.ones(len(ok), SL, device=s.dev), b["wh"][:, None]], 1)
            loss = (torch.maximum(s.lev * d, (s.lev - 1) * d).mean(2) * w).sum() / w.sum()
            u = sc[n0:, 0] * model.sqh[SL]
            med = q[n0:, SL, 10] * u
            yy = ys[n0:, SL].clamp(-10, 10) * u
            wg = ok[n0:].float()
            G = int(gid.max()) + 1
            cnt = torch.zeros(G, device=s.dev).index_add_(0, gid, wg)
            mu = lambda x: (torch.zeros(G, device=s.dev).index_add_(0, gid, x * wg) / cnt.clamp_min(1))[gid]
            xm, ym = (med - mu(med)) * wg, (yy - mu(yy)) * wg
            z = lambda x: torch.zeros(G, device=s.dev).index_add_(0, gid, x)
            corr = z(xm * ym) / (z(xm * xm) * z(ym * ym) + 1e-20).sqrt()
            good = (cnt >= 8).float()
            loss = loss - LAM * (corr * good).sum() / good.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                dk = min(0.998, (it + 1) / (it + 10))
                for k, v in model.state_dict().items():
                    ema[k].mul_(dk).add_(v.detach(), alpha=1 - dk) if v.dtype.is_floating_point else ema[k].copy_(v)
        model.load_state_dict(ema)
        os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
        f = ckpt(H, seed, D.FOLDS[fi][0])
        torch.save(dict(sd=ema, cfg=cfg, hz=hz, lam=LAM, seed=seed, fold=D.FOLDS[fi][0], limit_abs=smp[4], steps=STEPS, sec=time.time() - t0 - paused), f + ".tmp")
        os.replace(f + ".tmp", f)
        return model

    def load(s, path, fn=None, s_abs=None):
        c = torch.load(path, map_location=s.dev, weights_only=False)
        if fn is not None:
            assert c["fold"] == fn and c["limit_abs"] <= s_abs - 168, path
        m = BMT2(horizons=tuple(c.get("hz", S.HZ)), **c["cfg"]).to(s.dev); m.load_state_dict(c["sd"])
        return m, c

    def med(s, model, a, t, fi, slots):
        return s.predict(model, a, t, 1, fold=fi)[:, slots, 10].cpu().numpy()


def labels(b, a, t):
    """realised log returns at 24 / 72 / 168 h; NaN where there is no bar h hours ahead."""
    lp = b.g["logp"].cpu().numpy()
    Y = np.full((len(a), len(HEV)), np.nan, np.float32)
    for n, h in enumerate(HEV):
        ok = (t + h < b.T)
        ok[ok] &= b.real[a[ok], t[ok] + h]
        Y[ok, n] = lp[a[ok], t[ok] + h] - lp[a[ok], t[ok]]
    return Y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "pred"])
    ap.add_argument("--hs", default="72,168")
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    hs, seeds = [int(x) for x in a.hs.split(",")], [int(x) for x in a.seeds.split(",")]
    torch.backends.cuda.matmul.allow_tf32 = True
    R = RH("cuda", vram_gb=3.2)
    b = R.bank
    os.makedirs(os.path.join(OUT, "val"), exist_ok=True)
    v = R.val["hourly"]
    hsl = [S.HZ.index(h) for h in D.TASKS["hourly"]["horizons"]]
    if a.cmd == "train":
        for H in hs:
            for sd in seeds:
                fo = os.path.join(OUT, "val", f"{arm_of(H)}_s{sd}.npz")
                if os.path.exists(fo):
                    continue
                med, qq, secs = np.zeros((len(v["y"]), 2), np.float32), np.zeros((len(v["y"]), 3, 21), np.float16), []
                for fi, (fn, s_abs, _) in enumerate(D.FOLDS):
                    if os.path.exists(ckpt(H, sd, fn)):
                        model, c = R.load(ckpt(H, sd, fn), fn, s_abs)
                    else:
                        model = R.train_h(H, fi, sd)
                        c = torch.load(ckpt(H, sd, fn), map_location="cpu", weights_only=False)
                    secs.append(c["sec"])
                    i = np.flatnonzero(v["fold"] == fi)
                    q = R.predict(model, v["a"][i], v["t"][i], 1, fold=fi).cpu().numpy()
                    med[i], qq[i] = q[:, [H24, len(S.HZ)], 10], q[:, hsl].astype(np.float16)
                    del model; torch.cuda.empty_cache()
                np.savez(fo + ".tmp.npz", med=med, q=qq); os.replace(fo + ".tmp.npz", fo)
                rec = dict(arm=arm_of(H), seed=sd, sec=round(sum(secs), 1), at=time.strftime("%FT%TZ", time.gmtime()))
                print(json.dumps(rec), flush=True)
                with open(os.path.join(OUT, "ledger.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        print("TRAIN DONE", flush=True)
    else:
        rows = D.load_rows("hourly")
        sets = {"V": dict(a=v["a"].astype(np.int64), t=v["t"].astype(np.int64), fold=v["fold"].astype(np.int64)),
                "T": dict(a=np.array([b.ix[x] for x in rows["syms"]])[rows["a"]].astype(np.int64), t=(rows["t_abs"] - b.t0).astype(np.int64), fold=rows["fold"].astype(np.int64))}
        A, TT, FF = [], [], []
        for fi, (fn, s_abs, e_abs) in enumerate(D.FOLDS):
            t = np.arange(s_abs, e_abs) - b.t0
            t = t[((t + b.t0) % 24 == 23) & (t < b.T)]
            Ai, Ti = np.meshgrid([b.ix[x] for x in BOT_COINS], t, indexing="ij")
            Ai, Ti = Ai.ravel(), Ti.ravel()
            okr = b.real[Ai, Ti] & (Ti - b.first[Ai] >= 6200) & (Ti <= b.last[Ai])
            A.append(Ai[okr]); TT.append(Ti[okr]); FF.append(np.full(int(okr.sum()), fi))
        sets["B"] = dict(a=np.concatenate(A).astype(np.int64), t=np.concatenate(TT).astype(np.int64), fold=np.concatenate(FF).astype(np.int64))
        assert np.allclose(labels(b, sets["T"]["a"], sets["T"]["t"])[:, 0], rows["y"][:, 2], atol=1e-5, equal_nan=False)
        models = [("mkt+r24", sd, lambda fn, sd=sd: os.path.join(R24CK, f"mkt+r24_s{sd}_{fn}.pt"), {"": H24}) for sd in seeds]
        for H in hs:
            models += [(arm_of(H), sd, lambda fn, sd=sd, H=H: ckpt(H, sd, fn), {"": len(S.HZ), "@24": H24}) for sd in seeds]
        for nm, r in sets.items():
            out = dict(sym=np.array(b.syms)[r["a"]], t_abs=r["t"] + b.t0, fold=r["fold"], y=labels(b, r["a"], r["t"]))
            for arm, sd, pf, slots in models:
                P = np.zeros((len(r["a"]), len(slots)), np.float32)
                for fi, (fn, s_abs, _) in enumerate(D.FOLDS):
                    model, _ = R.load(pf(fn), fn, s_abs)
                    i = np.flatnonzero(r["fold"] == fi)
                    P[i] = R.med(model, r["a"][i], r["t"][i], fi, list(slots.values()))
                    del model
                for j, sfx in enumerate(slots):
                    out[f"{arm}{sfx}_s{sd}"] = P[:, j]
            np.savez(os.path.join(OUT, f"pred_{nm}.tmp.npz"), **out); os.replace(os.path.join(OUT, f"pred_{nm}.tmp.npz"), os.path.join(OUT, f"pred_{nm}.npz"))
            print("pred", nm, len(r["a"]), flush=True)
        print("PRED DONE", flush=True)
