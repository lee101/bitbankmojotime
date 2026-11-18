"""24 h ranking objective with the fixed validation set-up (agent/PREREG-24H.md).

  python -m train.rank24 train --arms inc,mkt,r24,mkt+r24,mkt+r24x3 --seeds 0,1     validation predictions -> runs-agent/r24/val
  python -m train.rank24 val                                                         promotion table (CPU)
  python -m train.rank24 test --arms inc,<winner>                                    the one test scoring (registered rows)
  python -m train.rank24 botpred --arms inc,<winner>                                 daily 00:00 UTC forecasts for the ALGOS coins
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from bmtbench.metrics import Boot, ci
from bmtbench.tradable import by_time
from train import search as S
from train.bank import flip
from train.model2 import BMT2

OUT = os.environ.get("BMT_R24_OUT", os.path.join(D.RUNS, "r24"))
ARMS = {"inc": ([], 0.0), "mkt": (["mkt"], 0.0), "r24": ([], 0.1), "mkt+r24": (["mkt"], 0.1), "mkt+r24x3": (["mkt"], 0.3)}
BASE = ["ohlc", "cal"]
STEPS, H24 = int(os.environ.get("BMT_R24_STEPS", 1500)), S.HZ.index(24)
COST = 4e-4
ALGO_COINS = ['ADAUSDT', 'ALGOUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BANDUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'CHZUSDT', 'DASHUSDT', 'DOGEUSDT',
              'DOTUSDT', 'ENJUSDT', 'ETCUSDT', 'ETHUSDT', 'HBARUSDT', 'ICXUSDT', 'IOSTUSDT', 'IOTAUSDT', 'KAVAUSDT', 'LINKUSDT', 'LTCUSDT', 'NEOUSDT',
              'ONTUSDT', 'RVNUSDT', 'SOLUSDT', 'TFUELUSDT', 'TRXUSDT', 'VETUSDT', 'XLMUSDT', 'XRPUSDT', 'XTZUSDT', 'ZECUSDT', 'ZILUSDT', 'ZRXUSDT']


def cfg_of(arm):
    g, lam = ARMS[arm]
    return S.make_cfg(BASE + g), lam


def ckpt(arm, seed, fn):
    return os.path.join(OUT, "ckpt", f"{arm}_s{seed}_{fn}.pt")


class R24(S.Runner):
    def groups24(s, smp, G=32, n_as=32):
        los, ns, g, _, _, fi, his = smp
        a0 = torch.multinomial(ns[0], G, replacement=True, generator=g)
        t = los[0][a0] + (torch.rand(G, device=s.dev, generator=g) * ns[0][a0]).long().clamp_max((ns[0][a0] - 1).long())
        V = (los[0][None, :] <= t[:, None]) & (t[:, None] <= his[0][None, :])
        a = torch.multinomial(V.float() + 1e-6, n_as, replacement=False, generator=g)
        b = s.bank.gather(a.reshape(-1), t[:, None].expand(G, n_as).reshape(-1), 1, S.HZ, fold=fi)
        return b, torch.arange(G, device=s.dev)[:, None].expand(G, n_as).reshape(-1), V.gather(1, a).reshape(-1)

    def train24(s, arm, fi, seed, bs=1024, lr=1e-3):
        cfg, lam = cfg_of(arm)
        t0, paused = time.time(), 0.0
        torch.manual_seed(1000 * fi + seed)
        smp = s.sampler(fi, 1.0, seed)
        model = BMT2(horizons=S.HZ, **cfg).to(s.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / STEPS))))
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for it in range(STEPS):
            if it % 100 == 0:
                paused += s.yield_gpu()
            model.train()
            b = s.batch(smp, bs)
            m = torch.rand(len(b["y"]), device=s.dev) < 0.5
            n0 = len(b["y"])
            if lam > 0:
                bg, gid, gok = s.groups24(smp)
                b = {k: torch.cat([b[k], bg[k]]) for k in b}
                m = torch.cat([m, (torch.rand(int(gid.max()) + 1, device=s.dev) < 0.5)[gid]])
            fb = flip(b)
            b = {k: torch.where(m.view(-1, *[1] * (v.dim() - 1)), fb[k], v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.dev == "cuda"):
                q, sc = model(b)
            q = q.float()
            ok = (b["fine"][:, 0] != 0).float().mean(1) > 0.5
            if lam > 0:
                ok = ok & torch.cat([torch.ones(n0, dtype=torch.bool, device=s.dev), gok])
            ys = (b["y"] / (sc * model.sqh[None, :])).clamp(-40, 40)
            d = ys[ok][..., None] - q[ok]
            loss = torch.maximum(s.lev * d, (s.lev - 1) * d).mean()
            if lam > 0:
                u = sc[n0:, 0] * model.sqh[H24]
                med = q[n0:, H24, 10] * u
                yy = ys[n0:, H24].clamp(-10, 10) * u
                w = ok[n0:].float()
                G = int(gid.max()) + 1
                cnt = torch.zeros(G, device=s.dev).index_add_(0, gid, w)
                mu = lambda x: (torch.zeros(G, device=s.dev).index_add_(0, gid, x * w) / cnt.clamp_min(1))[gid]
                xm, ym = (med - mu(med)) * w, (yy - mu(yy)) * w
                z = lambda x: torch.zeros(G, device=s.dev).index_add_(0, gid, x)
                corr = z(xm * ym) / (z(xm * xm) * z(ym * ym) + 1e-20).sqrt()
                good = (cnt >= 8).float()
                loss = loss - lam * (corr * good).sum() / good.sum().clamp_min(1)
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
        torch.save(dict(sd=ema, cfg=cfg, lam=lam, seed=seed, fold=D.FOLDS[fi][0], limit_abs=smp[4], steps=STEPS), ckpt(arm, seed, D.FOLDS[fi][0]))
        sec = time.time() - t0 - paused
        return model, sec

    def val_pred(s, model, fi):
        v = s.val["hourly"]
        i = np.flatnonzero(v["fold"] == fi)
        q = s.predict(model, v["a"][i], v["t"][i], 1, fold=fi).cpu().numpy()
        hs = [S.HZ.index(h) for h in D.TASKS["hourly"]["horizons"]]
        return dict(rows=i, med=q[:, hs, 10], q=q[:, hs].astype(np.float16))


def seed_series(arm, seed, R=None):
    z = np.load(os.path.join(OUT, "val", f"{arm}_s{seed}.npz"))
    return z


def stats_val():
    from bmtbench.metrics import pinball_levels
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "val", "rows.npz")
    rw = np.load(f)
    y24, tk, blkh = rw["y"][:, 2].astype(np.float64), rw["tkey"], 168
    have = {a: [s for s in (0, 1) if os.path.exists(os.path.join(OUT, "val", f"{a}_s{s}.npz"))] for a in ARMS}
    have["registered"] = [0] if os.path.exists(os.path.join(OUT, "val", "registered.npz")) else []
    ser = {}
    for a, ss in have.items():
        for s_ in ss:
            z = np.load(os.path.join(OUT, "val", "registered.npz" if a == "registered" else f"{a}_s{s_}.npz"))
            T, IC, LS = by_time(z["med"][:, 2].astype(np.float64), y24, tk)
            q = z["q"].astype(np.float32)
            spl = {}
            for n, h in enumerate(D.TASKS["hourly"]["horizons"]):
                pl = pinball_levels(np.sort(q[:, n], 1), rw["y"][:, n]).mean(1) / rw["scale"]
                spl[h] = float(np.mean([pl[rw["asset"] == x].mean() for x in np.unique(rw["asset"])]))
            ser[(a, s_)] = dict(T=T, IC=IC, LS=LS - 2 * COST, spl=spl)
    T0 = ser[("inc", 0)]["T"]
    for k in ser:
        assert np.array_equal(ser[k]["T"], T0), k
    b = Boot(np.zeros(len(T0), int), T0 // blkh)
    avg = lambda a, key: np.mean([ser[(a, s_)][key] for s_ in have[a]], 0)
    out = {}
    for a in ARMS:
        if len(have[a]) < 2:
            continue
        r = dict(ic=ci(b.pooled(avg(a, "IC"))), ls_net=ci(b.pooled(avg(a, "LS")) * 1e4),
                 ic_seed=[float(ser[(a, s_)]["IC"].mean()) for s_ in have[a]], ls_seed=[float(ser[(a, s_)]["LS"].mean() * 1e4) for s_ in have[a]],
                 spl={h: float(np.mean([ser[(a, s_)]["spl"][h] for s_ in have[a]])) for h in (1, 6, 24)})
        if a != "inc":
            dl = avg(a, "LS") - avg("inc", "LS")
            r["d_ls"] = ci(b.pooled(dl) * 1e4)
            r["d_ic"] = ci(b.pooled(avg(a, "IC") - avg("inc", "IC")))
            r["d_ls_seed"] = [float((ser[(a, s_)]["LS"] - avg("inc", "LS")).mean() * 1e4) for s_ in have[a]]
            r["d_spl_pct"] = {h: 100 * (1 - r["spl"][h] / out["inc"]["spl"][h]) for h in (1, 6, 24)}
            r["promoted"] = bool(r["d_ls"][1] > 0 and min(r["d_ls_seed"]) > 0)
        out[a] = r
    if have["registered"]:
        x = ser[("registered", 0)]
        out["registered"] = dict(ic=ci(b.pooled(x["IC"])), ls_net=ci(b.pooled(x["LS"]) * 1e4), spl=x["spl"], d_ic_vs_inc=ci(b.pooled(x["IC"] - avg("inc", "IC"))))
    cands = [a for a in out if a not in ("inc", "registered")]
    prom = [a for a in cands if out[a]["promoted"]]
    pick = max(prom or cands, key=lambda a: out[a]["d_ls"][0]) if cands else None
    res = dict(arms=out, promoted=prom, selected=pick, selected_promoted=bool(prom), origins=int(len(T0)))
    json.dump(res, open(os.path.join(OUT, "val.json"), "w"), indent=1)
    return res


def val_table(res):
    f = lambda x, d=2: f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]"
    md = ["| arm | val 24 h rank IC | L/S net at 4 bp, bp | IC per seed | L/S net per seed | d L/S vs inc, bp | d L/S per seed | d IC vs inc | sPL change % h1 / h6 / h24 | result |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in res["arms"].items():
        if a == "registered":
            md.append(f"| registered `bmt2-ohlc+cal~8bf4ed9e` (1 seed, sPL-selected checkpoint) | {f(r['ic'], 4)} | {f(r['ls_net'])} | | | | | {f(r['d_ic_vs_inc'], 4)} | | reference |")
            continue
        dd = r.get("d_spl_pct")
        md.append(f"| `{a}` | {f(r['ic'], 4)} | {f(r['ls_net'])} | {' / '.join(f'{x:.4f}' for x in r['ic_seed'])} | {' / '.join(f'{x:+.2f}' for x in r['ls_seed'])} | "
                  + (f"{f(r['d_ls'])} | {' / '.join(f'{x:+.2f}' for x in r['d_ls_seed'])} | {f(r['d_ic'], 4)} | {' / '.join(f'{dd[h]:+.2f}' for h in (1, 6, 24))} | {'PROMOTED' if r['promoted'] else 'not promoted'} |"
                     if a != "inc" else "| | | | incumbent |"))
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "val", "test", "botpred"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--coins", default="")
    ap.add_argument("--out", default="botpred.npz")
    a = ap.parse_args()
    arms, seeds = a.arms.split(","), [int(x) for x in a.seeds.split(",")]
    if a.cmd == "val":
        r = stats_val()
        print(val_table(r)); print("selected", r["selected"], "promoted", r["promoted"])
        open(os.path.join(OUT, "val.md"), "w").write(val_table(r) + f"\n\nselected `{r['selected']}` ({'promoted' if r['selected_promoted'] else 'not promoted'}), {r['origins']} validation origins\n")
        sys.exit()
    torch.backends.cuda.matmul.allow_tf32 = True
    R = R24("cuda", vram_gb=3.2)
    os.makedirs(os.path.join(OUT, "val"), exist_ok=True)
    if a.cmd == "train":
        v = R.val["hourly"]
        if not os.path.exists(os.path.join(OUT, "val", "rows.npz")):
            np.savez(os.path.join(OUT, "val", "rows.npz"), y=v["y"], tkey=v["tkey"], asset=v["asset"], scale=v["scale"], fold=v["fold"])
        if not os.path.exists(os.path.join(OUT, "val", "registered.npz")):
            med, qq = np.zeros((len(v["y"]), 3), np.float32), np.zeros((len(v["y"]), 3, 21), np.float16)
            for fi, (fn, _, _) in enumerate(D.FOLDS):
                c = torch.load(os.path.join(D.RUNS, "ckpt_search", f"ohlc+cal_{fn}.pt"), map_location="cuda", weights_only=False)
                m = BMT2(horizons=S.HZ, **c["cfg"]).cuda(); m.load_state_dict(c["sd"])
                p = R.val_pred(m, fi)
                med[p["rows"]], qq[p["rows"]] = p["med"], p["q"]
            np.savez(os.path.join(OUT, "val", "registered.npz"), med=med, q=qq)
        for arm in arms:
            for sd in seeds:
                fo = os.path.join(OUT, "val", f"{arm}_s{sd}.npz")
                if os.path.exists(fo):
                    continue
                med, qq, secs = np.zeros((len(v["y"]), 3), np.float32), np.zeros((len(v["y"]), 3, 21), np.float16), []
                for fi in range(len(D.FOLDS)):
                    model, sec = R.train24(arm, fi, sd)
                    p = R.val_pred(model, fi)
                    med[p["rows"]], qq[p["rows"]] = p["med"], p["q"]
                    secs.append(sec)
                    del model; torch.cuda.empty_cache()
                np.savez(fo, med=med, q=qq)
                rec = dict(arm=arm, seed=sd, sec=round(sum(secs), 1), at=time.strftime("%FT%TZ", time.gmtime()))
                print(json.dumps(rec), flush=True)
                with open(os.path.join(OUT, "ledger.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    elif a.cmd == "test":
        b = R.bank
        for arm in arms:
            for sd in seeds:
                for task, k in (("hourly", 1), ("daily", 24)):
                    rows = D.load_rows(task)
                    ai = np.array([b.ix[x] for x in rows["syms"]])[rows["a"]]
                    tb = (rows["t_abs"] - b.t0) if task == "hourly" else rows["t_abs"] * 24 + 23 - b.t0
                    hs = [S.HZ.index(h) for h in D.TASKS[task]["horizons"]]
                    q = np.zeros((len(tb), len(hs), 21), np.float32)
                    paths = [ckpt(arm, sd, fn) for fn, _, _ in D.FOLDS]
                    c0 = torch.load(paths[0], map_location="cpu", weights_only=False)
                    mh = D.model_hash(dict(c0["cfg"], lam=c0["lam"], seed=sd), paths)
                    row = f"bmt2-r24-{arm}-s{sd}~{mh[:8]}"
                    for fi, (fn, s_abs, _) in enumerate(D.FOLDS):
                        c = torch.load(paths[fi], map_location="cuda", weights_only=False)
                        assert c["limit_abs"] <= s_abs - 168 and c["fold"] == fn
                        m = BMT2(horizons=S.HZ, **c["cfg"]).cuda(); m.load_state_dict(c["sd"])
                        i = np.flatnonzero(rows["fold"] == fi)
                        q[i] = R.predict(m, ai[i], tb[i], k, fold=fi)[:, hs].cpu().numpy()
                    info = dict(model_hash=mh, ckpt_sha256={os.path.basename(p): D.file_sha(p) for p in paths}, cfg=c0["cfg"], steps=STEPS, tier="research",
                                params=f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M", note=f"PREREG-24H arm {arm} seed {sd}, lam {c0['lam']}")
                    D.save_pred(task, row, q, dict(passes=1, model_hash=mh, cfg=c0["cfg"]))
                    D.register(task, row, passes=1, **info)
                    print(row, task, "done", flush=True)
    elif a.cmd == "botpred":
        b = R.bank
        syms = [x for x in (a.coins.split(",") if a.coins else ALGO_COINS) if x in b.ix]
        print("botpred coins", len(syms), "missing", [x for x in a.coins.split(",") if a.coins and x not in b.ix], flush=True)
        out = {}
        for fi, (fn, s_abs, e_abs) in enumerate(D.FOLDS):
            t = np.arange(s_abs, e_abs) - b.t0
            t = t[((t + b.t0) % 24 == 23) & (t < b.T)]
            A, TT = np.meshgrid([b.ix[x] for x in syms], t, indexing="ij")
            A, TT = A.ravel(), TT.ravel()
            okr = b.real[A, TT] & (TT - b.first[A] >= 6200) & (TT <= b.last[A])
            A, TT = A[okr], TT[okr]
            out.setdefault("a", []).append(A); out.setdefault("t_abs", []).append(TT + b.t0)
            for arm in arms:
                for sd in seeds:
                    c = torch.load(ckpt(arm, sd, fn), map_location="cuda", weights_only=False)
                    m = BMT2(horizons=S.HZ, **c["cfg"]).cuda(); m.load_state_dict(c["sd"])
                    out.setdefault(f"{arm}_s{sd}", []).append(R.predict(m, A, TT, 1, fold=fi)[:, H24, 10].cpu().numpy())
            c = torch.load(os.path.join(D.RUNS, "ckpt_search", f"ohlc+cal_{fn}.pt"), map_location="cuda", weights_only=False)
            m = BMT2(horizons=S.HZ, **c["cfg"]).cuda(); m.load_state_dict(c["sd"])
            out.setdefault("registered", []).append(R.predict(m, A, TT, 1, fold=fi)[:, H24, 10].cpu().numpy())
        out = {k: np.concatenate(v) for k, v in out.items()}
        out["sym"] = np.array([b.syms[i] for i in out.pop("a")])
        np.savez(os.path.join(OUT, a.out), **out)
        print("botpred", {k: v.shape for k, v in out.items()}, flush=True)
