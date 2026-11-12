"""Input search on validation rows only (bench/PROTOCOL.md section 6b). One long-lived process: the feature bank stays on the GPU.

  python -m train.search                     greedy forward selection over input groups with successive halving
Everything is appended to runs/search/ledger.jsonl; finalists' fold checkpoints go to runs/search/ckpt/. No test row is ever loaded here.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from bmtbench.gpu import wait_for_gpu, cap_torch, pause_if_needed
from bmtbench.metrics import Boot, ci
from train.bank import Bank, flip, L, HIST, KS
from train.model import QLEV, SIZES
from train.model2 import BMT2

OUT = os.path.join(D.RUNS, "search")
HZ = (1, 5, 6, 24)
PURGE, VAL = 168, 90 * 24
KP = (0.7, 0.3)
PAIRS = (("hourly", 1), ("hourly", 6), ("hourly", 24), ("daily", 1), ("daily", 5))
GROUPS = {"ohlc": dict(fine=["hl", "gk"]), "vol": dict(fine=["v"]), "tb": dict(fine=["tb"]), "rvf": dict(fine=["rv"]), "har": dict(scal=["har"]),
          "vov": dict(scal=["vov"]), "coarse": dict(coarse=["r", "rv"]), "coarse4": dict(coarse=["r", "rv", "hl", "v"]), "btc": dict(fine=["btc"]),
          "eth": dict(fine=["eth"]), "mkt": dict(fine=["mkt", "disp", "brd"]), "cal": dict(cal=True),
          # second wave (symbols with a 5-minute archive): intra-hour realised measures, peers, BTC range
          "rv5": dict(fine=["rv5"]), "rsn": dict(fine=["rsn"]), "har5": dict(scal=["har5"]), "peer": dict(fine=["peer"]), "bvol": dict(fine=["bvol"]),
          # third wave (ranking objective): weight of the cross-sectional correlation loss on the median
          "ic10": dict(lam=0.1), "ic30": dict(lam=0.3)}
WAVE1 = ("ohlc", "vol", "tb", "rvf", "har", "vov", "coarse", "coarse4", "btc", "eth", "mkt", "cal")


def make_cfg(groups, size="s"):
    c = dict(fine=["r"], coarse=[], scal=[], cal=False)
    for g in groups:
        for k, v in GROUPS[g].items():
            if k == "cal":
                c["cal"] = True
            elif k == "lam":
                c["lam"] = v
            elif k == "coarse":
                c["coarse"] = sorted(set(c["coarse"]) | set(v), key=["r", "rv", "hl", "v"].index)
            else:
                c[k] = c[k] + [x for x in v if x not in c[k]]
    return dict(c, **SIZES[size])


def name_of(groups, size="s", frac=1.0):
    return ("+".join(groups) or "base") + ("" if size == "s" else f"@{size}") + ("" if frac == 1 else f"@f{frac}")


class Runner:
    def __init__(s, dev, placebo_seed=None, vram_gb=3.2, use5m=False):
        wait_for_gpu(3500); cap_torch(vram_gb)
        s.dev, u = dev, D.universe()
        if use5m:                                                    # second wave: the 64 evaluation symbols, trained on the seen ones only
            have = [x for x in u["eval"] if os.path.exists(os.path.join(D.CACHE, "rv5", x + ".npz"))]
            tr = [x for x in u["seen"] if x in have]
            s.bank = b = Bank(have, tr, dev, placebo_seed, use5m=True)
        else:
            tr = u["train"]
            s.bank = b = Bank(u["pool"], tr, dev, placebo_seed)
        s.train_idx = np.array([b.ix[x] for x in tr if x in b.ix])
        seen = [x for x in u["seen"] if x in b.ix]
        lp = b.g["logp"][[b.ix[x] for x in seen]].cpu().numpy()
        live = np.zeros_like(lp, bool)
        for i, x in enumerate(seen):
            live[i, b.first[b.ix[x]]:b.last[b.ix[x]] + 1] = True
        Ph = D.Panel(seen, b.t0, np.where(live, lp, np.nan), b.real[[b.ix[x] for x in seen]] & live)
        Pd = D.to_daily(Ph)
        sidx = np.array([b.ix[x] for x in seen])
        s.val = {}
        for task, P, stride in (("hourly", Ph, 1), ("daily", Pd, None)):
            r = D.build_rows(P, task, D.val_bounds(task), stride=stride)
            tb = r["t"] if task == "hourly" else (r["t"] + P.t0) * 24 + 23 - b.t0
            s.val[task] = dict(a=sidx[r["a"]], t=tb.astype(np.int64), y=r["y"], scale=r["scale"], fold=r["fold"], blk=(r["t"] + P.t0) // D.TASKS[task]["block"],
                               asset=r["a"], tkey=r["t"] + P.t0)
        s.lev = torch.tensor(QLEV, device=dev)
        s.gpu_s, s.cache = 0.0, {}
        s.mode = "spl"

    def yield_gpu(s):
        """quiet window: park the feature bank on the CPU and wait; nearly full card: release the cache and wait. Returns paused seconds."""
        from bmtbench.gpu import in_blackout
        t0 = time.time()
        if in_blackout():
            g = s.bank.g
            s.bank.g = {k: v.cpu() for k, v in g.items()}
            del g
            torch.cuda.empty_cache()
            while in_blackout():
                time.sleep(30)
            wait_for_gpu(3000)
            s.bank.g = {k: v.to(s.dev) for k, v in s.bank.g.items()}
        pause_if_needed()
        return time.time() - t0

    def sampler(s, fi, frac, seed):
        b = s.bank
        limit = D.FOLDS[fi][1] - PURGE - VAL - PURGE - b.t0
        keep = np.zeros(b.A, bool)
        idx = s.train_idx if frac >= 1 else np.random.default_rng(seed).permutation(s.train_idx)[:max(2, int(round(frac * len(s.train_idx))))]
        keep[idx] = True
        los, ns, his = [], [], []
        for k in KS:
            lo = b.first + max(k * L, HIST) + 25
            hi = np.minimum(b.last, limit - 1) - k * max(HZ)
            n = np.where(keep, np.maximum(hi - lo + 1, 0), 0)
            assert not n.sum() or int((hi + k * max(HZ))[n > 0].max()) < limit
            los.append(torch.tensor(lo, device=s.dev)); ns.append(torch.tensor(n, dtype=torch.float32, device=s.dev))
            his.append(torch.tensor(np.where(n > 0, hi, -1), device=s.dev))
        g = torch.Generator(device=s.dev); g.manual_seed(1000 * fi + seed)
        return los, ns, g, int(keep.sum()), limit + b.t0, fi, his

    def batch(s, smp, N):
        los, ns, g, _, _, fi, _ = smp
        ki = torch.multinomial(torch.tensor(KP, device=s.dev), N, replacement=True, generator=g)
        parts = []
        for j, k in enumerate(KS):
            m = int((ki == j).sum())
            if m:
                a = torch.multinomial(ns[j], m, replacement=True, generator=g)
                t = los[j][a] + (torch.rand(m, device=s.dev, generator=g) * ns[j][a]).long().clamp_max((ns[j][a] - 1).long())
                parts.append(s.bank.gather(a, t, k, HZ, fold=fi))
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def batch_groups(s, smp, G=32, n_as=32):
        """G timestamps x n_as symbols that are valid at that timestamp: rows of one group share the origin (for the ranking loss).
        Returns (batch, group id per row, valid mask per row)."""
        los, ns, g, _, _, fi, his = smp
        parts, gid, ok, g0 = [], [], [], 0
        for j, k in enumerate(KS):
            m = int(round(G * KP[j]))
            a0 = torch.multinomial(ns[j], m, replacement=True, generator=g)
            t = los[j][a0] + (torch.rand(m, device=s.dev, generator=g) * ns[j][a0]).long().clamp_max((ns[j][a0] - 1).long())
            V = (los[j][None, :] <= t[:, None]) & (t[:, None] <= his[j][None, :])
            a = torch.multinomial(V.float() + 1e-6, n_as, replacement=False, generator=g)
            parts.append(s.bank.gather(a.reshape(-1), t[:, None].expand(m, n_as).reshape(-1), k, HZ, fold=fi))
            gid.append((g0 + torch.arange(m, device=s.dev))[:, None].expand(m, n_as).reshape(-1)); ok.append(V.gather(1, a).reshape(-1)); g0 += m
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}, torch.cat(gid), torch.cat(ok), g0

    @torch.no_grad()
    def predict(s, model, a, t, k, sym=False, bs=2048, fold=0):
        model.eval()
        out = []
        for i in range(0, len(t), bs):
            b = s.bank.gather(torch.as_tensor(a[i:i + bs], device=s.dev), torch.as_tensor(t[i:i + bs], device=s.dev), k, fold=fold)
            q = model.quantiles(b)
            if sym:
                q = 0.5 * (q - model.quantiles(flip(b)).flip(-1))
            out.append(q.float())
        return torch.cat(out)

    def val_loss(s, model, fi, sub=None):
        """per-row scaled pinball on the fold's validation rows: {(task, h): array}, plus row bookkeeping."""
        res = {}
        for task, k in (("hourly", 1), ("daily", 24)):
            v = s.val[task]
            i = np.flatnonzero(v["fold"] == fi)
            if sub and task == "hourly":
                i = i[np.random.default_rng(0).permutation(len(i))[:sub]]
            elif sub:
                continue
            q = s.predict(model, v["a"][i], v["t"][i], k, fold=fi)
            y, sc = torch.tensor(v["y"][i], device=s.dev), torch.tensor(v["scale"][i], device=s.dev)
            for n, h in enumerate(D.TASKS[task]["horizons"]):
                d = y[:, n, None] - q[:, HZ.index(h)]
                res[(task, h)] = (torch.maximum(s.lev * d, (s.lev - 1) * d).mean(1) / sc).cpu().numpy()
            res[(task, "rows")] = i
            if not sub:                                               # cross-sectional Spearman of median vs outcome per origin (>= 10 symbols)
                tk = v["tkey"][i]
                ut, inv = np.unique(tk, return_inverse=True)
                inv_t = torch.as_tensor(inv, device=s.dev)
                cnt = torch.bincount(inv_t, minlength=len(ut)).float()
                def rk(x):                                           # rank inside each origin
                    o = torch.argsort(x + inv_t.double() * 1e6)
                    r = torch.empty_like(o, dtype=torch.float64)
                    r[o] = torch.arange(len(o), device=s.dev, dtype=torch.float64)
                    return r - torch.zeros(len(ut), device=s.dev, dtype=torch.float64).scatter_reduce(0, inv_t, r, "amin", include_self=False)[inv_t]
                for n, h in enumerate(D.TASKS[task]["horizons"]):
                    med, yy = q[:, HZ.index(h), 10].double(), y[:, n].double()
                    med = (med - med.mean()) / med.std().clamp_min(1e-30)     # only for the tie-free ordering key below
                    a_, b_ = rk(med.clamp(-1e3, 1e3)), rk(((yy - yy.mean()) / yy.std().clamp_min(1e-30)).clamp(-1e3, 1e3))
                    mu = (cnt - 1) / 2
                    da, db = a_ - mu[inv_t], b_ - mu[inv_t]
                    z = torch.zeros(len(ut), device=s.dev, dtype=torch.float64)
                    cov, va, vb = z.clone().scatter_add_(0, inv_t, da * db), z.clone().scatter_add_(0, inv_t, da * da), z.clone().scatter_add_(0, inv_t, db * db)
                    ic_t = (cov / (va * vb).sqrt().clamp_min(1e-30)).cpu().numpy()
                    keep_t = (cnt >= 10).cpu().numpy()
                    res[(task, h, "ic")] = ic_t[keep_t]
                res[(task, "icblk")] = (ut // D.TASKS[task]["block"])[keep_t]
        return res

    def objective(s, res):
        """sPL mode: mean over (task, horizon) of asset-equal sPL / sqrt(h). IC mode: minus the mean rank IC over the (task, horizon) pairs."""
        if s.mode == "ic" and ("hourly", 1, "ic") in res:
            return -float(np.mean([res[(task, h, "ic")].mean() for task, h in PAIRS if (task, h, "ic") in res]))
        o = []
        for task, h in PAIRS:
            if (task, h) in res:
                a = s.val[task]["asset"][res[(task, "rows")]]
                o.append(np.mean([res[(task, h)][a == x].mean() for x in np.unique(a)]) / math.sqrt(h))
        return float(np.mean(o))

    def train_fold(s, cfg, fi, steps, frac=1.0, seed=0, bs=1024, lr=1e-3, save=None):
        key = (json.dumps(cfg, sort_keys=True), fi, steps, frac, seed)
        if key in s.cache:
            if save:                                                 # a fold trained at an earlier stage: write its checkpoint now
                c = s.cache[key]
                os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
                torch.save(dict(sd=c["sd"], cfg={k: v for k, v in cfg.items() if k != "lam"}, fold=D.FOLDS[fi][0], limit_abs=c["limit_abs"], steps=steps),
                           os.path.join(OUT, "ckpt", f"{save}_{D.FOLDS[fi][0]}.pt"))
            return s.cache[key]
        t0, paused = time.time(), 0.0
        torch.manual_seed(1000 * fi + seed)
        smp = s.sampler(fi, frac, seed)
        lam = cfg.get("lam", 0.0)
        cfg = {k: v for k, v in cfg.items() if k != "lam"}
        model = BMT2(horizons=HZ, **cfg).to(s.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps))))
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best, best_sd = 1e9, None
        for it in range(steps):
            if it % 100 == 0:
                paused += s.yield_gpu()
            model.train()
            if s.mode == "ic":
                b, gid, gok, ng = s.batch_groups(smp)
            else:
                b = s.batch(smp, bs)
            m = torch.rand(len(b["y"]), device=s.dev) < 0.5
            if s.mode == "ic":                                       # mirror whole groups, not single rows
                m = (torch.rand(ng, device=s.dev) < 0.5)[gid]
            fb = flip(b)
            b = {k: torch.where(m.view(-1, *[1] * (v.dim() - 1)), fb[k], v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.dev == "cuda"):
                q, sc = model(b)
            ok = (b["fine"][:, 0] != 0).float().mean(1) > 0.5
            ys = (b["y"] / (sc * model.sqh[None, :])).clamp(-40, 40)
            if s.mode == "ic":
                ok = ok & gok
            d = ys[ok][..., None] - q.float()[ok]
            loss = torch.maximum(s.lev * d, (s.lev - 1) * d).mean()
            if lam > 0:                                              # minus the within-origin Pearson correlation of median and outcome
                w = ok.float()[:, None]
                oh = torch.nn.functional.one_hot(gid, ng).float() * w                    # [rows, groups]
                nn_ = oh.sum(0).clamp_min(1)[:, None]
                med = q.float()[:, :, 10]
                cm = lambda x: x * w - oh @ ((oh.T @ (x * w)) / nn_)
                xm, ym = cm(med), cm(ys.clamp(-10, 10))
                cov, vx, vy = oh.T @ (xm * ym), oh.T @ (xm * xm), oh.T @ (ym * ym)
                corr = cov / (vx * vy + 1e-12).sqrt()
                good = (oh.sum(0) >= 8).float()[:, None]
                loss = loss - lam * (corr * good).sum() / good.sum().clamp_min(1) / corr.shape[1]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                dk = min(0.998, (it + 1) / (it + 10))
                for k, v in model.state_dict().items():
                    ema[k].mul_(dk).add_(v.detach(), alpha=1 - dk) if v.dtype.is_floating_point else ema[k].copy_(v)
            if (it + 1) % 250 == 0 or it + 1 == steps:
                raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
                v = s.objective(s.val_loss(model, fi, sub=None if s.mode == "ic" else 30000))
                if v < best:
                    best, best_sd = v, {k: x.clone() for k, x in ema.items()}
                model.load_state_dict(raw)
        model.load_state_dict(best_sd)
        res = s.val_loss(model, fi)
        sec = time.time() - t0 - paused
        s.gpu_s += sec
        out = dict(res=res, obj=s.objective(res), sec=sec, params=sum(p.numel() for p in model.parameters()), cost=model.cost(), assets=smp[3], limit_abs=smp[4],
                   sd={k: v.cpu() for k, v in best_sd.items()} if steps >= 1000 else None)
        if save:
            os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
            torch.save(dict(sd=best_sd, cfg=cfg, fold=D.FOLDS[fi][0], limit_abs=smp[4], steps=steps), os.path.join(OUT, "ckpt", f"{save}_{D.FOLDS[fi][0]}.pt"))
        s.cache[key] = out
        del model, opt
        return out

    def run(s, groups, folds, steps, size="s", frac=1.0, save=False):
        cfg, nm = make_cfg(groups, size), name_of(groups, size, frac)
        per = {fi: s.train_fold(cfg, fi, steps, frac, save=nm if save else None) for fi in folds}
        res = {k: np.concatenate([per[fi]["res"][k] for fi in folds]) for k in per[folds[0]]["res"]}
        f0 = per[folds[0]]
        spl = float(np.mean([np.mean([np.mean([per[f]["res"][(t_, h)][s.val[t_]["asset"][per[f]["res"][(t_, "rows")]] == x].mean() for x in np.unique(s.val[t_]["asset"][per[f]["res"][(t_, "rows")]])]) / math.sqrt(h)
                                      for t_, h in PAIRS]) for f in folds]))
        icv = {f"{t_}|h{h}": float(np.mean(res[(t_, h, "ic")])) for t_, h in PAIRS if (t_, h, "ic") in res}
        return dict(name=nm, groups=list(groups), size=size, frac=frac, folds=[D.FOLDS[f][0] for f in folds], steps=steps, res=res, obj=float(np.mean([per[f]["obj"] for f in folds])), spl=spl, ic=icv,
                    sec=float(sum(per[f]["sec"] for f in folds)), params=f0["params"], cost=f0["cost"], assets=f0["assets"])

    def compare(s, cand, inc):
        """paired block-bootstrap interval of the relative sPL improvement of cand over inc per (task, horizon), on identical rows."""
        out = {}
        if s.mode == "ic":
            for task, h in PAIRS:
                d = cand["res"][(task, h, "ic")] - inc["res"][(task, h, "ic")]
                blk = cand["res"][(task, "icblk")]
                out[f"{task}|h{h}"] = ci(Boot(np.zeros(len(d), int), blk).pooled(d))
            return out
        for task in ("hourly", "daily"):
            assert np.array_equal(cand["res"][(task, "rows")], inc["res"][(task, "rows")])
            v = s.val[task]
            i = cand["res"][(task, "rows")]
            boot = Boot(v["asset"][i], v["blk"][i])
            for h in D.TASKS[task]["horizons"]:
                out[f"{task}|h{h}"] = ci(1 - boot.asset_mean(cand["res"][(task, h)].astype(np.float64)) / boot.asset_mean(inc["res"][(task, h)].astype(np.float64)))
        return out


def log(rec):
    os.makedirs(OUT, exist_ok=True)
    rec = {k: v for k, v in rec.items() if k != "res"}
    rec["at"] = time.strftime("%FT%TZ", time.gmtime())
    with open(os.path.join(OUT, "ledger.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec)[:400], flush=True)


def decide(cmp, cand, base_cost):
    better = sum(v[1] > 0 for v in cmp.values())
    worse = sum(v[2] < 0 for v in cmp.values())
    return bool(better >= 2 and worse == 0 and cand["cost"] <= 4 * base_cost)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-h", type=float, default=12.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--use5m", action="store_true", help="second wave: bank of the evaluation symbols with 5-minute realised measures")
    ap.add_argument("--start", default="", help="groups of the starting incumbent")
    ap.add_argument("--groups", default=",".join(WAVE1), help="candidate groups")
    ap.add_argument("--out", default="search")
    ap.add_argument("--objective", default="spl", choices=["spl", "ic"], help="ic: validation rank IC of the median (grouped batches, ranking loss weights ic10 / ic30)")
    a = ap.parse_args()
    OUT = os.path.join(D.RUNS, a.out)
    R = Runner("cuda", use5m=a.use5m)
    R.mode = a.objective
    ALL, S1, S2 = [0, 1, 2, 3], [2], [0, 2]
    st1, st3 = (60, 120) if a.smoke else (500, 1500)
    trial, since, rnd = 0, 0, 0
    start = [g for g in a.start.split(",") if g]
    CAND = [g for g in a.groups.split(",") if g and g not in start]
    inc = R.run(start, ALL, st3, save=True)
    base_cost = BMT2(horizons=HZ, **make_cfg([])).cost()
    log(dict(inc, trial=0, round=0, stage="full", decision="incumbent (starting point)"))
    remaining = list(CAND)
    chain = [dict(name=inc["name"], groups=list(start), obj=inc["obj"], cmp=None)]
    pairs_done = False
    while remaining and R.gpu_s < a.budget_h * 3600 and since < 100:
        rnd += 1
        cands = [inc["groups"] + [g] for g in remaining] if not pairs_done or rnd == 1 or True else []
        if isinstance(remaining[0], tuple):
            cands = [inc["groups"] + list(g) for g in remaining]
        ref1 = R.run(inc["groups"], S1, st1)
        s1 = []
        for c in cands:
            r = R.run(c, S1, st1)
            trial += 1
            s1.append(r)
            log(dict(r, trial=trial, round=rnd, stage="s1", inc_obj=ref1["obj"], delta_pct=100 * (1 - r["obj"] / ref1["obj"])))
            if R.gpu_s > a.budget_h * 3600:
                break
        s1.sort(key=lambda r: r["obj"])
        top = s1[:max(min(2, len(s1)), math.ceil(len(s1) / 3))]
        ref2 = R.run(inc["groups"], S2, st3)
        s2 = []
        for r in top:
            r2 = R.run(r["groups"], S2, st3)
            s2.append(r2)
            log(dict(r2, trial=trial, round=rnd, stage="s2", inc_obj=ref2["obj"], delta_pct=100 * (1 - r2["obj"] / ref2["obj"])))
        s2.sort(key=lambda r: r["obj"])
        promoted = None
        for r in s2[:2]:                                             # the finalist, then one runner-up if the finalist is not promoted
            full = R.run(r["groups"], ALL, st3, save=True)
            cmp = R.compare(full, inc)
            ok = decide(cmp, full, base_cost)
            log(dict(full, trial=trial, round=rnd, stage="full", cmp={k: [round(100 * x, 3) for x in v] for k, v in cmp.items()}, inc=inc["name"],
                     decision="PROMOTED" if ok else "not promoted"))
            if ok:
                promoted = full
                chain.append(dict(name=full["name"], groups=full["groups"], obj=full["obj"], cmp={k: list(v) for k, v in cmp.items()}))
                break
        if promoted:
            new = [g for g in promoted["groups"] if g not in inc["groups"]]
            inc, since = promoted, 0
            remaining = [g for g in CAND if g not in inc["groups"] and not (g == "coarse" and "coarse4" in inc["groups"])]
            pairs_done = False
        else:
            since += len(cands)
            if pairs_done or isinstance(remaining[0], tuple) or len(s1) < 3:
                break
            best4 = [r["groups"][-1] for r in s1[:4]]                # one more round: pairs of the four best single additions
            remaining = [(x, y) for i, x in enumerate(best4) for y in best4[i + 1:] if {x, y} != {"coarse", "coarse4"}]
            pairs_done = True
        json.dump(dict(chain=chain, trials=trial, gpu_hours=R.gpu_s / 3600, rounds=rnd), open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    # learning curve with the final inputs (validation only)
    lc = []
    for size, frac in ((("t", 1.0), ("m", 1.0), ("s", 0.05), ("s", 0.25)) if not a.smoke else (("t", 1.0),)):
        if R.gpu_s > (a.budget_h + 1.5) * 3600:
            break
        try:
            r = R.run(inc["groups"], ALL, st3, size=size, frac=frac, save=frac == 1.0)
            cmp = R.compare(r, inc)
            trial += 1
            lc.append(dict(name=r["name"], size=size, frac=frac, params=r["params"], assets=r["assets"], obj=r["obj"], cmp={k: list(v) for k, v in cmp.items()}))
            log(dict(r, trial=trial, round=rnd, stage="lc", cmp={k: [round(100 * x, 3) for x in v] for k, v in cmp.items()}, inc=inc["name"]))
        except torch.OutOfMemoryError as e:
            log(dict(name=name_of(inc["groups"], size, frac), stage="lc", decision="out of memory under the VRAM cap"))
            torch.cuda.empty_cache()
    lc.append(dict(name=inc["name"], size="s", frac=1.0, params=inc["params"], assets=inc["assets"], obj=inc["obj"], cmp=None))
    json.dump(dict(chain=chain, trials=trial, gpu_hours=R.gpu_s / 3600, rounds=rnd, lc=lc, final=inc["name"], final_groups=inc["groups"],
                   stop=("budget" if R.gpu_s >= a.budget_h * 3600 else "100 trials without promotion" if since >= 100 else "proposal list exhausted")),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("DONE", inc["name"], round(R.gpu_s / 3600, 2), "GPU-hours", trial, "trials", flush=True)
