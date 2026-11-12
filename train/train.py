"""Train a bmt candidate walk-forward (one model per fold) and write benchmark predictions.

  python -m train.train --name bmt-s --size s                      # all folds, hourly + daily prediction files
  python -m train.train --name lc --size s --folds F4 --frac 0.25  # learning-curve point (logged to runs/lc.jsonl, no prediction file)

Training windows are drawn at 1h, 4h and 24h bar sizes from the training pool (never an E_unseen symbol), with every target bar
<= fold train_end - purge - validation - purge (bench/PROTOCOL.md section 2); the bound is asserted and stored in the run metadata.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from bmtbench.metrics import pinball_levels
from train.model import BMT, SIZES, QLEV, pinball, n_params

KS, KP = (1, 4, 24), (0.6, 0.15, 0.25)
PURGE, VAL = 168, 90 * 24


def cal_ids(t_abs, k):
    """calendar ids of the first target bar: start hour of day, day of week, bar-size id."""
    return np.stack([(t_abs + 1) % 24, ((t_abs + 1) // 24 + 4) % 7, np.full_like(t_abs, KS.index(k))], 1)


class Sampler:
    def __init__(s, P, limit, L, hmax, dev, frac=1.0, seed=0):
        """limit = first hourly bar index (panel coordinates) no training target may reach."""
        s.L, s.hmax, s.dev, s.t0 = L, hmax, dev, P.t0
        s.logp = torch.tensor(P.logp, dtype=torch.float64, device=dev)
        s.btc = P.ix["BTCUSDT"]
        last = P.T - 1 - P.live[:, ::-1].argmax(1)
        keep = np.ones(len(P.syms), bool)
        if frac < 1:
            keep[:] = False
            keep[np.random.default_rng(seed).permutation(len(P.syms))[:max(2, int(round(frac * len(P.syms))))]] = True
        s.lo, s.n, s.max_target = [], [], -1
        for k in KS:
            lo = P.first + k * L
            hi = np.minimum(last, limit - 1) - k * hmax
            n = np.where(keep, np.maximum(hi - lo + 1, 0), 0)
            if n.sum():
                s.max_target = max(s.max_target, int((hi + k * hmax)[n > 0].max()))
            s.lo.append(torch.tensor(lo, device=dev)); s.n.append(torch.tensor(n, dtype=torch.float32, device=dev))
        assert s.max_target < limit
        s.windows = int(sum(float(n.sum()) for n in s.n))
        s.g = torch.Generator(device=dev); s.g.manual_seed(seed)
        s.off = torch.arange(-L, 1, device=dev)

    def batch(s, N, horizons, cov=False):
        ki = torch.multinomial(torch.tensor(KP, device=s.dev), N, replacement=True, generator=s.g)
        r, y, b, cal = [], [], [], []
        for j, k in enumerate(KS):
            m = int((ki == j).sum())
            if not m:
                continue
            a = torch.multinomial(s.n[j], m, replacement=True, generator=s.g)
            t = s.lo[j][a] + (torch.rand(m, device=s.dev, generator=s.g) * s.n[j][a]).long().clamp_max((s.n[j][a] - 1).long())
            idx = t[:, None] + k * s.off[None, :]
            lp = s.logp[a[:, None], idx]
            r.append((lp[:, 1:] - lp[:, :-1]).float())
            y.append(torch.stack([s.logp[a, t + k * h] - lp[:, -1] for h in horizons], 1).float())
            if cov:
                lb = s.logp[s.btc, idx]
                b.append((lb[:, 1:] - lb[:, :-1]).float())
            ta = t + s.t0
            cal.append(torch.stack([(ta + 1) % 24, ((ta + 1) // 24 + 4) % 7, torch.full_like(ta, j)], 1))
        return torch.cat(r), torch.cat(y), (torch.cat(b) if cov else None), torch.cat(cal)


@torch.no_grad()
def predict(model, p, a, t, k, dev, bs=2048):
    """quantiles [N, H_model, 21] for rows (a, t) of panel p whose bars are k hours."""
    model.eval()
    out = []
    step = 1 if k == 1 else 24
    for i in range(0, len(t), bs):
        aa, tt = a[i:i + bs], t[i:i + bs]
        r = torch.tensor(D.ctx_ret(p, aa, tt, model.L), device=dev)
        b = torch.tensor(D.ctx_ret(p, np.full_like(aa, p.ix["BTCUSDT"]), tt, model.L), device=dev) if model.cov else None
        ta = (tt + p.t0) if k == 1 else (tt + p.t0) * 24 + 23
        cal = torch.tensor(cal_ids(ta, k), device=dev) if model.cal else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            q = model.quantiles(r, b, cal)
        out.append(q.float().cpu().numpy())
    return np.concatenate(out)


def spl(q, y, scale):
    return float((pinball_levels(q, y).mean(1) / scale).mean())


def train_fold(a, fi, P, evalp, rows, val, dev):
    name, s_abs, _ = D.FOLDS[fi]
    torch.manual_seed(a.seed + fi)
    hz = (1, 5, 6, 24)
    limit = s_abs - PURGE - VAL - PURGE - P.t0
    smp = Sampler(P, limit, a.ctx, max(hz), dev, a.frac, a.seed)
    model = BMT(hz, a.ctx, a.patch, cov=a.cov, cal=a.cal, **SIZES[a.size]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / a.steps))))
    lev, sqh = torch.tensor(QLEV, device=dev), model.sqh
    vi = np.flatnonzero(val["hourly"]["fold"] == fi)
    vi = vi[np.random.default_rng(0).permutation(len(vi))[:40000]]
    hj = [hz.index(h) for h in D.TASKS["hourly"]["horizons"]]
    best, best_sd, t0, hist, run = 1e9, None, time.time(), [], 0.0
    for it in range(a.steps):
        model.train()
        r, y, b, cal = smp.batch(a.bs, hz, a.cov)
        flip = (torch.rand(len(r), 1, device=dev) < 0.5).float() * 2 - 1    # mirror augmentation (asset and market together)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            q, sc = model(r * flip, b * flip if a.cov else None, cal if a.cal else None)
        loss = pinball(q.float(), (y * flip) / (sc * sqh[None, :]), lev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = loss.item() if it == 0 else 0.98 * run + 0.02 * loss.item()
        if (it + 1) % a.eval_every == 0 or it + 1 == a.steps:
            q = predict(model, evalp["hourly"], val["hourly"]["a"][vi], val["hourly"]["t"][vi], 1, dev)
            v = float(np.mean([spl(q[:, j], val["hourly"]["y"][vi, n], val["hourly"]["scale"][vi]) for n, j in enumerate(hj)]))
            hist.append((it + 1, round(run, 5), round(v, 5)))
            if v < best:
                best, best_sd = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
            print(f"{a.name} {name} step {it + 1} loss {run:.4f} val_sPL {v:.4f} best {best:.4f} {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_sd)
    meta = dict(fold=name, params=n_params(model), windows=smp.windows, max_target_bar_abs=smp.max_target + P.t0, train_limit_abs=limit + P.t0,
                test_start_abs=s_abs, val_sPL=best, seconds=round(time.time() - t0, 1), hist=hist, cfg=model.cfg, steps=a.steps, bs=a.bs)
    assert meta["max_target_bar_abs"] < s_abs - PURGE
    os.makedirs(os.path.join(D.RUNS, "ckpt"), exist_ok=True)
    torch.save(dict(sd=best_sd, cfg=model.cfg, meta=meta), os.path.join(D.RUNS, "ckpt", f"{a.name}_{name}.pt"))
    out = {}
    for task, k in (("hourly", 1), ("daily", 24)):
        i = np.flatnonzero(rows[task]["fold"] == fi)
        q = predict(model, evalp[task], rows[task]["a"][i], rows[task]["t"][i], k, dev)
        out[task] = (i, q[:, [hz.index(h) for h in D.TASKS[task]["horizons"]]])
        meta["test_sPL_" + task] = [spl(out[task][1][:, n], rows[task]["y"][i, n], rows[task]["scale"][i]) for n in range(len(D.TASKS[task]["horizons"]))]
    return meta, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="s")
    ap.add_argument("--folds", default="F1,F2,F3,F4")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--cov", action="store_true")
    ap.add_argument("--cal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--placebo", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    if dev == "cuda":                                                # shared GPU: wait for head-room, hard VRAM cap
        from bmtbench.gpu import wait_for_gpu, cap_torch
        wait_for_gpu(); cap_torch()
    u = D.universe()
    assert not set(u["train"]) & set(u["unseen"])
    P = D.load_hourly(u["train"])
    E = D.load_hourly(u["eval"])
    if a.placebo:
        P, E = D.shuffle_time(P, a.seed), D.shuffle_time(E, a.seed)
    evalp = {"hourly": E, "daily": D.to_daily(E)}
    if a.placebo:
        rows = {t: D.build_rows(evalp[t], t) for t in D.TASKS}
    else:
        rows = {t: D.load_rows(t) for t in D.TASKS}
    seen = [E.ix[s] for s in u["seen"]]
    val = {}
    for t in ("hourly",):
        v = D.build_rows(evalp[t], t, D.val_bounds(t))
        m = np.isin(v["a"], seen)
        val[t] = {k: x[m] for k, x in v.items()}
    folds = [i for i, f in enumerate(D.FOLDS) if f[0] in a.folds.split(",")]
    metas, preds = [], {t: np.full((len(rows[t]["t"]), len(D.TASKS[t]["horizons"]), 21), np.nan, np.float32) for t in D.TASKS}
    for fi in folds:
        meta, out = train_fold(a, fi, P, evalp, rows, val, dev)
        metas.append(meta)
        for t, (i, q) in out.items():
            preds[t][i] = q
        print(a.name, meta["fold"], "params", meta["params"], "val", round(meta["val_sPL"], 4), "test hourly", [round(x, 4) for x in meta["test_sPL_hourly"]],
              "daily", [round(x, 4) for x in meta["test_sPL_daily"]], f"{meta['seconds']}s", flush=True)
        with open(os.path.join(D.RUNS, "lc.jsonl"), "a") as f:
            f.write(json.dumps(dict(name=a.name, size=a.size, frac=a.frac, cov=a.cov, cal=a.cal, placebo=a.placebo, **{k: v for k, v in meta.items() if k != "cfg"})) + "\n")
    if len(folds) == len(D.FOLDS):
        for t in D.TASKS:
            if a.placebo:
                os.makedirs(os.path.join(D.RUNS, "placebo"), exist_ok=True)
                np.savez(os.path.join(D.RUNS, "placebo", f"{a.name}_{t}.npz"), q=np.sort(preds[t], 2), **rows[t])
            else:
                D.save_pred(t, a.name, preds[t], dict(params=metas[0]["params"], folds=metas, size=a.size, cov=a.cov, cal=a.cal, steps=a.steps, bs=a.bs))
