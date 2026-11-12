"""Permissive-tier training: the promoted architecture trained on SYNTHETIC bars only (no market data touches the weights).

  python -m train.synth --groups ohlc,cal --size s --name syn-ohlc+cal

The simulator is domain-randomised: every parameter is drawn per series from the broad priors below, which were written down from
textbook stylised facts of financial returns (volatility clustering on several time scales, heavy tails, leverage, intraday / weekly
seasonality, jumps, tick-size discreteness, bid-ask bounce). No parameter was fitted to, or tuned against, any licensed dataset;
checkpoint selection uses a held-out synthetic set. Evaluation on real unseen folds happens afterwards (train/final_eval.py).

One series = 512 context bars + 24 future bars of one bar type (70 % "hourly": diurnal + weekend seasonality; 30 % "daily").
  log sigma_t = x1 + x2 + x3 + season(t)          x_j AR(1), half-lives log-uniform in [1, 8], [8, 100], [100, 2000] bars, stationary std U[0.1, 0.6]
  leverage: corr(innovation of x1, previous return shock) ~ U[-0.5, 0.3];  volatility jumps: rate log-U[1e-3, 2e-2] / bar, size Exp(0.7)
  r_t = sigma_t * Student-t(nu) (unit variance), nu ~ U[2.5, 12], as a normal variance mixture over 6 sub-steps per bar
  price jumps: rate log-U[1e-4, 5e-3] / bar, size N(0, (c sigma_t)^2), c ~ U[3, 10];  MA(1) bounce: coefficient ~ U[-0.15, 0.05]
  high / low: extremes of the 6 sub-steps plus Brownian-bridge extremes inside each sub-step
  tick grid on 25 % of the series: tick = log-U[0.01, 1.5] x median bar sigma (closes, highs and lows are rounded)
  season: hourly: A cos(2 pi (h - phi) / 24) + A c2 / 2 cos(4 pi (h - phi2) / 24) + W [weekend], A ~ U[0, 0.5], W ~ U[-0.5, 0.1]; daily: W only
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from bmtbench.gpu import wait_for_gpu, cap_torch, pause_if_needed
from train.bank import FINE, COARSE, L, LC, flip
from train.model import QLEV, SIZES
from train.model2 import BMT2
from train.search import make_cfg, HZ

BURN, HMAX, M = 200, 24, 6


def U(g, n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, device=dev, generator=g)


def LU(g, n, lo, hi, dev):
    return torch.exp(U(g, n, math.log(lo), math.log(hi), dev))


@torch.no_grad()
def simulate(N, g, dev):
    """-> batch dict in the Bank.gather layout (fine r / hl / gk filled, the rest zero), cal ids and y[N, 4]."""
    T = BURN + L + HMAX
    hourly = torch.rand(N, device=dev, generator=g) < 0.7
    t0 = torch.randint(0, 168 * 1000, (N,), device=dev, generator=g)
    tt = t0[:, None] + torch.arange(T, device=dev)[None, :]                      # bar index (hours for hourly, days for daily)
    hod = tt % 24
    wk = lambda day: (((day + 4) % 7) == 6) | (((day + 4) % 7) == 0)             # Saturday / Sunday in the id convention of the real rows
    A, phi, c2, phi2 = U(g, N, 0, 0.5, dev), U(g, N, 0, 24, dev), U(g, N, -1, 1, dev), U(g, N, 0, 24, dev)
    Wd = U(g, N, -0.5, 0.1, dev)
    seas_h = A[:, None] * (torch.cos(2 * math.pi * (hod - phi[:, None]) / 24) + 0.5 * c2[:, None] * torch.cos(4 * math.pi * (hod - phi2[:, None]) / 24)) + Wd[:, None] * wk(tt // 24)
    seas_d = 0.8 * Wd[:, None] * wk(tt)
    seas = torch.where(hourly[:, None], seas_h, seas_d)
    hl_ = torch.stack([LU(g, N, 1, 8, dev), LU(g, N, 8, 100, dev), LU(g, N, 100, 2000, dev)], 1)
    rho = torch.exp(-math.log(2.0) / hl_)
    sd = U(g, N * 3, 0.1, 0.6, dev).view(N, 3)
    eta = sd * torch.sqrt(1 - rho * rho)
    lam = U(g, N, -0.5, 0.3, dev)
    pj_v, pj_r, cj = LU(g, N, 1e-3, 2e-2, dev), LU(g, N, 1e-4, 5e-3, dev), U(g, N, 3, 10, dev)
    nu = U(g, N, 2.5, 12, dev)
    ma = U(g, N, -0.15, 0.05, dev)
    x = sd * torch.randn(N, 3, device=dev, generator=g)
    chi = torch.distributions.Chi2(nu[:, None].expand(N, T)).sample()            # variance mixture -> Student-t bar returns with unit variance
    mix = ((nu[:, None] - 2) / chi).clamp(0.02, 50.0)
    z = torch.randn(N, T, M, device=dev, generator=g)
    xi = torch.randn(N, T, 3, device=dev, generator=g)
    vj = (torch.rand(N, T, device=dev, generator=g) < pj_v[:, None]) * torch.distributions.Exponential(1 / 0.7).sample((N, T)).to(dev)
    sig = torch.empty(N, T, device=dev)
    prev = torch.zeros(N, device=dev)
    for t in range(T):                                               # the AR(1) volatility components with leverage
        sh = xi[:, t]
        sh[:, 0] = lam * prev + torch.sqrt(1 - lam * lam) * sh[:, 0]
        x = rho * x + eta * sh
        x[:, 0] = x[:, 0] + vj[:, t]
        sig[:, t] = torch.exp(x.sum(1) + seas[:, t])
        prev = z[:, t].sum(1) / math.sqrt(M)
    v = (sig * sig * mix)[:, :, None] / M                            # variance of each sub-step
    u = z * torch.sqrt(v)
    jr = (torch.rand(N, T, device=dev, generator=g) < pj_r[:, None]) * torch.randn(N, T, device=dev, generator=g) * cj[:, None] * sig
    u[:, :, M // 2] = u[:, :, M // 2] + jr
    path = torch.cumsum(u, 2)                                        # log price inside the bar relative to its open
    a = torch.cat([torch.zeros(N, T, 1, device=dev), path[:, :, :-1]], 2)
    d2 = (path - a) ** 2
    e1 = -2 * v * torch.log(torch.rand(N, T, M, device=dev, generator=g).clamp_min(1e-12))
    e2 = -2 * v * torch.log(torch.rand(N, T, M, device=dev, generator=g).clamp_min(1e-12))
    hi = (0.5 * (a + path + torch.sqrt(d2 + e1))).amax(2)
    lo = (0.5 * (a + path - torch.sqrt(d2 + e2))).amin(2)
    e = path[:, :, -1]                                               # bar innovation
    r = e.clone()
    r[:, 1:] = r[:, 1:] + ma[:, None] * e[:, :-1]                    # MA(1) bounce
    op = torch.cumsum(r, 1) - r                                      # log open of each bar
    cl, hi, lo = op + r, op + torch.maximum(hi, r.clamp_min(0)), op + torch.minimum(lo, r.clamp_max(0))
    tick = torch.where(torch.rand(N, device=dev, generator=g) < 0.25, LU(g, N, 0.01, 1.5, dev) * sig.median(1).values, torch.zeros(N, device=dev))[:, None]
    q = lambda p: torch.where(tick > 0, torch.round(p / tick.clamp_min(1e-12)) * tick, p)
    cl, hi, lo = q(cl), q(hi), q(lo)
    r = torch.diff(cl, dim=1, prepend=cl[:, :1] * 0 + q(op[:, :1]))
    hl = (hi - lo).clamp_min(0)
    s0, s1 = BURN, BURN + L
    rr, hh = r[:, s0:s1].float(), hl[:, s0:s1].float()
    fine = torch.zeros(N, len(FINE), L, device=dev)
    fine[:, FINE.index("r")], fine[:, FINE.index("hl")] = rr, hh
    fine[:, FINE.index("gk")] = (0.5 * hh * hh - 0.3862944 * rr * rr).clamp_min(0).sqrt()
    fine[:, FINE.index("tb")] = 0.5
    fine[:, FINE.index("rsn")] = 0.5
    cs = torch.cumsum(r[:, s1:], 1)
    y = torch.stack([cs[:, h - 1] for h in HZ], 1).float()
    tl = tt[:, s1 - 1]                                               # origin bar
    cal_h = torch.stack([(tl + 1) % 24, ((tl + 1) // 24 + 4) % 7, torch.zeros_like(tl)], 1)      # same id formulas as Bank.gather
    cal_d = torch.stack([torch.zeros_like(tl), (tl + 1 + 4) % 7, torch.ones_like(tl)], 1)
    return dict(fine=fine, coarse=torch.zeros(N, len(COARSE), LC, device=dev), scal=torch.zeros(N, 11, device=dev), cal=torch.where(hourly[:, None], cal_h, cal_d), y=y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="ohlc,cal")
    ap.add_argument("--size", default="s")
    ap.add_argument("--name", default="syn-ohlc+cal")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=512)
    a = ap.parse_args()
    dev = "cuda"
    wait_for_gpu(3500); cap_torch(3.0)
    torch.manual_seed(a.seed)
    cfg = make_cfg([x for x in a.groups.split(",") if x], a.size)
    assert set(cfg["fine"]) <= {"r", "hl", "gk"} and not cfg["coarse"] and not cfg["scal"], "the simulator only produces r / hl / gk and calendar inputs"
    cfg["ctx"] = a.ctx
    model = BMT2(horizons=HZ, **cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / a.steps))))
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    gv = torch.Generator(device=dev); gv.manual_seed(10_000 + a.seed)
    val = [simulate(2048, gv, dev) for _ in range(8)]                # held-out synthetic validation set
    lev = torch.tensor(QLEV, device=dev)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best, best_sd, t0, hist = 1e9, None, time.time(), []

    def loss_of(b, autocast=True):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            q, sc = model(b)
        ok = (b["fine"][:, 0] != 0).float().mean(1) > 0.5
        ys = (b["y"] / (sc * model.sqh[None, :])).clamp(-40, 40)
        d = ys[ok][..., None] - q.float()[ok]
        return torch.maximum(lev * d, (lev - 1) * d).mean()

    for it in range(a.steps):
        if it % 100 == 0:
            pause_if_needed()
        model.train()
        if it % 4 == 0:                                              # one simulator call feeds four steps
            pool = simulate(4 * a.bs, g, dev)
        b = {k: v[(it % 4) * a.bs:(it % 4 + 1) * a.bs] for k, v in pool.items()}
        m = torch.rand(a.bs, device=dev) < 0.5
        fb = flip(b)
        b = {k: torch.where(m.view(-1, *[1] * (v.dim() - 1)), fb[k], v) for k, v in b.items()}
        loss = loss_of(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            dk = min(0.998, (it + 1) / (it + 10))
            for k, v in model.state_dict().items():
                ema[k].mul_(dk).add_(v.detach(), alpha=1 - dk) if v.dtype.is_floating_point else ema[k].copy_(v)
        if (it + 1) % 500 == 0 or it + 1 == a.steps:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema); model.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_of(vb, False).item() for vb in val]))
            model.load_state_dict(raw)
            hist.append((it + 1, round(loss.item(), 5), round(vl, 5)))
            if vl < best:
                best, best_sd = vl, {k: x.clone() for k, x in ema.items()}
            print(f"{a.name} step {it + 1} loss {loss.item():.4f} synthetic val {vl:.5f} best {best:.5f} {time.time() - t0:.0f}s", flush=True)
    out = os.path.join(D.RUNS, "synth", "ckpt")
    os.makedirs(out, exist_ok=True)
    for fn, _, _ in D.FOLDS:                                         # one model for every fold: it has seen no market data at all
        torch.save(dict(sd=best_sd, cfg=cfg, fold=fn, limit_abs=0, steps=a.steps, synthetic=True, hist=hist), os.path.join(out, f"{a.name}_{fn}.pt"))
    print("saved", a.name, "params", sum(p.numel() for p in model.parameters()), "synthetic val", best)
