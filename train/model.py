"""bmt: compact patch transformer for scale-normalised log returns with monotone multi-horizon quantile heads.

Input  r[N, L] log returns (any bar size), optional market covariate b[N, L] (BTC returns on the same bars), optional calendar ids.
Scale  s = RMS of the context returns; x = asinh(r / s); tokens = Linear([x, x^2] per patch) + learned position.
Output q[N, H, 21] quantiles of the cumulative log return at each horizon, = s * sqrt(h) * monotone(head), never crossing.
The same arithmetic is implemented in src/bmt.mojo (CPU SIMD and GPU); tests/ pins parity.
"""
import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

QLEV = [0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99]
SIZES = {"t": dict(d=64, layers=2, heads=4), "s": dict(d=128, layers=4, heads=4), "m": dict(d=256, layers=6, heads=8)}


class Block(nn.Module):
    def __init__(s, d, heads):
        super().__init__()
        s.h, s.n1, s.n2 = heads, nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.f1, s.f2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(s, x):
        N, T, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(N, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, T, D)
        x = x + s.o(a)
        return x + s.f2(F.gelu(s.f1(s.n2(x)), approximate="tanh"))


class BMT(nn.Module):
    def __init__(s, horizons=(1, 5, 6, 24), L=512, patch=16, d=128, layers=4, heads=4, cov=False, cal=False):
        super().__init__()
        s.cfg = dict(horizons=list(horizons), L=L, patch=patch, d=d, layers=layers, heads=heads, cov=cov, cal=cal)
        s.L, s.p, s.cov, s.cal, s.H = L, patch, cov, cal, len(horizons)
        s.register_buffer("sqh", torch.tensor([math.sqrt(h) for h in horizons]))
        s.embed = nn.Linear(patch * (5 if cov else 2), d)
        s.pos = nn.Parameter(torch.zeros(L // patch, d))
        nn.init.normal_(s.pos, std=0.02)
        s.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        s.nf = nn.LayerNorm(d)
        if cal:
            s.e_hour, s.e_dow, s.e_bar = nn.Embedding(24, d), nn.Embedding(7, d), nn.Embedding(3, d)
            for e in (s.e_hour, s.e_dow, s.e_bar):
                nn.init.zeros_(e.weight)
        s.head = nn.Linear(d + (1 if cov else 0), s.H * 21)
        z = torch.distributions.Normal(0, 1).icdf(torch.tensor(QLEV))
        inc = torch.diff(z)                                           # start at the Gaussian quantiles of the scaled return
        b = torch.cat([torch.log(torch.expm1(inc[:10])), torch.zeros(1), torch.log(torch.expm1(inc[10:]))])
        nn.init.zeros_(s.head.weight)
        with torch.no_grad():
            s.head.bias.copy_(b.repeat(s.H))

    def forward(s, r, b=None, cal=None):
        """r[N, L] -> (q_scaled[N, H, 21] in units of s*sqrt(h), s[N, 1])."""
        N = r.shape[0]
        sc = r.float().pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        x = torch.asinh(r.float() / sc)
        feats = [x, x * x]
        if s.cov:
            sb = b.float().pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
            xb = torch.asinh(b.float() / sb)
            feats += [xb, xb * xb, x * xb]
        t = s.embed(torch.stack(feats, -1).view(N, s.L // s.p, -1)) + s.pos
        for blk in s.blocks:
            t = blk(t)
        z = s.nf(t[:, -1])
        if s.cal:
            z = z + s.e_hour(cal[:, 0]) + s.e_dow(cal[:, 1]) + s.e_bar(cal[:, 2])
        if s.cov:
            z = torch.cat([z, torch.log(sc / sb).clamp(-5, 5)], 1)
        o = s.head(z).float().view(N, s.H, 21)
        med = o[..., 10:11]
        dn = torch.cumsum(F.softplus(o[..., :10]).flip(-1), -1).flip(-1)
        up = torch.cumsum(F.softplus(o[..., 11:]), -1)
        return torch.cat([med - dn, med, med + up], -1), sc

    def quantiles(s, r, b=None, cal=None):
        q, sc = s(r, b, cal)
        return q * sc[:, :, None] * s.sqh[None, :, None]


def pinball(q, y, lev):
    """q[N, H, 21], y[N, H] (same units) -> scalar."""
    d = y[..., None] - q
    return torch.maximum(lev * d, (lev - 1) * d).mean()


def n_params(m):
    return sum(p.numel() for p in m.parameters())
