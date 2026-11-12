"""bmt2: the bmt patch transformer with configurable inputs (train/bank.py): extra per-bar channels, a coarse 24 h token stream,
long-memory scalars as one token, calendar embeddings. Normalisation of the raw inputs happens here so the Mojo kernel can mirror it.

Channel kinds:  RET  x/s -> [asinh, asinh^2], s = RMS of the series (the first channel of each stream defines s)
                POS  non-negative, return units -> [asinh(x / s)]
                VOL  volume-like -> [log1p(x / mean over the context)]
                SHR  share in [0, 1] -> [4 (x - 0.5)]
                XRT  another series' return, own RMS s_j -> [asinh(x / s_j), asinh(x / s_j) * asinh(r / s)]
                RAW  -> [x]
Scalars: har, har5 -> log(x / s) clamped to +-4 (5 values each), vov -> as is.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from train.model import Block, QLEV, SIZES
from train.bank import FINE, COARSE, L, LC

KIND = {"r": "RET", "hl": "POS", "gk": "POS", "rv": "POS", "v": "VOL", "tb": "SHR", "btc": "XRT", "eth": "XRT", "mkt": "XRT", "disp": "VOL", "brd": "RAW",
        "rv5": "POS", "rsn": "SHR", "peer": "XRT", "bvol": "VOL"}
NF = {"RET": 2, "POS": 1, "VOL": 1, "SHR": 1, "XRT": 2, "RAW": 1}
SCAL = {"har": (0, 5), "vov": (5, 6), "har5": (6, 11)}
LOGRATIO = ("har", "har5")


def rms(x):
    return x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)


def norm_stream(x, names):
    """x[N, C, T] raw channels (names[0] must be 'r') -> (features [N, T, F], s [N, 1])."""
    s = rms(x[:, 0])
    xr = torch.asinh(x[:, 0] / s)
    f = [xr, xr * xr]
    for c, n in enumerate(names[1:], 1):
        v, k = x[:, c], KIND[n]
        if k == "POS":
            f.append(torch.asinh(v / s))
        elif k == "VOL":
            f.append(torch.log1p(v / v.mean(1, keepdim=True).clamp_min(1e-12)))
        elif k == "SHR":
            f.append((v - 0.5) * 4)
        elif k == "XRT":
            xb = torch.asinh(v / rms(v))
            f += [xb, xb * xr]
        else:
            f.append(v)
    return torch.stack(f, -1), s


class BMT2(nn.Module):
    def __init__(s, fine=("r",), coarse=(), scal=(), cal=False, horizons=(1, 5, 6, 24), patch=16, patch_c=16, d=128, layers=4, heads=4):
        super().__init__()
        assert fine[0] == "r" and (not coarse or coarse[0] == "r")
        s.cfg = dict(fine=list(fine), coarse=list(coarse), scal=list(scal), cal=cal, horizons=list(horizons), patch=patch, patch_c=patch_c, d=d, layers=layers, heads=heads)
        s.fine, s.coarse, s.scal, s.cal, s.p, s.pc, s.H = list(fine), list(coarse), list(scal), cal, patch, patch_c, len(horizons)
        s.fi, s.ci = [FINE.index(n) for n in fine], [COARSE.index(n) for n in coarse]
        s.register_buffer("sqh", torch.tensor([math.sqrt(h) for h in horizons]))
        s.embed = nn.Linear(patch * sum(NF[KIND[n]] for n in fine), d)
        s.nt = L // patch + (LC // patch_c if coarse else 0) + (1 if scal else 0)
        if coarse:
            s.embed_c = nn.Linear(patch_c * sum(NF[KIND[n]] for n in coarse), d)
        if scal:
            s.ns = sum(SCAL[n][1] - SCAL[n][0] for n in scal)
            s.embed_s = nn.Linear(s.ns, d)
        s.pos = nn.Parameter(torch.zeros(s.nt, d))
        nn.init.normal_(s.pos, std=0.02)
        s.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        s.nf = nn.LayerNorm(d)
        if cal:
            s.e_hour, s.e_dow, s.e_bar = nn.Embedding(24, d), nn.Embedding(7, d), nn.Embedding(3, d)
            for e in (s.e_hour, s.e_dow, s.e_bar):
                nn.init.zeros_(e.weight)
        s.head = nn.Linear(d, s.H * 21)
        z = torch.distributions.Normal(0, 1).icdf(torch.tensor(QLEV))
        inc = torch.diff(z)
        b = torch.cat([torch.log(torch.expm1(inc[:10])), torch.zeros(1), torch.log(torch.expm1(inc[10:]))])
        nn.init.zeros_(s.head.weight)
        with torch.no_grad():
            s.head.bias.copy_(b.repeat(s.H))

    def scalars(s, raw, sc):
        """raw[N, 6] (bank order) -> normalised scalar vector in the order of s.scal."""
        out = []
        for n in s.scal:
            a, b = SCAL[n]
            out.append(torch.log(raw[:, a:b].clamp_min(1e-12) / sc).clamp(-4, 4) if n in LOGRATIO else raw[:, a:b])
        return torch.cat(out, 1)

    def forward(s, b):
        """b: dict from Bank.gather -> (q_scaled[N, H, 21] in units of s * sqrt(h), s[N, 1])."""
        N = b["fine"].shape[0]
        f, sc = norm_stream(b["fine"][:, s.fi].float(), s.fine)
        toks = [s.embed(f.reshape(N, L // s.p, -1))]
        if s.coarse:
            fc, _ = norm_stream(b["coarse"][:, s.ci].float(), s.coarse)
            toks.insert(0, s.embed_c(fc.reshape(N, LC // s.pc, -1)))
        if s.scal:
            toks.insert(0, s.embed_s(s.scalars(b["scal"].float(), sc))[:, None])
        t = torch.cat(toks, 1) + s.pos
        for blk in s.blocks:
            t = blk(t)
        z = s.nf(t[:, -1])
        if s.cal:
            c = b["cal"]
            z = z + s.e_hour(c[:, 0]) + s.e_dow(c[:, 1]) + s.e_bar(c[:, 2])
        o = s.head(z).float().view(N, s.H, 21)
        med = o[..., 10:11]
        dn = torch.cumsum(F.softplus(o[..., :10]).flip(-1), -1).flip(-1)
        up = torch.cumsum(F.softplus(o[..., 11:]), -1)
        return torch.cat([med - dn, med, med + up], -1), sc

    def quantiles(s, b):
        q, sc = s(b)
        return q * sc[:, :, None] * s.sqh[None, :, None]

    def cost(s):
        """FLOP proxy per row (multiply-adds of the encoder), for the latency budget."""
        d = s.cfg["d"]
        return s.nt * 12 * d * d * s.cfg["layers"]
