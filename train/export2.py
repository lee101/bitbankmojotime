"""Export a bmt2 checkpoint to the flat file src/bmt2.mojo reads + a self-test vector (synthetic raw inputs -> fp32 PyTorch reference).

  python -m train.export2 <ckpt.pt | --random fine=r,hl coarse=r,rv scal=har,vov cal=1 size=t> <out_dir>
bmt2.bin: 64 int32 (magic 'BMT2', L, P, LC, PC, D, layers, heads, H, cal, n_fine, n_coarse, har, vov, 4 horizons, 0.., fine kinds at
[20..35], coarse kinds at [36..39], fine channel ids (index into train.bank.FINE) at [44..59]) then float32 tensors: embed, [embed_c], [embed_s with columns in the order har(5), har5(5), vov(1)], pos,
blocks, nf, [calendar embeddings], head.
Input block of N rows: fine[N, n_fine, L], coarse[N, n_coarse, 256], scal[N, n_scalars], cal[N, 3] (float32, channel-major).
"""
import os, struct, sys
import numpy as np, torch
from train.model2 import BMT2, KIND, SCAL
from train.model import SIZES
from train.bank import FINE, COARSE, L, LC

MAGIC, CODE = 0x32544D42, {"RET": 0, "POS": 1, "VOL": 2, "SHR": 3, "XRT": 4, "RAW": 5}
ORDER = ("har", "har5", "vov")                                       # scalar order in the file: log-ratio scalars first


def synth(m, n=32, seed=20261112):
    """plausible raw inputs in the bank layout: dict(fine[n, 11, 512], coarse[n, 4, 256], scal[n, 6], cal[n, 3])."""
    g = np.random.default_rng(seed)
    vol = 0.003 * np.exp(g.normal(0, 1, (n, 1)))
    def ret(T, k=1.0):
        return (g.standard_t(3, (n, T)) * vol * k).astype(np.float32)
    r = ret(L)
    r[3, 100:120] = 0; r[5, 300] = 0.25; r[7] = 0
    f = {"r": r, "hl": np.abs(r) * 1.5 + np.abs(ret(L, 0.5)), "v": np.exp(g.normal(12, 1.5, (n, L))).astype(np.float32), "tb": g.uniform(0.2, 0.8, (n, L)).astype(np.float32),
         "rv": np.abs(ret(L)), "btc": ret(L, 0.7), "eth": ret(L, 0.9), "mkt": ret(L, 0.8), "disp": np.abs(ret(L)) + 1e-4, "brd": g.uniform(-0.5, 0.5, (n, L)).astype(np.float32)}
    f["gk"] = np.sqrt(np.maximum(0.5 * f["hl"] ** 2 - 0.386 * r ** 2, 0)).astype(np.float32)
    f.update(rv5=np.abs(ret(L)) + 1e-5, rsn=g.uniform(0.1, 0.9, (n, L)).astype(np.float32), peer=ret(L, 0.8), bvol=np.abs(ret(L, 0.6)) + 1e-4)
    f["v"][7] = 0
    rc = ret(LC, 4.9)
    c = {"r": rc, "rv": np.abs(ret(LC, 4.9)), "hl": np.abs(rc) * 1.5 + np.abs(ret(LC, 2)), "v": np.exp(g.normal(15, 1.5, (n, LC))).astype(np.float32)}
    scal = np.concatenate([(vol * np.exp(g.normal(0, 0.3, (n, 5)))).astype(np.float32), g.uniform(0.3, 1.5, (n, 1)).astype(np.float32),
                           (vol * np.exp(g.normal(0, 0.3, (n, 5)))).astype(np.float32)], 1)
    cal = np.stack([g.integers(0, 24, n), g.integers(0, 7, n), g.integers(0, 2, n)], 1)
    return dict(fine=torch.tensor(np.stack([f[x] for x in FINE], 1)), coarse=torch.tensor(np.stack([c[x] for x in COARSE], 1)), scal=torch.tensor(scal), cal=torch.tensor(cal))


def export(m, out):
    m = m.eval().float().cpu()
    c = m.cfg
    os.makedirs(out, exist_ok=True)
    hdr = [0] * 64
    hdr[:14] = [MAGIC, c.get("ctx", L), c["patch"], LC, c["patch_c"], c["d"], c["layers"], c["heads"], len(c["horizons"]), int(c["cal"]), len(c["fine"]), len(c["coarse"]),
                5 * sum(n in c["scal"] for n in ORDER[:2]), int("vov" in c["scal"])]
    hdr[14:14 + len(c["horizons"])] = c["horizons"]
    hdr[20:20 + len(c["fine"])] = [CODE[KIND[x]] for x in c["fine"]]
    hdr[36:36 + len(c["coarse"])] = [CODE[KIND[x]] for x in c["coarse"]]
    hdr[44:44 + len(c["fine"])] = [FINE.index(x) for x in c["fine"]]       # channel ids, so a client can build the inputs it knows
    sd = m.state_dict()
    parts = [sd["embed.weight"], sd["embed.bias"]]
    if c["coarse"]:
        parts += [sd["embed_c.weight"], sd["embed_c.bias"]]
    if c["scal"]:
        w, cols, o = sd["embed_s.weight"], {}, 0
        for n in c["scal"]:
            k = SCAL[n][1] - SCAL[n][0]
            cols[n] = w[:, o:o + k]; o += k
        parts += [torch.cat([cols[n] for n in ORDER if n in cols], 1), sd["embed_s.bias"]]
    parts.append(sd["pos"])
    for i in range(c["layers"]):
        parts += [sd[f"blocks.{i}.{k}"] for k in ("n1.weight", "n1.bias", "qkv.weight", "qkv.bias", "o.weight", "o.bias", "n2.weight", "n2.bias", "f1.weight", "f1.bias", "f2.weight", "f2.bias")]
    parts += [sd["nf.weight"], sd["nf.bias"]]
    if c["cal"]:
        parts += [sd["e_hour.weight"], sd["e_dow.weight"], sd["e_bar.weight"]]
    parts += [sd["head.weight"], sd["head.bias"]]
    with open(os.path.join(out, "bmt2.bin"), "wb") as f:
        f.write(struct.pack("<64i", *hdr))
        for p in parts:
            f.write(p.detach().float().contiguous().numpy().astype("<f4").tobytes())
    b = synth(m)
    with torch.no_grad():
        q = m.quantiles(b).numpy()
    blk = [b["fine"][:, m.fi, -m.L:].numpy().ravel()]
    if c["coarse"]:
        blk.append(b["coarse"][:, m.ci].numpy().ravel())
    if c["scal"]:
        blk.append(torch.cat([b["scal"][:, SCAL[n][0]:SCAL[n][1]] for n in ORDER if n in c["scal"]], 1).numpy().ravel())
    if c["cal"]:
        blk.append(b["cal"].numpy().astype(np.float32).ravel())
    np.concatenate(blk).astype("<f4").tofile(os.path.join(out, "selftest_in.f32"))
    q.astype("<f4").tofile(os.path.join(out, "selftest_ref.f32"))


if __name__ == "__main__":
    if sys.argv[1] == "--random":
        kv = dict(x.split("=") for x in sys.argv[2:-1])
        torch.manual_seed(0)
        m = BMT2(fine=kv.get("fine", "r").split(","), coarse=[x for x in kv.get("coarse", "").split(",") if x], scal=[x for x in kv.get("scal", "").split(",") if x],
                 cal=kv.get("cal", "0") == "1", **SIZES[kv.get("size", "s")])
        for n_, p in m.named_parameters():
            torch.nn.init.normal_(p, mean=1.0, std=0.1) if n_.endswith(("n1.weight", "n2.weight", "nf.weight")) else torch.nn.init.normal_(p, std=0.05 if p.dim() > 1 else 0.3)
    else:
        c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
        m = BMT2(horizons=c["cfg"].pop("horizons", (1, 5, 6, 24)), **c["cfg"]) if "horizons" in c["cfg"] else BMT2(**c["cfg"])
        m.load_state_dict(c["sd"])
    export(m, sys.argv[-1])
