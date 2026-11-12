"""Export a bmt checkpoint to the flat weight file src/bmt.mojo reads, plus a self-test vector (fixed input -> reference quantiles
from the fp32 PyTorch forward on CPU).

  python -m train.export runs/ckpt/bmt-s_F4.pt out_dir     ->  out_dir/bmt.bin, selftest_in.f32, selftest_ref.f32

bmt.bin: 16 int32 (magic 'BMT1', L, patch, d, layers, heads, n_horizons, cov, cal, horizons[<=6], 0) then float32 tensors in the
order of BMT.state_dict() with every Linear weight row-major [out, in].
"""
import os, struct, sys
import numpy as np, torch
from train.model import BMT

MAGIC = 0x31544D42


def export(ckpt, out):
    c = torch.load(ckpt, map_location="cpu", weights_only=False) if isinstance(ckpt, str) else ckpt
    cfg = c["cfg"]
    assert not cfg.get("zm"), "zero-median ablation models are not exportable"
    m = BMT(**cfg)
    m.load_state_dict(c["sd"])
    m.eval().float()
    os.makedirs(out, exist_ok=True)
    hz = cfg["horizons"] + [0] * (6 - len(cfg["horizons"]))
    hdr = struct.pack("<16i", MAGIC, cfg["L"], cfg["patch"], cfg["d"], cfg["layers"], cfg["heads"], len(cfg["horizons"]), int(cfg["cov"]), int(cfg["cal"]), *hz, 0)
    sd = m.state_dict()
    names = ["embed.weight", "embed.bias", "pos"]
    for i in range(cfg["layers"]):
        names += [f"blocks.{i}.{k}" for k in ("n1.weight", "n1.bias", "qkv.weight", "qkv.bias", "o.weight", "o.bias", "n2.weight", "n2.bias",
                                              "f1.weight", "f1.bias", "f2.weight", "f2.bias")]
    names += ["nf.weight", "nf.bias"]
    if cfg["cal"]:
        names += ["e_hour.weight", "e_dow.weight", "e_bar.weight"]
    names += ["head.weight", "head.bias"]
    with open(os.path.join(out, "bmt.bin"), "wb") as f:
        f.write(hdr)
        for n in names:
            f.write(sd[n].detach().float().numpy().astype("<f4").tobytes())
    g = np.random.default_rng(20261112)
    n, L = 32, cfg["L"]
    vol = 0.002 * np.exp(g.normal(0, 1, (n, 1)))
    r = (g.standard_t(3, (n, L)) * vol * np.exp(0.5 * np.cumsum(g.normal(0, 0.05, (n, L)), 1))).astype(np.float32)
    r[3, 100:120] = 0                                                # an outage, a jump and a dead series
    r[5, 300] = 0.25
    r[7] = 0
    b = (g.standard_t(3, (n, L)) * 0.004).astype(np.float32)
    cal = np.stack([g.integers(0, 24, n), g.integers(0, 7, n), g.integers(0, 3, n)], 1)
    with torch.no_grad():
        q = m.quantiles(torch.tensor(r), torch.tensor(b) if cfg["cov"] else None, torch.tensor(cal) if cfg["cal"] else None).numpy()
    parts = [r] + ([b] if cfg["cov"] else []) + ([cal.astype(np.float32)] if cfg["cal"] else [])
    np.concatenate([p.ravel() for p in parts]).astype("<f4").tofile(os.path.join(out, "selftest_in.f32"))
    q.astype("<f4").tofile(os.path.join(out, "selftest_ref.f32"))
    return m, r, q


if __name__ == "__main__":
    if sys.argv[1] == "--random":                                    # untrained model of a given size, for kernel tests
        from train.model import SIZES
        torch.manual_seed(0)
        m = BMT(cov="x" in sys.argv[4:], cal="x" in sys.argv[4:], **SIZES[sys.argv[2]])
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, mean=0.5 if "n" in "" else 0.0, std=0.3)
        for n_, p in m.named_parameters():
            if n_.endswith(("n1.weight", "n2.weight", "nf.weight")):
                torch.nn.init.normal_(p, mean=1.0, std=0.1)
        export(dict(sd=m.state_dict(), cfg=m.cfg), sys.argv[3])
    else:
        export(sys.argv[1], sys.argv[2])
