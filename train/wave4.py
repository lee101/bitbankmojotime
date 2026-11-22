"""Wave 4: training-side search on validation rows only (seed ensembles, schedule, batch). Same Runner, folds, rows and promotion rule
as train/search.py; the incumbent is the promoted input set at the given size.

  python -m train.wave4 --size t --variants ens4,long4k,bs4k
Ledger: runs/wave4/ledger.jsonl; member checkpoints runs/wave4/ckpt/<name>-s<seed>_<fold>.pt.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train import search as S
from train.model2 import BMT2

VARIANTS = {"inc": dict(seeds=[0]), "ens2": dict(seeds=[0, 1]), "ens4": dict(seeds=[0, 1, 2, 3]), "ens8": dict(seeds=list(range(8))),
            "long4k": dict(seeds=[0], steps=4000), "bs4k": dict(seeds=[0], bs=4096, lr=2e-3), "lr3": dict(seeds=[0], lr=3e-3),
            "ens4long": dict(seeds=[0, 1, 2, 3], steps=4000)}


class Ens(torch.nn.Module):
    """quantile averaging (Vincentisation) of members; monotone because every member is."""
    def __init__(s, ms):
        super().__init__()
        s.ms = torch.nn.ModuleList(ms)

    def quantiles(s, b):
        return torch.stack([m.quantiles(b) for m in s.ms]).mean(0)


def members(R, cfg, fi, v, name):
    out = []
    for sd in v["seeds"]:
        r = R.train_fold(cfg, fi, v.get("steps", 1500), seed=sd, bs=v.get("bs", 1024), lr=v.get("lr", 1e-3), save=f"{name}-s{sd}")
        m = BMT2(horizons=S.HZ, **{k: x for k, x in cfg.items() if k != "lam"}).to(R.dev)
        m.load_state_dict({k: x.to(R.dev) for k, x in r["sd"].items()})
        out.append((m, r))
    return out


def evaluate(R, cfg, v, name, size):
    per, sec = {}, 0.0
    for fi in range(4):
        ms = members(R, cfg, fi, v, name)
        sec += sum(r["sec"] for _, r in ms)
        e = Ens([m for m, _ in ms]) if len(ms) > 1 else ms[0][0]
        per[fi] = R.val_loss(e, fi)
        del ms, e
        torch.cuda.empty_cache()
    res = {k: np.concatenate([per[f][k] for f in range(4)]) for k in per[0]}
    obj = float(np.mean([R.objective(per[f]) for f in range(4)]))
    m0 = BMT2(horizons=S.HZ, **{k: x for k, x in cfg.items() if k != "lam"})
    icv = {f"{t_}|h{h}": float(np.mean(res[(t_, h, "ic")])) for t_, h in S.PAIRS if (t_, h, "ic") in res}
    return dict(name=name, size=size, variant=v, res=res, obj=obj, ic=icv, sec=sec, params=sum(p.numel() for p in m0.parameters()) * len(v["seeds"]),
                cost=m0.cost() * len(v["seeds"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="t")
    ap.add_argument("--groups", default="ohlc,cal")
    ap.add_argument("--variants", default="ens4,long4k,bs4k")
    ap.add_argument("--out", default="wave4")
    a = ap.parse_args()
    S.OUT = os.path.join(D.RUNS, a.out)
    os.makedirs(S.OUT, exist_ok=True)
    R = S.Runner("cuda", vram_gb=3.0)
    groups = [g for g in a.groups.split(",") if g]
    cfg = S.make_cfg(groups, a.size)
    base_cost = BMT2(horizons=S.HZ, **S.make_cfg([])).cost()
    tag = "+".join(groups) + ("" if a.size == "s" else f"@{a.size}")
    inc = evaluate(R, cfg, VARIANTS["inc"], f"{tag}-inc", a.size)
    S.log(dict(inc, stage="w4", decision="incumbent"))
    for vn in a.variants.split(","):
        c = evaluate(R, cfg, VARIANTS[vn], f"{tag}-{vn}", a.size)
        cmp = R.compare(c, inc)
        ok = S.decide(cmp, c, base_cost)
        S.log(dict(c, stage="w4", inc=inc["name"], delta_pct=100 * (1 - c["obj"] / inc["obj"]), cmp={k: [round(100 * x, 3) for x in v] for k, v in cmp.items()},
                   decision="PROMOTED" if ok else "not promoted"))
    print("DONE", round(R.gpu_s / 3600, 2), "GPU-hours", flush=True)
