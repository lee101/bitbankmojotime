"""Wave 5: seed-robust hyperparameter search on validation rows only. Every config is trained with the same K seeds; the candidate's
K-seed quantile-averaged ensemble is compared with the incumbent's K-seed ensemble on identical rows (paired block bootstrap, the
search's promotion rule), and in addition every seed is compared with its own incumbent seed: a promotion also needs the candidate
to win the objective for at least K - 1 of the K seed pairs. So a lucky seed cannot carry a config.

  python -m train.wave5 --seeds 3 --variants ema999,lr2e-3,wd0.1
Ledger: runs/wave5/ledger.jsonl; checkpoints runs/wave5/ckpt/<variant>-s<seed>_<fold>.pt.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train import search as S
from train.model2 import BMT2
from train.wave4 import Ens

TRAIN = ("steps", "bs", "lr", "wd", "ema_d", "warm", "teacher", "kd")
VARIANTS = {"inc": {}, "ema999": dict(ema_d=0.999), "ema995": dict(ema_d=0.995), "lr2e-3": dict(lr=2e-3), "lr5e-4": dict(lr=5e-4), "wd0.1": dict(wd=0.1),
            "steps3k": dict(steps=3000), "warm100": dict(warm=100), "patch8": dict(patch=8), "layers3": dict(layers=3), "d96": dict(d=96),
            "heads8": dict(heads=8), "steps3k-lr2e-3": dict(steps=3000, lr=2e-3),
            # distillation from the wave-4 4-seed ensemble of the same fold (the teacher saw the same training window only)
            "kd0.5": dict(teacher=":".join(os.path.join(D.RUNS, "wave4", "ckpt", f"ohlc+cal@t-ens4-s{i}") for i in range(4)), kd=0.5),
            "kd0.8": dict(teacher=":".join(os.path.join(D.RUNS, "wave4", "ckpt", f"ohlc+cal@t-ens4-s{i}") for i in range(4)), kd=0.8)}


def split(v):
    return {k: x for k, x in v.items() if k not in TRAIN}, {k: x for k, x in v.items() if k in TRAIN}


def train_cfg(R, cfg, v, seed, fi, name):
    co, tr = split(v)
    c = dict(cfg, **co)
    m = BMT2(horizons=S.HZ, **c).to(R.dev)
    ck = os.path.join(S.OUT, "ckpt", f"{name}-s{seed}_{D.FOLDS[fi][0]}.pt")
    if os.path.exists(ck) and getattr(R, "reuse", False):           # finished by an earlier / parallel run of this wave: score it, do not retrain
        m.load_state_dict(torch.load(ck, map_location=R.dev, weights_only=False)["sd"])
        res = R.val_loss(m, fi)
        return m, dict(obj=R.objective(res), sec=0.0)
    r = R.train_fold(c, fi, tr.get("steps", 1500), seed=seed, bs=tr.get("bs", 1024), lr=tr.get("lr", 1e-3), wd=tr.get("wd", 0.01),
                     ema_d=tr.get("ema_d", 0.998), warm=tr.get("warm", 300), teacher=tr.get("teacher"), kd=tr.get("kd", 0.5), save=f"{name}-s{seed}")
    m.load_state_dict({k: x.to(R.dev) for k, x in r["sd"].items()})
    return m, r


def evaluate(R, cfg, v, name, seeds):
    per, single, sec = {}, {sd: {} for sd in seeds}, 0.0
    for fi in range(4):
        ms = []
        for sd in seeds:
            m, r = train_cfg(R, cfg, v, sd, fi, name)
            single[sd][fi] = r["obj"]
            sec += r["sec"]
            ms.append(m)
        per[fi] = R.val_loss(Ens(ms) if len(ms) > 1 else ms[0], fi)
        del ms
        torch.cuda.empty_cache()
    res = {k: np.concatenate([per[f][k] for f in range(4)]) for k in per[0]}
    co, _ = split(v)
    m0 = BMT2(horizons=S.HZ, **dict(cfg, **co))
    return dict(name=name, variant=v, res=res, obj=float(np.mean([R.objective(per[f]) for f in range(4)])),
                single={sd: float(np.mean(list(single[sd].values()))) for sd in seeds}, sec=sec,
                params=sum(p.numel() for p in m0.parameters()), cost=m0.cost())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="t")
    ap.add_argument("--groups", default="ohlc,cal")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="ema999,lr2e-3,wd0.1,steps3k,patch8,layers3,lr5e-4,warm100,d96")
    ap.add_argument("--out", default="wave5")
    ap.add_argument("--reuse", action="store_true", help="score checkpoints already in <out>/ckpt instead of retraining them")
    a = ap.parse_args()
    S.OUT = os.path.join(D.RUNS, a.out)
    os.makedirs(S.OUT, exist_ok=True)
    R = S.Runner("cuda", vram_gb=3.0)
    R.reuse = a.reuse
    cfg = S.make_cfg([g for g in a.groups.split(",") if g], a.size)
    seeds = list(range(a.seeds))
    base_cost = BMT2(horizons=S.HZ, **S.make_cfg([])).cost()
    tag = a.groups.replace(",", "+") + ("" if a.size == "s" else f"@{a.size}")
    inc = evaluate(R, cfg, {}, f"{tag}-inc", seeds)
    S.log(dict(inc, stage="w5", decision="incumbent"))
    for vn in a.variants.split(","):
        c = evaluate(R, cfg, VARIANTS[vn], f"{tag}-{vn}", seeds)
        cmp = R.compare(c, inc)
        wins = sum(c["single"][sd] < inc["single"][sd] for sd in seeds)
        ok = S.decide(cmp, c, base_cost) and wins >= len(seeds) - 1
        S.log(dict(c, stage="w5", inc=inc["name"], delta_pct=100 * (1 - c["obj"] / inc["obj"]), seed_wins=f"{wins}/{len(seeds)}",
                   single_delta_pct={sd: round(100 * (1 - c["single"][sd] / inc["single"][sd]), 3) for sd in seeds},
                   cmp={k: [round(100 * x, 3) for x in v] for k, v in cmp.items()}, decision="PROMOTED" if ok else "not promoted"))
    print("DONE", round(R.gpu_s / 3600, 2), "GPU-hours", flush=True)
