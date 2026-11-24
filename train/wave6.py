"""Wave 6: CPU-only 5-seed rerun of the wave-5 lead patch8 (docs/RESEARCH_LOG.md, wave 6 preregistration). Seeds 0-2 reuse the wave-5
checkpoints of both arms (GPU-trained), seeds 3-4 are trained here for both arms on CPU, so every seed pair is like-for-like.
Same Runner, folds, validation rows, objective and wave-5 rule with K = 5 (ensemble paired rule and seed wins >= K - 1).

  python -m train.wave6 --dev cpu --threads 8 --seeds 0,1,2,3,4 --variants patch8
Resumable: finished fold checkpoints in runs/wave6/ckpt are scored, not retrained.
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train import search as S
from train import wave5 as W
from train.model2 import BMT2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--size", default="t")
    ap.add_argument("--groups", default="ohlc,cal")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--variants", default="patch8")
    ap.add_argument("--out", default="wave6")
    ap.add_argument("--only-train", action="store_true", help="train and save the fold checkpoints of the listed seeds, no comparison")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    S.OUT = os.path.join(D.RUNS, a.out)
    os.makedirs(os.path.join(S.OUT, "ckpt"), exist_ok=True)
    R = S.Runner(a.dev, vram_gb=3.0)
    R.reuse = True
    cfg = S.make_cfg([g for g in a.groups.split(",") if g], a.size)
    seeds = [int(x) for x in a.seeds.split(",")]
    base_cost = BMT2(horizons=S.HZ, **S.make_cfg([])).cost()
    tag = a.groups.replace(",", "+") + ("" if a.size == "s" else f"@{a.size}")
    if a.only_train:
        for vn in ["inc"] + a.variants.split(","):
            for fi in range(4):
                for sd in seeds:
                    W.train_cfg(R, cfg, W.VARIANTS[vn], sd, fi, f"{tag}-{vn}")
                    print("trained", vn, sd, D.FOLDS[fi][0], round(R.gpu_s / 3600, 3), "h", flush=True)
        sys.exit(0)
    inc = W.evaluate(R, cfg, {}, f"{tag}-inc", seeds)
    S.log(dict(inc, stage="w6", seeds=seeds, decision="incumbent"))
    for vn in a.variants.split(","):
        c = W.evaluate(R, cfg, W.VARIANTS[vn], f"{tag}-{vn}", seeds)
        cmp = R.compare(c, inc)
        wins = sum(c["single"][sd] < inc["single"][sd] for sd in seeds)
        ok = S.decide(cmp, c, base_cost) and wins >= len(seeds) - 1
        S.log(dict(c, stage="w6", seeds=seeds, inc=inc["name"], delta_pct=100 * (1 - c["obj"] / inc["obj"]), seed_wins=f"{wins}/{len(seeds)}",
                   single_delta_pct={sd: round(100 * (1 - c["single"][sd] / inc["single"][sd]), 3) for sd in seeds},
                   cmp={k: [round(100 * x, 3) for x in v] for k, v in cmp.items()}, decision="PROMOTED" if ok else "not promoted"))
    print("DONE", round(R.gpu_s / 3600, 2), "train-hours", flush=True)
