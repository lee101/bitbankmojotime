"""Validation-only comparison of a checkpoint ensemble against a single checkpoint (same rows, paired block bootstrap, the search's
promotion rule).   python -m train.ens_val --ckpt-dir runs/synth/ckpt --inc syn-ohlc+cal@t --cand syn-ohlc+cal@t:syn-ohlc+cal@t-s1:...
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train import search as S
from train.model2 import BMT2
from train.wave4 import Ens


def load(R, d, names, fi):
    ms = []
    for n in names:
        c = torch.load(os.path.join(d, f"{n}_{D.FOLDS[fi][0]}.pt"), map_location=R.dev, weights_only=False)
        m = BMT2(horizons=S.HZ, **c["cfg"]).to(R.dev)
        m.load_state_dict(c["sd"])
        ms.append(m)
    return Ens(ms) if len(ms) > 1 else ms[0], sum(p.numel() for p in ms[0].parameters()), ms[0].cost()


def run(R, d, names, label):
    per = {}
    for fi in range(4):
        m, npar, cost = load(R, d, names, fi)
        per[fi] = R.val_loss(m, fi)
    res = {k: np.concatenate([per[f][k] for f in range(4)]) for k in per[0]}
    return dict(name=label, res=res, obj=float(np.mean([R.objective(per[f]) for f in range(4)])), params=npar * len(names), cost=cost * len(names))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--inc", required=True)
    ap.add_argument("--cand", required=True, help="colon-separated members")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="wave4")
    a = ap.parse_args()
    S.OUT = os.path.join(D.RUNS, a.out)
    R = S.Runner("cuda", vram_gb=2.5)
    inc = run(R, a.ckpt_dir, a.inc.split(":"), a.inc)
    c = run(R, a.ckpt_dir, a.cand.split(":"), a.label or a.cand)
    cmp = R.compare(c, inc)
    ok = S.decide(cmp, c, BMT2(horizons=S.HZ, **S.make_cfg([])).cost())
    S.log(dict(c, stage="ens_val", inc=inc["name"], inc_obj=inc["obj"], delta_pct=100 * (1 - c["obj"] / inc["obj"]),
               cmp={k: [round(100 * x, 3) for x in v] for k, v in cmp.items()}, decision="PROMOTED" if ok else "not promoted"))
