"""Wave 6 secondary: the 5-seed ensembles of both arms vs the promoted size-s bmt2-ohlc+cal (search checkpoints, single pass) on the
same validation rows; paired block bootstrap. Reported, not decisive."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train import search as S
from train.model2 import BMT2
from train.wave4 import Ens

torch.set_num_threads(8)
R = S.Runner("cpu")
W6 = os.path.join(D.RUNS, "wave6", "ckpt")
PROM = "/nvme0n1-disk/code/bitbankmojotime/runs/search/ckpt"


def load(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    m = BMT2(horizons=S.HZ, **c["cfg"]); m.load_state_dict(c["sd"]); m.eval()
    return m


def score(models_of_fold):
    per = {fi: R.val_loss(models_of_fold(fi), fi) for fi in range(4)}
    res = {k: np.concatenate([per[f][k] for f in range(4)]) for k in per[0]}
    return dict(res=res, obj=float(np.mean([R.objective(per[f]) for f in range(4)])))


arms = {"promoted ohlc+cal (s)": lambda fi: load(f"{PROM}/ohlc+cal_{D.FOLDS[fi][0]}.pt")}
for v in ("inc", "patch8"):
    arms[f"ohlc+cal@t-{v}-ens5"] = (lambda v: lambda fi: Ens([load(f"{W6}/ohlc+cal@t-{v}-s{s}_{D.FOLDS[fi][0]}.pt") for s in range(5)]))(v)
sc = {k: score(f) for k, f in arms.items()}
ref = sc["promoted ohlc+cal (s)"]
out = {}
for k, v in sc.items():
    out[k] = dict(obj=v["obj"], vs_promoted=None if v is ref else {p: [round(100 * x, 3) for x in iv] for p, iv in R.compare(v, ref).items()})
    print(k, json.dumps(out[k]), flush=True)
json.dump(out, open(os.path.join(D.RUNS, "wave6", "vs_promoted.json"), "w"), indent=1)
