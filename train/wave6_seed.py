"""Wave 6 secondary: per-seed paired intervals, patch8 vs incumbent, for the CPU-trained seeds."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train import search as S
from train.model2 import BMT2

torch.set_num_threads(8)
R = S.Runner("cpu")
W6 = os.path.join(D.RUNS, "wave6", "ckpt")


def arm(v, sd):
    per = {}
    for fi in range(4):
        c = torch.load(f"{W6}/ohlc+cal@t-{v}-s{sd}_{D.FOLDS[fi][0]}.pt", map_location="cpu", weights_only=False)
        m = BMT2(horizons=S.HZ, **c["cfg"]); m.load_state_dict(c["sd"]); m.eval()
        per[fi] = R.val_loss(m, fi)
    return dict(res={k: np.concatenate([per[f][k] for f in range(4)]) for k in per[0]}, obj=float(np.mean([R.objective(per[f]) for f in range(4)])))


out = {}
for sd in (3, 4):
    a, b = arm("patch8", sd), arm("inc", sd)
    out[sd] = dict(obj_patch8=a["obj"], obj_inc=b["obj"], cmp={k: [round(100 * x, 3) for x in v] for k, v in R.compare(a, b).items()})
    print(sd, json.dumps(out[sd]), flush=True)
json.dump(out, open(os.path.join(D.RUNS, "wave6", "per_seed.json"), "w"), indent=1)
