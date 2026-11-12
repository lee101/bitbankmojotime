"""Retrain single folds of a search config with the search's seeds and save the checkpoints (used once: the first search run cached
the folds trained at stage 2 without writing them).   python -m train.refit --configs ohlc,ohlc+cal --folds F1,F3"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train import search as S

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--folds", default="F1,F3")
    ap.add_argument("--use5m", action="store_true")
    ap.add_argument("--out", default="search")
    a = ap.parse_args()
    S.OUT = os.path.join(D.RUNS, a.out)
    R = S.Runner("cuda", vram_gb=2.5, use5m=a.use5m)
    for name in a.configs.split(","):
        groups = [g for g in name.split("+") if g != "base"]
        for fi, f in enumerate(D.FOLDS):
            if f[0] in a.folds.split(","):
                r = R.train_fold(S.make_cfg(groups), fi, 1500, save=name)
                rec = dict(name=name, fold=f[0], stage="refit", obj=r["obj"], sec=r["sec"])
                print(json.dumps(rec), flush=True)
                with open(os.path.join(S.OUT, "refit.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
