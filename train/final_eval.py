"""Score promoted search configs on the unseen test folds from the checkpoints the search trained (no retraining).

  python -m train.final_eval --names base,ohlc+har        ->  runs/preds/<task>/bmt2-<name>[-sym].npy
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bmtbench import data as D
from train.search import Runner, OUT, HZ
from train.model2 import BMT2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--ckpt-dir", default=os.path.join(OUT, "ckpt"))
    ap.add_argument("--placebo-seed", type=int, default=None)
    ap.add_argument("--use5m", action="store_true")
    ap.add_argument("--prefix", default="bmt2-")
    ap.add_argument("--tier", default="research")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    R = Runner("cuda", a.placebo_seed, vram_gb=2.5, use5m=a.use5m)
    b = R.bank
    for name in a.names.split(","):
        for task, k in (("hourly", 1), ("daily", 24)):
            rows = D.load_rows(task)
            ai = np.array([b.ix[s] for s in rows["syms"]])[rows["a"]]
            tb = (rows["t_abs"] - b.t0) if task == "hourly" else rows["t_abs"] * 24 + 23 - b.t0
            hs = [HZ.index(h) for h in D.TASKS[task]["horizons"]]
            q, qs, meta = np.zeros((len(tb), len(hs), 21), np.float32), np.zeros((len(tb), len(hs), 21), np.float32), {}
            paths = [os.path.join(a.ckpt_dir, f"{name}_{fn}.pt") for fn, _, _ in D.FOLDS]
            cfg0 = torch.load(paths[0], map_location="cpu", weights_only=False)["cfg"]
            mh = D.model_hash(cfg0, paths)                           # the row name carries the content hash of config + checkpoints
            row = f"{a.prefix}{name}~{mh[:8]}"
            for fi, (fn, s_abs, _) in enumerate(D.FOLDS):
                c = torch.load(os.path.join(a.ckpt_dir, f"{name}_{fn}.pt"), map_location="cuda", weights_only=False)
                assert c["limit_abs"] <= s_abs - 168 and c["fold"] == fn
                m = BMT2(horizons=HZ, **c["cfg"]).cuda()
                m.load_state_dict(c["sd"])
                i = np.flatnonzero(rows["fold"] == fi)
                q[i] = R.predict(m, ai[i], tb[i], k, fold=fi)[:, hs].cpu().numpy()
                qs[i] = R.predict(m, ai[i], tb[i], k, sym=True, fold=fi)[:, hs].cpu().numpy()
                meta = dict(params=sum(p.numel() for p in m.parameters()), cfg=c["cfg"], steps=c["steps"])
            if a.placebo_seed is None:
                info = dict(model_hash=mh, ckpt_sha256={os.path.basename(p): D.file_sha(p) for p in paths}, cfg=cfg0, steps=meta["steps"], tier=a.tier,
                            params=f"{meta['params'] / 1e6:.2f}M" + a.note)
                for rn, pr, np_ in ((row, q, 1), (row.replace("~", "-sym~"), qs, 2)):
                    D.save_pred(task, rn, pr, dict(meta, passes=np_, model_hash=mh))
                    D.register(task, rn, passes=np_, **info)
            print(row, task, "done", flush=True)
