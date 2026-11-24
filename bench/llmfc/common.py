import os, sys, json
import numpy as np
LIVE = "/nvme0n1-disk/code/bitbankmojotime"
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BMT_RUNS", os.path.join(LIVE, "runs"))
os.environ.setdefault("BMT_DATA", os.path.join(LIVE, "data", "cache"))
sys.path.insert(0, os.path.join(LIVE, "bench"))
import bmtbench.data as D
from bmtbench.metrics import Boot, summarize, ci
TASK = "hourly"
HOR = D.TASKS[TASK]["horizons"]
SUB = os.path.join(WT, "bench", "llmfc", "subsample.json")
WT_PREDS = os.path.join(WT, "runs", "preds", TASK)
SPEND = os.path.expanduser("~/research_data/llm_spend/spend.jsonl")
CAP_USD = 5.0
SEED = 20261124
REF_MODELS = ["naive", "empirical", "chronos-2", "chronos-2-ret", "timestral-fast", "timestral-max", "timestral-official", "bmt2-ohlc+cal-sym", "bmt2-ohlc+cal@t-kd"]
BMT2 = "bmt2-ohlc+cal-sym"


def load_pred(model):
    for base in (WT_PREDS, os.path.join(LIVE, "runs", "preds", TASK)):
        p = os.path.join(base, model + ".npy")
        if os.path.exists(p):
            return np.load(p, mmap_mode="r")
    raise FileNotFoundError(model)


def spent():
    if not os.path.exists(SPEND):
        return 0.0
    return sum(json.loads(l)["est_cost_usd"] for l in open(SPEND) if l.strip())
