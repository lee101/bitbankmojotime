"""LLM quantile forecaster over the openpaths gateway on the preregistered subsample.
  python llm.py --model deepseek-v4-flash --name llm-flash [--refine] [--rows secondary] [--limit N] [--dry]
Writes runs/preds/hourly/<name>.npy (naive rows outside the subsample and for failures), <name>.json (failures, cost)."""
import argparse, json, os, re, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from common import *

URL = "https://openpaths.io/v1/chat/completions"
PRICE = {"deepseek-v4-flash": (0.30, 1.20), "gpt-6.1-terra": (1.00, 5.00), "gpt-6.1-sol": (2.00, 10.00)}   # USD per 1M in/out, peak prices for flash
CTX = 256
LEV = [float(x) for x in D.QLEV]
LOCK = threading.Lock()
TOTAL = [spent()]

SYS = "You are a quantitative forecaster. Answer with one JSON object only, no prose, no markdown."


def prompt(ret_bp, hint=None):
    s = " ".join(str(int(x)) for x in ret_bp)
    p = (f"Below are the last {len(ret_bp)} consecutive 1-hour log returns of a liquid traded asset, oldest first, in basis points "
         f"(1 bp = 0.0001 log return). The asset, venue and dates are withheld.\n\nRETURNS: {s}\n\n"
         f"Forecast the cumulative log return over the next 1, 6 and 24 bars after the last bar. For each horizon give the 21 quantiles "
         f"at levels {LEV} in basis points, as a JSON object of the exact form "
         '{"1": [q01, q05, ..., q99], "6": [...], "24": [...]} with 21 numbers per list in non-decreasing order. '
         "Calibration matters: the quantiles will be scored with the pinball loss against the realised returns.")
    if hint is not None:
        h = {str(k): [int(round(x)) for x in v] for k, v in hint.items()}
        p += ("\n\nA specialised purpose-trained forecasting model produced these quantiles (bp) for the same horizons: "
              f"{json.dumps(h, separators=(',', ':'))}. Use them as a strong prior and refine them where you believe the series justifies it; "
              "output the final quantiles in the same JSON form.")
    return p


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no json")
    o = json.loads(m.group(0))
    out = []
    for h in HOR:
        v = o[str(h)]
        if not isinstance(v, list) or len(v) != 21:
            raise ValueError(f"h{h} len")
        v = np.array(v, dtype=np.float64)
        if not np.isfinite(v).all() or np.abs(v).max() > 1e5:
            raise ValueError(f"h{h} values")
        out.append(np.sort(v))
    return np.stack(out) / 1e4


def call(model, msgs, key, effort="low", max_tokens=2500, timeout=180):
    body = json.dumps(dict(model=model, messages=msgs, routing_strategy="config", reasoning_effort=effort, max_tokens=max_tokens, temperature=0)).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.load(r)
    u = j.get("usage", {})
    pi, po = PRICE[model]
    tin, tout = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
    cost = (tin * pi + tout * po) / 1e6
    with LOCK:
        TOTAL[0] += cost
        with open(SPEND, "a") as f:
            f.write(json.dumps(dict(ts=time.time(), slot="PMLLMFC", model=model, tokens_in=tin, tokens_out=tout, est_cost_usd=round(cost, 6),
                                    running_total_usd=round(TOTAL[0], 4), latency_s=round(time.time() - t0, 1))) + "\n")
    return j["choices"][0]["message"]["content"] or "", tin, tout, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--refine", action="store_true"); ap.add_argument("--rows", default="rows")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    key = os.environ.get("OPENPATHS_API_KEY")
    if not key and not a.dry:
        raise SystemExit("no OPENPATHS_API_KEY in env")
    r = D.load_rows(TASK)
    sub = json.load(open(SUB))
    idx = np.array(sub["secondary_rows" if a.rows == "secondary" else "rows"])
    if a.limit:
        idx = idx[:a.limit]
    u = D.universe(); p = D.load_hourly(u["eval"])
    ctx = D.ctx_ret(p, r["a"][idx], r["t"][idx], CTX)
    ctx_bp = np.rint(np.clip(ctx, -0.3, 0.3) * 1e4)
    hint = np.asarray(load_pred(BMT2)[idx]) * 1e4 if a.refine else None
    naive = np.asarray(load_pred("naive"))
    out_p = os.path.join(WT_PREDS, a.name + ".npy")
    meta_p = out_p[:-4] + ".json"
    Q = np.array(naive, dtype=np.float32, copy=True)
    meta = dict(model=a.model, refine=a.refine, rows=a.rows, ctx=CTX, reasoning_effort="low", results={})
    if os.path.exists(meta_p):
        old = json.load(open(meta_p)); meta["results"] = old.get("results", {})
        if os.path.exists(out_p):
            Q = np.load(out_p)
    if a.dry:
        pr = prompt(ctx_bp[0], None if hint is None else {h: hint[0, j] for j, h in enumerate(HOR)})
        print(pr); print("chars", len(pr), "est tokens ~", len(pr) // 3)
        return

    def one(k):
        i = int(idx[k])
        if str(i) in meta["results"] and meta["results"][str(i)]["ok"]:
            return
        if TOTAL[0] >= CAP_USD:
            return
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt(ctx_bp[k], None if hint is None else {h: hint[k, j] for j, h in enumerate(HOR)})}]
        rec = dict(ok=False, tries=0, tin=0, tout=0, cost=0.0, err=None)
        for attempt in range(2):
            if TOTAL[0] >= CAP_USD:
                rec["err"] = "cap"; break
            rec["tries"] += 1
            try:
                txt, tin, tout, cost = call(a.model, msgs, key)
                rec["tin"] += tin; rec["tout"] += tout; rec["cost"] += cost
                q = parse(txt)
                with LOCK:
                    Q[i] = q.astype(np.float32)
                rec["ok"] = True; rec["err"] = None
                break
            except urllib.error.HTTPError as e:
                rec["err"] = f"http {e.code} {e.read()[:200]!r}"
                time.sleep(3)
            except Exception as e:
                rec["err"] = f"{type(e).__name__}: {str(e)[:120]}"
        with LOCK:
            meta["results"][str(i)] = rec
            n = len(meta["results"])
            if n % 25 == 0:
                print(f"{n}/{len(idx)} ok={sum(v['ok'] for v in meta['results'].values())} spent={TOTAL[0]:.3f}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, range(len(idx))))
    res = meta["results"]
    meta.update(n=len(idx), n_done=len(res), n_ok=sum(v["ok"] for v in res.values()), fail_rate=1 - sum(v["ok"] for v in res.values()) / max(len(res), 1),
                tokens_in=sum(v["tin"] for v in res.values()), tokens_out=sum(v["tout"] for v in res.values()), cost_usd=sum(v["cost"] for v in res.values()),
                running_total_usd=TOTAL[0])
    np.save(out_p, np.sort(Q, axis=2))
    json.dump(meta, open(meta_p, "w"))
    print({k: v for k, v in meta.items() if k != "results"})


if __name__ == "__main__":
    main()
