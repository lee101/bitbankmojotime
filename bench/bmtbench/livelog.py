"""Post-release score of the LIVE Timestral-209M hourly pair forecasts (timestral-serve run/pair_forecasts.jsonl, 2026-11-04 ..; every row is
after the checkpoint release 2026-10-15, so none can be in its pretraining corpus).  python -m bmtbench.livelog [JSONL] [CLOSES]
Row = (record, pair): forecast_for_hour_starting H, last_close = close at H (closes.json key H), 21 quantiles of y = log(close[H + 1h] /
last_close). References on the same rows from the same closes (trailing window up to 512 h ending at H, >= 168 h): naive = zero-mean
Gaussian with the window SD (protocol 'naive'), empirical = the window's empirical quantiles. Scale per pair = mean |r| over the closes
before the first forecast (pre-evaluation). Paired bootstrap over 24 h calendar blocks (Boot, 1000 resamples). IC = mean over hours of the
cross-pair Spearman of the median vs y (>= 10 pairs)."""
import json, os, sys
import numpy as np
from .data import QLEV
from .metrics import Boot, ci, ic, pinball_levels
from scipy.stats import norm

J = sys.argv[1] if len(sys.argv) > 1 else "/nvme0n1-disk/code/timestral-serve/run/pair_forecasts.jsonl"
CL = sys.argv[2] if len(sys.argv) > 2 else "/nvme0n1-disk/code/timestral-serve/run/closes.json"
closes = {p: {int(k): float(v) for k, v in d.items()} for p, d in json.load(open(CL)).items()}
recs = [json.loads(l) for l in open(J) if l.strip()]
t_first = min(v["forecast_for_hour_starting"] for r in recs for v in r["f"].values())
rows = {}
for r in recs:                                                       # last record per (pair, H) wins (re-forecasts of the same hour)
    for p, v in r["f"].items():
        rows[(p, v["forecast_for_hour_starting"])] = (r["model"], v.get("passes", 1), v["last_close"], np.array(v["q_logret"], float))
Q, Y, A, H, M, PS, QN, QE, SC = [], [], [], [], [], [], [], [], []
pairs = sorted({p for p, _ in rows}); pix = {p: i for i, p in enumerate(pairs)}
for (p, h), (model, passes, lc, q) in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    c = closes.get(p, {}); nxt = c.get(h + 3600)
    if nxt is None or len(q) != 21: continue
    ks = np.array(sorted(k for k in c if k <= h)); ks = ks[-513:]
    if len(ks) < 169 or not np.all(np.diff(ks) == 3600): continue
    px = np.array([c[k] for k in ks]); rr = np.diff(np.log(px))
    pre = np.array([c[k] for k in sorted(c) if k <= t_first]); scale = np.mean(np.abs(np.diff(np.log(pre)))) if len(pre) > 24 else np.mean(np.abs(rr))
    Q.append(np.sort(q)); Y.append(np.log(nxt / lc)); A.append(pix[p]); H.append(h); M.append(model); PS.append(passes); SC.append(scale)
    QN.append(norm.ppf(QLEV) * rr.std()); QE.append(np.quantile(rr, QLEV))
Q, Y, A, H, QN, QE, SC, PS = map(np.array, (Q, Y, A, H, QN, QE, SC, PS)); M = np.array(M)
def rep(mask, lab):
    if mask.sum() < 100: print(f"{lab}: {mask.sum()} rows, skipped"); return
    b = Boot(A[mask], H[mask] // 86400); out = {}
    for nm, q in (("timestral-live", Q), ("naive", QN), ("empirical", QE)):
        out[nm] = b.asset_mean(pinball_levels(q[mask], Y[mask]).mean(1) / SC[mask])
    print(f"{lab}: rows {mask.sum()}, pairs {len(np.unique(A[mask]))}, hours {len(np.unique(H[mask]))}, "
          f"{np.datetime64(int(H[mask].min()), 's')} .. {np.datetime64(int(H[mask].max()), 's')}")
    for nm in out:
        s = ci(out[nm]); line = f"  {nm:15s} sPL {s[0]:.4f} [{s[1]:.4f}, {s[2]:.4f}]"
        if nm != "timestral-live": sk = ci(1 - out["timestral-live"] / out[nm]); line += f" | Timestral skill vs it {100 * sk[0]:+.2f}% [{100 * sk[1]:+.2f}, {100 * sk[2]:+.2f}]"
        print(line)
    med = Q[mask][:, 10]; icv = ic(med, Y[mask], H[mask])
    per = []
    for hh in np.unique(H[mask]):
        k = mask & (H == hh)
        if k.sum() >= 10 and Q[k][:, 10].std() > 0: per.append(np.corrcoef(np.argsort(np.argsort(Q[k][:, 10])), np.argsort(np.argsort(Y[k])))[0, 1])
    per = np.array(per); print(f"  median rank IC {icv:+.4f} (t {per.mean() / per.std() * np.sqrt(len(per)):+.2f}, {len(per)} hours, iid-hour t)")
print("Timestral-209M live hourly pair forecasts, post-release only (checkpoint released 2026-10-15)")
rep(np.ones(len(Y), bool), "all rows")
for p_ in sorted(set(PS.tolist())): rep(PS == p_, f"passes {p_}")
for m_ in sorted(set(M.tolist())): rep(M == m_, f"model {m_}")
