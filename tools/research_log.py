#!/usr/bin/env python3
"""Write docs/RESEARCH_LOG.md from runs/search/ledger.jsonl + runs/search/summary.json (header text in docs/research_log_head.md)."""
import json, os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.get("BMT_RUNS", os.path.join(R, "runs"))
OUT = os.environ.get("BMT_REPORT_DIR", os.path.join(R, "docs"))
IDEA = {"ohlc": "bar range (log high/low) and Garman-Klass sigma per bar", "vol": "quote volume per bar", "tb": "taker-buy share per bar", "rvf": "realised sigma of each bar from hourly returns",
        "har": "long-memory scalars: RMS of hourly returns over 1 d / 1 w / 30 d / 90 d / 250 d", "vov": "vol-of-vol scalar (std of log daily RV, 30 d)",
        "coarse": "second token stream: 256 daily bars (return, realised sigma)", "coarse4": "daily token stream with return, realised sigma, range, volume",
        "btc": "BTC returns on the same bars", "eth": "ETH returns on the same bars", "mkt": "market state: pool mean return, cross-sectional dispersion, breadth",
        "cal": "calendar: start hour (covers funding / session times), weekday, bar size", "base": "returns only (the phase-2 model in the new pipeline)"}
def fmt(rows):
    out = []
    for r in rows:
        g = r.get("groups", [])
        idea = IDEA.get(g[-1], g[-1]) if g else IDEA["base"]
        if r.get("stage") == "lc":
            idea = f"learning curve: size {r.get('size')}, {r.get('frac', 1):.0%} of training symbols"
        elif r.get("trial") == 0:
            idea = "starting incumbent"
        c = r.get("cmp")
        cs = "; ".join(f"{k.replace('|', ' ')} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for k, v in c.items()) if c else ""
        res = r.get("decision", {"s1": "screen", "s2": "screen", "lc": "curve point"}.get(r.get("stage"), ""))
        d = f"{r['delta_pct']:+.3f}" if "delta_pct" in r else ""
        out.append(f"| {r.get('trial', '')} | {r.get('round', '')} | {r.get('stage', '')} | `{r['name']}` | {idea} | {' '.join(r.get('folds', []))} | {r.get('steps', '')} | "
                   f"{r.get('obj', float('nan')):.5f} | {d} | {cs} | {res} |")
    return out


IDEA.update(ic10="ranking loss: within-origin correlation of median and outcome, weight 0.1", ic30="ranking loss, weight 0.3", rv5="realised sigma of each bar from 5-minute returns", rsn="share of the bar's 5-minute variance from negative returns", har5="long-memory scalars from 5-minute realised variance",
            peer="mean return of the 5 most correlated training symbols", bvol="BTC bar range on the same bars")
hp = os.path.join(OUT, "research_log_head.md")
out = [open(hp if os.path.exists(hp) else os.path.join(R, "docs", "research_log_head.md")).read().rstrip()]
for wave, d in ((1, "search"), (2, "search2"), (3, "search3")):
    if not os.path.exists(os.path.join(RUNS, d, "summary.json")):
        continue
    rows = [json.loads(l) for l in open(os.path.join(RUNS, d, "ledger.jsonl"))]
    S = json.load(open(os.path.join(RUNS, d, "summary.json")))
    out += ["", f"## Wave {wave}", "", f"Trials: {S['trials']} configurations, {S['gpu_hours']:.2f} GPU-hours, {S['rounds']} rounds. Stop reason: {S.get('stop', 'running')}. Final inputs: `{S.get('final', '?')}`.", "",
            "| # | round | stage | config | idea added | folds | steps | validation objective | vs incumbent at that stage % | paired improvement % by (task, horizon), 95 % | result |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    out += fmt(rows)
open(os.path.join(OUT, "RESEARCH_LOG.md"), "w").write("\n".join(out) + "\n")
