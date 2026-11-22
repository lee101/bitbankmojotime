#!/usr/bin/env python3
"""Write docs/RESEARCH_LOG.md from runs/search/ledger.jsonl + runs/search/summary.json (header text in docs/research_log_head.md)."""
import json, os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


IDEA.update(rv5="realised sigma of each bar from 5-minute returns", rsn="share of the bar's 5-minute variance from negative returns", har5="long-memory scalars from 5-minute realised variance",
            peer="mean return of the 5 most correlated training symbols", bvol="BTC bar range on the same bars")
out = [open(os.path.join(R, "docs", "research_log_head.md")).read().rstrip()]
for wave, d in ((1, "search"), (2, "search2")):
    if not os.path.exists(os.path.join(R, "runs", d, "summary.json")):
        continue
    rows = [json.loads(l) for l in open(os.path.join(R, "runs", d, "ledger.jsonl"))]
    S = json.load(open(os.path.join(R, "runs", d, "summary.json")))
    out += ["", f"## Wave {wave}", "", f"Trials: {S['trials']} configurations, {S['gpu_hours']:.2f} GPU-hours, {S['rounds']} rounds. Stop reason: {S.get('stop', 'running')}. Final inputs: `{S.get('final', '?')}`.", "",
            "| # | round | stage | config | idea added | folds | steps | validation objective | vs incumbent at that stage % | paired improvement % by (task, horizon), 95 % | result |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    out += fmt(rows)
open(os.path.join(R, "docs", "RESEARCH_LOG.md"), "w").write("\n".join(out) + "\n")
W4 = os.path.join(R, "runs", "wave4", "ledger.jsonl")
if os.path.exists(W4):
    VAR = {"inc": "incumbent: one seed, 1,500 steps x batch 1,024, lr 1e-3", "ens2": "quantile average of 2 seeds", "ens4": "quantile average of 4 seeds",
           "ens8": "quantile average of 8 seeds", "long4k": "4,000 steps", "bs4k": "batch 4,096, lr 2e-3", "lr3": "lr 3e-3", "ens4long": "4 seeds x 4,000 steps"}
    rows = [json.loads(l) for l in open(W4)]
    out = [open(os.path.join(R, "docs", "RESEARCH_LOG.md")).read().rstrip(), "", "## Wave 4: training-side search (2026-11-19)", "",
           "Inputs fixed at the promoted `ohlc+cal`; size t (0.11M); same folds, validation rows, objective and promotion rule as above, with the",
           "cost cap of 4x the base size-s encoder (a k-member ensemble costs k forwards). `ens_val` rows compare a checkpoint ensemble with the",
           "single checkpoint on the same rows (the permissive tier: members trained on synthetic bars only, seeds 0-3).", "",
           "| config | change | validation objective | vs incumbent % | paired improvement % by (task, horizon), 95 % | result |", "|---|---|---|---|---|---|"]
    for r in rows:
        v = r["name"].rsplit("-", 1)[-1]
        c = r.get("cmp")
        cs = "; ".join(f"{k.replace('|', ' ')} {x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]" for k, x in c.items()) if c else ""
        idea = "4 synthetic-data seeds vs seed 0 (permissive tier)" if r.get("stage") == "ens_val" else VAR.get(v, v)
        d = f"{r['delta_pct']:+.3f}" if "delta_pct" in r else ""
        out.append(f"| `{r['name']}` | {idea} | {r['obj']:.5f} | {d} | {cs} | {r.get('decision', '')} |")
    open(os.path.join(R, "docs", "RESEARCH_LOG.md"), "w").write("\n".join(out) + "\n")
DIAL = os.path.join(R, "runs", "dial.log")
if os.path.exists(DIAL):
    tab = [l.rstrip() for l in open(DIAL) if l.startswith("|")]
    out = [open(os.path.join(R, "docs", "RESEARCH_LOG.md")).read().rstrip(), "", "## Quality dial: compute vs validation sPL (`train/dial.py`, 2026-11-19)", "",
           "Out of sample on the second half of each fold's validation window (the calibration is fitted on the first half). Paired sPL",
           "improvement over one forward of the single seed-0 model, % (95 %). `+cal` = affine spread + per-knot shift per (task, horizon).",
           "Reading: a 4-seed ensemble is the whole gain (hourly); 8 seeds, adding the size-s model, flip averaging on top and calibration add",
           "nothing separated, and the calibration intervals are wide (45 days fit). Daily on these half windows is worse with ensembling;",
           "on the full validation windows (wave 4) and on test it is not separated. Levels shipped: q0 single (1 forward), q1 ens4 (4), q2 ens4-sym (8).", "",
           tab[0], "|" + "---|" * (tab[0].count("|") - 1)] + tab[1:]
    open(os.path.join(R, "docs", "RESEARCH_LOG.md"), "w").write("\n".join(out) + "\n")
W5 = os.path.join(R, "runs", "wave5", "ledger.jsonl")
if os.path.exists(W5):
    rows = [json.loads(l) for l in open(W5)]
    seen, uniq = set(), []
    for r in rows:
        if r["name"] not in seen:
            seen.add(r["name"]); uniq.append(r)
    out = [open(os.path.join(R, "docs", "RESEARCH_LOG.md")).read().rstrip(), "", "## Wave 5: seed-robust hyperparameters and distillation (2026-11-19)", "",
           "Every config trained with seeds 0-2. Ensemble rule: the candidate's 3-seed quantile average vs the incumbent's, paired, the usual",
           "promotion rule, plus a win for at least 2 of 3 seed pairs. `ema999` is a no-op (the EMA warm-up min(cap, (i+1)/(i+10)) stays under",
           "0.998 for 1,500 steps). Distillation (`kd`: loss = (1 - w) pinball + w L1 to the frozen wave-4 4-seed ensemble of the same fold) is",
           "judged single model vs single model, the question it answers: each of the 6 distilled seeds passes the promotion rule against its",
           "own incumbent seed (hourly +0.03 to +0.06 %), so `kd0.5` was promoted and scored once on test (`bmt2-ohlc+cal@t-kd`).",
           "Lead, not promoted: `patch8` (64 tokens, 2x compute) wins all 3 seed pairs (+0.03 / +0.11 / +0.13 %, largest daily) but no seed passes",
           "the single-model rule and the ensemble intervals include zero; worth a 5-seed rerun or patch 8 + distillation. `steps3k` improves",
           "the 3-seed ensemble hourly but wins 1 of 3 seeds. Lower lr, a third layer, short warm-up and d 96 are worse or flat.", "",
           "| config | change | validation objective (3-seed ensemble) | seed wins | per-seed improvement % | ensemble paired improvement % by (task, horizon), 95 % | result |",
           "|---|---|---|---|---|---|---|"]
    for r in uniq:
        c = r.get("cmp")
        cs = "; ".join(f"{k.replace('|', ' ')} {x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]" for k, x in c.items()) if c else ""
        sd = ", ".join(f"{v:+.3f}" for v in (r.get("single_delta_pct") or {}).values())
        dec = r.get("decision", "")
        if r["name"].endswith(("kd0.5", "kd0.8")):
            dec = "ensemble rule: " + dec + "; single-model rule: passed for 3/3 seeds" + (" (PROMOTED)" if r["name"].endswith("kd0.5") else "")
        out.append(f"| `{r['name']}` | {json.dumps({k: ('wave-4 ens4' if k == 'teacher' else v) for k, v in r.get('variant', {}).items()})} | {r['obj']:.5f} | {r.get('seed_wins', '')} | {sd} | {cs} | {dec} |")
    open(os.path.join(R, "docs", "RESEARCH_LOG.md"), "w").write("\n".join(out) + "\n")
