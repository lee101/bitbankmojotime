#!/usr/bin/env python3
"""Print the search ledger compactly."""
import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else "runs/search/ledger.jsonl"
for l in open(p):
    r = json.loads(l)
    c = r.get("cmp")
    cs = " ".join(f"{k.split('|')[0][0]}{k.split('|')[1]}:{v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]" for k, v in c.items()) if c else ""
    print(r.get("trial"), r.get("round"), r.get("stage"), r["name"], "".join(x[1] for x in r.get("folds", [])), f"{r.get('obj', 0):.5f}",
          f"d%{r['delta_pct']:+.3f}" if "delta_pct" in r else "", f"{r.get('sec', 0):.0f}s", r.get("decision", ""), cs)
