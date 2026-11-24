"""Render bench/llmfc/RESULTS.md tables from score_sub outputs and the llm-*.json metas.  python results.py"""
import json, os
import numpy as np
from common import *
from score_sub import score

f3 = lambda x: f"{x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]"
f4 = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"


def meta(name):
    p = os.path.join(WT_PREDS, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def table(models, rows, r, title):
    res = score(models, rows, r)
    out = [f"\n### {title} ({len(rows)} rows)\n", "| model | h | sPL [95 %] | paired dsPL vs bmt2 | paired dsPL vs naive | MASE [95 %] | IC | cov 90 % |", "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for h in list(HOR) + ["mean"]:
            k = f"{m}|h{h}"
            if k not in res:
                continue
            s = res[k]
            d1 = f4(s[f"dsPL_{BMT2}"]) if f"dsPL_{BMT2}" in s else "ref"
            d2 = f4(s["dsPL_naive"]) if "dsPL_naive" in s else "ref"
            if h == "mean":
                out.append(f"| {m} | mean | {f3(s['sPL'])} | {d1} | {d2} | | | |")
            else:
                out.append(f"| {m} | {h} | {f3(s['sPL'])} | {d1} | {d2} | {f3(s['mase'])} | {s['ic']:.4f} | {s['c90'][0]*100:.1f} |")
    return "\n".join(out), res


if __name__ == "__main__":
    r = D.load_rows(TASK)
    sub = json.load(open(SUB))
    llm_full = [m for m in ("llm-flash", "llm-flash-refine") if os.path.exists(os.path.join(WT_PREDS, m + ".npy"))]
    llm_sec = llm_full + [m for m in ("llm-sol", "llm-terra", "llm-sol-refine") if os.path.exists(os.path.join(WT_PREDS, m + ".npy"))]
    md = ["## Run summary\n", "| arm | model | effort | rows called | parsed | failure rate | tokens in / out | cost USD (peak prices) |", "|---|---|---|---|---|---|---|---|"]
    for m in ("llm-flash", "llm-flash-refine", "llm-sol", "llm-terra", "llm-sol-refine"):
        t = meta(m)
        if t:
            md.append(f"| {m} | {t['model']} | {t['reasoning_effort']} | {t['n_done']} | {t['n_ok']} | {t['fail_rate']*100:.1f} % | {t['tokens_in']} / {t['tokens_out']} | {t['cost_usd']:.3f} |")
    md.append(f"\nTotal logged spend for the slot (spend.jsonl, incl. pilots and debug calls): USD {spent():.3f}.")
    t1, res_full = table(REF_MODELS + llm_full, sub["rows"], r, "Preregistered 600-row subsample")
    t2, res_sec = table(REF_MODELS + llm_sec, sub["secondary_rows"], r, "Secondary 150-row subset")
    md += [t1, t2]
    json.dump(dict(full=res_full, secondary=res_sec), open("scores_sub.json", "w"))
    open("results_tables.md", "w").write("\n".join(md))
    print("\n".join(md))
