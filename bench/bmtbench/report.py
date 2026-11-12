"""Assemble bench/LEADERBOARD.md from runs/ (scores, paired extras, placebo, learning curve, speed).  python -m bmtbench.report"""
import json, os
from . import data as D
from .score import tables

R = D.RUNS
HEAD = open(os.path.join(D.REPO, "bench", "leaderboard_head.md")).read()


def f3(x, k=100, d=2):
    return f"{k * x[0]:+.{d}f} [{k * x[1]:+.{d}f}, {k * x[2]:+.{d}f}]"


def speed_md():
    s = json.load(open(os.path.join(R, "speed.json")))
    names = {"bmt-mojo-gpu": "bmt-s, Mojo GPU (fp32)", "bmt-mojo-gpu:sym": "bmt-s-sym (2 passes), Mojo GPU", "bmt-mojo-gpu:t": "bmt-t, Mojo GPU (fp32)",
             "bmt-torch": "bmt-s, PyTorch fp32", "bmt-mojo-cpu": "bmt-s, Mojo CPU", "bmt-mojo-cpu@12": "bmt-s, Mojo CPU", "bmt-torch-cpu@12": "bmt-s, PyTorch fp32",
             "bmt-mojo-cpu:t": "bmt-t, Mojo CPU", "bmt-mojo-cpu@12:t": "bmt-t, Mojo CPU", "timestral-fast": "Timestral-209M fast (tsfm16 kernel, ctx 256, 1 horizon)",
             "timestral-max": "Timestral-209M max (tsfm16, flip-averaged ctx 512, 1 horizon)", "timestral-official": "Timestral-209M reference PyTorch (ctx 512, h 24, flip-averaged)",
             "chronos-2": "Chronos-2 reference PyTorch (ctx 512, h 24)", "chronos-2-cpu@12": "Chronos-2 reference PyTorch", "timestral-official-cpu@12": "Timestral-209M reference PyTorch"}
    out = ["| model / path | device | batch 1: rows/s, p50 / p99 ms | batch 64: rows/s, p50 / p99 ms | batch 4096: rows/s, p50 / p99 ms | VRAM MiB | binary | weights | other tenants' GPU load at start |",
           "|---|---|---|---|---|---|---|---|---|"]
    mb = lambda v: "-" if not isinstance(v, int) else (f"{v / 1e6:.1f} MB" if v > 2e6 else f"{v / 1e3:.0f} KB")
    for k, r in s.items():
        b = r["batches"]
        g = lambda x: f"{b[x]['rows_s']:,.0f}, {b[x]['p50_ms']:.2f} / {b[x]['p99_ms']:.2f}" if x in b else "not run (minutes per call)"
        nm = names.get(k, (k.split(":")[1] + (", Mojo GPU (fp32)" if "gpu" in k else ", Mojo CPU")) if ":" in k and k.split(":")[1].startswith("bmt2") else k)
        out.append(f"| {nm} | {r.get('device', '')} | {g('1')} | {g('64')} | {g('4096')} | {r.get('vram_mib') or '-'} | {mb(r.get('size_bytes'))} | {mb(r.get('weights_bytes'))} | "
                   f"{r['before']['gpu_util_pct']} %, {r['before']['gpu_mem_used_mib'] / 1024:.1f} GB |")
    out += ["", "Batch pipelines of the statistical baselines (whole benchmark run, numpy / numba, not a latency measurement):", "",
            "| model | rows/s | note |", "|---|---|---|"]
    for m, note in (("naive", "1 process"), ("ewma", "1 process"), ("ewma-seasonal", "1 process"), ("garch", "1 process, includes 248 fits"),
                    ("autoets", "10 processes, refit every 30 days"), ("autoarima", "10 processes, refit every 30 days")):
        j = json.load(open(D.pred_path("hourly", m)[:-4] + ".json"))
        out.append(f"| {m} | {j['rows'] / j['seconds']:,.0f} | {note} |")
    return "\n".join(out)


def lc_md():
    rows = [json.loads(l) for l in open(os.path.join(R, "lc.jsonl"))]
    keep = {}
    for r in rows:
        n = r["name"]
        if r["fold"] == "F4" and not r.get("placebo") and (n.startswith("lc-") or n in ("bmt-t", "bmt-s", "bmt-m")):
            keep[(r["size"], r["frac"])] = r
    out = ["| size | parameters | share of training symbols | symbols | validation score | test sPL h1 / h6 / h24 (row-pooled, F4) |", "|---|---|---|---|---|---|"]
    for z in "tsm":
        for fr in (0.05, 0.25, 1.0):
            r = keep.get((z, fr))
            if r:
                out.append(f"| {z} | {r['params']:,} | {fr:.0%} | {r.get('assets', '-')} | {r['val_sPL']:.4f} | " + " / ".join(f"{x:.4f}" for x in r["test_sPL_hourly"]) + " |")
    return "\n".join(out)


def placebo_md():
    d = json.load(open(os.path.join(R, "placebo", "bmt-s-placebo.json")))
    out = ["| task, horizon | candidate vs `empirical` % (pre-registered, pass < 1 %) | verdict | `naive` vs `empirical` % | candidate vs better unconditional % (amendment A1) | verdict |",
           "|---|---|---|---|---|---|"]
    for k, v in d.items():
        out.append(f"| {k.replace('|', ', ')} | {f3(v['skill_vs_empirical'])} | {'PASS' if v['passed'] else 'FAIL'} | {f3(v['skill_naive_vs_empirical'])} | "
                   f"{f3(v['skill_vs_best_unconditional'])} | {'PASS' if v['passed_amended'] else 'FAIL'} |")
    return "\n".join(out)


def extra_md():
    d = json.load(open(os.path.join(R, "paired_extra.json")))
    out = ["| horizon | model | vs timestral-fast % | vs timestral-max % | vs timestral-official-2048 % |", "|---|---|---|---|---|"]
    for h in (1, 6, 24):
        for m in ("bmt-t", "bmt-s-sym"):
            out.append(f"| {h} | {m} | " + " | ".join(f3(d[f"h{h}|{m}|{r}"]) for r in ("timestral-fast", "timestral-max", "timestral-official-2048")) + " |")
    return "\n".join(out)


def search_md():
    S = json.load(open(os.path.join(R, "search", "summary.json")))
    out = [f"{S['trials']} configurations tried in {S['rounds']} rounds, {S['gpu_hours']:.2f} GPU-hours; stop reason: {S.get('stop', '?')}. Full ledger: `docs/RESEARCH_LOG.md`.", "",
           "| step | config | validation objective | paired validation improvement over the previous incumbent, % (95 %) |", "|---|---|---|---|"]
    for i, c in enumerate(S["chain"]):
        cs = "; ".join(f"{k.replace('|', ' ')} {100 * v[0]:+.2f} [{100 * v[1]:+.2f}, {100 * v[2]:+.2f}]" for k, v in c["cmp"].items()) if c.get("cmp") else "(baseline: returns only)"
        out.append(f"| {i} | `{c['name']}` | {c['obj']:.5f} | {cs} |")
    return "\n".join(out)


def lc2_md():
    S = json.load(open(os.path.join(R, "search", "summary.json")))
    out = ["| size | parameters | share of training symbols | symbols | validation objective | vs size s at 100 %, paired % (95 %): hourly h1 / h6 / h24, daily h1 / h5 |", "|---|---|---|---|---|---|"]
    for r in sorted(S.get("lc", []), key=lambda r: ("tsm".index(r["size"]), r["frac"])):
        cs = " / ".join(f"{100 * v[0]:+.2f} [{100 * v[1]:+.2f}, {100 * v[2]:+.2f}]" for v in r["cmp"].values()) if r.get("cmp") else "reference"
        out.append(f"| {r['size']} | {r['params']:,} | {r['frac']:.0%} | {r['assets']} | {r['obj']:.5f} | {cs} |")
    return "\n".join(out)


def placebo2_md():
    S = json.load(open(os.path.join(R, "search", "summary.json")))
    p = os.path.join(R, "placebo", f"placebo-{S.get('final', 'base')}.json")
    if not os.path.exists(p):
        return "(not run)"
    d = json.load(open(p))
    out = ["| task, horizon | candidate vs the better unconditional baseline % (pass < 1 %) | reference | verdict |", "|---|---|---|---|"]
    for k, v in d.items():
        out.append(f"| {k.replace('|', ', ')} | {f3(v['skill_vs_best_unconditional'])} | {v['reference']} | {'PASS' if v['passed'] else 'FAIL'} |")
    return "\n".join(out)


def read(name):
    p = os.path.join(R, name)
    return open(p).read() if os.path.exists(p) else "(not run)"


if __name__ == "__main__":
    info = json.load(open(os.path.join(D.REPO, "bench", "models.json")))
    tabs = "\n".join(tables(t, json.load(open(os.path.join(R, f"scores_{t}.json"))), info) for t in D.TASKS)
    HEAD = HEAD.replace("{{SEARCH}}", search_md()).replace("{{LC2}}", lc2_md()).replace("{{PLACEBO2}}", placebo2_md()).replace("{{TRADABLE}}", read("tradable.md")).replace("{{VOLTIMING}}", read("voltiming.md"))
    md = HEAD.replace("{{EXTRA}}", extra_md()).replace("{{SPEED}}", speed_md()).replace("{{LC}}", lc_md()).replace("{{PLACEBO}}", placebo_md()).replace("{{TABLES}}", tabs)
    open(os.path.join(D.REPO, "bench", "LEADERBOARD.md"), "w").write(md)
    print(len(md), "chars")
