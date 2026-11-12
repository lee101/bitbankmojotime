"""Speed table (bench/PROTOCOL.md section 8): rows/s, p50 / p99 per call at batch 1 / 64 / 4096, VRAM, sizes.

  python -m bmtbench.speed --models bmt-mojo-cpu,bmt-mojo-gpu,bmt-torch,timestral-fast,timestral-max,chronos-2,timestral-official
One row = one 512-bar context -> all horizons x 21 quantiles (Timestral kernel: one horizon, 256 / flip-averaged 512 bars).
>= 30 timed calls after 5 warm-ups, fewer when a call takes longer than a second (the count is recorded).
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np
from . import data as D
from .gpu import wait_for_gpu

BATCHES = (1, 64, 4096)
CUDA_LIB = os.environ.get("BMT_CUDA_LIB", "/usr/local/cuda-12.9/lib64")
PTXAS = os.environ.get("BMT_PTXAS", "/usr/local/cuda-12.9/bin/ptxas")


def gpu_state():
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    u, m = (int(x) for x in o.split(","))
    return {"gpu_util_pct": u, "gpu_mem_used_mib": m}


def vram_of(pid):
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for l in o.splitlines():
        p, m = (x.strip() for x in l.split(","))
        if int(p) == pid:
            return int(m)
    return None


def timeit(fn, batch, budget=25.0):
    for _ in range(5):
        t0 = time.perf_counter(); fn(); w = time.perf_counter() - t0
        if w > 2:
            break
    n = int(max(5, min(30 if w > 0.05 else 200, budget / max(w, 1e-4))))
    lat = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); lat.append(time.perf_counter() - t0)
    lat = np.sort(lat)
    return {"rows_s": round(batch * n / lat.sum(), 1), "p50_ms": round(1e3 * lat[n // 2], 3), "p99_ms": round(1e3 * lat[min(n - 1, int(n * 0.99))], 3), "calls": n}


def synth(n, L=512, seed=0):
    return (np.random.default_rng(seed).standard_t(4, (n, L)) * 0.004).astype(np.float32)


def mojo_bench(binary, weights, threads=None, env=None):
    out = {}
    for b in BATCHES:
        it = 200 if b == 1 else (40 if b == 64 else 8)
        cmd = ["nice", "-n", "10", binary, "bench", weights, str(b), str(it)] + ([str(threads)] if threads else [])
        o = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **(env or {}))).stdout
        m = re.search(r"rows/s ([\d.e+]+) p50_ms ([\d.e+-]+) p99_ms ([\d.e+-]+)", o)
        out[str(b)] = {"rows_s": round(float(m.group(1)), 1), "p50_ms": round(float(m.group(2)), 3), "p99_ms": round(float(m.group(3)), 3), "calls": it}
    return out


def run(model, a):
    res = {"model": model, "before": gpu_state()}
    if model.startswith("bmt-mojo-cpu"):
        thr = int(model.split("@")[1]) if "@" in model else 1
        cb = os.environ.get("BMT_CPU_BIN", os.path.join(D.REPO, "build", "bmt"))
        res.update(device=f"CPU {thr} thread" + ("s" if thr > 1 else ""), batches=mojo_bench(cb, a.weights, thr),
                   size_bytes=os.path.getsize(cb), weights_bytes=os.path.getsize(a.weights), vram_mib=0)
    elif model == "bmt-mojo-gpu":
        wait_for_gpu(2500)
        env = {"LD_LIBRARY_PATH": CUDA_LIB, "MODULAR_NVPTX_COMPILER_PATH": PTXAS}
        b = os.environ.get("BMT_GPU_BIN", os.path.join(D.REPO, "build", "bmt_gpu_sm120"))
        p = subprocess.Popen([b, "serve", a.weights, "4096"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(os.environ, **env))
        p.stderr.readline()
        res.update(vram_mib=vram_of(p.pid))
        p.kill(); p.wait()
        res.update(device="GPU", batches=mojo_bench(b, a.weights, None, env), size_bytes=os.path.getsize(b), weights_bytes=os.path.getsize(a.weights))
    elif model.startswith("bmt-torch"):
        import torch
        from train.model import BMT
        dev = "cpu" if "cpu" in model else "cuda"
        if dev == "cpu":
            torch.set_num_threads(int(model.split("@")[1]) if "@" in model else 1)
        else:
            wait_for_gpu(2500)
        c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        m = BMT(**c["cfg"]); m.load_state_dict(c["sd"]); m.eval().to(dev)
        res.update(device="GPU" if dev == "cuda" else f"CPU {torch.get_num_threads()} thr", batches={}, weights_bytes=os.path.getsize(a.weights))
        for b in BATCHES:
            x = synth(b)
            def f():
                with torch.inference_mode():
                    q = m.quantiles(torch.from_numpy(x).to(dev)).cpu().numpy()
            res["batches"][str(b)] = timeit(f, b)
        res["vram_mib"] = vram_of(os.getpid()) if dev == "cuda" else 0
    elif model in ("timestral-fast", "timestral-max"):
        from .foundation import TslKernel, TSL_SRC
        wait_for_gpu(2500)
        k = TslKernel()
        L, sa = (512, True) if model.endswith("max") else (256, False)
        res.update(device="GPU", batches={}, size_bytes=os.path.getsize(os.path.join(TSL_SRC, "tsfm16_sm120")),
                   weights_bytes=sum(os.path.getsize(os.path.join(TSL_SRC, "mojo", "data", f)) for f in os.listdir(os.path.join(TSL_SRC, "mojo", "data"))))
        for b in BATCHES:
            x = synth(b, L)
            res["batches"][str(b)] = timeit(lambda: k.run(x, sa), b)
        res["vram_mib"] = vram_of(k.p.pid)
        k.close()
    elif model.startswith("chronos-2"):
        import torch
        from chronos import Chronos2Pipeline
        dev = "cpu" if "cpu" in model else "cuda"
        if dev == "cpu":
            torch.set_num_threads(int(model.split("@")[1]) if "@" in model else 1)
        else:
            wait_for_gpu(2500)
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
        res.update(device="GPU" if dev == "cuda" else f"CPU {torch.get_num_threads()} thr", batches={})
        for b in BATCHES if dev == "cuda" else BATCHES[:2]:
            x = torch.from_numpy(np.cumsum(synth(b), 1))[:, None, :]
            res["batches"][str(b)] = timeit(lambda: pipe.predict_quantiles(x, prediction_length=24, quantile_levels=[float(v) for v in D.QLEV], batch_size=256), b)
        res["vram_mib"] = vram_of(os.getpid()) if dev == "cuda" else 0
    elif model.startswith("timestral-official"):
        import torch
        from timestral import TimestralForecaster
        dev = "cpu" if "cpu" in model else "cuda"
        if dev == "cpu":
            torch.set_num_threads(int(model.split("@")[1]) if "@" in model else 1)
        else:
            wait_for_gpu(2500)
        f = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device=dev, local_files_only=True)
        res.update(device="GPU" if dev == "cuda" else f"CPU {torch.get_num_threads()} thr", batches={})

        def go(x):
            with torch.inference_mode():
                for i in range(0, len(x), 256):
                    f.forecast(x[i:i + 256], prediction_length=24, clip_nonnegative=False).quantiles.float().cpu()
        for b in BATCHES if dev == "cuda" else BATCHES[:2]:
            x = torch.from_numpy(np.cumsum(synth(b), 1))
            res["batches"][str(b)] = timeit(lambda: go(x), b)
        res["vram_mib"] = vram_of(os.getpid()) if dev == "cuda" else 0
    res["after"] = gpu_state()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--weights", default=os.path.join(D.RUNS, "export", "bmt-s", "bmt.bin"))
    ap.add_argument("--ckpt", default=os.path.join(D.RUNS, "ckpt", "bmt-s_F4.pt"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = os.path.join(D.RUNS, "speed.json")
    allr = json.load(open(path)) if os.path.exists(path) else {}
    for m in a.models.split(","):
        r = run(m, a)
        allr[m + a.tag] = r
        print(json.dumps(r), flush=True)
        json.dump(allr, open(path, "w"), indent=1)
