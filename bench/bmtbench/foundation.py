"""Zero-shot foundation-model baselines, each in its own process on the shared GPU (bench/PROTOCOL.md section 7).

timestral-fast / timestral-max : a private copy of the production tsfm16 kernel (never the live service): one-step quantiles of the
    last 256 log returns / of the flip-averaged last 512, inputs clipped to +-0.3 (the service contract).
chronos-2 / timestral-official : the vendors' reference PyTorch pipelines on the last 512 log prices, native multi-step quantiles.
"""
import os, select, shutil, struct, subprocess, time
import numpy as np
from .data import NQ, CTX, TASKS, RUNS, ctx_ret, ctx_logp, ctx_agg

TSL_SRC = os.environ.get("BMT_TSL_SRC", "/nvme0n1-disk/code/timestral-serve")
PTXAS = os.environ.get("BMT_PTXAS", "/usr/local/cuda-12.9/bin/ptxas")


class TslKernel:
    """resident tsfm16 process in a private working directory (weights and runtime libs are symlinked read-only)."""

    def __init__(self, tokens=8704, binary="tsfm16_sm120"):
        d = os.path.join(RUNS, "tsl")
        os.makedirs(os.path.join(d, "home"), exist_ok=True)
        for n in ("mojo", "lib"):
            if not os.path.lexists(os.path.join(d, n)):
                os.symlink(os.path.join(TSL_SRC, n), os.path.join(d, n))
        b = os.path.join(d, binary)
        if not os.path.exists(b):
            shutil.copy2(os.path.join(TSL_SRC, binary), b)
        env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(d, "lib"), MODULAR_NVPTX_COMPILER_PATH=PTXAS, HOME=os.path.join(d, "home"),
                   XDG_CACHE_HOME=os.path.join(d, "home", "cache"))
        self.tokens, self.gpu_s = tokens, 0.0
        self.p = subprocess.Popen(["nice", "-n", "10", b, "serve", str(tokens)], cwd=d, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        t0 = time.time()
        while True:
            l = self.p.stderr.readline().decode(errors="replace")
            if l.startswith("ready"):
                self.info = l.strip()
                break
            if not l and self.p.poll() is not None or time.time() - t0 > 180:
                raise RuntimeError("tsfm16 did not start")

    def _read(self, n):
        b = bytearray()
        while len(b) < n:
            if not select.select([self.p.stdout], [], [], 120)[0]:
                raise RuntimeError("tsfm16 timeout")
            c = os.read(self.p.stdout.fileno(), n - len(b))
            if not c:
                raise RuntimeError("tsfm16 died")
            b += c
        return bytes(b)

    def cap(self, L, sa):
        return max(1, self.tokens // (L + 1) // (2 if sa else 1))

    def run(self, x, sa=False):
        """x[N, 256 or 512] float32 log returns -> q[N, 21]."""
        x = np.ascontiguousarray(np.clip(x, -0.3, 0.3), np.float32)
        L, out, cap = x.shape[1] // 32, [], 0
        cap = self.cap(L, sa)
        for i in range(0, len(x), cap):
            b = x[i:i + cap]
            self.p.stdin.write(struct.pack("<4i", len(b), L, 1 if sa else 0, 0) + b.tobytes())
            self.p.stdin.flush()
            n, us = struct.unpack("<2i", self._read(8))
            o = self._read(len(b) * NQ * 4)
            if n != len(b):
                raise RuntimeError("tsfm16 bad output")
            self.gpu_s += us / 1e6
            out.append(np.frombuffer(o, np.float32).reshape(len(b), NQ))
        return np.concatenate(out)

    def close(self):
        self.p.kill(); self.p.wait()


def timestral_kernel(p, rows, task, quality, k=None):
    """h = 1: the series itself. h > 1: one step on non-overlapping h-bar returns ending at the origin (temporal aggregation); max uses
    the flip-averaged 512 context when the guaranteed history covers 512 h-bars, else the flip-averaged 256."""
    H, own = TASKS[task]["horizons"], k is None
    k = k or TslKernel()
    out = np.zeros((len(rows["t"]), len(H), NQ), np.float32)
    try:
        for j, h in enumerate(H):
            L = 512 if quality == "max" and 512 * h + 1 <= TASKS[task]["min_hist"] else 256
            if L * h + 1 > TASKS[task]["min_hist"]:
                return out[:, :j] if j else None                    # not every row has the history: no entry for this horizon (protocol section 3)
            for i in range(0, len(out), 8192):
                a, t = rows["a"][i:i + 8192], rows["t"][i:i + 8192]
                x = ctx_ret(p, a, t, L) if h == 1 else ctx_agg(p, a, t, h, L)
                out[i:i + 8192, j] = k.run(x, sa=quality == "max")
    finally:
        if own:
            k.close()
    return out


def chronos2(p, rows, task, inp="logp", batch=256, dtype="float32"):
    import torch
    from chronos import Chronos2Pipeline
    from .data import QLEV
    from .gpu import wait_for_gpu, cap_torch
    wait_for_gpu(2500); cap_torch(2.0)
    H = TASKS[task]["horizons"]
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", torch_dtype=getattr(torch, dtype))
    out = np.zeros((len(rows["t"]), len(H), NQ), np.float32)
    hm = max(H) if inp == "logp" else 1
    for i in range(0, len(out), 4096):
        a, t = rows["a"][i:i + 4096], rows["t"][i:i + 4096]
        x = ctx_logp(p, a, t) if inp == "logp" else ctx_ret(p, a, t).astype(np.float64)
        base = x[:, -1:] if inp == "logp" else 0.0
        q, _ = pipe.predict_quantiles(torch.tensor(x - (base if inp == "logp" else 0.0), dtype=torch.float32)[:, None, :], prediction_length=hm,
                                      quantile_levels=[float(v) for v in QLEV], batch_size=batch)
        q = torch.stack([v[0] for v in q]).float().cpu().numpy()     # [n, hm, 21]
        for j, h in enumerate(H):
            if inp == "logp":
                out[i:i + 4096, j] = q[:, h - 1]
            elif h == 1:
                out[i:i + 4096, j] = q[:, 0]
    return out if inp == "logp" else out[:, :1]


def timestral_official(p, rows, task, batch=256, ctx=CTX):
    import torch
    from timestral import TimestralForecaster
    from .gpu import wait_for_gpu, cap_torch
    wait_for_gpu(2500); cap_torch(2.5)
    H = TASKS[task]["horizons"]
    f = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", local_files_only=True)
    out = np.zeros((len(rows["t"]), len(H), NQ), np.float32)
    for i in range(0, len(out), batch):
        a, t = rows["a"][i:i + batch], rows["t"][i:i + batch]
        x = ctx_logp(p, a, t, ctx)
        base = x[:, -1:]
        with torch.inference_mode():
            q = f.forecast(torch.tensor(x - base, dtype=torch.float32), prediction_length=max(H), clip_nonnegative=False).quantiles
        q = q.float().cpu().numpy()[:, 0]                           # [n, hm, 21]
        for j, h in enumerate(H):
            out[i:i + batch, j] = q[:, h - 1]
    return out
