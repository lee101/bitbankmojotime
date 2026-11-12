"""bmt CPU inference: compact patch transformer -> 21 quantiles per horizon (see train/model.py for the reference arithmetic).

  bmt selftest <dir>                      bmt.bin + selftest_in.f32 vs selftest_ref.f32, exit 1 above tolerance
  bmt run <bmt.bin> <in.f32> <out.f32> <rows> [threads]
  bmt bench <bmt.bin> <batch> <iters> [threads]
  bmt serve <bmt.bin> [threads]           stdin: int32 rows, then the input block; stdout: int32 rows, rows*H*21 float32

Input block for N rows: r[N, L] float32 log returns (oldest first), then b[N, L] (market returns) if the model has the covariate, then
cal[N, 3] float32 (start hour, weekday, bar id) if it has calendar inputs. Output: q[N, H, 21] quantiles of the cumulative log return.
fp32 throughout, AVX2-width SIMD (8 lanes), rows in parallel. The last layer only computes the last token (the only one the head reads).
"""

from std.math import sqrt, exp, asinh, log, log1p
from std.sys import argv, stderr
from std.os import getenv
from std.ffi import external_call
from std.time import perf_counter_ns
from max.algorithm import parallelize

comptime FP = Pointer[Float32, MutAnyOrigin]
comptime IP = Pointer[Int, MutAnyOrigin]
comptime V = SIMD[DType.float32, 8]
comptime MAGIC = 0x31544D42
comptime NQ = 21
# cfg slots
comptime C_L = 0
comptime C_P = 1
comptime C_D = 2
comptime C_NL = 3
comptime C_NH = 4
comptime C_H = 5
comptime C_COV = 6
comptime C_CAL = 7
comptime C_HZ = 8          # 6 horizons
comptime C_T = 14
comptime C_F = 15
comptime O_EMB = 16        # weight offsets (float32 index)
comptime O_POS = 17
comptime O_LAYER = 18
comptime O_NF = 19
comptime O_CAL = 20
comptime O_HEAD = 21
comptime C_TOTAL = 22
comptime C_SCRATCH = 23
comptime NCFG = 24


@always_inline
def fp(l: List[Float32]) -> FP:
    return l.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def ip(l: List[Int]) -> IP:
    return l.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def fd_read(fd: Int, dst: Pointer[UInt8, MutAnyOrigin], cap: Int) -> Int:
    var off = 0
    while off < cap:
        var r = external_call["read", Int](fd, dst + off, cap - off)
        if r <= 0:
            break
        off += r
    return off


def fd_write(fd: Int, src: Pointer[UInt8, MutAnyOrigin], n: Int) raises:
    var off = 0
    while off < n:
        var w = external_call["write", Int](fd, src + off, n - off)
        if w <= 0:
            raise Error("write failed")
        off += w


def read_f32(path: String, mut dst: List[Float32], skip_bytes: Int) raises -> Int:
    """float32 values of a file after skip_bytes -> dst; returns the count."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    var n = (len(bytes) - skip_bytes) // 4
    dst = List[Float32](length=n, fill=0)
    var src = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
    var d = fp(dst)
    for i in range(n):
        d[i] = src[skip_bytes // 4 + i]
    return n


def load_model(path: String, mut cfg: List[Int], mut w: List[Float32]) raises:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    if len(bytes) < 64:
        raise Error(path + ": too short")
    var hp = bytes.unsafe_ptr().unsafe_bitcast[Int32]()
    if Int(hp[0]) != MAGIC:
        raise Error(path + ": bad magic")
    cfg = List[Int](length=NCFG, fill=0)
    for i in range(8):
        cfg[i] = Int(hp[1 + i])
    for i in range(6):
        cfg[C_HZ + i] = Int(hp[9 + i])
    var L = cfg[C_L]
    var P = cfg[C_P]
    var D = cfg[C_D]
    var T = L // P
    var F = P * (5 if cfg[C_COV] != 0 else 2)
    cfg[C_T] = T
    cfg[C_F] = F
    var o = 0
    cfg[O_EMB] = o
    o += D * F + D
    cfg[O_POS] = o
    o += T * D
    cfg[O_LAYER] = o
    o += cfg[C_NL] * (12 * D * D + 13 * D)
    cfg[O_NF] = o
    o += 2 * D
    cfg[O_CAL] = o
    if cfg[C_CAL] != 0:
        o += 34 * D
    cfg[O_HEAD] = o
    var hin = D + (1 if cfg[C_COV] != 0 else 0)
    o += cfg[C_H] * NQ * hin + cfg[C_H] * NQ
    cfg[C_TOTAL] = o
    cfg[C_SCRATCH] = T * F + 2 * T * D + T * 3 * D + T * D + T * 4 * D + T + 2 * D + 8 + cfg[C_H] * NQ
    if len(bytes) != 64 + o * 4:
        raise Error(path + ": expected " + String(64 + o * 4) + " bytes, got " + String(len(bytes)))
    w = List[Float32](length=o, fill=0)
    var src = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
    var d = fp(w)
    for i in range(o):
        d[i] = src[16 + i]


@always_inline
def dot(a: FP, b: FP, n: Int) -> Float32:
    """n is a multiple of 8."""
    var s0 = V(0)
    var s1 = V(0)
    var i = 0
    while i + 16 <= n:
        s0 = a.unsafe_load[width=8](i).fma(b.unsafe_load[width=8](i), s0)
        s1 = a.unsafe_load[width=8](i + 8).fma(b.unsafe_load[width=8](i + 8), s1)
        i += 16
    if i < n:
        s0 = a.unsafe_load[width=8](i).fma(b.unsafe_load[width=8](i), s0)
    return (s0 + s1).reduce_add()


def linear(dst: FP, src: FP, w: FP, b: FP, rows: Int, n_in: Int, n_out: Int, relu: Bool):
    """dst[r, j] = b[j] + sum_k src[r, k] * w[j, k]; four rows share each pass over a weight row."""
    var n8 = n_in - n_in % 8
    var r = 0
    while r + 4 <= rows:
        var x0 = src + r * n_in
        var x1 = x0 + n_in
        var x2 = x1 + n_in
        var x3 = x2 + n_in
        var d0 = dst + r * n_out
        for j in range(n_out):
            var wj = w + j * n_in
            var a0 = V(0)
            var a1 = V(0)
            var a2 = V(0)
            var a3 = V(0)
            var k = 0
            while k < n8:
                var wv = wj.unsafe_load[width=8](k)
                a0 = wv.fma(x0.unsafe_load[width=8](k), a0)
                a1 = wv.fma(x1.unsafe_load[width=8](k), a1)
                a2 = wv.fma(x2.unsafe_load[width=8](k), a2)
                a3 = wv.fma(x3.unsafe_load[width=8](k), a3)
                k += 8
            var y0 = a0.reduce_add() + b[j]
            var y1 = a1.reduce_add() + b[j]
            var y2 = a2.reduce_add() + b[j]
            var y3 = a3.reduce_add() + b[j]
            while k < n_in:
                y0 += wj[k] * x0[k]
                y1 += wj[k] * x1[k]
                y2 += wj[k] * x2[k]
                y3 += wj[k] * x3[k]
                k += 1
            if relu:
                y0 = max(y0, 0)
                y1 = max(y1, 0)
                y2 = max(y2, 0)
                y3 = max(y3, 0)
            d0[j] = y0
            d0[n_out + j] = y1
            d0[2 * n_out + j] = y2
            d0[3 * n_out + j] = y3
        r += 4
    while r < rows:
        var x = src + r * n_in
        var d = dst + r * n_out
        for j in range(n_out):
            var wj = w + j * n_in
            var y = dot(wj, x, n8) + b[j]
            var k = n8
            while k < n_in:
                y += wj[k] * x[k]
                k += 1
            d[j] = max(y, 0) if relu else y
        r += 1


def layernorm(dst: FP, src: FP, g: FP, b: FP, rows: Int, D: Int):
    for r in range(rows):
        var x = src + r * D
        var y = dst + r * D
        var s = V(0)
        for i in range(0, D, 8):
            s += x.unsafe_load[width=8](i)
        var mean = s.reduce_add() / Float32(D)
        var mv = V(mean)
        var q = V(0)
        for i in range(0, D, 8):
            var c = x.unsafe_load[width=8](i) - mv
            q = c.fma(c, q)
        var inv = V(1.0 / sqrt(q.reduce_add() / Float32(D) + 1e-5))
        for i in range(0, D, 8):
            y.store(i, ((x.unsafe_load[width=8](i) - mv) * inv).fma(g.unsafe_load[width=8](i), b.unsafe_load[width=8](i)))


def attention(dst: FP, qkv: FP, sc: FP, T: Int, D: Int, NH: Int, first: Int):
    """softmax(q k^T / sqrt(hd)) v per head for query tokens first..T-1; qkv[t] = [q | k | v], each D wide."""
    var hd = D // NH
    var inv = Float32(1.0) / sqrt(Float32(hd))
    for h in range(NH):
        for i in range(first, T):
            var qi = qkv + i * 3 * D + h * hd
            var mx = Float32(-1e30)
            for j in range(T):
                var s = dot(qi, qkv + j * 3 * D + D + h * hd, hd) * inv
                sc[j] = s
                mx = max(mx, s)
            var tot = Float32(0)
            for j in range(T):
                var e = exp(sc[j] - mx)
                sc[j] = e
                tot += e
            var o = dst + i * D + h * hd
            for k in range(0, hd, 8):
                o.store(k, V(0))
            for j in range(T):
                var p = V(sc[j] / tot)
                var vj = qkv + j * 3 * D + 2 * D + h * hd
                for k in range(0, hd, 8):
                    o.store(k, p.fma(vj.unsafe_load[width=8](k), o.unsafe_load[width=8](k)))


@always_inline
def softplus(x: Float32) -> Float32:
    return max(x, 0) + log1p(exp(-abs(x)))


def forward_row(c: IP, w: FP, r: FP, bm: FP, cal: FP, res: FP, s: FP):
    """one series -> out[H * 21]. s = scratch of cfg[C_SCRATCH] floats."""
    var L = c[C_L]
    var P = c[C_P]
    var D = c[C_D]
    var T = c[C_T]
    var F = c[C_F]
    var NL = c[C_NL]
    var NH = c[C_NH]
    var H = c[C_H]
    var cov = c[C_COV] != 0
    var nf = F // P
    var feat = s
    var x = feat + T * F
    var hb = x + T * D
    var qkv = hb + T * D
    var att = qkv + T * 3 * D
    var u = att + T * D
    var scb = u + T * 4 * D
    var z = scb + T
    var o = z + 2 * D + 8
    # scale + features
    var acc = V(0)
    for i in range(0, L, 8):
        var v = r.unsafe_load[width=8](i)
        acc = v.fma(v, acc)
    var sc = max(sqrt(acc.reduce_add() / Float32(L)), Float32(1e-8))
    var sb = Float32(1)
    if cov:
        var accb = V(0)
        for i in range(0, L, 8):
            var v = bm.unsafe_load[width=8](i)
            accb = v.fma(v, accb)
        sb = max(sqrt(accb.reduce_add() / Float32(L)), Float32(1e-8))
    for i in range(L):
        var xv = asinh(r[i] / sc)
        feat[i * nf] = xv
        feat[i * nf + 1] = xv * xv
        if cov:
            var xb = asinh(bm[i] / sb)
            feat[i * nf + 2] = xb
            feat[i * nf + 3] = xb * xb
            feat[i * nf + 4] = xv * xb
    linear(x, feat, w + c[O_EMB], w + c[O_EMB] + D * F, T, F, D, False)
    var pos = w + c[O_POS]
    for i in range(0, T * D, 8):
        x.store(i, x.unsafe_load[width=8](i) + pos.unsafe_load[width=8](i))
    var lsz = 12 * D * D + 13 * D
    for l in range(NL):
        var lw = w + c[O_LAYER] + l * lsz
        var n1 = lw
        var wq = n1 + 2 * D
        var bq = wq + 3 * D * D
        var wo = bq + 3 * D
        var bo = wo + D * D
        var n2 = bo + D
        var w1 = n2 + 2 * D
        var b1 = w1 + 4 * D * D
        var w2 = b1 + 4 * D
        var b2 = w2 + 4 * D * D
        var first = T - 1 if l == NL - 1 else 0                      # the head reads only the last token
        var nr = T - first
        layernorm(hb, x, n1, n1 + D, T, D)
        linear(qkv, hb, wq, bq, T, D, 3 * D, False)
        attention(att, qkv, scb, T, D, NH, first)
        linear(hb + first * D, att + first * D, wo, bo, nr, D, D, False)
        for i in range(first * D, T * D, 8):
            x.store(i, x.unsafe_load[width=8](i) + hb.unsafe_load[width=8](i))
        layernorm(hb + first * D, x + first * D, n2, n2 + D, nr, D)
        linear(u + first * 4 * D, hb + first * D, w1, b1, nr, D, 4 * D, True)
        linear(hb + first * D, u + first * 4 * D, w2, b2, nr, 4 * D, D, False)
        for i in range(first * D, T * D, 8):
            x.store(i, x.unsafe_load[width=8](i) + hb.unsafe_load[width=8](i))
    layernorm(z, x + (T - 1) * D, w + c[O_NF], w + c[O_NF] + D, 1, D)
    if c[C_CAL] != 0:
        var e0 = w + c[O_CAL] + Int(cal[0]) * D
        var e1 = w + c[O_CAL] + (24 + Int(cal[1])) * D
        var e2 = w + c[O_CAL] + (31 + Int(cal[2])) * D
        for i in range(0, D, 8):
            z.store(i, z.unsafe_load[width=8](i) + e0.unsafe_load[width=8](i) + e1.unsafe_load[width=8](i) + e2.unsafe_load[width=8](i))
    var hin = D
    if cov:
        z[D] = min(max(log(sc / sb), Float32(-5)), Float32(5))
        hin = D + 1
    linear(o, z, w + c[O_HEAD], w + c[O_HEAD] + H * NQ * hin, 1, hin, H * NQ, False)
    for h in range(H):
        var oh = o + h * NQ
        var qo = res + h * NQ
        var k = sc * sqrt(Float32(c[C_HZ + h]))
        var med = oh[10]
        qo[10] = med * k
        var a = med
        for j in range(11, NQ):
            a += softplus(oh[j])
            qo[j] = a * k
        a = med
        var j = 9
        while j >= 0:
            a -= softplus(oh[j])
            qo[j] = a * k
            j -= 1


def infer(cfg: List[Int], w: List[Float32], inp: FP, res: FP, n: Int, threads: Int):
    var c = ip(cfg)
    var wp = fp(w)
    var L = cfg[C_L]
    var HQ = cfg[C_H] * NQ
    var cov = cfg[C_COV] != 0
    var hascal = cfg[C_CAL] != 0
    var ns = cfg[C_SCRATCH]
    var chunk = max(1, min(16, (n + threads - 1) // max(threads, 1)))
    var nchunks = (n + chunk - 1) // chunk
    var pb = inp + (n * L if cov else 0)
    var pc = inp + n * L * (2 if cov else 1)

    def work(i: Int) {var c, var wp, var inp, var pb, var pc, var res, var n, var L, var HQ, var ns, var chunk, var hascal}:
        var scratch = List[Float32](length=ns, fill=0)
        var s = scratch.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
        var hi = min(n, (i + 1) * chunk)
        for row in range(i * chunk, hi):
            forward_row(c, wp, inp + row * L, pb + row * L, pc + (row * 3 if hascal else 0), res + row * HQ, s)

    if threads <= 1 or nchunks == 1:
        for i in range(nchunks):
            work(i)
    else:
        parallelize(work, nchunks, min(threads, nchunks))


def infer_sym(cfg: List[Int], w: List[Float32], inp: FP, res: FP, n: Int, threads: Int):
    """Flip-averaged forecast (2 passes): q = (q(r) - mirror(q(-r))) / 2, the market covariate is flipped with the series."""
    var L = cfg[C_L]
    var HQ = cfg[C_H] * NQ
    var nl = n * L * (2 if cfg[C_COV] != 0 else 1)
    var ncal = n * 3 if cfg[C_CAL] != 0 else 0
    var x2 = List[Float32](length=2 * (nl + ncal), fill=0)
    var o2 = List[Float32](length=2 * n * HQ, fill=0)
    var nb = nl // n                                                 # floats per row in each of the r / b blocks
    for blk in range(nl // (n * L)):                                 # r block, then b block: [rows | negated rows]
        for i in range(n * L):
            x2[blk * 2 * n * L + i] = inp[blk * n * L + i]
            x2[blk * 2 * n * L + n * L + i] = -inp[blk * n * L + i]
    for i in range(ncal):
        x2[2 * nl + i] = inp[nl + i]
        x2[2 * nl + ncal + i] = inp[nl + i]
    infer(cfg, w, fp(x2), fp(o2), 2 * n, threads)
    for row in range(n):
        for h in range(cfg[C_H]):
            for j in range(NQ):
                res[row * HQ + h * NQ + j] = 0.5 * (o2[row * HQ + h * NQ + j] - o2[(n + row) * HQ + h * NQ + (NQ - 1 - j)])


def run_infer(cfg: List[Int], w: List[Float32], inp: FP, res: FP, n: Int, threads: Int):
    """BMT_SYM=1 in the environment selects the flip-averaged (2-pass) forecast."""
    if getenv("BMT_SYM", "0") == "1":
        infer_sym(cfg, w, inp, res, n, threads)
    else:
        infer(cfg, w, inp, res, n, threads)


def in_floats(cfg: List[Int], n: Int) -> Int:
    return n * (cfg[C_L] * (2 if cfg[C_COV] != 0 else 1) + (3 if cfg[C_CAL] != 0 else 0))


def selftest(dir: String) raises -> Bool:
    var cfg = List[Int]()
    var w = List[Float32]()
    load_model(dir + "/bmt.bin", cfg, w)
    var xin = List[Float32]()
    var ref_ = List[Float32]()
    var ni = read_f32(dir + "/selftest_in.f32", xin, 0)
    var nr = read_f32(dir + "/selftest_ref.f32", ref_, 0)
    var HQ = cfg[C_H] * NQ
    var n = nr // HQ
    if in_floats(cfg, n) != ni:
        raise Error("selftest vector does not match the model")
    var out = List[Float32](length=nr, fill=0)
    infer(cfg, w, fp(xin), fp(out), n, 4)
    var maxabs = Float32(0)
    var maxrel = Float32(0)
    for row in range(n):
        var acc = Float32(0)
        for i in range(cfg[C_L]):
            acc += xin[row * cfg[C_L] + i] * xin[row * cfg[C_L] + i]
        var sc = max(sqrt(acc / Float32(cfg[C_L])), Float32(1e-8))
        for h in range(cfg[C_H]):
            var k = sc * sqrt(Float32(cfg[C_HZ + h]))
            for j in range(NQ):
                var e = abs(out[row * HQ + h * NQ + j] - ref_[row * HQ + h * NQ + j])
                maxabs = max(maxabs, e)
                maxrel = max(maxrel, e / k)
    var tol = Float32(2e-4)
    print("selftest rows", n, "max abs err", maxabs, "max err in scale units", maxrel, "tol", tol, "PASS" if maxrel <= tol else "FAIL")
    return maxrel <= tol


def bench(path: String, batch: Int, iters: Int, threads: Int) raises:
    var cfg = List[Int]()
    var w = List[Float32]()
    load_model(path, cfg, w)
    var ni = in_floats(cfg, batch)
    var xin = List[Float32](length=ni, fill=0)
    var seed: UInt64 = 12345
    for i in range(ni):
        seed = seed * 6364136223846793005 + 1442695040888963407
        xin[i] = (Float32(Int((seed >> 40) & 0xFFFF)) / 65536.0 - 0.5) * 0.02
    if cfg[C_CAL] != 0:
        for i in range(batch * 3):
            xin[ni - batch * 3 + i] = Float32(i % 3)
    var out = List[Float32](length=batch * cfg[C_H] * NQ, fill=0)
    for _ in range(3):
        run_infer(cfg, w, fp(xin), fp(out), batch, threads)
    var lat = List[Float64](length=iters, fill=0)
    var t00 = perf_counter_ns()
    for it in range(iters):
        var t0 = perf_counter_ns()
        run_infer(cfg, w, fp(xin), fp(out), batch, threads)
        lat[it] = Float64(perf_counter_ns() - t0) / 1e6
    var total = Float64(perf_counter_ns() - t00) / 1e9
    sort(lat)
    print("batch", batch, "threads", threads, "rows/s", Float64(batch * iters) / total, "p50_ms", lat[iters // 2], "p99_ms", lat[min(iters - 1, (iters * 99) // 100)],
          "check", out[0])


def serve(path: String, threads: Int) raises:
    var cfg = List[Int]()
    var w = List[Float32]()
    load_model(path, cfg, w)
    print("ready L", cfg[C_L], "H", cfg[C_H], "cov", cfg[C_COV], "cal", cfg[C_CAL], "threads", threads, file=stderr)
    var hdr = List[Int32](length=1, fill=0)
    var bi = List[Float32]()
    var bo = List[Float32]()
    var hp = hdr.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    while True:
        if fd_read(0, hp, 4) != 4:
            return
        var n = Int(hdr[0])
        if n <= 0:
            return
        var ni = in_floats(cfg, n)
        if len(bi) < ni:
            bi = List[Float32](length=ni, fill=0)
        if len(bo) < n * cfg[C_H] * NQ:
            bo = List[Float32](length=n * cfg[C_H] * NQ, fill=0)
        if fd_read(0, bi.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), ni * 4) != ni * 4:
            raise Error("short request body")
        run_infer(cfg, w, fp(bi), fp(bo), n, threads)
        fd_write(1, hp, 4)
        fd_write(1, bo.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), n * cfg[C_H] * NQ * 4)


def main() raises:
    var a = argv()
    if len(a) >= 3 and a[1] == "selftest":
        if not selftest(String(a[2])):
            raise Error("selftest failed")
    elif len(a) >= 5 and a[1] == "bench":
        bench(String(a[2]), Int(a[3]), Int(a[4]), Int(a[5]) if len(a) > 5 else 1)
    elif len(a) >= 3 and a[1] == "serve":
        serve(String(a[2]), Int(a[3]) if len(a) > 3 else 1)
    elif len(a) >= 6 and a[1] == "run":
        var cfg = List[Int]()
        var w = List[Float32]()
        load_model(String(a[2]), cfg, w)
        var n = Int(a[5])
        var xin = List[Float32]()
        if read_f32(String(a[3]), xin, 0) != in_floats(cfg, n):
            raise Error("input size does not match rows")
        var out = List[Float32](length=n * cfg[C_H] * NQ, fill=0)
        run_infer(cfg, w, fp(xin), fp(out), n, Int(a[6]) if len(a) > 6 else 1)
        var nb = n * cfg[C_H] * NQ * 4
        var b = List[UInt8](length=nb, fill=0)
        var pb = b.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(nb // 4):
            pb[i] = out[i]
        var f = open(String(a[4]), "w")
        f.write_bytes(Span(b))
        f.close()
    else:
        print("usage: bmt selftest <dir> | run <bmt.bin> <in.f32> <out.f32> <rows> [threads] | bench <bmt.bin> <batch> <iters> [threads] | serve <bmt.bin> [threads]")
