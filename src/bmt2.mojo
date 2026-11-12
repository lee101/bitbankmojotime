"""bmt2 CPU inference: the bmt patch transformer with configurable inputs (train/model2.py is the reference arithmetic).

  bmt2 selftest <dir>                     bmt2.bin + selftest_in.f32 vs selftest_ref.f32, exit 1 above tolerance
  bmt2 bench <bmt2.bin> <batch> <iters> [threads]
  bmt2 serve <bmt2.bin> [threads]         stdin: int32 rows, then the input block; stdout: int32 rows, rows*H*21 float32
  bmt2 rotation <bmt2.bin> <history.json> scores for a rotation consumer from its hourly candle history (docs/INTEGRATION.md)

Input block of N rows (float32, channel-major, layout written by train/export2.py): fine[N, n_fine, 512], coarse[N, n_coarse, 256],
scal[N, n_scalars], cal[N, 3]. Channel kinds: 0 RET, 1 POS, 2 VOL, 3 SHR, 4 XRT, 5 RAW. BMT_SYM=1 selects the flip-averaged forecast.
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
comptime MAGIC = 0x32544D42
comptime NQ = 21
comptime C_L = 0
comptime C_P = 1
comptime C_LC = 2
comptime C_PC = 3
comptime C_D = 4
comptime C_NL = 5
comptime C_NH = 6
comptime C_H = 7
comptime C_CAL = 8
comptime C_NFC = 9          # fine channels
comptime C_NCC = 10         # coarse channels
comptime C_HAR = 11         # number of log-ratio scalars (first in the scalar block)
comptime C_VOV = 12         # number of raw scalars
comptime C_HZ = 13          # 4 horizons
comptime C_FK = 20          # 16 fine kinds
comptime C_CK = 36          # 4 coarse kinds
comptime C_TF = 40
comptime C_TC = 41
comptime C_TS = 42
comptime C_NT = 43
comptime C_FF = 44          # features per fine step
comptime C_FC = 45
comptime C_NS = 46
comptime O_EMB = 47
comptime O_EMBC = 48
comptime O_EMBS = 49
comptime O_POS = 50
comptime O_LAYER = 51
comptime O_NF = 52
comptime O_CAL = 53
comptime O_HEAD = 54
comptime C_TOTAL = 55
comptime C_SCRATCH = 56
comptime C_ROWIN = 57
comptime NCFG = 64


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



def nfeat(kind: Int) -> Int:
    return 2 if kind == 0 or kind == 4 else 1


def load_model(path: String, mut cfg: List[Int], mut w: List[Float32]) raises:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    if len(bytes) < 256:
        raise Error(path + ": too short")
    var hp = bytes.unsafe_ptr().unsafe_bitcast[Int32]()
    if Int(hp[0]) != MAGIC:
        raise Error(path + ": bad magic")
    cfg = List[Int](length=NCFG, fill=0)
    for i in range(13):
        cfg[i] = Int(hp[1 + i])
    for i in range(4):
        cfg[C_HZ + i] = Int(hp[14 + i])
    for i in range(16):
        cfg[C_FK + i] = Int(hp[20 + i])
    for i in range(4):
        cfg[C_CK + i] = Int(hp[36 + i])
    var D = cfg[C_D]
    var ff = 0
    for i in range(cfg[C_NFC]):
        ff += nfeat(cfg[C_FK + i])
    var fc = 0
    for i in range(cfg[C_NCC]):
        fc += nfeat(cfg[C_CK + i])
    var ns = cfg[C_HAR] + cfg[C_VOV]
    var tf = cfg[C_L] // cfg[C_P]
    var tc = cfg[C_LC] // cfg[C_PC] if cfg[C_NCC] > 0 else 0
    var ts = 1 if ns > 0 else 0
    var nt = tf + tc + ts
    cfg[C_TF] = tf
    cfg[C_TC] = tc
    cfg[C_TS] = ts
    cfg[C_NT] = nt
    cfg[C_FF] = ff
    cfg[C_FC] = fc
    cfg[C_NS] = ns
    var o = 0
    cfg[O_EMB] = o
    o += D * cfg[C_P] * ff + D
    cfg[O_EMBC] = o
    if tc > 0:
        o += D * cfg[C_PC] * fc + D
    cfg[O_EMBS] = o
    if ts > 0:
        o += D * ns + D
    cfg[O_POS] = o
    o += nt * D
    cfg[O_LAYER] = o
    o += cfg[C_NL] * (12 * D * D + 13 * D)
    cfg[O_NF] = o
    o += 2 * D
    cfg[O_CAL] = o
    if cfg[C_CAL] != 0:
        o += 34 * D
    cfg[O_HEAD] = o
    o += cfg[C_H] * NQ * D + cfg[C_H] * NQ
    cfg[C_TOTAL] = o
    cfg[C_SCRATCH] = cfg[C_L] * ff + cfg[C_LC] * fc + 2 * nt * D + nt * 3 * D + nt * D + nt * 4 * D + nt + D + 16 + cfg[C_H] * NQ + 16
    cfg[C_ROWIN] = cfg[C_NFC] * cfg[C_L] + cfg[C_NCC] * cfg[C_LC] + ns + (3 if cfg[C_CAL] != 0 else 0)
    if len(bytes) != 256 + o * 4:
        raise Error(path + ": expected " + String(256 + o * 4) + " bytes, got " + String(len(bytes)))
    w = List[Float32](length=o, fill=0)
    var src = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
    var d = fp(w)
    for i in range(o):
        d[i] = src[64 + i]


def rms(x: FP, n: Int) -> Float32:
    var acc = V(0)
    for i in range(0, n, 8):
        var v = x.unsafe_load[width=8](i)
        acc = v.fma(v, acc)
    return max(sqrt(acc.reduce_add() / Float32(n)), Float32(1e-8))


def norm_stream(feat: FP, x: FP, kinds: IP, nch: Int, T: Int, F: Int, sgn: Float32) -> Float32:
    """Raw channels x[nch, T] -> feat[T, F]; returns the scale of channel 0. sgn = -1 evaluates the mirrored series."""
    var s = rms(x, T)
    for i in range(T):
        var xr = asinh(sgn * x[i] / s)
        feat[i * F] = xr
        feat[i * F + 1] = xr * xr
    var f = 2
    for c in range(1, nch):
        var v = x + c * T
        var k = kinds[c]
        if k == 1:
            for i in range(T):
                feat[i * F + f] = asinh(v[i] / s)
        elif k == 2:
            var m = Float32(0)
            for i in range(T):
                m += v[i]
            m = max(m / Float32(T), Float32(1e-12))
            for i in range(T):
                feat[i * F + f] = log1p(v[i] / m)
        elif k == 3:
            for i in range(T):
                var sh = v[i] if sgn > 0 else Float32(1) - v[i]
                feat[i * F + f] = (sh - 0.5) * 4
        elif k == 4:
            var sj = rms(v, T)
            for i in range(T):
                var xb = asinh(sgn * v[i] / sj)
                feat[i * F + f] = xb
                feat[i * F + f + 1] = xb * feat[i * F]
            f += 1
        else:
            for i in range(T):
                feat[i * F + f] = sgn * v[i]
        f += 1
    return s


def forward_row(c: IP, w: FP, fine: FP, coarse: FP, scal: FP, cal: FP, res: FP, s: FP, sgn: Float32):
    """One row -> res[H * 21]. s = scratch of cfg[C_SCRATCH] floats."""
    var L = c[C_L]
    var D = c[C_D]
    var T = c[C_NT]
    var TF = c[C_TF]
    var TC = c[C_TC]
    var TS = c[C_TS]
    var FF = c[C_FF]
    var FC = c[C_FC]
    var NL = c[C_NL]
    var NH = c[C_NH]
    var H = c[C_H]
    var feat = s
    var featc = feat + L * FF
    var x = featc + c[C_LC] * FC
    var hb = x + T * D
    var qkv = hb + T * D
    var att = qkv + T * 3 * D
    var u = att + T * D
    var scb = u + T * 4 * D
    var z = scb + T
    var o = z + D + 16
    var sv = o + H * NQ
    var sc = norm_stream(feat, fine, c + C_FK, c[C_NFC], L, FF, sgn)
    linear(x + (TS + TC) * D, feat, w + c[O_EMB], w + c[O_EMB] + D * c[C_P] * FF, TF, c[C_P] * FF, D, False)
    if TC > 0:
        _ = norm_stream(featc, coarse, c + C_CK, c[C_NCC], c[C_LC], FC, sgn)
        linear(x + TS * D, featc, w + c[O_EMBC], w + c[O_EMBC] + D * c[C_PC] * FC, TC, c[C_PC] * FC, D, False)
    if TS > 0:
        var ns = c[C_NS]
        for i in range(c[C_HAR]):
            sv[i] = min(max(log(max(scal[i], Float32(1e-12)) / sc), Float32(-4)), Float32(4))
        for i in range(c[C_HAR], ns):
            sv[i] = scal[i]
        linear(x, sv, w + c[O_EMBS], w + c[O_EMBS] + D * ns, 1, ns, D, False)
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
        var first = T - 1 if l == NL - 1 else 0
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
    linear(o, z, w + c[O_HEAD], w + c[O_HEAD] + H * NQ * D, 1, D, H * NQ, False)
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


def infer(cfg: List[Int], w: List[Float32], inp: FP, res: FP, n: Int, threads: Int, sym: Bool):
    var c = ip(cfg)
    var wp = fp(w)
    var HQ = cfg[C_H] * NQ
    var nfl = cfg[C_NFC] * cfg[C_L]
    var ncl = cfg[C_NCC] * cfg[C_LC]
    var ns = cfg[C_NS]
    var ncal = 3 if cfg[C_CAL] != 0 else 0
    var pc = inp + n * nfl
    var ps = pc + n * ncl
    var pk = ps + n * ns
    var nscr = cfg[C_SCRATCH]
    var chunk = max(1, min(16, (n + threads - 1) // max(threads, 1)))
    var nchunks = (n + chunk - 1) // chunk

    def work(i: Int) {var c, var wp, var inp, var pc, var ps, var pk, var res, var n, var HQ, var nfl, var ncl, var ns, var ncal, var nscr, var chunk, var sym}:
        var scratch = List[Float32](length=nscr + 2 * HQ, fill=0)
        var s = scratch.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
        var tmp = s + nscr
        var hi = min(n, (i + 1) * chunk)
        for row in range(i * chunk, hi):
            forward_row(c, wp, inp + row * nfl, pc + row * ncl, ps + row * ns, pk + row * ncal, res + row * HQ, s, 1)
            if sym:
                forward_row(c, wp, inp + row * nfl, pc + row * ncl, ps + row * ns, pk + row * ncal, tmp, s, -1)
                var q = res + row * HQ
                for h in range(HQ // NQ):
                    for j in range(NQ):
                        tmp[HQ + h * NQ + j] = 0.5 * (q[h * NQ + j] - tmp[h * NQ + NQ - 1 - j])
                for j in range(HQ):
                    q[j] = tmp[HQ + j]

    if threads <= 1 or nchunks == 1:
        for i in range(nchunks):
            work(i)
    else:
        parallelize(work, nchunks, min(threads, nchunks))


def use_sym() -> Bool:
    return getenv("BMT_SYM", "0") == "1"


def selftest(dir: String) raises -> Bool:
    var cfg = List[Int]()
    var w = List[Float32]()
    load_model(dir + "/bmt2.bin", cfg, w)
    var xin = List[Float32]()
    var ref_ = List[Float32]()
    var ni = read_f32(dir + "/selftest_in.f32", xin, 0)
    var nr = read_f32(dir + "/selftest_ref.f32", ref_, 0)
    var HQ = cfg[C_H] * NQ
    var n = nr // HQ
    if n * cfg[C_ROWIN] != ni:
        raise Error("selftest vector does not match the model")
    var out = List[Float32](length=nr, fill=0)
    infer(cfg, w, fp(xin), fp(out), n, 4, False)
    var maxrel = Float32(0)
    var nfl = cfg[C_NFC] * cfg[C_L]
    for row in range(n):
        var sc = rms(fp(xin) + row * nfl, cfg[C_L])
        for h in range(cfg[C_H]):
            var k = sc * sqrt(Float32(cfg[C_HZ + h]))
            for j in range(NQ):
                maxrel = max(maxrel, abs(out[row * HQ + h * NQ + j] - ref_[row * HQ + h * NQ + j]) / k)
    var tol = Float32(2e-4)
    print("selftest rows", n, "tokens", cfg[C_NT], "max err in scale units", maxrel, "tol", tol, "PASS" if maxrel <= tol else "FAIL")
    return maxrel <= tol


def synth(cfg: List[Int], batch: Int) -> List[Float32]:
    var ni = batch * cfg[C_ROWIN]
    var xin = List[Float32](length=ni, fill=0)
    var seed: UInt64 = 12345
    for i in range(ni):
        seed = seed * 6364136223846793005 + 1442695040888963407
        xin[i] = (Float32(Int((seed >> 40) & 0xFFFF)) / 65536.0) * 0.02 + 0.0001
    if cfg[C_CAL] != 0:
        for i in range(batch * 3):
            xin[ni - batch * 3 + i] = Float32(i % 2)
    return xin^


def bench(path: String, batch: Int, iters: Int, threads: Int) raises:
    var cfg = List[Int]()
    var w = List[Float32]()
    load_model(path, cfg, w)
    var xin = synth(cfg, batch)
    var out = List[Float32](length=batch * cfg[C_H] * NQ, fill=0)
    var sym = use_sym()
    for _ in range(3):
        infer(cfg, w, fp(xin), fp(out), batch, threads, sym)
    var lat = List[Float64](length=iters, fill=0)
    var t00 = perf_counter_ns()
    for it in range(iters):
        var t0 = perf_counter_ns()
        infer(cfg, w, fp(xin), fp(out), batch, threads, sym)
        lat[it] = Float64(perf_counter_ns() - t0) / 1e6
    var total = Float64(perf_counter_ns() - t00) / 1e9
    sort(lat)
    print("batch", batch, "threads", threads, "rows/s", Float64(batch * iters) / total, "p50_ms", lat[iters // 2], "p99_ms", lat[min(iters - 1, (iters * 99) // 100)],
          "check", out[0])


def serve(path: String, threads: Int) raises:
    var cfg = List[Int]()
    var w = List[Float32]()
    load_model(path, cfg, w)
    var sym = use_sym()
    print("ready rowin", cfg[C_ROWIN], "H", cfg[C_H], "tokens", cfg[C_NT], "threads", threads, file=stderr)
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
        var ni = n * cfg[C_ROWIN]
        if len(bi) < ni:
            bi = List[Float32](length=ni, fill=0)
        if len(bo) < n * cfg[C_H] * NQ:
            bo = List[Float32](length=n * cfg[C_H] * NQ, fill=0)
        if fd_read(0, bi.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), ni * 4) != ni * 4:
            raise Error("short request body")
        infer(cfg, w, fp(bi), fp(bo), n, threads, sym)
        fd_write(1, hp, 4)
        fd_write(1, bo.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), n * cfg[C_H] * NQ * 4)


def find5(p: Pointer[UInt8, MutAnyOrigin], n: Int, a: Int, b: Int, c: Int, d: Int, e: Int) -> Int:
    """Index just after the '[' that follows the JSON key spelled by the five bytes a..e, or -1."""
    for i in range(n - 8):
        if Int(p[i]) == 34 and Int(p[i + 1]) == a and Int(p[i + 2]) == b and Int(p[i + 3]) == c and Int(p[i + 4]) == d and Int(p[i + 5]) == e and Int(p[i + 6]) == 34:
            var j = i + 7
            while j < n and Int(p[j]) != 91:
                j += 1
            return j + 1
    return -1


def numbers(p: Pointer[UInt8, MutAnyOrigin], n: Int, start: Int, mut vals: List[Float64]) raises -> Int:
    """Every number of the (possibly nested) JSON array that starts at `start`; returns the index after its closing bracket."""
    var depth = 1
    var i = start
    var tok = String("")
    while i < n and depth > 0:
        var ch = Int(p[i])
        var isnum = (ch >= 48 and ch <= 57) or ch == 45 or ch == 43 or ch == 46 or ch == 101 or ch == 69
        if isnum:
            tok += chr(ch)
        else:
            if tok.byte_length() > 0:
                vals.append(atof(tok))
                tok = String("")
            if ch == 91:
                depth += 1
            elif ch == 93:
                depth -= 1
        i += 1
    return i


def fmt(v: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0:
            s += ", "
        s += String(v[i])
    return s + "]"


def rotation(wpath: String, hpath: String, threads: Int) raises:
    """Scores for a rotation consumer from its own candle history file (schema rotation-completed-history-v1:
    {"pairs": [...], "hours": [...], "ohlcv": [[[open, high, low, close, volume] per pair] per hour]}), printed as one JSON object."""
    var cfg = List[Int]()
    var w = List[Float32]()
    load_model(wpath, cfg, w)
    var wf = open(wpath, "r")
    var wb = wf.read_bytes()
    var hp = wb.unsafe_ptr().unsafe_bitcast[Int32]()
    var L = cfg[C_L]
    if cfg[C_NCC] != 0 or cfg[C_NS] != 0:
        raise Error("this model needs coarse / scalar inputs that a candle history file does not provide")
    for c in range(cfg[C_NFC]):
        if Int(hp[44 + c]) > 3:
            raise Error("this model needs input channels beyond return / range / Garman-Klass / volume")
    var f = open(hpath, "r")
    var bytes = f.read_bytes()
    var n = len(bytes)
    var p = bytes.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var ip_ = find5(p, n, 112, 97, 105, 114, 115)
    var ih = find5(p, n, 104, 111, 117, 114, 115)
    var io = find5(p, n, 111, 104, 108, 99, 118)
    if ip_ < 0 or ih < 0 or io < 0:
        raise Error("history file: pairs / hours / ohlcv not found")
    var pairs = String("[")
    var npair = 0
    var i = ip_
    while Int(p[i]) != 93:
        if Int(p[i]) == 34:
            npair += 1
        pairs += chr(Int(p[i]))
        i += 1
    pairs += "]"
    npair = npair // 2
    var hv = List[Float64]()
    _ = numbers(p, n, ih, hv)
    var ov = List[Float64]()
    _ = numbers(p, n, io, ov)
    var nh = len(hv)
    if len(ov) != nh * npair * 5:
        raise Error("history file: ohlcv has " + String(len(ov)) + " numbers, expected hours x pairs x 5 = " + String(nh * npair * 5))
    if nh < L + 1:
        raise Error("history file has " + String(nh) + " hours; this model needs " + String(L + 1) + " completed hourly candles per pair")
    for k in range(nh - L, nh):
        if Int(hv[k]) != Int(hv[k - 1]) + 1:
            raise Error("history file: the last " + String(L + 1) + " hours are not consecutive")
    var hr = Int(hv[nh - 1])
    var rowin = cfg[C_ROWIN]
    var nfc = cfg[C_NFC]
    var xin = List[Float32](length=npair * rowin, fill=0)
    for j in range(npair):
        for t in range(L):
            var a = ((nh - L + t) * npair + j) * 5
            var b = ((nh - L + t - 1) * npair + j) * 5
            if ov[a + 3] <= 0 or ov[b + 3] <= 0 or ov[a + 2] <= 0 or ov[a + 1] < ov[a + 2]:
                raise Error("history file: non-positive or inconsistent candle")
            var r = Float32(log(ov[a + 3] / ov[b + 3]))
            var hl = Float32(log(ov[a + 1] / ov[a + 2]))
            for c in range(nfc):
                var id = Int(hp[44 + c])
                var v = r
                if id == 1:
                    v = hl
                elif id == 2:
                    v = sqrt(max(Float32(0.5) * hl * hl - Float32(0.3862944) * r * r, Float32(0)))
                elif id == 3:
                    v = Float32(ov[a + 4])
                xin[(j * nfc + c) * L + t] = v
        if cfg[C_CAL] != 0:
            var base = npair * nfc * L + j * 3
            xin[base] = Float32((hr + 1) % 24)
            xin[base + 1] = Float32(((hr + 1) // 24 + 4) % 7)
            xin[base + 2] = 0
    var HQ = cfg[C_H] * NQ
    var q = List[Float32](length=npair * HQ, fill=0)
    infer(cfg, w, fp(xin), fp(q), npair, threads, use_sym())
    var hsel = cfg[C_H] - 1                                            # the longest horizon (24 bars)
    var med = List[Float64]()
    var sig = List[Float64]()
    var q05 = List[Float64]()
    var q95 = List[Float64]()
    var sc = List[Float64]()
    var mean = Float64(0)
    for j in range(npair):
        var o = j * HQ + hsel * NQ
        med.append(Float64(q[o + 10]))
        sig.append(Float64(q[o + 17] - q[o + 3]) / 2.0728)             # (q85 - q15) / (2 x 1.0364)
        q05.append(Float64(q[o + 1]))
        q95.append(Float64(q[o + 19]))
        sc.append(med[j] / max(sig[j], 1e-12))
        mean += sc[j]
    mean = mean / Float64(npair)
    var sd = Float64(0)
    for j in range(npair):
        sd += (sc[j] - mean) * (sc[j] - mean)
    sd = sqrt(sd / Float64(max(npair - 1, 1)))
    for j in range(npair):
        sc[j] = (sc[j] - mean) / max(sd, 1e-12)
    print("{\"schema\": \"bmt-rotation-forecast-v1\", \"pairs\": " + pairs + ", \"issued_hour\": " + String(hr) + ", \"usable_execution_hour\": " + String(hr + 1)
          + ", \"horizon_hours\": " + String(cfg[C_HZ + hsel]) + ", \"raw_scores\": " + fmt(sc) + ", \"median_logret\": " + fmt(med) + ", \"sigma_logret\": " + fmt(sig)
          + ", \"q05_logret\": " + fmt(q05) + ", \"q95_logret\": " + fmt(q95) + ", \"flip_averaged\": " + ("true" if use_sym() else "false")
          + ", \"warning\": \"Ranks, not expected returns. Research inference only; no orders or fills.\"}")


def main() raises:
    var a = argv()
    if len(a) >= 3 and a[1] == "selftest":
        if not selftest(String(a[2])):
            raise Error("selftest failed")
    elif len(a) >= 5 and a[1] == "bench":
        bench(String(a[2]), Int(a[3]), Int(a[4]), Int(a[5]) if len(a) > 5 else 1)
    elif len(a) >= 3 and a[1] == "serve":
        serve(String(a[2]), Int(a[3]) if len(a) > 3 else 1)
    elif len(a) >= 4 and a[1] == "rotation":
        rotation(String(a[2]), String(a[3]), Int(a[4]) if len(a) > 4 else 1)
    else:
        print("usage: bmt2 selftest <dir> | bench <bmt2.bin> <batch> <iters> [threads] | serve <bmt2.bin> [threads] | rotation <bmt2.bin> <history.json> [threads]")
