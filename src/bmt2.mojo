"""bmt2 CPU inference: the bmt patch transformer with configurable inputs (train/model2.py is the reference arithmetic).

  bmt2 selftest <dir>                     bmt2.bin + selftest_in.f32 vs selftest_ref.f32, exit 1 above tolerance
  bmt2 bench <bmt2.bin> <batch> <iters> [threads]
  (bench / serve / rotation take a comma-separated list of weight files: a seed ensemble, quantile-averaged)
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
comptime P_EMB = 58         # offsets of the packed (transposed, 16-column panel) weights appended after the raw ones
comptime P_EMBC = 59
comptime P_EMBS = 60
comptime P_LAYER = 61
comptime P_HEAD = 62
comptime C_TP = 63          # tokens padded to a multiple of 8
comptime NCFG = 64
comptime W16 = SIMD[DType.float32, 16]


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


@always_inline
def tile[MR: Int](dst: FP, src: FP, wp: FP, bias: FP, r: Int, K: Int, n_out: Int, j0: Int, relu: Bool):
    """dst[r..r+MR, j0..j0+16) = src rows @ panel + bias; one 16-wide weight row load feeds MR broadcast FMAs."""
    var x0 = src + r * K
    var a0 = W16(0)
    var a1 = W16(0)
    var a2 = W16(0)
    var a3 = W16(0)
    var a4 = W16(0)
    var a5 = W16(0)
    for k in range(K):
        var wv = wp.unsafe_load[width=16](k * 16)
        a0 = W16(x0[k]).fma(wv, a0)
        comptime if MR > 1:
            a1 = W16(x0[K + k]).fma(wv, a1)
        comptime if MR > 2:
            a2 = W16(x0[2 * K + k]).fma(wv, a2)
        comptime if MR > 3:
            a3 = W16(x0[3 * K + k]).fma(wv, a3)
        comptime if MR > 4:
            a4 = W16(x0[4 * K + k]).fma(wv, a4)
        comptime if MR > 5:
            a5 = W16(x0[5 * K + k]).fma(wv, a5)
    var full = j0 + 16 <= n_out
    var bv = W16(0)
    if full:
        bv = bias.unsafe_load[width=16](j0)
    else:
        for jj in range(n_out - j0):
            bv[jj] = bias[j0 + jj]
    var d = dst + r * n_out + j0
    comptime for m in range(MR):
        var y = a0
        comptime if m == 1:
            y = a1
        comptime if m == 2:
            y = a2
        comptime if m == 3:
            y = a3
        comptime if m == 4:
            y = a4
        comptime if m == 5:
            y = a5
        y = y + bv
        if relu:
            y = max(y, W16(0))
        if full:
            d.store(m * n_out, y)
        else:
            for jj in range(n_out - j0):
                d[m * n_out + jj] = y[jj]


def linear(dst: FP, src: FP, wp: FP, b: FP, rows: Int, n_in: Int, n_out: Int, relu: Bool):
    """dst[r, j] = b[j] + sum_k src[r, k] * w[j, k], w packed by pack_w: panel-major, k-major inside, 16 columns."""
    for p in range((n_out + 15) // 16):
        var wpp = wp + p * n_in * 16
        var j0 = p * 16
        var r = 0
        while r + 6 <= rows:
            tile[6](dst, src, wpp, b, r, n_in, n_out, j0, relu)
            r += 6
        var rem = rows - r
        if rem == 5:
            tile[5](dst, src, wpp, b, r, n_in, n_out, j0, relu)
        elif rem == 4:
            tile[4](dst, src, wpp, b, r, n_in, n_out, j0, relu)
        elif rem == 3:
            tile[3](dst, src, wpp, b, r, n_in, n_out, j0, relu)
        elif rem == 2:
            tile[2](dst, src, wpp, b, r, n_in, n_out, j0, relu)
        elif rem == 1:
            tile[1](dst, src, wpp, b, r, n_in, n_out, j0, relu)


def packed_size(n_in: Int, n_out: Int) -> Int:
    return ((n_out + 15) // 16) * 16 * n_in


def pack_w(dst: FP, w: FP, n_in: Int, n_out: Int):
    """w[n_out, n_in] row-major -> dst[panel][k][16], zero beyond n_out."""
    for p in range((n_out + 15) // 16):
        for k in range(n_in):
            for jj in range(16):
                var j = p * 16 + jj
                dst[(p * n_in + k) * 16 + jj] = w[j * n_in + k] if j < n_out else Float32(0)


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


def attention(dst: FP, qkv: FP, kt: FP, sc: FP, T: Int, Tp: Int, D: Int, NH: Int, first: Int):
    """softmax(q k^T / sqrt(hd)) v per head for query tokens first..T-1; qkv[t] = [q | k | v], each D wide. Scores are computed 8 keys
    at a time from a per-head transposed copy of k (kt[hd, Tp], zero padded)."""
    var hd = D // NH
    var inv = Float32(1.0) / sqrt(Float32(hd))
    for h in range(NH):
        for j in range(T):
            var kj = qkv + j * 3 * D + D + h * hd
            for d in range(hd):
                kt[d * Tp + j] = kj[d] * inv
        for j in range(T, Tp):
            for d in range(hd):
                kt[d * Tp + j] = 0
        for i in range(first, T):
            var qi = qkv + i * 3 * D + h * hd
            var mxv = V(-1e30)
            for jb in range(0, Tp, 8):
                var acc = V(0)
                for d in range(hd):
                    acc = V(qi[d]).fma(kt.unsafe_load[width=8](d * Tp + jb), acc)
                sc.store(jb, acc)
            for j in range(T, Tp):
                sc[j] = -1e30
            for jb in range(0, Tp, 8):
                mxv = max(mxv, sc.unsafe_load[width=8](jb))
            var mx = mxv.reduce_max()
            var tv = V(0)
            for jb in range(0, Tp, 8):
                var e = exp(sc.unsafe_load[width=8](jb) - V(mx))
                sc.store(jb, e)
                tv += e
            var it = Float32(1) / tv.reduce_add()
            var o = dst + i * D + h * hd
            for k in range(0, hd, 8):
                o.store(k, V(0))
            for j in range(T):
                var p = V(sc[j] * it)
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
    var tp = (nt + 7) // 8 * 8
    cfg[C_TP] = tp
    cfg[C_SCRATCH] = cfg[C_L] * ff + cfg[C_LC] * fc + 2 * nt * D + nt * 3 * D + nt * D + nt * 4 * D + tp + D + 16 + cfg[C_H] * NQ + 16 + D * tp + 16
    var po = o
    cfg[P_EMB] = po
    po += packed_size(cfg[C_P] * ff, D)
    cfg[P_EMBC] = po
    if tc > 0:
        po += packed_size(cfg[C_PC] * fc, D)
    cfg[P_EMBS] = po
    if ts > 0:
        po += packed_size(ns, D)
    cfg[P_LAYER] = po
    po += cfg[C_NL] * (packed_size(D, 3 * D) + packed_size(D, D) + packed_size(D, 4 * D) + packed_size(4 * D, D))
    cfg[P_HEAD] = po
    po += packed_size(D, cfg[C_H] * NQ)
    cfg[C_ROWIN] = cfg[C_NFC] * cfg[C_L] + cfg[C_NCC] * cfg[C_LC] + ns + (3 if cfg[C_CAL] != 0 else 0)
    if len(bytes) != 256 + o * 4:
        raise Error(path + ": expected " + String(256 + o * 4) + " bytes, got " + String(len(bytes)))
    w = List[Float32](length=po, fill=0)
    var src = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
    var d = fp(w)
    for i in range(o):
        d[i] = src[64 + i]
    pack_w(d + cfg[P_EMB], d + cfg[O_EMB], cfg[C_P] * ff, D)
    if tc > 0:
        pack_w(d + cfg[P_EMBC], d + cfg[O_EMBC], cfg[C_PC] * fc, D)
    if ts > 0:
        pack_w(d + cfg[P_EMBS], d + cfg[O_EMBS], ns, D)
    var lsz = 12 * D * D + 13 * D
    var psz = packed_size(D, 3 * D) + packed_size(D, D) + packed_size(D, 4 * D) + packed_size(4 * D, D)
    for l in range(cfg[C_NL]):
        var lw = d + cfg[O_LAYER] + l * lsz
        var pw = d + cfg[P_LAYER] + l * psz
        pack_w(pw, lw + 2 * D, D, 3 * D)
        pack_w(pw + 3 * D * D, lw + 2 * D + 3 * D * D + 3 * D, D, D)
        pack_w(pw + 4 * D * D, lw + 2 * D + 4 * D * D + 4 * D + 2 * D, D, 4 * D)
        pack_w(pw + 8 * D * D, lw + 2 * D + 8 * D * D + 10 * D, 4 * D, D)
    pack_w(d + cfg[P_HEAD], d + cfg[O_HEAD], D, cfg[C_H] * NQ)


def rms(x: FP, n: Int) -> Float32:
    var acc = V(0)
    for i in range(0, n, 8):
        var v = x.unsafe_load[width=8](i)
        acc = v.fma(v, acc)
    return max(sqrt(acc.reduce_add() / Float32(n)), Float32(1e-8))


@always_inline
def vasinh(x: V) -> V:
    var a = abs(x)
    var y = log1p(a + a * a / (V(1) + sqrt(a * a + V(1))))
    return x.lt(V(0)).select(-y, y)


def norm_stream(feat: FP, x: FP, kinds: IP, nch: Int, T: Int, F: Int, sgn: Float32) -> Float32:
    """Raw channels x[nch, T] -> feat[T, F]; returns the scale of channel 0. sgn = -1 evaluates the mirrored series. T % 8 == 0."""
    var s = rms(x, T)
    var g = V(sgn / s)
    for i in range(0, T, 8):
        var xr = vasinh(x.unsafe_load[width=8](i) * g)
        var x2 = xr * xr
        for u in range(8):
            feat[(i + u) * F] = xr[u]
            feat[(i + u) * F + 1] = x2[u]
    var f = 2
    for c in range(1, nch):
        var v = x + c * T
        var k = kinds[c]
        if k == 1:
            var gs = V(Float32(1) / s)
            for i in range(0, T, 8):
                var y = vasinh(v.unsafe_load[width=8](i) * gs)
                for u in range(8):
                    feat[(i + u) * F + f] = y[u]
        elif k == 2:
            var mv = V(0)
            for i in range(0, T, 8):
                mv += v.unsafe_load[width=8](i)
            var m = max(mv.reduce_add() / Float32(T), Float32(1e-12))
            var gm = V(Float32(1) / m)
            for i in range(0, T, 8):
                var y = log1p(v.unsafe_load[width=8](i) * gm)
                for u in range(8):
                    feat[(i + u) * F + f] = y[u]
        elif k == 3:
            for i in range(T):
                var sh = v[i] if sgn > 0 else Float32(1) - v[i]
                feat[i * F + f] = (sh - 0.5) * 4
        elif k == 4:
            var gj = V(sgn / rms(v, T))
            for i in range(0, T, 8):
                var xb = vasinh(v.unsafe_load[width=8](i) * gj)
                for u in range(8):
                    feat[(i + u) * F + f] = xb[u]
                    feat[(i + u) * F + f + 1] = xb[u] * feat[(i + u) * F]
            f += 1
        else:
            for i in range(T):
                feat[i * F + f] = sgn * v[i]
        f += 1
    return s


comptime RMAX = 8


def scratch_size(c: IP, R: Int) -> Int:
    var D = c[C_D]
    var T = c[C_NT]
    var Tp = c[C_TP]
    return c[C_L] * c[C_FF] + c[C_LC] * c[C_FC] + R * T * D * 10 + Tp + D * Tp + R * (D + 16) + R * c[C_H] * NQ + 16 + R * 7 * D + 64


def forward_block(c: IP, w: FP, rows: IP, sgns: FP, R: Int, fine: FP, coarse: FP, scal: FP, cal: FP, dst: FP, s: FP):
    """R sequences (row rows[i] of the input block, mirrored when sgns[i] < 0) through the network together, so each weight panel
    is streamed once per R * T tokens. out[i * H * 21] = raw quantiles in return units."""
    var L = c[C_L]
    var D = c[C_D]
    var T = c[C_NT]
    var Tp = c[C_TP]
    var TF = c[C_TF]
    var TC = c[C_TC]
    var TS = c[C_TS]
    var FF = c[C_FF]
    var FC = c[C_FC]
    var NL = c[C_NL]
    var NH = c[C_NH]
    var H = c[C_H]
    var M = R * T
    var nfl = c[C_NFC] * L
    var ncl = c[C_NCC] * c[C_LC]
    var ns = c[C_NS]
    var feat = s
    var featc = feat + L * FF
    var x = featc + c[C_LC] * FC
    var hb = x + M * D
    var qkv = hb + M * D
    var att = qkv + M * 3 * D
    var u = att + M * D
    var scb = u + M * 4 * D
    var kt = scb + Tp
    var z = kt + D * Tp
    var o = z + R * (D + 16)
    var sv = o + R * H * NQ
    var xl = sv + 16
    var al = xl + R * D
    var hl = al + R * D
    var ul = hl + R * D
    var scs = List[Float32](length=R, fill=0)
    var pos = w + c[O_POS]
    for r in range(R):
        var row = rows[r]
        var sg = sgns[r]
        var xr = x + r * T * D
        var sc = norm_stream(feat, fine + row * nfl, c + C_FK, c[C_NFC], L, FF, sg)
        scs[r] = sc
        linear(xr + (TS + TC) * D, feat, w + c[P_EMB], w + c[O_EMB] + D * c[C_P] * FF, TF, c[C_P] * FF, D, False)
        if TC > 0:
            _ = norm_stream(featc, coarse + row * ncl, c + C_CK, c[C_NCC], c[C_LC], FC, sg)
            linear(xr + TS * D, featc, w + c[P_EMBC], w + c[O_EMBC] + D * c[C_PC] * FC, TC, c[C_PC] * FC, D, False)
        if TS > 0:
            var sp = scal + row * ns
            for i in range(c[C_HAR]):
                sv[i] = min(max(log(max(sp[i], Float32(1e-12)) / sc), Float32(-4)), Float32(4))
            for i in range(c[C_HAR], ns):
                sv[i] = sp[i]
            linear(xr, sv, w + c[P_EMBS], w + c[O_EMBS] + D * ns, 1, ns, D, False)
        for i in range(0, T * D, 8):
            xr.store(i, xr.unsafe_load[width=8](i) + pos.unsafe_load[width=8](i))
    var lsz = 12 * D * D + 13 * D
    var psz = 12 * D * D
    for l in range(NL):
        var lw = w + c[O_LAYER] + l * lsz
        var pw = w + c[P_LAYER] + l * psz
        var n1 = lw
        var bq = n1 + 2 * D + 3 * D * D
        var bo = bq + 3 * D + D * D
        var n2 = bo + D
        var b1 = n2 + 2 * D + 4 * D * D
        var b2 = b1 + 4 * D + 4 * D * D
        layernorm(hb, x, n1, n1 + D, M, D)
        linear(qkv, hb, pw, bq, M, D, 3 * D, False)
        if l < NL - 1:
            for r in range(R):
                attention(att + r * T * D, qkv + r * T * 3 * D, kt, scb, T, Tp, D, NH, 0)
            linear(hb, att, pw + 3 * D * D, bo, M, D, D, False)
            for i in range(0, M * D, 8):
                x.store(i, x.unsafe_load[width=8](i) + hb.unsafe_load[width=8](i))
            layernorm(hb, x, n2, n2 + D, M, D)
            linear(u, hb, pw + 4 * D * D, b1, M, D, 4 * D, True)
            linear(hb, u, pw + 8 * D * D, b2, M, 4 * D, D, False)
            for i in range(0, M * D, 8):
                x.store(i, x.unsafe_load[width=8](i) + hb.unsafe_load[width=8](i))
        else:                                                         # last layer: only the last token of each sequence is read
            for r in range(R):
                attention(att + r * T * D, qkv + r * T * 3 * D, kt, scb, T, Tp, D, NH, T - 1)
                var src = att + (r * T + T - 1) * D
                var xs = x + (r * T + T - 1) * D
                for i in range(0, D, 8):
                    al.store(r * D + i, src.unsafe_load[width=8](i))
                    xl.store(r * D + i, xs.unsafe_load[width=8](i))
            linear(hl, al, pw + 3 * D * D, bo, R, D, D, False)
            for i in range(0, R * D, 8):
                xl.store(i, xl.unsafe_load[width=8](i) + hl.unsafe_load[width=8](i))
            layernorm(hl, xl, n2, n2 + D, R, D)
            linear(ul, hl, pw + 4 * D * D, b1, R, D, 4 * D, True)
            linear(hl, ul, pw + 8 * D * D, b2, R, 4 * D, D, False)
            for i in range(0, R * D, 8):
                xl.store(i, xl.unsafe_load[width=8](i) + hl.unsafe_load[width=8](i))
    if NL == 0:
        for r in range(R):
            for i in range(0, D, 8):
                xl.store(r * D + i, x.unsafe_load[width=8]((r * T + T - 1) * D + i))
    layernorm(z, xl, w + c[O_NF], w + c[O_NF] + D, R, D)
    if c[C_CAL] != 0:
        for r in range(R):
            var ck = cal + rows[r] * 3
            var e0 = w + c[O_CAL] + Int(ck[0]) * D
            var e1 = w + c[O_CAL] + (24 + Int(ck[1])) * D
            var e2 = w + c[O_CAL] + (31 + Int(ck[2])) * D
            var zr = z + r * D
            for i in range(0, D, 8):
                zr.store(i, zr.unsafe_load[width=8](i) + e0.unsafe_load[width=8](i) + e1.unsafe_load[width=8](i) + e2.unsafe_load[width=8](i))
    linear(o, z, w + c[P_HEAD], w + c[O_HEAD] + H * NQ * D, R, D, H * NQ, False)
    for r in range(R):
        for h in range(H):
            var oh = o + (r * H + h) * NQ
            var qo = dst + (r * H + h) * NQ
            var k = scs[r] * sqrt(Float32(c[C_HZ + h]))
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
    var pc = inp + n * cfg[C_NFC] * cfg[C_L]
    var ps = pc + n * cfg[C_NCC] * cfg[C_LC]
    var pk = ps + n * cfg[C_NS]
    var G = RMAX // 2 if sym else RMAX                               # input rows per block
    var nblk = (n + G - 1) // G
    var per = max(1, (nblk + threads - 1) // max(threads, 1))
    per = min(per, 4)
    var nchunks = (nblk + per - 1) // per

    def work(i: Int) {var c, var wp, var inp, var pc, var ps, var pk, var res, var n, var HQ, var G, var per, var sym}:
        var scratch = List[Float32](length=scratch_size(c, RMAX) + RMAX * HQ, fill=0)
        var s = scratch.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
        var tmp = s + scratch_size(c, RMAX)
        var rl = List[Int](length=RMAX, fill=0)
        var sl = List[Float32](length=RMAX, fill=1)
        for blk in range(i * per, min((i + 1) * per, (n + G - 1) // G)):
            var r0 = blk * G
            var g = min(G, n - r0)
            var R = 2 * g if sym else g
            for j in range(g):
                rl[j] = r0 + j
                sl[j] = 1
                if sym:
                    rl[g + j] = r0 + j
                    sl[g + j] = -1
            forward_block(c, wp, ip(rl), fp(sl), R, inp, pc, ps, pk, tmp, s)
            for j in range(g):
                var q = res + (r0 + j) * HQ
                var a = tmp + j * HQ
                if sym:
                    var b = tmp + (g + j) * HQ
                    for h in range(HQ // NQ):
                        for k in range(NQ):
                            q[h * NQ + k] = 0.5 * (a[h * NQ + k] - b[h * NQ + NQ - 1 - k])
                else:
                    for k in range(HQ):
                        q[k] = a[k]

    if threads <= 1 or nchunks == 1:
        for i in range(nchunks):
            work(i)
    else:
        parallelize(work, nchunks, min(threads, nchunks))


def use_sym() -> Bool:
    return getenv("BMT_SYM", "0") == "1"


def expand_spec(spec: String, mut sym: Bool) raises -> String:
    """A weight spec is a comma-separated list of bmt2.bin files, or @level.txt: whitespace-separated member files (relative to the
    level file) and the optional token `sym` (flip averaging). Returns the comma-separated member list."""
    if not spec.startswith("@"):
        return spec
    var path = String(spec[byte=1:])
    var dir = String("")
    var cut = -1
    for i in range(path.byte_length()):
        if path[byte=i] == "/":
            cut = i
    if cut >= 0:
        dir = String(path[byte=:cut + 1])
    var txt = open(path, "r").read()
    var out = String("")
    for line in txt.splitlines():
        var l = String(line.strip())
        if l.byte_length() == 0 or l.startswith("#"):
            continue
        for tok in l.split():
            var t = String(tok)
            if t == "sym":
                sym = True
            else:
                if out.byte_length() > 0:
                    out += ","
                out += t if t.startswith("/") else dir + t
    return out


struct Ens(Movable):
    """Members that share the input contract; the forecast is the quantile average (monotone because every member is)."""
    var cfgs: List[List[Int]]
    var ws: List[List[Float32]]
    var sym: Bool

    def __init__(out self, spec_in: String) raises:
        self.cfgs = List[List[Int]]()
        self.ws = List[List[Float32]]()
        var sy = use_sym()
        var spec = expand_spec(spec_in, sy)
        self.sym = sy
        for p in spec.split(","):
            var cfg = List[Int]()
            var w = List[Float32]()
            load_model(String(p), cfg, w)
            if len(self.cfgs) > 0:
                var c0 = self.cfgs[0].copy()
                if cfg[C_ROWIN] != c0[C_ROWIN] or cfg[C_H] != c0[C_H] or cfg[C_NFC] != c0[C_NFC] or cfg[C_NCC] != c0[C_NCC] or cfg[C_NS] != c0[C_NS] or cfg[C_CAL] != c0[C_CAL]:
                    raise Error(String(p) + ": ensemble members must share the input contract")
                for i in range(4):
                    if cfg[C_HZ + i] != c0[C_HZ + i] or cfg[C_FK + i] != c0[C_FK + i]:
                        raise Error(String(p) + ": ensemble members must share horizons and channel kinds")
            self.cfgs.append(cfg^)
            self.ws.append(w^)

    def cfg(self) -> List[Int]:
        return self.cfgs[0].copy()

    def infer(self, inp: FP, res: FP, n: Int, threads: Int, sym: Bool):
        var K = len(self.cfgs)
        infer(self.cfgs[0], self.ws[0], inp, res, n, threads, sym)
        if K == 1:
            return
        var HQ = self.cfgs[0][C_H] * NQ
        var tmp = List[Float32](length=n * HQ, fill=0)
        for m in range(1, K):
            infer(self.cfgs[m], self.ws[m], inp, fp(tmp), n, threads, sym)
            for i in range(n * HQ):
                res[i] += tmp[i]
        var a = Float32(1) / Float32(K)
        for i in range(n * HQ):
            res[i] *= a


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
    var ens = Ens(path)
    var cfg = ens.cfg()
    var xin = synth(cfg, batch)
    var out = List[Float32](length=batch * cfg[C_H] * NQ, fill=0)
    var sym = ens.sym
    for _ in range(3):
        ens.infer(fp(xin), fp(out), batch, threads, sym)
    var lat = List[Float64](length=iters, fill=0)
    var t00 = perf_counter_ns()
    for it in range(iters):
        var t0 = perf_counter_ns()
        ens.infer(fp(xin), fp(out), batch, threads, sym)
        lat[it] = Float64(perf_counter_ns() - t0) / 1e6
    var total = Float64(perf_counter_ns() - t00) / 1e9
    sort(lat)
    print("batch", batch, "threads", threads, "members", len(ens.cfgs), "rows/s", Float64(batch * iters) / total, "p50_ms", lat[iters // 2], "p99_ms", lat[min(iters - 1, (iters * 99) // 100)],
          "check", out[0])


def serve(path: String, threads: Int) raises:
    var ens = Ens(path)
    var cfg = ens.cfg()
    var sym = ens.sym
    print("ready rowin", cfg[C_ROWIN], "H", cfg[C_H], "tokens", cfg[C_NT], "threads", threads, "members", len(ens.cfgs), file=stderr)
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
        ens.infer(fp(bi), fp(bo), n, threads, sym)
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
    var ens = Ens(wpath)
    var cfg = ens.cfg()
    var sy0 = False
    var wf = open(String(expand_spec(wpath, sy0).split(",")[0]), "r")
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
    ens.infer(fp(xin), fp(q), npair, threads, ens.sym)
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
          + ", \"q05_logret\": " + fmt(q05) + ", \"q95_logret\": " + fmt(q95) + ", \"flip_averaged\": " + ("true" if ens.sym else "false")
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
