"""GPU inference for bmt2 (configurable inputs): fp32 GEMMs + fused kernels, same arithmetic as src/bmt2.mojo / train/model2.py.

  bmt2_gpu selftest <dir> | bench <bmt2.bin> <batch> <iters> | serve <bmt2.bin> [capacity]

Dimensions are compile-time defines (tools/gpu_defs.py prints them for a weight file): D, NL, NH, NFC, NCC, FF, FC, NS, HAR, CAL.
One build per model configuration and GPU arch. BMT_SYM=1 selects the flip-averaged (2-pass) forecast.
"""

from std.math import sqrt, exp, log, log1p, ceildiv
from std.sys import argv, stderr, get_defined_int
from std.os import getenv
from std.ffi import external_call
from std.memory import stack_allocation
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major, Idx
import linalg.matmul.vendor.blas as vendor_blas

comptime F32 = DType.float32
comptime D = get_defined_int["D", 128]()
comptime NL = get_defined_int["NL", 4]()
comptime NH = get_defined_int["NH", 4]()
comptime NFC = get_defined_int["NFC", 1]()      # fine channels
comptime NCC = get_defined_int["NCC", 0]()      # coarse channels
comptime FF = get_defined_int["FF", 2]()        # features per fine step
comptime FC = get_defined_int["FC", 2]()        # features per coarse step (unused when NCC = 0)
comptime NS = get_defined_int["NS", 0]()        # scalar inputs
comptime HAR = get_defined_int["HAR", 0]()      # number of log-ratio scalars
comptime CAL = get_defined_int["CAL", 0]()
comptime HD = D // NH
comptime L = 512
comptime P = 16
comptime LC = 256
comptime PC = 16
comptime TF = L // P
comptime TC = LC // PC if NCC > 0 else 0
comptime TS = 1 if NS > 0 else 0
comptime T = TF + TC + TS
comptime H = 4
comptime NQ = 21
comptime HQ = H * NQ
comptime MAGIC = 0x32544D42
comptime LSZ = 12 * D * D + 13 * D
comptime O_EMB = 0
comptime O_EMBC = D * P * FF + D
comptime O_EMBS = O_EMBC + (D * PC * FC + D if NCC > 0 else 0)
comptime O_POS = O_EMBS + (D * NS + D if NS > 0 else 0)
comptime O_LAYER = O_POS + T * D
comptime O_NF = O_LAYER + NL * LSZ
comptime O_CAL = O_NF + 2 * D
comptime O_HEAD = O_CAL + (34 * D if CAL > 0 else 0)
comptime NW = O_HEAD + HQ * D + HQ
comptime ROWIN = NFC * L + NCC * LC + NS + 3 * CAL
comptime NCS = NCC if NCC > 0 else 1
comptime NSS = NS if NS > 0 else 1

comptime BufF = DeviceBuffer[F32]
comptime FP = Pointer[Float32, MutAnyOrigin]
comptime BP = Pointer[UInt8, MutAnyOrigin]


@always_inline
def mpf(b: BufF) -> FP:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def fd_read(fd: Int, dst: BP, cap: Int) -> Int:
    var off = 0
    while off < cap:
        var r = external_call["read", Int](fd, dst + off, cap - off)
        if r <= 0:
            break
        off += r
    return off


def fd_write(fd: Int, src: BP, n: Int) raises:
    var off = 0
    while off < n:
        var w = external_call["write", Int](fd, src + off, n - off)
        if w <= 0:
            raise Error("write failed")
        off += w


def read_f32(path: String, mut dst: List[Float32], skip: Int) raises -> Int:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    var n = len(bytes) // 4 - skip
    dst = List[Float32](length=n, fill=0)
    var src = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(n):
        dst[i] = src[skip + i]
    return n


def gemm[N: Int, K: Int](ctx: DeviceContext, c: FP, a: FP, w: FP, m: Int) raises:
    """c[m, N] = a[m, K] @ w[N, K]^T."""
    var tc = TileTensor(c, row_major(m, Idx[N]))
    var ta = TileTensor(a, row_major(m, Idx[K]))
    var tw = TileTensor(w, row_major(Idx[N], Idx[K]))
    vendor_blas.matmul(ctx, tc, ta, tw, c_row_major=True, transpose_b=True)


def k_ln(ctx: DeviceContext, dst: FP, src: FP, g: FP, M: Int) raises:
    """LayerNorm over D per token row; g = [gain D | bias D]."""
    var m64 = Int64(M)

    def k() {var dst, var src, var g, var m64}:
        var r = Int(block_idx.x * block_dim.x + thread_idx.x)
        if r >= Int(m64):
            return
        var x = src + r * D
        var mean: Float32 = 0
        for i in range(D):
            mean += x[i]
        mean = mean / Float32(D)
        var v: Float32 = 0
        for i in range(D):
            var c = x[i] - mean
            v += c * c
        var inv = Float32(1.0) / sqrt(v / Float32(D) + Float32(1e-5))
        var y = dst + r * D
        for i in range(D):
            y[i] = (x[i] - mean) * inv * g[i] + g[D + i]

    ctx.enqueue_function(k, grid_dim=ceildiv(M, 64), block_dim=64)


def k_resid(ctx: DeviceContext, x: FP, hbp: FP, bias: FP, M: Int) raises:
    """x += h + bias."""
    var tot64 = Int64(M * D)

    def k() {var x, var hbp, var bias, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            x[i] = x[i] + hbp[i] + bias[i % D]

    ctx.enqueue_function(k, grid_dim=ceildiv(M * D, 256), block_dim=256)


def k_bias_relu(ctx: DeviceContext, u: FP, bias: FP, M: Int) raises:
    var tot64 = Int64(M * 4 * D)

    def k() {var u, var bias, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            u[i] = max(u[i] + bias[i % (4 * D)], Float32(0))

    ctx.enqueue_function(k, grid_dim=ceildiv(M * 4 * D, 256), block_dim=256)


@always_inline
def softplus(x: Float32) -> Float32:
    return max(x, Float32(0)) + log1p(exp(-abs(x)))


struct Engine(Movable):
    var cap: Int
    var w: BufF
    var xf: BufF
    var xc: BufF
    var xs: BufF
    var sc: BufF
    var feat: BufF
    var featc: BufF
    var ef: BufF
    var ec: BufF
    var x: BufF
    var hb: BufF
    var qkv: BufF
    var att: BufF
    var u: BufF
    var res: BufF
    var hin: HostBuffer[F32]
    var hout: HostBuffer[F32]
    var kf: Int64
    var kc: Int64
    var k0: Float32
    var k1: Float32
    var k2: Float32
    var k3: Float32

    def __init__(out self, ctx: DeviceContext, path: String, cap: Int) raises:
        var f = open(path, "r")
        var bytes = f.read_bytes()
        var hp = bytes.unsafe_ptr().unsafe_bitcast[Int32]()
        var ok = len(bytes) == 256 + NW * 4 and Int(hp[0]) == MAGIC and Int(hp[5]) == D and Int(hp[6]) == NL and Int(hp[7]) == NH and Int(hp[9]) == CAL
        ok = ok and Int(hp[10]) == NFC and Int(hp[11]) == NCC and Int(hp[12]) == HAR and Int(hp[12]) + Int(hp[13]) == NS
        if not ok:
            raise Error(path + ": weight file does not match this build")
        var kf: Int64 = 0
        var kc: Int64 = 0
        var ff = 0
        var fc = 0
        for i in range(NFC):
            kf = kf | (Int64(Int(hp[20 + i])) << Int64(4 * i))
            ff += 2 if Int(hp[20 + i]) == 0 or Int(hp[20 + i]) == 4 else 1
        for i in range(NCC):
            kc = kc | (Int64(Int(hp[36 + i])) << Int64(4 * i))
            fc += 2 if Int(hp[36 + i]) == 0 or Int(hp[36 + i]) == 4 else 1
        if ff != FF or (NCC > 0 and fc != FC):
            raise Error(path + ": feature counts do not match this build")
        self.kf = kf
        self.kc = kc
        self.k0 = sqrt(Float32(Int(hp[14])))
        self.k1 = sqrt(Float32(Int(hp[15])))
        self.k2 = sqrt(Float32(Int(hp[16])))
        self.k3 = sqrt(Float32(Int(hp[17])))
        var src = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
        var hbuf = ctx.enqueue_create_host_buffer[F32](NW)
        for i in range(NW):
            hbuf[i] = src[64 + i]
        self.w = ctx.enqueue_create_buffer[F32](NW)
        ctx.enqueue_copy(self.w, hbuf)
        ctx.synchronize()
        self.cap = cap
        var M = cap * T
        self.xf = ctx.enqueue_create_buffer[F32](cap * NFC * L)
        self.xc = ctx.enqueue_create_buffer[F32](cap * NCS * LC)
        self.xs = ctx.enqueue_create_buffer[F32](cap * (NSS + 3))
        self.sc = ctx.enqueue_create_buffer[F32](cap)
        self.feat = ctx.enqueue_create_buffer[F32](cap * L * FF)
        self.featc = ctx.enqueue_create_buffer[F32](cap * LC * FC)
        self.ef = ctx.enqueue_create_buffer[F32](cap * TF * D)
        self.ec = ctx.enqueue_create_buffer[F32](cap * (TC if TC > 0 else 1) * D)
        self.x = ctx.enqueue_create_buffer[F32](M * D)
        self.hb = ctx.enqueue_create_buffer[F32](M * D)
        self.qkv = ctx.enqueue_create_buffer[F32](M * 3 * D)
        self.att = ctx.enqueue_create_buffer[F32](M * D)
        self.u = ctx.enqueue_create_buffer[F32](M * 4 * D)
        self.res = ctx.enqueue_create_buffer[F32](cap * HQ)
        self.hin = ctx.enqueue_create_host_buffer[F32](cap * ROWIN)
        self.hout = ctx.enqueue_create_host_buffer[F32](cap * HQ)


@always_inline
def asinh_(x: Float32) -> Float32:
    """asinh without libm (not available on the device)."""
    var a = abs(x)
    var y = log1p(a + a * a / (Float32(1) + sqrt(a * a + Float32(1))))
    return y if x >= 0 else -y


def k_norm[NCH: Int, TT: Int, FT: Int](ctx: DeviceContext, px: FP, pf: FP, ps: FP, kinds: Int64, n: Int, write_s: Bool) raises:
    """Per row: raw channels x[NCH, TT] -> features [TT, FT] (kinds: 4 bits per channel); optionally stores the scale of channel 0."""
    var n64 = Int64(n)
    var ws = write_s

    def k() {var px, var pf, var ps, var kinds, var n64, var ws}:
        var r = Int(block_idx.x * block_dim.x + thread_idx.x)
        if r >= Int(n64):
            return
        var x = px + r * (NCH * TT)
        var f = pf + r * (TT * FT)
        var acc: Float32 = 0
        for i in range(TT):
            acc += x[i] * x[i]
        var s = max(sqrt(acc / Float32(TT)), Float32(1e-8))
        if ws:
            ps[r] = s
        for i in range(TT):
            var xr = asinh_(x[i] / s)
            f[i * FT] = xr
            f[i * FT + 1] = xr * xr
        var o = 2
        for c in range(1, NCH):
            var v = x + c * TT
            var kd = Int((kinds >> Int64(4 * c)) & 15)
            if kd == 1:
                for i in range(TT):
                    f[i * FT + o] = asinh_(v[i] / s)
            elif kd == 2:
                var m: Float32 = 0
                for i in range(TT):
                    m += v[i]
                m = max(m / Float32(TT), Float32(1e-12))
                for i in range(TT):
                    f[i * FT + o] = log1p(v[i] / m)
            elif kd == 3:
                for i in range(TT):
                    f[i * FT + o] = (v[i] - 0.5) * 4
            elif kd == 4:
                var a2: Float32 = 0
                for i in range(TT):
                    a2 += v[i] * v[i]
                var sj = max(sqrt(a2 / Float32(TT)), Float32(1e-8))
                for i in range(TT):
                    var xb = asinh_(v[i] / sj)
                    f[i * FT + o] = xb
                    f[i * FT + o + 1] = xb * f[i * FT]
                o += 1
            else:
                for i in range(TT):
                    f[i * FT + o] = v[i]
            o += 1

    ctx.enqueue_function(k, grid_dim=ceildiv(n, 64), block_dim=64)


def k_assemble(ctx: DeviceContext, e: Engine, n: Int) raises:
    """Token sequence per row: [scalar token][coarse tokens][fine tokens] + embedding bias + position."""
    var px = mpf(e.x)
    var pe = mpf(e.ef)
    var pc = mpf(e.ec)
    var pxs = mpf(e.xs)
    var ps = mpf(e.sc)
    var w = mpf(e.w)
    var tot64 = Int64(n * T * D)

    def k() {var px, var pe, var pc, var pxs, var ps, var w, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i >= Int(tot64):
            return
        var d = i % D
        var tok = (i // D) % T
        var r = i // (T * D)
        var v: Float32 = 0
        if tok >= TS + TC:
            v = pe[(r * TF + tok - TS - TC) * D + d] + w[O_EMB + D * P * FF + d]
        elif tok >= TS:
            v = pc[(r * TC + tok - TS) * D + d] + w[O_EMBC + D * PC * FC + d]
        else:
            v = w[O_EMBS + D * NS + d]
            var sr = pxs + r * (NSS + 3)
            for j in range(NS):
                var sv = sr[j]
                if j < HAR:
                    sv = min(max(log(max(sv, Float32(1e-12)) / ps[r]), Float32(-4)), Float32(4))
                v += sv * w[O_EMBS + d * NS + j]
        px[i] = v + w[O_POS + tok * D + d]

    ctx.enqueue_function(k, grid_dim=ceildiv(n * T * D, 256), block_dim=256)


def k_attn(ctx: DeviceContext, e: Engine, bq: FP, n: Int) raises:
    """One block per (row, head), one thread per query token; k and v of the block live in shared memory."""
    var pq = mpf(e.qkv)
    var po = mpf(e.att)

    def k() {var pq, var po, var bq}:
        var h = Int(block_idx.x) % NH
        var b = Int(block_idx.x) // NH
        var t = Int(thread_idx.x)
        var ks = stack_allocation[T * HD, Float32, address_space=AddressSpace.SHARED]()
        var vs = stack_allocation[T * HD, Float32, address_space=AddressSpace.SHARED]()
        var row = pq + (b * T + t) * (3 * D) + h * HD
        var bb = bq + h * HD
        var q = stack_allocation[HD, Float32]()
        var sm = stack_allocation[T, Float32]()
        for d in range(HD):
            q[d] = row[d] + bb[d]
            ks[t * HD + d] = row[D + d] + bb[D + d]
            vs[t * HD + d] = row[2 * D + d] + bb[2 * D + d]
        barrier()
        var inv = Float32(1.0) / sqrt(Float32(HD))
        var mx: Float32 = -3.0e38
        for j in range(T):
            var a: Float32 = 0
            for d in range(HD):
                a += q[d] * ks[j * HD + d]
            a = a * inv
            sm[j] = a
            mx = max(mx, a)
        var tot: Float32 = 0
        for j in range(T):
            var ev = exp(sm[j] - mx)
            sm[j] = ev
            tot += ev
        var it = Float32(1.0) / tot
        var o = po + (b * T + t) * D + h * HD
        for d in range(HD):
            var a: Float32 = 0
            for j in range(T):
                a += sm[j] * vs[j * HD + d]
            o[d] = a * it

    ctx.enqueue_function(k, grid_dim=n * NH, block_dim=T)


def k_head(ctx: DeviceContext, e: Engine, n: Int) raises:
    """Per row: final LayerNorm of the last token (+ calendar embeddings), head, monotone quantiles, scale back."""
    var px = mpf(e.x)
    var ps = mpf(e.sc)
    var po = mpf(e.res)
    var pxs = mpf(e.xs)
    var w = mpf(e.w)
    var n64 = Int64(n)
    var k0 = e.k0
    var k1 = e.k1
    var k2 = e.k2
    var k3 = e.k3

    def k() {var px, var ps, var po, var pxs, var w, var n64, var k0, var k1, var k2, var k3}:
        var r = Int(block_idx.x * block_dim.x + thread_idx.x)
        if r >= Int(n64):
            return
        var x = px + (r * T + T - 1) * D
        var pn = w + O_NF
        var ph = w + O_HEAD
        var z = stack_allocation[D, Float32]()
        var o = stack_allocation[NQ, Float32]()
        var mean: Float32 = 0
        for i in range(D):
            mean += x[i]
        mean = mean / Float32(D)
        var v: Float32 = 0
        for i in range(D):
            var c = x[i] - mean
            v += c * c
        var inv = Float32(1.0) / sqrt(v / Float32(D) + Float32(1e-5))
        for i in range(D):
            z[i] = (x[i] - mean) * inv * pn[i] + pn[D + i]
        comptime if CAL > 0:
            var cr = pxs + r * (NSS + 3) + NSS
            var e0 = w + O_CAL + Int(cr[0]) * D
            var e1 = w + O_CAL + (24 + Int(cr[1])) * D
            var e2 = w + O_CAL + (31 + Int(cr[2])) * D
            for i in range(D):
                z[i] = z[i] + e0[i] + e1[i] + e2[i]
        for h in range(H):
            for j in range(NQ):
                var a = ph[HQ * D + h * NQ + j]
                var wr = ph + (h * NQ + j) * D
                for i in range(D):
                    a += z[i] * wr[i]
                o[j] = a
            var kk = k0
            if h == 1:
                kk = k1
            elif h == 2:
                kk = k2
            elif h == 3:
                kk = k3
            kk = kk * ps[r]
            var q = po + r * HQ + h * NQ
            var med = o[10]
            q[10] = med * kk
            var a = med
            for j in range(11, NQ):
                a += softplus(o[j])
                q[j] = a * kk
            a = med
            for jj in range(10):
                var j = 9 - jj
                a -= softplus(o[j])
                q[j] = a * kk

    ctx.enqueue_function(k, grid_dim=ceildiv(n, 64), block_dim=64)


def forward(ctx: DeviceContext, e: Engine, n: Int) raises:
    var M = n * T
    var w = mpf(e.w)
    var x = mpf(e.x)
    var hb = mpf(e.hb)
    var qkv = mpf(e.qkv)
    var att = mpf(e.att)
    var u = mpf(e.u)
    k_norm[NFC, L, FF](ctx, mpf(e.xf), mpf(e.feat), mpf(e.sc), e.kf, n, True)
    gemm[D, P * FF](ctx, mpf(e.ef), mpf(e.feat), w + O_EMB, n * TF)
    comptime if NCC > 0:
        k_norm[NCC, LC, FC](ctx, mpf(e.xc), mpf(e.featc), mpf(e.sc), e.kc, n, False)
        gemm[D, PC * FC](ctx, mpf(e.ec), mpf(e.featc), w + O_EMBC, n * TC)
    k_assemble(ctx, e, n)
    for l in range(NL):
        var lw = w + (O_LAYER + l * LSZ)
        var wq = lw + 2 * D
        var bq = wq + 3 * D * D
        var wo = bq + 3 * D
        var bo = wo + D * D
        var n2 = bo + D
        var w1 = n2 + 2 * D
        var b1 = w1 + 4 * D * D
        var w2 = b1 + 4 * D
        var b2 = w2 + 4 * D * D
        k_ln(ctx, hb, x, lw, M)
        gemm[3 * D, D](ctx, qkv, hb, wq, M)
        k_attn(ctx, e, bq, n)
        gemm[D, D](ctx, hb, att, wo, M)
        k_resid(ctx, x, hb, bo, M)
        k_ln(ctx, hb, x, n2, M)
        gemm[4 * D, D](ctx, u, hb, w1, M)
        k_bias_relu(ctx, u, b1, M)
        gemm[D, 4 * D](ctx, hb, u, w2, M)
        k_resid(ctx, x, hb, b2, M)
    k_head(ctx, e, n)


@always_inline
def lp(l: List[Float32]) -> FP:
    return l.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def infer(ctx: DeviceContext, e: Engine, inp: FP, res: FP, n: Int) raises:
    """Host block (fine | coarse | scal | cal, each for all n rows) -> host quantiles, in slices of the engine capacity."""
    var pc = inp + n * NFC * L
    var ps = pc + n * NCC * LC
    var pk = ps + n * NS
    var done = 0
    while done < n:
        var cnt = min(e.cap, n - done)
        var o = 0
        for i in range(cnt * NFC * L):
            e.hin[i] = inp[done * NFC * L + i]
        ctx.enqueue_copy(e.xf, e.hin)
        ctx.synchronize()
        comptime if NCC > 0:
            for i in range(cnt * NCC * LC):
                e.hin[i] = pc[done * NCC * LC + i]
            ctx.enqueue_copy(e.xc, e.hin)
            ctx.synchronize()
        comptime if NS + CAL > 0:
            for r in range(cnt):
                for j in range(NS):
                    e.hin[r * (NSS + 3) + j] = ps[(done + r) * NS + j]
                comptime if CAL > 0:
                    for j in range(3):
                        e.hin[r * (NSS + 3) + NSS + j] = pk[(done + r) * 3 + j]
            ctx.enqueue_copy(e.xs, e.hin)
            ctx.synchronize()
        forward(ctx, e, cnt)
        ctx.enqueue_copy(e.hout, e.res)
        ctx.synchronize()
        for i in range(cnt * HQ):
            res[done * HQ + i] = e.hout[i]
        done += cnt


def infer_sym(ctx: DeviceContext, e: Engine, inp: FP, res: FP, n: Int) raises:
    """Flip-averaged forecast: q = (q(x) - mirror(q(flip x))) / 2; RET / XRT / RAW channels change sign, SHR mirrors."""
    var x2 = List[Float32](length=n * ROWIN, fill=0)
    var o2 = List[Float32](length=n * HQ, fill=0)
    for i in range(n * ROWIN):
        x2[i] = inp[i]
    for r in range(n):
        for c in range(NFC):
            var kd = Int((e.kf >> Int64(4 * c)) & 15)
            for i in range(L):
                var j = (r * NFC + c) * L + i
                if kd == 0 or kd == 4 or kd == 5:
                    x2[j] = -inp[j]
                elif kd == 3:
                    x2[j] = Float32(1) - inp[j]
        for i in range(LC if NCC > 0 else 0):
            var j = n * NFC * L + r * NCC * LC + i
            x2[j] = -inp[j]
    infer(ctx, e, inp, res, n)
    infer(ctx, e, lp(x2), lp(o2), n)
    for row in range(n):
        for h in range(H):
            for j in range(NQ // 2 + 1):
                var a = res[row * HQ + h * NQ + j]
                var b = res[row * HQ + h * NQ + NQ - 1 - j]
                res[row * HQ + h * NQ + j] = 0.5 * (a - o2[row * HQ + h * NQ + NQ - 1 - j])
                res[row * HQ + h * NQ + NQ - 1 - j] = 0.5 * (b - o2[row * HQ + h * NQ + j])


def run_infer(ctx: DeviceContext, e: Engine, inp: FP, res: FP, n: Int) raises:
    if getenv("BMT_SYM", "0") == "1":
        infer_sym(ctx, e, inp, res, n)
    else:
        infer(ctx, e, inp, res, n)


def selftest(dir: String) raises -> Bool:
    var xin = List[Float32]()
    var ref_ = List[Float32]()
    var ni = read_f32(dir + "/selftest_in.f32", xin, 0)
    var nr = read_f32(dir + "/selftest_ref.f32", ref_, 0)
    var n = nr // HQ
    if ni != n * ROWIN:
        raise Error("selftest vector does not match this build")
    var got = List[Float32](length=nr, fill=0)
    with DeviceContext() as ctx:
        var e = Engine(ctx, dir + "/bmt2.bin", 64)
        infer(ctx, e, lp(xin), lp(got), n)
        var maxrel = Float32(0)
        for row in range(n):
            var acc = Float32(0)
            for i in range(L):
                acc += xin[row * NFC * L + i] * xin[row * NFC * L + i]
            var s = max(sqrt(acc / Float32(L)), Float32(1e-8))
            for j in range(HQ):
                maxrel = max(maxrel, abs(got[row * HQ + j] - ref_[row * HQ + j]) / s)
        var tol = Float32(2e-4)
        print("selftest rows", n, "tokens", T, "max err in scale units", maxrel, "tol", tol, "PASS" if maxrel <= tol else "FAIL")
        return maxrel <= tol


def bench(path: String, batch: Int, iters: Int) raises:
    var xin = List[Float32](length=batch * ROWIN, fill=0)
    var seed: UInt64 = 12345
    for i in range(batch * ROWIN):
        seed = seed * 6364136223846793005 + 1442695040888963407
        xin[i] = (Float32(Int((seed >> 40) & 0xFFFF)) / 65536.0) * 0.02 + 0.0001
    comptime if CAL > 0:
        for i in range(batch * 3):
            xin[batch * ROWIN - batch * 3 + i] = Float32(i % 2)
    var got = List[Float32](length=batch * HQ, fill=0)
    with DeviceContext() as ctx:
        var e = Engine(ctx, path, min(batch, 4096))
        for _ in range(5):
            run_infer(ctx, e, lp(xin), lp(got), batch)
        var lat = List[Float64](length=iters, fill=0)
        var t00 = perf_counter_ns()
        for it in range(iters):
            var t0 = perf_counter_ns()
            run_infer(ctx, e, lp(xin), lp(got), batch)
            lat[it] = Float64(perf_counter_ns() - t0) / 1e6
        var total = Float64(perf_counter_ns() - t00) / 1e9
        sort(lat)
        print("batch", batch, "rows/s", Float64(batch * iters) / total, "p50_ms", lat[iters // 2], "p99_ms", lat[min(iters - 1, (iters * 99) // 100)], "check", got[0])


def serve(path: String, cap: Int) raises:
    with DeviceContext() as ctx:
        var e = Engine(ctx, path, cap)
        print("ready rowin", ROWIN, "H", H, "tokens", T, "cap", cap, file=stderr)
        var hdr = List[Int32](length=1, fill=0)
        var hp = hdr.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
        var bi = List[Float32]()
        var bo = List[Float32]()
        while True:
            if fd_read(0, hp, 4) != 4:
                return
            var n = Int(hdr[0])
            if n <= 0:
                return
            if len(bi) < n * ROWIN:
                bi = List[Float32](length=n * ROWIN, fill=0)
                bo = List[Float32](length=n * HQ, fill=0)
            if fd_read(0, bi.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), n * ROWIN * 4) != n * ROWIN * 4:
                raise Error("short request body")
            run_infer(ctx, e, lp(bi), lp(bo), n)
            fd_write(1, hp, 4)
            fd_write(1, bo.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), n * HQ * 4)


def main() raises:
    var a = argv()
    if len(a) >= 3 and a[1] == "selftest":
        if not selftest(String(a[2])):
            raise Error("selftest failed")
    elif len(a) >= 5 and a[1] == "bench":
        bench(String(a[2]), Int(a[3]), Int(a[4]))
    elif len(a) >= 3 and a[1] == "serve":
        serve(String(a[2]), Int(a[3]) if len(a) > 3 else 1024)
    else:
        print("usage: bmt2_gpu selftest <dir> | bench <bmt2.bin> <batch> <iters> | serve <bmt2.bin> [capacity]")
