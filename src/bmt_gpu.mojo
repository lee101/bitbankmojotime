"""GPU inference for bmt (univariate model): fp32 GEMMs + fused custom kernels, same arithmetic as src/bmt.mojo and train/model.py.

  bmt_gpu selftest <dir>                 bmt.bin + selftest_in.f32 vs selftest_ref.f32, exit 1 above tolerance
  bmt_gpu bench <bmt.bin> <batch> <iters>
  bmt_gpu serve <bmt.bin> [capacity]     stdin: int32 rows, rows*512 float32; stdout: int32 rows, rows*H*21 float32

Model dimensions are compile-time (-D D=.. -D NL=.. -D NH=..; defaults = size "s"); the weight file header must match.
One build per GPU arch (--target-accelerator=nvidia:sm_120 for the RTX 5090).
"""

from std.math import sqrt, exp, log1p, ceildiv
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
comptime HD = D // NH
comptime L = 512
comptime P = 16
comptime T = L // P
comptime F = 2 * P
comptime H = 4
comptime NQ = 21
comptime HQ = H * NQ
comptime MAGIC = 0x31544D42
comptime LSZ = 12 * D * D + 13 * D
comptime O_EMB = 0
comptime O_POS = D * F + D
comptime O_LAYER = O_POS + T * D
comptime O_NF = O_LAYER + NL * LSZ
comptime O_HEAD = O_NF + 2 * D
comptime NW = O_HEAD + HQ * D + HQ

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


struct Engine(Movable):
    var cap: Int
    var w: BufF
    var xin: BufF
    var sc: BufF
    var feat: BufF
    var x: BufF
    var hb: BufF
    var qkv: BufF
    var att: BufF
    var u: BufF
    var res: BufF
    var hin: HostBuffer[F32]
    var hout: HostBuffer[F32]
    var k0: Float32
    var k1: Float32
    var k2: Float32
    var k3: Float32

    def __init__(out self, ctx: DeviceContext, path: String, cap: Int) raises:
        var f = open(path, "r")
        var bytes = f.read_bytes()
        var hp = bytes.unsafe_ptr().unsafe_bitcast[Int32]()
        if len(bytes) != 64 + NW * 4 or Int(hp[0]) != MAGIC or Int(hp[1]) != L or Int(hp[2]) != P or Int(hp[3]) != D or Int(hp[4]) != NL or Int(hp[5]) != NH or Int(hp[6]) != H or Int(hp[7]) != 0 or Int(hp[8]) != 0:
            raise Error(path + ": weight file does not match this build (univariate, D " + String(D) + ", layers " + String(NL) + ", heads " + String(NH) + ")")
        self.k0 = sqrt(Float32(Int(hp[9])))
        self.k1 = sqrt(Float32(Int(hp[10])))
        self.k2 = sqrt(Float32(Int(hp[11])))
        self.k3 = sqrt(Float32(Int(hp[12])))
        var src = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
        var hbuf = ctx.enqueue_create_host_buffer[F32](NW)
        for i in range(NW):
            hbuf[i] = src[16 + i]
        self.w = ctx.enqueue_create_buffer[F32](NW)
        ctx.enqueue_copy(self.w, hbuf)
        ctx.synchronize()
        self.cap = cap
        var M = cap * T
        self.xin = ctx.enqueue_create_buffer[F32](cap * L)
        self.sc = ctx.enqueue_create_buffer[F32](cap)
        self.feat = ctx.enqueue_create_buffer[F32](M * F)
        self.x = ctx.enqueue_create_buffer[F32](M * D)
        self.hb = ctx.enqueue_create_buffer[F32](M * D)
        self.qkv = ctx.enqueue_create_buffer[F32](M * 3 * D)
        self.att = ctx.enqueue_create_buffer[F32](M * D)
        self.u = ctx.enqueue_create_buffer[F32](M * 4 * D)
        self.res = ctx.enqueue_create_buffer[F32](cap * HQ)
        self.hin = ctx.enqueue_create_host_buffer[F32](cap * L)
        self.hout = ctx.enqueue_create_host_buffer[F32](cap * HQ)


@always_inline
def asinh_(x: Float32) -> Float32:
    """asinh without libm (not available on the device): sign(x) * log1p(|x| + x^2 / (1 + sqrt(x^2 + 1)))."""
    var a = abs(x)
    var y = log1p(a + a * a / (Float32(1) + sqrt(a * a + Float32(1))))
    return y if x >= 0 else -y


def k_feat(ctx: DeviceContext, e: Engine, n: Int) raises:
    """Per row: RMS scale, then [asinh(r / s), asinh(r / s)^2] per step."""
    var px = mpf(e.xin)
    var ps = mpf(e.sc)
    var pf = mpf(e.feat)
    var n64 = Int64(n)

    def k() {var px, var ps, var pf, var n64}:
        var r = Int(block_idx.x * block_dim.x + thread_idx.x)
        if r >= Int(n64):
            return
        var acc: Float32 = 0
        for i in range(L):
            var v = px[r * L + i]
            acc += v * v
        var s = max(sqrt(acc / Float32(L)), Float32(1e-8))
        ps[r] = s
        for i in range(L):
            var xv = asinh_(px[r * L + i] / s)
            pf[(r * L + i) * 2] = xv
            pf[(r * L + i) * 2 + 1] = xv * xv

    ctx.enqueue_function(k, grid_dim=ceildiv(n, 64), block_dim=64)


def k_addpos(ctx: DeviceContext, e: Engine, M: Int) raises:
    var px = mpf(e.x)
    var pb = mpf(e.w) + (O_EMB + D * F)
    var pp = mpf(e.w) + O_POS
    var tot64 = Int64(M * D)

    def k() {var px, var pb, var pp, var tot64}:
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Int(tot64):
            px[i] = px[i] + pb[i % D] + pp[i % (T * D)]

    ctx.enqueue_function(k, grid_dim=ceildiv(M * D, 256), block_dim=256)


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


def k_attn(ctx: DeviceContext, e: Engine, bq: FP, n: Int) raises:
    """One block per (row, head), one thread per query token; k and v of the block live in shared memory. qkv[token] = [q | k | v]
    without bias (added here)."""
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


def k_head(ctx: DeviceContext, e: Engine, n: Int) raises:
    """Per row: final LayerNorm of the last token, head, monotone quantiles, scale back to log-return units."""
    var px = mpf(e.x)
    var ps = mpf(e.sc)
    var po = mpf(e.res)
    var pn = mpf(e.w) + O_NF
    var ph = mpf(e.w) + O_HEAD
    var n64 = Int64(n)
    var k0 = e.k0
    var k1 = e.k1
    var k2 = e.k2
    var k3 = e.k3

    def k() {var px, var ps, var po, var pn, var ph, var n64, var k0, var k1, var k2, var k3}:
        var r = Int(block_idx.x * block_dim.x + thread_idx.x)
        if r >= Int(n64):
            return
        var x = px + (r * T + T - 1) * D
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
    """e.xin[n, L] -> e.res[n, H * 21]."""
    var M = n * T
    var w = mpf(e.w)
    var x = mpf(e.x)
    var hb = mpf(e.hb)
    var qkv = mpf(e.qkv)
    var att = mpf(e.att)
    var u = mpf(e.u)
    k_feat(ctx, e, n)
    gemm[D, F](ctx, x, mpf(e.feat), w + O_EMB, M)
    k_addpos(ctx, e, M)
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


def infer(ctx: DeviceContext, e: Engine, inp: Pointer[Float32, MutAnyOrigin], res: Pointer[Float32, MutAnyOrigin], n: Int) raises:
    """Host rows -> host quantiles, in slices of the engine capacity."""
    var done = 0
    while done < n:
        var cnt = min(e.cap, n - done)
        for i in range(cnt * L):
            e.hin[i] = inp[done * L + i]
        ctx.enqueue_copy(e.xin, e.hin)
        forward(ctx, e, cnt)
        ctx.enqueue_copy(e.hout, e.res)
        ctx.synchronize()
        for i in range(cnt * HQ):
            res[done * HQ + i] = e.hout[i]
        done += cnt


def infer_sym(ctx: DeviceContext, e: Engine, inp: FP, res: FP, n: Int) raises:
    """Flip-averaged forecast (2 passes): q = (q(r) - mirror(q(-r))) / 2."""
    var x2 = List[Float32](length=2 * n * L, fill=0)
    var o2 = List[Float32](length=2 * n * HQ, fill=0)
    for i in range(n * L):
        x2[i] = inp[i]
        x2[n * L + i] = -inp[i]
    infer(ctx, e, x2.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), o2.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), 2 * n)
    for row in range(n):
        for h in range(H):
            for j in range(NQ):
                res[row * HQ + h * NQ + j] = 0.5 * (o2[row * HQ + h * NQ + j] - o2[(n + row) * HQ + h * NQ + (NQ - 1 - j)])


def run_infer(ctx: DeviceContext, e: Engine, inp: FP, res: FP, n: Int) raises:
    """BMT_SYM=1 in the environment selects the flip-averaged (2-pass) forecast."""
    if getenv("BMT_SYM", "0") == "1":
        infer_sym(ctx, e, inp, res, n)
    else:
        infer(ctx, e, inp, res, n)


@always_inline
def lp(l: List[Float32]) -> FP:
    return l.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def selftest(dir: String) raises -> Bool:
    var xin = List[Float32]()
    var ref_ = List[Float32]()
    var ni = read_f32(dir + "/selftest_in.f32", xin, 0)
    var nr = read_f32(dir + "/selftest_ref.f32", ref_, 0)
    var n = nr // HQ
    if ni != n * L:
        raise Error("selftest vector does not match the model")
    var got = List[Float32](length=nr, fill=0)
    with DeviceContext() as ctx:
        var e = Engine(ctx, dir + "/bmt.bin", 64)
        infer(ctx, e, lp(xin), lp(got), n)
        var maxrel = Float32(0)
        for row in range(n):
            var acc = Float32(0)
            for i in range(L):
                acc += xin[row * L + i] * xin[row * L + i]
            var s = max(sqrt(acc / Float32(L)), Float32(1e-8))
            for j in range(HQ):
                maxrel = max(maxrel, abs(got[row * HQ + j] - ref_[row * HQ + j]) / s)
        var tol = Float32(2e-4)
        print("selftest rows", n, "max err in scale units", maxrel, "tol", tol, "PASS" if maxrel <= tol else "FAIL")
        return maxrel <= tol


def bench(path: String, batch: Int, iters: Int) raises:
    var xin = List[Float32](length=batch * L, fill=0)
    var seed: UInt64 = 12345
    for i in range(batch * L):
        seed = seed * 6364136223846793005 + 1442695040888963407
        xin[i] = (Float32(Int((seed >> 40) & 0xFFFF)) / 65536.0 - 0.5) * 0.02
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
        var wi = List[Float32](length=L, fill=0.001)
        var wo = List[Float32](length=HQ, fill=0)
        infer(ctx, e, lp(wi), lp(wo), 1)
        print("ready L", L, "H", H, "cap", cap, file=stderr)
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
            if len(bi) < n * L:
                bi = List[Float32](length=n * L, fill=0)
                bo = List[Float32](length=n * HQ, fill=0)
            if fd_read(0, bi.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), n * L * 4) != n * L * 4:
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
        print("usage: bmt_gpu selftest <dir> | bench <bmt.bin> <batch> <iters> | serve <bmt.bin> [capacity]")
