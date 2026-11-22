"""Mojo inference vs the PyTorch reference on a stored vector (random-weight size-t model). Skipped when the binary is not built."""
import os, subprocess
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN, VEC = os.path.join(ROOT, "build", "bmt"), os.path.join(ROOT, "tests", "vectors", "bmt-t-random")


@pytest.mark.skipif(not os.path.exists(BIN), reason="build/bmt not built (./build.sh)")
def test_cpu_kernel_matches_stored_reference():
    r = subprocess.run([BIN, "selftest", VEC], capture_output=True, text=True)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr


def test_stored_vector_matches_pytorch():
    torch = pytest.importorskip("torch")
    from train.model import BMT, SIZES
    import struct
    hdr = struct.unpack("<16i", open(os.path.join(VEC, "bmt.bin"), "rb").read(64))
    assert hdr[0] == 0x31544D42 and hdr[1:6] == (512, 16, 64, 2, 4)
    w = np.fromfile(os.path.join(VEC, "bmt.bin"), np.float32, offset=64)
    m = BMT(**SIZES["t"])
    assert sum(p.numel() for p in m.parameters()) == len(w)
    sd, o = m.state_dict(), 0
    names = ["embed.weight", "embed.bias", "pos"] + [f"blocks.{i}.{k}" for i in range(2) for k in (
        "n1.weight", "n1.bias", "qkv.weight", "qkv.bias", "o.weight", "o.bias", "n2.weight", "n2.bias", "f1.weight", "f1.bias", "f2.weight", "f2.bias")] + [
        "nf.weight", "nf.bias", "head.weight", "head.bias"]
    for n in names:
        k = sd[n].numel()
        sd[n] = torch.tensor(w[o:o + k].reshape(sd[n].shape)); o += k
    m.load_state_dict(sd); m.eval()
    x = np.fromfile(os.path.join(VEC, "selftest_in.f32"), np.float32).reshape(32, 512)
    ref = np.fromfile(os.path.join(VEC, "selftest_ref.f32"), np.float32).reshape(32, 4, 21)
    with torch.no_grad():
        q = m.quantiles(torch.tensor(x)).numpy()
    sc = np.maximum(np.sqrt((x.astype(np.float64) ** 2).mean(1)), 1e-8)[:, None, None]
    assert (np.abs(q - ref) / sc).max() < 2e-4
    assert (np.diff(q, axis=2) >= 0).all()                           # quantiles never cross


BIN2, VEC2 = os.path.join(ROOT, "build", "bmt2"), os.path.join(ROOT, "tests", "vectors", "bmt2-t-random")
HIST = os.path.join(ROOT, "tests", "vectors", "history_synthetic.json")


@pytest.mark.skipif(not os.path.exists(BIN2), reason="build/bmt2 not built")
def test_v2_kernel_matches_stored_reference():
    r = subprocess.run([BIN2, "selftest", VEC2], capture_output=True, text=True)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(not os.path.exists(BIN2), reason="build/bmt2 not built")
def test_rotation_cli_matches_pytorch_on_a_candle_history_file():
    """the consumer hook: features built by the Mojo CLI from a history file = the reference feature code + model."""
    torch = pytest.importorskip("torch")
    import json, struct
    from train.model2 import BMT2
    from train.model import SIZES
    from train.bank import FINE, COARSE, LC
    out = json.loads(subprocess.run([BIN2, "rotation", os.path.join(VEC2, "bmt2.bin"), HIST], capture_output=True, text=True, check=True).stdout)
    h = json.load(open(HIST))
    x = np.array(h["ohlcv"], np.float64)[-513:]                      # [hours, pairs, (o, h, l, c, v)]
    r = np.diff(np.log(x[:, :, 3]), axis=0).T.astype(np.float32)
    hl = np.log(x[1:, :, 1] / x[1:, :, 2]).T.astype(np.float32)
    n = r.shape[0]
    fine = np.zeros((n, len(FINE), 512), np.float32)
    fine[:, 0], fine[:, 1], fine[:, 2] = r, hl, np.sqrt(np.maximum(0.5 * hl ** 2 - np.float32(0.3862944) * r ** 2, 0))
    hr = h["hours"][-1]
    cal = np.tile([(hr + 1) % 24, ((hr + 1) // 24 + 4) % 7, 0], (n, 1))
    m = BMT2(fine=["r", "hl", "gk"], cal=True, **SIZES["t"])
    w = np.fromfile(os.path.join(VEC2, "bmt2.bin"), np.float32, offset=256)
    sd, o = m.state_dict(), 0
    names = ["embed.weight", "embed.bias", "pos"] + [f"blocks.{i}.{k}" for i in range(2) for k in (
        "n1.weight", "n1.bias", "qkv.weight", "qkv.bias", "o.weight", "o.bias", "n2.weight", "n2.bias", "f1.weight", "f1.bias", "f2.weight", "f2.bias")] + [
        "nf.weight", "nf.bias", "e_hour.weight", "e_dow.weight", "e_bar.weight", "head.weight", "head.bias"]
    for k in names:
        c = sd[k].numel()
        sd[k] = torch.tensor(w[o:o + c].reshape(sd[k].shape)); o += c
    assert o == len(w)
    m.load_state_dict(sd); m.eval()
    with torch.no_grad():
        q = m.quantiles(dict(fine=torch.tensor(fine), coarse=torch.zeros(n, len(COARSE), LC), scal=torch.zeros(n, 11), cal=torch.tensor(cal))).numpy()[:, 3]
    assert out["pairs"] == h["pairs"] and out["issued_hour"] == hr and out["horizon_hours"] == 24
    assert np.allclose(out["median_logret"], q[:, 10], atol=2e-6) and np.allclose(out["q05_logret"], q[:, 1], atol=5e-6)
    sc = q[:, 10] / ((q[:, 17] - q[:, 3]) / 2.0728)
    assert np.allclose(out["raw_scores"], (sc - sc.mean()) / sc.std(ddof=1), atol=2e-3)


GPU2 = os.path.join(ROOT, "build", "bmt2_gpu_t_sm120")
GPU_ENV = {"LD_LIBRARY_PATH": os.environ.get("BMT_CUDA_LIB", "/usr/local/cuda-12.9/lib64")}


def _serve(binary, wfile, x, n, sym=False, cap=8):
    env = dict(os.environ, BMT_SYM="1" if sym else "0", **(GPU_ENV if "gpu" in binary else {}))
    p = subprocess.run([binary, "serve", wfile] + ([str(cap)] if "gpu" in binary else ["3"]), input=np.int32([n]).tobytes() + x.astype(np.float32).tobytes() + np.int32([0]).tobytes(),
                       capture_output=True, env=env, check=True)
    assert np.frombuffer(p.stdout[:4], np.int32)[0] == n
    return np.frombuffer(p.stdout[4:], np.float32).reshape(n, 4, 21)


def _rows(x, n_all, idx, nfc):
    """subset of rows of an input block laid out as fine[N, nfc, 512] | cal[N, 3]."""
    f = x[:n_all * nfc * 512].reshape(n_all, nfc, 512)[idx]
    c = x[n_all * nfc * 512:].reshape(n_all, 3)[idx]
    return np.concatenate([f.ravel(), c.ravel()]), f, c


@pytest.mark.parametrize("binary", [BIN2, GPU2])
def test_v2_partial_blocks_and_flip_average(binary):
    """odd row counts (partial multi-row blocks, several GPU slices) match the stored reference; BMT_SYM=1 = (q(x) - mirror(q(flip x))) / 2
    from the same kernel."""
    if not os.path.exists(binary):
        pytest.skip(binary + " not built (./build.sh gpu)")
    wf = os.path.join(VEC2, "bmt2.bin")
    x = np.fromfile(os.path.join(VEC2, "selftest_in.f32"), np.float32)
    ref = np.fromfile(os.path.join(VEC2, "selftest_ref.f32"), np.float32).reshape(-1, 4, 21)
    n_all = ref.shape[0]
    nfc = struct_nfc(wf)
    idx = np.arange(13)
    xs, f, c = _rows(x, n_all, idx, nfc)
    s = np.sqrt((f[:, 0] ** 2).mean(1)).clip(1e-8)
    q = _serve(binary, wf, xs, 13)
    assert (np.abs(q - ref[idx]) / s[:, None, None]).max() < 2e-4
    kinds = np.fromfile(wf, np.int32, 64)[20:20 + nfc]
    ff = f.copy()
    ff[:, kinds == 0] *= -1
    qf = _serve(binary, wf, np.concatenate([ff.ravel(), c.ravel()]), 13)
    qs = _serve(binary, wf, xs, 13, sym=True)
    assert (np.abs(qs - 0.5 * (q - qf[..., ::-1])) / s[:, None, None]).max() < 1e-5
    if "gpu" in binary:                                              # one slice: request read straight into the pinned upload buffer
        for sym in (False, True):
            a, b = _serve(binary, wf, xs, 13, sym, cap=8), _serve(binary, wf, xs, 13, sym, cap=64)
            assert (np.abs(a - b) / s[:, None, None]).max() < 1e-5


def struct_nfc(wf):
    return int(np.fromfile(wf, np.int32, 64)[10])


@pytest.mark.parametrize("binary", [BIN2, GPU2])
def test_v2_seed_ensemble_is_the_quantile_average(binary, tmp_path):
    if not os.path.exists(binary):
        pytest.skip(binary + " not built")
    wf = os.path.join(VEC2, "bmt2.bin")
    raw = np.fromfile(wf, np.uint8)
    w = raw[256:].view(np.float32).copy()
    w[-84:] += np.random.default_rng(0).normal(0, 0.3, 84).astype(np.float32)     # another member: perturbed head bias
    w[:4096] *= 1.1
    w2 = str(tmp_path / "m2.bin")
    np.concatenate([raw[:256], w.view(np.uint8)]).tofile(w2)
    x = np.fromfile(os.path.join(VEC2, "selftest_in.f32"), np.float32)
    n = np.fromfile(os.path.join(VEC2, "selftest_ref.f32"), np.float32).size // 84
    for sym in (False, True):
        a, b = _serve(binary, wf, x, n, sym), _serve(binary, w2, x, n, sym)
        e = _serve(binary, wf + "," + w2, x, n, sym)
        assert np.abs(a - b).max() > 1e-3 and np.abs(e - 0.5 * (a + b)).max() < 1e-6


@pytest.mark.parametrize("binary", [BIN2, GPU2])
def test_v2_level_file_equals_member_list_with_sym(binary, tmp_path):
    """@level.txt (relative member paths + `sym`) gives the same forecast as the explicit comma list with BMT_SYM=1."""
    if not os.path.exists(binary):
        pytest.skip(binary + " not built")
    wf = os.path.join(VEC2, "bmt2.bin")
    lv = tmp_path / "lv"
    lv.mkdir()
    os.symlink(wf, lv / "a.bin")
    (lv / "q.txt").write_text("# test level\nsym\na.bin a.bin\n")
    x = np.fromfile(os.path.join(VEC2, "selftest_in.f32"), np.float32)
    n = np.fromfile(os.path.join(VEC2, "selftest_ref.f32"), np.float32).size // 84
    a = _serve(binary, "@" + str(lv / "q.txt"), x, n)
    b = _serve(binary, wf + "," + wf, x, n, sym=True)
    assert np.abs(a - b).max() < 1e-6
