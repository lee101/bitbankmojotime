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
