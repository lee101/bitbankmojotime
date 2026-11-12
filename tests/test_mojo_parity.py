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
