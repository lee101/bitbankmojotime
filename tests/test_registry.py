"""A table generator must refuse a prediction file that is not the one its model row was registered with."""
import numpy as np
import pytest
from bmtbench import data as D


def test_open_pred_refuses_replaced_or_unregistered_files(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(D, "REG", str(tmp_path / "models.json"))
    q = np.sort(np.random.default_rng(0).normal(size=(50, 2, 21)).astype(np.float32), 2)
    D.save_pred("daily", "m~abc12345", q)
    with pytest.raises(RuntimeError, match="not a registered"):
        D.open_pred("daily", "m~abc12345")
    D.register("daily", "m~abc12345", model_hash="abc12345" * 8)
    assert np.array_equal(np.asarray(D.open_pred("daily", "m~abc12345")), q)
    assert D.resolve("m") == "m~abc12345" and D.models_of("daily") == ["m~abc12345"] and D.models_of("hourly") == []
    D.save_pred("daily", "m~abc12345", q + 1e-3)                    # someone else writes a different model under the same name
    with pytest.raises(RuntimeError, match="does not match"):
        D.open_pred("daily", "m~abc12345")


def test_model_hash_depends_on_config_and_checkpoint_bytes(tmp_path):
    a, b = tmp_path / "a_F1.pt", tmp_path / "b_F1.pt"
    a.write_bytes(b"one"); b.write_bytes(b"two")
    h = D.model_hash({"d": 64}, [str(a)])
    assert h == D.model_hash({"d": 64}, [str(a)]) and h != D.model_hash({"d": 128}, [str(a)]) and h != D.model_hash({"d": 64}, [str(b)])
