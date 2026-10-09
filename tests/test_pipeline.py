"""Small synthetic end-to-end verification. Never interpreted as trading performance."""

import copy
import importlib.util
import zipfile

import numpy as np
import pytest
from test_features import candles

from upbit_bot.common import load_config, save_json
from upbit_bot.data import CandleStore
from upbit_bot.features import FEATURES
from upbit_bot.package import package
from upbit_bot.runtime import Bundle


def test_train_export_and_package(tmp_path):
    pytest.importorskip("torch")
    from upbit_bot.train import train

    cfg = copy.deepcopy(load_config())
    cfg["data"]["minimum_days"] = 5
    cfg["model"].update(lookback=16, hidden=8, layers=1, dropout=0)
    cfg["training"].update(epochs=1, batch_size=32, samples_per_epoch=64, validation_days=1, test_days=1)
    directory, out = tmp_path / "data", tmp_path / "artifacts"
    save_json(directory / "universe.json", {"markets": ["KRW-BTC", "KRW-ETH"], "selected_at": "synthetic"})
    store = CandleStore(directory)
    for market in ("KRW-BTC", "KRW-ETH"):
        store.put(market, candles(2400))
    store.close()
    manifest = train(cfg, directory, out, allow_cpu=True)
    assert manifest["approved_for_live"] is False
    assert manifest["onnx_max_error_bps"] < 0.005
    bundle = Bundle(out)
    y = bundle.session.run(None, {"features": np.zeros((5, 16, len(FEATURES) + 2), np.float32)})[0]
    assert len(y) == 5 and np.isfinite(y).all()
    destination = package(artifacts=out, destination=tmp_path / "oracle.zip")
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert "artifacts/model.onnx" in names
        assert "requirements-runtime.txt" in names
        assert "deploy/install.sh" in names
        assert not any("checkpoint" in n or n == ".env" or "sqlite" in n for n in names)
    if importlib.util.find_spec("hmmlearn"):
        import pandas as pd

        from upbit_bot.hmm_research import train_hmm

        cfg["hmm"].update(iterations=4, restarts=1, max_fit_rows=400, block_size=100)
        comparison_dir = out / "hmm"
        comparison = train_hmm(cfg, directory, comparison_dir, out)
        assert comparison["snapshot_verified_against_lstm"]
        assert comparison["lstm_model_sha256"] == manifest["model_sha256"]
        assert set(comparison["results"]) == {"HMM", "LSTM"}
        for split in ("val", "test"):
            hmm = pd.read_csv(comparison_dir / f"hmm_{split}_predictions.csv")
            lstm = pd.read_csv(comparison_dir / f"lstm_{split}_predictions.csv")
            pd.testing.assert_frame_equal(
                hmm.drop(columns="prediction_bps"), lstm.drop(columns="prediction_bps")
            )
            # The new evaluator reproduces the original LSTM forecast on the same data.
            actual = comparison["results"]["LSTM"]["forecast"][split]
            original = bundle.report["forecast"][split]
            assert abs(actual["mae_bps"] - original["mae_bps"]) < 0.01
    # Mismatched model bytes must fail before any inference/order.
    with (out / "model.onnx").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        Bundle(out)
