import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from upbit_bot.common import load_config  # noqa: E402
from upbit_bot.model import (  # noqa: E402
    ARCHITECTURES,
    Ensemble,
    build_member,
    export_onnx,
    member_architectures,
)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_member_shapes_causality_and_onnx(arch, tmp_path):
    mc = {"hidden": 16, "layers": 2, "dropout": 0.0, "lookback": 24}
    torch.manual_seed(0)
    net = build_member(arch, 7, mc).eval()
    x = torch.randn(3, 24, 7)
    y = net(x)
    assert y.shape == (3,) and torch.isfinite(y).all()
    y.sum().backward()
    error = export_onnx(Ensemble([net, build_member(arch, 7, mc)]), 7, 24, tmp_path / "m.onnx")
    assert error < 0.005


def test_tcn_is_causal_until_pooling():
    from upbit_bot.model import ReturnTCN

    net = ReturnTCN(5, hidden=8, dropout=0).eval()
    x = torch.randn(1, 40, 5)
    z = net.proj(net.norm(x).transpose(1, 2))
    a = net.blocks(z)[:, :, :20]
    x2 = x.clone()
    x2[:, 30:] += 5
    b = net.blocks(net.proj(net.norm(x2).transpose(1, 2)))[:, :, :20]
    assert torch.allclose(a, b, atol=1e-6)


def test_mixed_list_cycles_and_validation():
    assert member_architectures({"architecture": ["lstm", "tcn"]}, 3) == ["lstm", "tcn", "lstm"]
    assert member_architectures({}, 2) == ["lstm", "lstm"]
    cfg = load_config()
    bad = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        member_architectures({"architecture": "cnn"}, 1)
    import tomllib  # noqa: F401

    from upbit_bot.model_names import check_architecture

    bad["model"].update(architecture="transformer", hidden=30)
    with pytest.raises(ValueError):
        check_architecture(bad["model"])


def test_compare_models_end_to_end(tmp_path):
    from test_features import candles

    from upbit_bot.common import save_json
    from upbit_bot.compare import compare
    from upbit_bot.data import CandleStore
    from upbit_bot.runtime import Bundle

    cfg = copy.deepcopy(load_config())
    cfg["data"]["minimum_days"] = 5
    cfg["model"].update(lookback=16, hidden=8, layers=1, dropout=0, ensemble=1)
    cfg["training"].update(epochs=1, batch_size=32, samples_per_epoch=64, validation_days=1, test_days=1)
    data = tmp_path / "data"
    save_json(data / "universe.json", {"markets": ["KRW-BTC", "KRW-ETH"], "selected_at": "synthetic"})
    store = CandleStore(data)
    for market in ("KRW-BTC", "KRW-ETH"):
        store.put(market, candles(2400))
    store.close()
    result = compare(cfg, data, tmp_path / "cmp", ["tcn", "mixed"], allow_cpu=True)
    assert not result["failures"]
    assert {r["candidate"] for r in result["ranking"]} == {"tcn", "mixed"}
    bundle = Bundle(tmp_path / "cmp" / "mixed")
    assert bundle.manifest["architectures"] == ["lstm", "tcn", "transformer"]
    y, spread = bundle.session.run(None, {"features": np.zeros((2, 16, len(bundle.manifest["features"]) + 2), np.float32)})
    assert np.isfinite(y).all() and (spread >= 0).all()
