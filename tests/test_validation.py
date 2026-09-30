import pytest
from test_features import candles

from upbit_bot.common import load_config, save_json
from upbit_bot.data import CandleStore


def test_scaler_fits_train_only_even_when_test_price_changes(tmp_path):
    pytest.importorskip("torch")
    from upbit_bot.train import prepare

    cfg = load_config()
    cfg["data"]["minimum_days"] = 5
    cfg["training"].update(validation_days=1, test_days=1)
    cfg["model"]["lookback"] = 16
    directory = tmp_path / "data"
    save_json(directory / "universe.json", {"markets": ["KRW-BTC"]})
    raw = candles(2400)
    store = CandleStore(directory)
    store.put("KRW-BTC", raw)
    original = prepare(cfg, directory, tmp_path / "one")
    raw.iloc[-200:, :4] *= 3
    store.put("KRW-BTC", raw)
    changed = prepare(cfg, directory, tmp_path / "two")
    store.close()
    assert original[3] == changed[3]
