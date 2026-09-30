import numpy as np
import pandas as pd

from upbit_bot.data import candle_frame, regularize
from upbit_bot.features import feature_frame, split_indices


def candles(n=1000, start="2024-01-01"):
    rng = np.random.default_rng(4)
    c = 10000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.001,
            "low": c * 0.999,
            "close": c,
            "volume": rng.uniform(1, 100, n),
            "value": c * rng.uniform(1, 100, n),
        },
        index=pd.date_range(start, periods=n, freq="5min", tz="UTC"),
    )


def test_features_are_causal():
    raw = candles()
    _, before = feature_frame(raw)
    changed = raw.copy()
    changed.iloc[600:, :4] *= 3
    _, after = feature_frame(changed)
    np.testing.assert_equal(before.iloc[:600].to_numpy(), after.iloc[:600].to_numpy())


def test_no_backfill_and_missing_bar():
    raw = candles(100).drop(candles(100).index[10])
    frame = regularize(raw)
    assert len(frame) == 100
    assert frame.iloc[10]["close"] == frame.iloc[9]["close"]
    assert frame.iloc[10]["volume"] == frame.iloc[10]["observed"] == 0
    assert frame.index[0] == raw.index[0]


def test_split_purges_forward_labels():
    f, x = feature_frame(candles())
    a, b = f.index[600], f.index[800]
    parts = split_indices(f, x, 96, 6, a, b)
    assert (f.index[parts["train"] + 7] < a).all()
    assert (f.index[parts["val"]] >= a).all()
    assert (f.index[parts["val"] + 7] < b).all()
    assert (f.index[parts["test"]] >= b).all()
    assert not set(parts["train"]) & set(parts["val"])


def test_candle_sort_deduplicate():
    row = {
        "candle_date_time_utc": "2024-01-01T00:00:00",
        "opening_price": 10,
        "high_price": 11,
        "low_price": 9,
        "trade_price": 10,
        "candle_acc_trade_volume": 2,
        "candle_acc_trade_price": 20,
    }
    frame = candle_frame([row, row])
    assert len(frame) == 1
    assert str(frame.index.tz) == "UTC"


def test_live_rolling_window_matches_historical_features():
    raw = candles()
    _, full = feature_frame(raw)
    _, recent = feature_frame(raw.iloc[-156:])
    np.testing.assert_allclose(full.iloc[-96:].to_numpy(), recent.iloc[-96:].to_numpy(), atol=1e-5)
