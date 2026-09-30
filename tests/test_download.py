import pandas as pd
import pytest

from upbit_bot.common import load_config, read_json, save_json
from upbit_bot.data import CandleStore, download


def row(ts):
    return {
        "candle_date_time_utc": ts,
        "opening_price": 100,
        "high_price": 101,
        "low_price": 99,
        "trade_price": 100,
        "candle_acc_trade_volume": 1,
        "candle_acc_trade_price": 100,
    }


def test_download_resumes_persisted_cursor_after_failure(tmp_path):
    cfg = load_config()
    cfg["data"]["start"] = "2024-01-01"
    save_json(tmp_path / "universe.json", {"markets": ["KRW-BTC"]})

    class FailingAPI:
        calls = 0

        def candles(self, market, to):
            self.calls += 1
            if self.calls == 1:
                return [row("2024-01-01T00:15:00"), row("2024-01-01T00:10:00")]
            raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        download(FailingAPI(), cfg, tmp_path)
    store = CandleStore(tmp_path)
    assert len(store.frame("KRW-BTC")) == 2
    store.close()

    class ResumedAPI:
        def __init__(self):
            self.calls = []

        def candles(self, market, to):
            self.calls.append(pd.Timestamp(to))
            if pd.Timestamp(to) == pd.Timestamp("2024-01-01T00:10:00Z"):
                return [row("2024-01-01T00:05:00"), row("2024-01-01T00:00:00")]
            return [row("2024-01-01T00:20:00"), row("2024-01-01T00:15:00")]

    api = ResumedAPI()
    download(api, cfg, tmp_path)
    assert api.calls[0] == pd.Timestamp("2024-01-01T00:10:00Z")
    coverage = read_json(tmp_path / "coverage.json")
    assert coverage["KRW-BTC"]["observed_candles"] == 5
    store = CandleStore(tmp_path)
    assert store.frame("KRW-BTC").index.is_unique
    store.close()
