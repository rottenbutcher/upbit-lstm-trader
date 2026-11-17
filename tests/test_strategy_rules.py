import math
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from upbit_bot.backtest import simulate
from upbit_bot.common import load_config
from upbit_bot.runtime import process_cycle
from upbit_bot.state import State
from upbit_bot.strategy import decide, order_key
from upbit_bot.train import rank_ic, validation_score

NOW = datetime(2026, 9, 30, 0, 5, 15, tzinfo=UTC)
BAR = int(NOW.timestamp() // 300) - 1


def strategy(**kw):
    s = load_config()["strategy"]
    s.update({"trailing_stop": 0.0, "min_hold_bars": 0, "cooldown_bars": 0, **kw})
    return s


def pos(entry=100.0, bar=0, peak=None):
    return {"qty": 1.0, "entry": entry, "bar": bar, "peak": peak}


def test_old_strategy_without_new_keys_is_unchanged():
    s = load_config()["strategy"]
    for key in ("trailing_stop", "min_hold_bars", "cooldown_bars"):
        s.pop(key, None)
    assert decide(-5, pos(), 100, 1, s, 20) == ("sell", "forecast")
    assert decide(25, None, 100, 1, s, 20) == ("buy", "signal")


def test_trailing_stop_only_after_gain():
    s = strategy(trailing_stop=0.01, take_profit=0.5)
    assert decide(50, pos(peak=100), 99.5, 1, s, 20)[0] == "hold"  # never above entry
    assert decide(50, pos(peak=103), 102.5, 1, s, 20)[0] == "hold"
    assert decide(50, pos(peak=103), 101.9, 1, s, 20) == ("sell", "trailing_stop")


def test_stop_loss_beats_min_hold_but_forecast_exit_waits():
    s = strategy(min_hold_bars=3)
    assert decide(-50, pos(bar=0), 100, 1, s, 20) == ("hold", "holding")
    assert decide(-50, pos(bar=0), 100, 3, s, 20) == ("sell", "forecast")
    assert decide(50, pos(bar=0), 97, 1, s, 20) == ("sell", "stop_loss")


def test_cooldown_blocks_reentry_after_stop():
    s = strategy(cooldown_bars=6)
    assert decide(100, None, 100, 10, s, 20, last_stop_bar=5) == ("hold", "cooldown")
    assert decide(100, None, 100, 11, s, 20, last_stop_bar=5) == ("buy", "signal")


def test_strongest_buy_first_sells_before_buys():
    rows = [("buy", "KRW-A", 30), ("buy", "KRW-B", 90), ("sell", "KRW-C", -10)]
    assert [m for _, m, _ in sorted(rows, key=lambda r: order_key(*r))] == ["KRW-C", "KRW-B", "KRW-A"]


def test_backtest_buys_strongest_when_exposure_allows_one():
    index = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    f = pd.DataFrame({"open": 100.0, "close": 100.0, "observed": 1}, index=index)
    s = strategy(position_fraction=0.5, max_exposure=0.5)
    preds = {"KRW-A": pd.Series([30.0], index[:1]), "KRW-B": pd.Series([90.0], index[:1])}
    result = simulate({"KRW-A": f, "KRW-B": f}, preds, s, 20)
    assert [t["market"] for t in result["trades"]] == ["KRW-B"]


def test_backtest_trailing_stop_and_reason():
    index = pd.date_range("2024-01-01", periods=6, freq="5min", tz="UTC")
    prices = [100, 100, 105, 103, 103, 103]
    f = pd.DataFrame({"open": prices, "close": prices, "observed": 1}, index=index, dtype=float)
    s = strategy(trailing_stop=0.01, take_profit=0.5, slippage_bps=0)
    pred = pd.Series([100.0] * 5, index=index[:5])
    result = simulate({"KRW-A": f}, {"KRW-A": pred}, s, 20)
    assert result["trades"][0]["reason"] == "trailing_stop"


def test_rank_ic_and_score():
    y = np.array([1.0, 2, 3, 4, 100])
    assert rank_ic(y, y) == pytest.approx(1)
    assert rank_ic(np.zeros(5), y) == 0.0
    assert validation_score(y, y, "ic") < validation_score(-y, y, "ic")


def test_state_migrates_old_positions_table(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE positions (market TEXT PRIMARY KEY, qty REAL, entry REAL, cost REAL, bar INT)")
    db.execute("INSERT INTO positions VALUES ('KRW-A', 1, 100, 100, 5)")
    db.commit()
    db.close()
    state = State(path, 100000, "paper")
    state.update_peaks({"KRW-A": 110})
    state.update_peaks({"KRW-A": 105})
    assert state.positions()["KRW-A"]["peak"] == 110
    state.close()


class Book:
    def __init__(self, bid=10000, ask=10001):
        self.bid, self.ask = bid, ask

    def markets(self):
        return [{"market": "KRW-BTC"}, {"market": "KRW-ETH"}]

    def orderbook(self, market):
        return {
            "timestamp": NOW.timestamp() * 1000,
            "orderbook_units": [{"ask_price": self.ask, "bid_price": self.bid, "ask_size": 50, "bid_size": 50}],
        }


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("upbit_bot.runtime.utcnow", lambda: NOW)
    cfg = load_config()
    state = State(tmp_path / "s.sqlite", 100000, "paper")

    def bundle(pred, closes):
        manifest = {
            "strategy": cfg["strategy"],
            "data_end": "2026-09-29T00:00:00+00:00",
            "approved_for_live": True,
            "threshold_bps": 20,
            "markets": ["KRW-BTC", "KRW-ETH"],
            "model_sha256": "abc",
        }
        return SimpleNamespace(
            manifest=manifest, report={"approved_for_live": True}, infer=lambda api, end: (pred, closes)
        )

    yield cfg, state, bundle, tmp_path
    state.close()


def test_missing_candle_market_holds_without_failing_cycle(env):
    cfg, state, bundle, path = env
    b = bundle({"KRW-BTC": math.nan, "KRW-ETH": 100.0}, {"KRW-BTC": 10000, "KRW-ETH": 10000})
    result = process_cycle(cfg, b, Book(), state, "paper", path, NOW)
    assert result["status"] == "ok"
    assert set(state.positions()) == {"KRW-ETH"}
    assert result["predictions_bps"]["KRW-BTC"] is None


def test_dust_position_does_not_block_reentry(env):
    cfg, state, bundle, path = env
    with state.db:
        state.db.execute(
            "INSERT INTO positions (market,qty,entry,cost,bar,peak) VALUES ('KRW-BTC',0.0001,10000,1,0,10000)"
        )
    b = bundle({"KRW-BTC": 100.0, "KRW-ETH": 0.0}, {"KRW-BTC": 10000, "KRW-ETH": 10000})
    process_cycle(cfg, b, Book(), state, "paper", path, NOW)
    assert state.positions()["KRW-BTC"]["qty"] > 0.5


def test_stop_loss_sells_through_wide_spread_and_sets_cooldown(env):
    cfg, state, bundle, path = env
    with state.db:
        state.db.execute(
            "INSERT INTO positions (market,qty,entry,cost,bar,peak) VALUES ('KRW-BTC',1,10500,10500,0,10500)"
        )
        state.set("cash", 100000 - 10500)
    b = bundle({"KRW-BTC": 100.0, "KRW-ETH": 0.0}, {"KRW-BTC": 10000, "KRW-ETH": 10000})
    # 30 bps spread: above the 20 bps entry limit, within the protective limit.
    process_cycle(cfg, b, Book(bid=10000, ask=10030), state, "paper", path, NOW)
    assert "KRW-BTC" not in state.positions()
    assert state.last_stop("KRW-BTC") == BAR
