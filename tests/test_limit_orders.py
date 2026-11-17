import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from upbit_bot.api import APIError, Upbit
from upbit_bot.backtest import simulate
from upbit_bot.common import load_config
from upbit_bot.runtime import process_cycle, reconcile
from upbit_bot.state import State

NOW = datetime(2026, 9, 30, 0, 5, 15, tzinfo=UTC)
BAR = int(NOW.timestamp() // 300) - 1


def limit_strategy(**kw):
    s = load_config()["strategy"]
    s.update({"entry_order": "limit", "limit_offset_bps": 0.0, "trailing_stop": 0.0, "min_hold_bars": 0,
              "cooldown_bars": 0, "slippage_bps": 0.0, "fee_rate": 0.0, **kw})
    return s


def frame(lows):
    index = pd.date_range("2024-01-01", periods=len(lows), freq="5min", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": lows, "close": 100.0, "observed": 1}, index=index)


def test_backtest_limit_fills_only_when_traded_through():
    s = limit_strategy(position_fraction=0.5, max_exposure=0.5)
    pred = pd.Series([100.0], index=frame([100] * 3).index[:1])
    touched = simulate({"KRW-A": frame([100.0, 100.0, 100.0])}, {"KRW-A": pred}, s, 20)
    assert touched["metrics"]["trades"] == 0 and touched["metrics"]["missed_limit_entries"] == 1
    through = simulate({"KRW-A": frame([100.0, 99.9, 100.0])}, {"KRW-A": pred}, s, 20)
    assert through["metrics"]["trades"] == 1  # filled at 100, closed at the end at 100
    assert through["trades"][0]["pnl"] == pytest.approx(0)


def test_backtest_limit_gap_down_fills_at_open():
    f = frame([100.0, 97.0, 97.0])
    f.loc[f.index[1], "open"] = 98.0
    f.loc[f.index[1]:, "close"] = 98.0
    s = limit_strategy(position_fraction=0.5, max_exposure=0.5, stop_loss=0.5)
    result = simulate({"KRW-A": f}, {"KRW-A": pd.Series([100.0], index=f.index[:1])}, s, 20)
    assert result["trades"][0]["pnl"] == pytest.approx(0)  # bought at 98 (open), valued at 98


class Paper:
    def markets(self):
        return [{"market": "KRW-BTC"}]

    def orderbook(self, market):
        return {"timestamp": self.now.timestamp() * 1000,
                "orderbook_units": [{"ask_price": 10001, "bid_price": 10000, "ask_size": 50, "bid_size": 50}]}


def bundle(cfg, pred, low):
    manifest = {"strategy": {**cfg["strategy"], "entry_order": "limit"}, "data_end": "2026-09-29T00:00:00+00:00",
                "approved_for_live": True, "threshold_bps": 20, "markets": ["KRW-BTC"], "model_sha256": "abc"}
    return SimpleNamespace(manifest=manifest, report={"approved_for_live": True},
                           infer=lambda api, end: ({"KRW-BTC": pred}, {"KRW-BTC": 10000.0}, {"KRW-BTC": 0.0},
                                                   {"KRW-BTC": {"low": low, "high": 10001.0, "volatility_bps": 30.0}}))


@pytest.mark.parametrize(("low", "filled"), [(9999.0, True), (10000.0, False)])
def test_paper_limit_rests_then_fills_or_cancels(tmp_path, monkeypatch, low, filled):
    cfg = load_config()
    clock = {"now": NOW}
    monkeypatch.setattr("upbit_bot.runtime.utcnow", lambda: clock["now"])
    state = State(tmp_path / "p.sqlite", 100000, "paper")
    api = Paper()
    api.now = NOW
    process_cycle(cfg, bundle(cfg, 100.0, 10000.0), api, state, "paper", tmp_path, NOW)
    assert state.positions() == {} and len(state.pending()) == 1 and state.get("cash") == 100000
    later = NOW + timedelta(minutes=5)
    clock["now"] = api.now = later
    process_cycle(cfg, bundle(cfg, 0.0, low), api, state, "paper", tmp_path, later)
    assert state.pending() == []
    assert ("KRW-BTC" in state.positions()) is filled
    assert state.get("cash") < 100000 if filled else state.get("cash") == 100000
    state.close()


def test_live_resting_limit_cancelled_next_bar_and_partial_fill_applied(tmp_path):
    state = State(tmp_path / "l.sqlite", 100000, "live")
    payload = {"market": "KRW-BTC", "side": "bid", "ord_type": "limit", "price": "10000", "volume": "1"}
    state.intent("ul-x", "KRW-BTC", "buy", BAR, payload)

    class Live:
        cancelled = False

        def order(self, identifier):
            if not self.cancelled:
                return {"market": "KRW-BTC", "side": "bid", "state": "wait"}
            return {"market": "KRW-BTC", "side": "bid", "state": "cancel", "executed_volume": "0.4",
                    "paid_fee": "2", "trades": [{"funds": "4000", "volume": "0.4"}]}

        def cancel(self, identifier):
            self.cancelled = True

    api = Live()
    assert reconcile(api, state, BAR) is False and not api.cancelled  # same bar: keep resting
    assert reconcile(api, state, BAR + 1) is True and api.cancelled
    assert state.positions()["KRW-BTC"]["qty"] == pytest.approx(0.4)
    assert state.get("cash") == pytest.approx(100000 - 4002)
    state.close()


def test_market_orders_are_never_cancelled(tmp_path):
    state = State(tmp_path / "m.sqlite", 100000, "live")
    state.intent("ul-m", "KRW-BTC", "buy", BAR, {"ord_type": "price"})

    class Live:
        def order(self, identifier):
            return {"market": "KRW-BTC", "side": "bid", "state": "wait"}

        def cancel(self, identifier):
            raise AssertionError("must not cancel market orders")

    assert reconcile(Live(), state, BAR + 5) is False
    state.close()


def test_cancel_uses_query_params_and_no_retry():
    calls = []

    class Session:
        def request(self, method, url, headers, timeout, **kw):
            calls.append((method, kw))
            return SimpleNamespace(status_code=500, ok=False, headers={}, json=lambda: {"error": {"name": "x"}})

    api = Upbit("a", "b", interval=0, session=Session())
    with pytest.raises(APIError):
        api.cancel("ul-x")
    assert calls == [("DELETE", {"params": {"identifier": "ul-x"}})]
    assert json.dumps(calls)


def test_volatility_gate_blocks_quiet_entries():
    from upbit_bot.strategy import decide

    s = limit_strategy(min_volatility_bps=20.0)
    assert decide(100, None, 100, 1, s, 20, volatility=10.0) == ("hold", "low_volatility")
    assert decide(100, None, 100, 1, s, 20, volatility=float("nan")) == ("hold", "low_volatility")
    assert decide(100, None, 100, 1, s, 20, volatility=25.0) == ("buy", "signal")
    assert decide(100, None, 100, 1, limit_strategy(), 20, volatility=None) == ("buy", "signal")


def test_volatility_bps_matches_pandas():
    import numpy as np

    from upbit_bot.strategy import volatility_bps

    closes = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.002, 200)))
    expected = pd.Series(np.log(closes)).diff().rolling(48).std().iloc[-1] * 10000
    assert volatility_bps(closes, 48) == pytest.approx(expected)
    assert np.isnan(volatility_bps(closes[:10], 48))


def exit_frame(highs):
    index = pd.date_range("2024-01-01", periods=len(highs), freq="5min", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": highs, "low": 99.0, "close": 100.0, "observed": 1}, index=index)


def test_backtest_limit_exit_fills_through_or_falls_back_to_market():
    s = limit_strategy(entry_order="market", exit_order="limit", position_fraction=0.5, max_exposure=0.5,
                       slippage_bps=10.0)
    # Buy on bar 0 (forecast 100), forecast exit on bar 1.
    pred = pd.Series([100.0, -50.0, -50.0], index=exit_frame([100.0] * 4).index[:3])
    through = simulate({"KRW-A": exit_frame([100.0, 100.0, 100.1, 100.0])}, {"KRW-A": pred}, s, 20)
    assert through["trades"][0]["reason"] == "forecast"
    assert through["metrics"]["missed_limit_exits"] == 0
    missed = simulate({"KRW-A": exit_frame([100.0] * 4)}, {"KRW-A": pred}, s, 20)
    assert missed["metrics"]["missed_limit_exits"] == 1
    # Fallback market sell pays slippage, the limit exit does not.
    assert missed["trades"][0]["pnl"] < through["trades"][0]["pnl"]


@pytest.mark.parametrize(("high", "filled"), [(10002.0, True), (10001.0, False)])
def test_paper_limit_sell_rests_then_fills_or_flags_fallback(tmp_path, monkeypatch, high, filled):
    cfg = load_config()
    clock = {"now": NOW}
    monkeypatch.setattr("upbit_bot.runtime.utcnow", lambda: clock["now"])
    state = State(tmp_path / "s.sqlite", 100000, "paper")
    with state.db:
        state.db.execute(
            "INSERT INTO positions (market,qty,entry,cost,bar,peak) VALUES ('KRW-BTC',1,10000,10000,0,10000)"
        )
        state.set("cash", 90000)
    api = Paper()
    api.now = NOW

    def b(pred, high):
        bundle_ = bundle(cfg, pred, 9990.0)
        bundle_.manifest["strategy"]["exit_order"] = "limit"
        bundle_.infer = lambda api_, end: ({"KRW-BTC": pred}, {"KRW-BTC": 10000.0}, {"KRW-BTC": 0.0},
                                          {"KRW-BTC": {"low": 9990.0, "high": high, "volatility_bps": 30.0}})
        return bundle_

    process_cycle(cfg, b(-100.0, 10000.0), api, state, "paper", tmp_path, NOW)
    assert len(state.pending()) == 1 and "KRW-BTC" in state.positions()
    later = NOW + timedelta(minutes=5)
    clock["now"] = api.now = later
    process_cycle(cfg, b(0.0, high), api, state, "paper", tmp_path, later)
    # Filled at the ask, or missed and then closed by the market-order fallback in the same cycle.
    assert "KRW-BTC" not in state.positions()
    orders = [json.loads(r[0])["ord_type"] for r in state.db.execute("SELECT payload FROM intents ORDER BY rowid")]
    assert orders == (["limit"] if filled else ["limit", "market"])
    state.close()
