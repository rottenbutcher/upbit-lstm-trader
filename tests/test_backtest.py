import pandas as pd
import pytest

from upbit_bot.backtest import simulate
from upbit_bot.common import load_config


def test_next_open_execution_charges_fees_and_no_future_fill():
    index = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
    f = pd.DataFrame(
        {"open": [100, 200, 220, 220], "close": [100, 200, 220, 220], "observed": 1}, index=index
    )
    pred = pd.Series([100, -100], index=index[:2])
    s = load_config()["strategy"]
    s.update(
        position_fraction=0.5,
        max_exposure=0.5,
        fee_rate=0.001,
        slippage_bps=0,
        stop_loss=0.9,
        take_profit=0.9,
        max_drawdown=0.9,
        daily_loss_limit=0.9,
    )
    result = simulate({"KRW-BTC": f}, {"KRW-BTC": pred}, s, 20)
    # Buy at 200, not signal close 100; then sell at 220.
    expected_cash = 100000 - 50000 * 1.001 + (50000 / 200) * 220 * 0.999
    assert result["equity"][-1][1] == pytest.approx(expected_cash)
    assert result["metrics"]["trades"] == 1


def test_no_trade_bar_does_not_fill():
    index = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    f = pd.DataFrame({"open": 100, "close": 100, "observed": [1, 0, 1]}, index=index)
    result = simulate(
        {"KRW-BTC": f}, {"KRW-BTC": pd.Series([100], index=index[:1])}, load_config()["strategy"], 20
    )
    assert result["metrics"]["trades"] == 0
    assert result["metrics"]["return_pct"] == 0
