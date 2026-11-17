"""Shared closed-bar decision rules for backtest, paper and live.

Optional keys (missing = disabled, so older bundles behave exactly as before):

* ``trailing_stop``  - exit when the close falls this fraction below the highest close since entry.
* ``min_hold_bars``  - forecast-driven exits wait this many bars (risk exits never wait). Cuts
  fee churn from noisy forecasts flipping sign bar to bar.
* ``cooldown_bars``  - after a stop/trailing exit, no re-entry in that market for this many bars.
* ``min_volatility_bps`` - enter only when the 5-minute return volatility over ``volatility_window``
  bars is at least this many bps. The forecast's edge is only tradable in active markets.
"""

import math
from itertools import pairwise

# Exits that are not urgent may rest as limit orders; stops and halts are always market orders.
LIMIT_EXIT_REASONS = ("forecast", "max_hold", "take_profit")


def volatility_bps(closes, window):
    """Std of the last ``window`` 5-minute log returns, in bps (closes: oldest first)."""
    values = [float(c) for c in closes][-(window + 1) :]
    if len(values) < window + 1:
        return float("nan")
    rets = [math.log(b / a) for a, b in pairwise(values)]
    mean = sum(rets) / len(rets)
    return math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) * 10000


def decide(prediction, position, price, bar, strategy, threshold, last_stop_bar=None, volatility=None):
    """Returns (decision, reason). decision is "buy", "sell" or "hold"."""
    if position and position["qty"] > 0:
        change = price / position["entry"] - 1
        held = bar - position["bar"]
        if change <= -strategy["stop_loss"]:
            return "sell", "stop_loss"
        trailing = float(strategy.get("trailing_stop", 0) or 0)
        peak = max(float(position.get("peak") or position["entry"]), price)
        if trailing > 0 and peak > position["entry"] and price <= peak * (1 - trailing):
            return "sell", "trailing_stop"
        if change >= strategy["take_profit"]:
            return "sell", "take_profit"
        if held >= strategy["max_hold_bars"]:
            return "sell", "max_hold"
        if prediction <= strategy["exit_bps"] and held >= int(strategy.get("min_hold_bars", 0) or 0):
            return "sell", "forecast"
        return "hold", "holding"
    cooldown = int(strategy.get("cooldown_bars", 0) or 0)
    if cooldown > 0 and last_stop_bar is not None and bar - last_stop_bar < cooldown:
        return "hold", "cooldown"
    floor = float(strategy.get("min_volatility_bps", 0) or 0)
    if floor > 0 and prediction >= threshold and not (volatility is not None and volatility >= floor):
        return "hold", "low_volatility"
    return ("buy", "signal") if prediction >= threshold else ("hold", "no_signal")


def action(prediction, position, price, bar, strategy, threshold, last_stop_bar=None, volatility=None):
    return decide(prediction, position, price, bar, strategy, threshold, last_stop_bar, volatility)[0]


STOP_REASONS = ("stop_loss", "trailing_stop")


def order_key(decision, market, signal):
    """Sells first (release cash), then the strongest buy signals, deterministic by market."""
    return (decision != "sell", -signal if decision == "buy" else 0.0, market)
