from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .strategy import STOP_REASONS, decide, order_key


def metrics(equity, trades, initial):
    series = pd.Series([e for _, e in equity], index=pd.to_datetime([t for t, _ in equity], utc=True))
    if series.empty:
        return {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_daily": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
    daily = series.resample("1D").last().ffill().pct_change().dropna()
    daily_std = daily.std()
    profit = sum(max(t["pnl"], 0) for t in trades)
    loss = -sum(min(t["pnl"], 0) for t in trades)
    peak = np.maximum.accumulate(np.r_[initial, series.to_numpy()])[1:]
    return {
        "return_pct": float((series.iloc[-1] / initial - 1) * 100),
        "max_drawdown_pct": float(np.max(1 - series.to_numpy() / peak) * 100),
        "sharpe_daily": float(daily.mean() / daily_std * math.sqrt(365))
        if len(daily) > 2 and np.isfinite(daily_std) and daily_std > 1e-12
        else 0.0,
        "trades": len(trades),
        "win_rate": sum(t["pnl"] > 0 for t in trades) / max(len(trades), 1),
        "profit_factor": min(profit / max(loss, 1e-9), 999.0),
    }


def simulate(frames, predictions, strategy, threshold, initial=100000.0, max_order=float("inf")):
    """Close[t] decision, open[t+1] fill. No intrabar stop-fill assumptions.

    ``entry_order = "limit"``: buys rest at close[t] x (1 - limit_offset_bps) for one bar and fill
    only if bar t+1 trades strictly below that price (touching is not enough: queue position is
    unknown). Unfilled signals are lost, exactly as the runtime cancels them. Sells stay market.
    """
    times = sorted(set().union(*(set(p.index) for p in predictions.values())))
    cash, positions, marks, trades, equity, last_stop = initial, {}, {}, [], [], {}
    peak, day, day_start, daily_halt, halted = initial, None, initial, False, False
    fee, slip = strategy["fee_rate"], strategy["slippage_bps"] / 10000
    limit_entry = strategy.get("entry_order", "market") == "limit"
    offset = float(strategy.get("limit_offset_bps", 0.0)) / 10000
    missed = 0
    arrays = {m: f.to_dict("index") for m, f in frames.items()}
    for ts in times:
        executable = {}
        for market, rows in arrays.items():
            row = rows.get(ts)
            next_row = rows.get(ts + pd.Timedelta(minutes=5))
            if row is not None:
                marks[market] = row["close"]
            if row and next_row and row["observed"] and next_row["observed"]:
                executable[market] = (row, next_row)
        eq = cash + sum(p["qty"] * marks[m] for m, p in positions.items())
        peak = max(peak, eq)
        if day != ts.date():
            day, day_start, daily_halt = ts.date(), eq, False
        daily_halt = daily_halt or eq < day_start * (1 - strategy["daily_loss_limit"])
        halted = halted or eq < peak * (1 - strategy["max_drawdown"])
        intents = []
        row_close = {m: v[0]["close"] for m, v in executable.items()}
        bar = int(ts.timestamp() // 300)
        for market, p in positions.items():
            if market in executable:
                p["peak"] = max(p["peak"], executable[market][0]["close"])
        for market, (row, nxt) in executable.items():
            pred = predictions[market].get(ts, np.nan)
            position = positions.get(market)
            if not np.isfinite(pred):
                if position is None:
                    continue
                # No forecast: risk exits still apply, forecast exit and entries do not.
                pred, gate = float("inf"), float("inf")
            else:
                pred, gate = float(pred), threshold
            decision, reason = decide(pred, position, row["close"], bar, strategy, gate, last_stop.get(market))
            if market in positions and (daily_halt or halted):
                decision, reason = "sell", "risk_halt"
            intents.append((decision, market, nxt, pred, reason))
        # Sells release cash first, then strongest signals. Same ordering as the runtime.
        for decision, market, nxt, pred, reason in sorted(intents, key=lambda x: order_key(x[0], x[1], x[3])):
            if decision == "sell" and market in positions:
                p = positions.pop(market)
                proceeds = p["qty"] * nxt["open"] * (1 - slip) * (1 - fee)
                cash += proceeds
                trades.append(
                    {
                        "market": market,
                        "time": ts.isoformat(),
                        "pnl": proceeds - p["cost"],
                        "bars": bar - p["bar"],
                        "reason": reason,
                    }
                )
                if reason in STOP_REASONS:
                    last_stop[market] = bar
            elif decision == "buy" and not (daily_halt or halted):
                exposure = sum(p["qty"] * marks[m] for m, p in positions.items())
                amount = min(
                    eq * strategy["position_fraction"],
                    eq * strategy["max_exposure"] - exposure,
                    cash / (1 + fee),
                    max_order,
                )
                if amount < strategy["min_order_krw"]:
                    continue
                if limit_entry:
                    limit = row_close[market] * (1 - offset)
                    if not nxt.get("low", np.inf) < limit:
                        missed += 1
                        continue
                    # A bar opening below the bid makes the order marketable at about the open.
                    price = min(limit, nxt["open"] * (1 + slip))
                else:
                    price = nxt["open"] * (1 + slip)
                cash -= amount * (1 + fee)
                positions[market] = {
                    "qty": amount / price,
                    "entry": price,
                    "peak": price,
                    "cost": amount * (1 + fee),
                    "bar": bar,
                }
        # Value after execution at the next open; last close marks only for missing markets.
        for market, (_, nxt) in executable.items():
            marks[market] = nxt["open"]
        equity.append(
            (ts + pd.Timedelta(minutes=5), cash + sum(p["qty"] * marks[m] for m, p in positions.items()))
        )
    if equity:
        # Charge realistic exit costs to close end-of-period positions for comparable reports.
        for market, p in positions.items():
            proceeds = p["qty"] * marks[market] * (1 - slip) * (1 - fee)
            cash += proceeds
            trades.append(
                {"market": market, "time": equity[-1][0].isoformat(), "pnl": proceeds - p["cost"], "reason": "end"}
            )
        equity[-1] = (equity[-1][0], cash)
    result_metrics = metrics(equity, trades, initial)
    if limit_entry:
        result_metrics["missed_limit_entries"] = missed
    return {
        "metrics": result_metrics,
        "trades": trades,
        "equity": [(t.isoformat(), e) for t, e in equity],
    }


def passed(result, gate):
    m = result["metrics"]
    return (
        m["trades"] >= gate["min_trades"]
        and m["return_pct"] > 0
        and m["sharpe_daily"] > 0
        and m["profit_factor"] >= gate["min_profit_factor"]
        and m["max_drawdown_pct"] <= gate["max_drawdown"] * 100
    )


def buy_hold(frames, predictions, initial, strategy):
    value = initial * (1 - strategy["max_exposure"])
    allocation = initial * strategy["max_exposure"] / len(frames)
    cost = strategy["fee_rate"] + strategy["slippage_bps"] / 10000
    for market, f in frames.items():
        indexes = predictions[market].index
        if len(indexes):
            first, last = indexes[0] + pd.Timedelta(minutes=5), indexes[-1] + pd.Timedelta(minutes=5)
            value += allocation * float(f.loc[last, "open"] / f.loc[first, "open"]) * (1 - cost) / (1 + cost)
        else:
            value += allocation
    return {"return_pct": (value / initial - 1) * 100, "exposure": strategy["max_exposure"]}
