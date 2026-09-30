def action(prediction, position, price, bar, strategy, threshold):
    """Shared closed-bar decision rules for backtest, paper and live."""
    if position and position["qty"] > 0:
        change = price / position["entry"] - 1
        if (
            change <= -strategy["stop_loss"]
            or change >= strategy["take_profit"]
            or bar - position["bar"] >= strategy["max_hold_bars"]
            or prediction <= strategy["exit_bps"]
        ):
            return "sell"
        return "hold"
    return "buy" if prediction >= threshold else "hold"
