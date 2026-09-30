from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from filelock import FileLock

from .api import APIError, Upbit, healthy_market
from .common import read_json, save_json, sha256, utcnow
from .data import candle_frame, iso
from .features import FEATURES, feature_frame, model_input
from .state import State
from .strategy import action

LOG = logging.getLogger(__name__)


class Bundle:
    def __init__(self, directory):
        path = Path(directory)
        self.manifest = m = read_json(path / "manifest.json")
        if m["schema"] != 1 or m["features"] != FEATURES:
            raise ValueError("Unsupported model/feature schema")
        if sha256(path / "model.onnx") != m["model_sha256"]:
            raise ValueError("Model hash mismatch")
        if sha256(path / "report.json") != m["report_sha256"]:
            raise ValueError("Report hash mismatch")
        self.report = read_json(path / "report.json")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def infer(self, api, end):
        m = self.manifest
        x, closes = [], {}
        count = m["lookback"] + 48 + 12
        for i, market in enumerate(m["markets"]):
            rows, before = [], end
            for _ in range(math.ceil(count / 200) + 3):
                page = api.candles(market, iso(before), min(200, count))
                if not page:
                    break
                rows.extend(page)
                before = pd.Timestamp(page[-1]["candle_date_time_utc"], tz="UTC")
                if len(rows) >= count:
                    break
            raw = candle_frame(rows)
            raw = raw.loc[raw.index < end]
            expected = end - pd.Timedelta(minutes=5)
            if raw.empty or raw.index[-1] != expected:
                raise ValueError(f"{market}: most recent closed candle missing; skip trading")
            frame, feat = feature_frame(raw)
            window = feat.iloc[-m["lookback"] :].to_numpy()
            if len(window) != m["lookback"] or not np.isfinite(window).all():
                raise ValueError(f"{market}: incomplete features")
            x.append(model_input(window, m["scaler"], i, len(m["markets"])))
            closes[market] = float(frame["close"].iloc[-1])
        pred = self.session.run(None, {"features": np.stack(x)})[0]
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite model output")
        return dict(zip(m["markets"], map(float, pred))), closes


def reconcile(api, state):
    for row in state.pending():
        try:
            order = api.order(row["id"])
        except APIError as exc:
            if exc.status == 404:
                raise RuntimeError(
                    f"Unknown order outcome {row['id']}; do not resend. Inspect on Upbit."
                ) from exc
            raise
        if (
            order.get("market") != row["market"]
            or order.get("side") != {"buy": "bid", "sell": "ask"}[row["side"]]
        ):
            raise ValueError("Order reconciliation identity mismatch")
        if order["state"] not in ("done", "cancel"):
            return False
        qty = float(order.get("executed_volume", 0))
        trades = order.get("trades", [])
        funds = sum(float(t["funds"]) for t in trades)
        traded_qty = sum(float(t["volume"]) for t in trades)
        if abs(traded_qty - qty) > max(1e-10, qty * 1e-7):
            raise ValueError("Trade details incomplete; cannot reconcile inventory")
        state.apply_fill(row["id"], qty, funds, float(order.get("paid_fee", 0)), order)
    return not state.pending()


def book_prices(book, now):
    if now.timestamp() * 1000 - float(book["timestamp"]) > 15000:
        raise ValueError("Stale order book")
    unit = book["orderbook_units"][0]
    bid, ask = float(unit["bid_price"]), float(unit["ask_price"])
    if bid <= 0 or ask < bid:
        raise ValueError("Invalid book")
    return bid, ask, (ask / bid - 1) * 10000


def risk_state(state, equity, day, strategy):
    with state.db:
        peak = max(state.get("peak"), equity)
        state.set("peak", peak)
        if state.get("day") != day:
            state.set("day", day)
            state.set("day_start", equity)
            state.set("daily_halt", False)
        if equity < state.get("day_start") * (1 - strategy["daily_loss_limit"]):
            state.set("daily_halt", True)
        if equity < peak * (1 - strategy["max_drawdown"]):
            state.set("risk_halt", True)
    return state.get("daily_halt", False) or state.get("risk_halt", False)


def process_cycle(cfg, bundle, api, state, mode, state_dir, now=None):
    now = now or utcnow()
    m, s, r = bundle.manifest, bundle.manifest["strategy"], cfg["runtime"]
    if mode == "live" and not reconcile(api, state):
        return {"status": "waiting_for_fill"}
    if (Path(state_dir) / "STOP").exists():
        return {"status": "stopped", "note": "STOP disables orders, including liquidation"}
    end = pd.Timestamp(now).floor("5min")
    bar = int(end.timestamp() // 300) - 1
    if state.db.execute("SELECT 1 FROM cycles WHERE bar=?", (bar,)).fetchone():
        return {"status": "already_processed", "bar": bar}
    pred, closes = bundle.infer(api, end)
    if utcnow().timestamp() - (end.timestamp() - 300) > r["max_candle_age_seconds"]:
        raise ValueError("Candle expired during inference")
    healthy = {row["market"] for row in api.markets() if healthy_market(row)}
    positions = state.positions()
    if set(positions) - set(m["markets"]):
        raise ValueError("Model universe changed while positions are open; restore previous bundle")
    cash = state.get("cash")
    equity = cash + sum(p["qty"] * closes[market] for market, p in positions.items())
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("Invalid portfolio equity")
    loss_halt = risk_state(state, equity, now.date().isoformat(), s)
    age_days = (pd.Timestamp(now) - pd.Timestamp(m["data_end"])).total_seconds() / 86400
    live_approved = (
        m["approved_for_live"]
        and bundle.report["approved_for_live"]
        and 0 <= age_days <= r["model_max_age_days"]
    )
    buy_allowed = not loss_halt and (mode == "paper" or live_approved)
    if not state.claim_cycle(bar):
        return {"status": "already_processed", "bar": bar}
    decisions = []
    for market in sorted(m["markets"]):
        decision = action(pred[market], positions.get(market), closes[market], bar, s, m["threshold_bps"])
        if loss_halt and market in positions:
            decision = "sell"
        decisions.append((decision, market))
    for decision, market in sorted(decisions, key=lambda x: (x[0] != "sell", x[1])):
        if decision == "hold" or (decision == "buy" and (not buy_allowed or market not in healthy)):
            continue
        current_time = utcnow()
        if (current_time - end.to_pydatetime()).total_seconds() > r["max_candle_age_seconds"] - 300:
            raise ValueError("Signal expired; no further orders this cycle")
        book = api.orderbook(market)
        bid, ask, spread = book_prices(book, current_time)
        if spread > r["max_spread_bps"]:
            LOG.warning("%s spread %.1f bps: skip %s", market, spread, decision)
            continue
        positions = state.positions()
        p = positions.get(market)
        fee = s["fee_rate"]
        minimum = s["min_order_krw"]
        available_cash = state.get("cash")
        available_qty = p["qty"] if p else 0.0
        max_total = float("inf")
        if mode == "live":
            chance = api.chance(market)
            if chance["market"]["state"] != "active":
                continue
            fee = float(chance["bid_fee"] if decision == "buy" else chance["ask_fee"])
            side_name = "bid" if decision == "buy" else "ask"
            supported = chance["market"].get(f"{side_name}_types", [])
            if ("price" if decision == "buy" else "market") not in supported:
                raise ValueError("Market order type unavailable")
            minimum = max(minimum, float(chance["market"][side_name]["min_total"]))
            max_total = float(chance["market"].get("max_total") or "inf")
            available_cash = min(available_cash, float(chance["bid_account"]["balance"]))
            account_qty = float(chance["ask_account"]["balance"])
            if p and account_qty + max(1e-10, p["qty"] * 1e-8) < p["qty"]:
                raise ValueError(
                    "Exchange balance below bot ledger (manual sale/locked order); stop and reconcile"
                )
            if decision == "buy" and fee > s["fee_rate"]:
                LOG.warning("Live fee exceeds backtest assumption; skip buy")
                continue
        identifier = (
            "ul-"
            + hashlib.sha256(f"{m['model_sha256']}:{mode}:{bar}:{market}:{decision}".encode()).hexdigest()[
                :48
            ]
        )
        payload = {"market": market, "side": "bid" if decision == "buy" else "ask", "identifier": identifier}
        if decision == "buy":
            exposure = sum(v["qty"] * closes[k] for k, v in positions.items())
            amount = min(
                equity * s["position_fraction"],
                equity * s["max_exposure"] - exposure,
                r["max_order_krw"],
                max_total,
                available_cash / (1 + fee),
            )
            amount = math.floor(amount)
            if amount < minimum:
                continue
            if float(book["orderbook_units"][0]["ask_size"]) * ask < amount:
                continue
            payload.update(ord_type="price", price=str(amount))
            price = ask * (1 + s["slippage_bps"] / 10000)
            qty, funds = amount / price, float(amount)
        else:
            if not p:
                continue
            qty = float(Decimal(str(available_qty)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN))
            if qty * bid < minimum:
                LOG.info("%s remaining dust below minimum; retain ledger", market)
                continue
            if qty * bid > max_total:
                raise ValueError("Sell exceeds exchange maximum")
            payload.update(ord_type="market", volume=f"{qty:.8f}")
            funds = qty * bid * (1 - s["slippage_bps"] / 10000)
        state.intent(identifier, market, decision, bar, payload)
        if mode == "paper":
            state.apply_fill(identifier, qty, funds, funds * fee, {"paper": True})
        else:
            # Do not wrap POST in retry. Pending intent survives timeouts, crashes and lost responses.
            api.submit(payload)
            for _ in range(5):
                if reconcile(api, state):
                    break
                time.sleep(1)
            if state.pending():
                return {"status": "waiting_for_fill", "bar": bar}
        LOG.info("%s %s %s (identifier=%s)", mode, decision, market, identifier)
    return {
        "status": "ok",
        "bar": bar,
        "equity_krw": equity,
        "predictions_bps": pred,
        "positions": state.positions(),
        "buy_allowed": buy_allowed,
        "risk_halt": loss_halt,
    }


def run_once(cfg, artifacts="artifacts", state_dir="state", mode="paper"):
    if mode not in ("paper", "live"):
        raise ValueError("Mode must be paper or live")
    if mode == "live" and os.environ.get("LIVE_ACK") != "I_UNDERSTAND_REAL_ORDERS":
        raise RuntimeError("Set LIVE_ACK=I_UNDERSTAND_REAL_ORDERS explicitly to enable real orders")
    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "bot.lock"), timeout=0):
        bundle = Bundle(artifacts)
        if (
            mode == "live"
            and cfg["runtime"]["capital_krw"] != bundle.manifest["config"]["runtime"]["capital_krw"]
        ):
            raise ValueError(
                "Live capital differs from evaluated capital; retrain with the desired configuration"
            )
        api = Upbit(
            os.environ.get("UPBIT_ACCESS_KEY", "") if mode == "live" else "",
            os.environ.get("UPBIT_SECRET_KEY", "") if mode == "live" else "",
        )
        state = State(directory / f"{mode}.sqlite", cfg["runtime"]["capital_krw"], mode)
        try:
            if state.get("errors", 0) >= cfg["runtime"]["max_errors"]:
                raise RuntimeError(
                    "Error circuit open; inspect logs and use reset-errors after resolving cause"
                )
            result = process_cycle(cfg, bundle, api, state, mode, directory)
            with state.db:
                state.set("errors", 0)
            save_json(directory / "health.json", {"at": utcnow().isoformat(), "mode": mode, **result})
            LOG.info("cycle %s", result)
            return result
        except Exception as exc:
            with state.db:
                state.set("errors", state.get("errors", 0) + 1)
            save_json(
                directory / "health.json",
                {
                    "at": utcnow().isoformat(),
                    "mode": mode,
                    "status": "error",
                    "error_type": type(exc).__name__,
                },
            )
            raise
        finally:
            state.close()


def seconds_to_next(now, delay=15):
    return (math.floor(now / 300) + 1) * 300 + delay - now


def run(cfg, artifacts, state_dir, mode, once=False):
    if once:
        return run_once(cfg, artifacts, state_dir, mode)
    while True:
        # Start only on a fresh candle boundary plus the configured publication delay.
        time.sleep(seconds_to_next(time.time(), cfg["runtime"]["delay_seconds"]))
        try:
            run_once(cfg, artifacts, state_dir, mode)
        except Exception:
            LOG.exception("Cycle failed; persisted error counter will stop repeated failures")


def status(state_dir):
    path = Path(state_dir) / "health.json"
    result = read_json(path)
    age = (utcnow() - datetime.fromisoformat(result["at"])).total_seconds()
    return {**result, "heartbeat_age_seconds": age, "healthy": age < 660 and result["status"] == "ok"}
