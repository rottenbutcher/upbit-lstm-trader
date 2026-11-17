from __future__ import annotations

import hashlib
import json
import os
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def utcnow():
    return datetime.now(UTC)


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    s, r, m, t = (cfg[k] for k in ("strategy", "runtime", "model", "training"))
    if not 0 < s["position_fraction"] <= s["max_exposure"] <= 1:
        raise ValueError("Invalid position/exposure limits")
    if not 0 < s["daily_loss_limit"] <= s["max_drawdown"] < 1:
        raise ValueError("Invalid loss limits")
    if r["capital_krw"] <= 0 or r["max_order_krw"] < s["min_order_krw"]:
        raise ValueError("Capital/order limits are invalid")
    if min(m["lookback"], m["horizon"], t["batch_size"], t["stride"]) < 1:
        raise ValueError("Model/training sizes must be positive")
    if s["fee_rate"] < 0 or s["slippage_bps"] < 0:
        raise ValueError("Trading costs must be nonnegative")
    if int(m.get("ensemble", 1)) < 1 or int(m.get("feature_version", 1)) not in (1, 2):
        raise ValueError("model.ensemble must be >= 1 and model.feature_version 1 or 2")
    from .model_names import check_architecture

    check_architecture(m)
    if t.get("lr_schedule", "constant") not in ("constant", "cosine"):
        raise ValueError("training.lr_schedule must be 'constant' or 'cosine'")
    if t.get("early_stop_metric", "mae") not in ("mae", "ic"):
        raise ValueError("training.early_stop_metric must be 'mae' or 'ic'")
    if s.get("entry_order", "market") not in ("market", "limit"):
        raise ValueError("strategy.entry_order must be 'market' or 'limit'")
    if s.get("exit_order", "market") not in ("market", "limit"):
        raise ValueError("strategy.exit_order must be 'market' or 'limit'")
    if int(s.get("volatility_window", 48)) < 2:
        raise ValueError("strategy.volatility_window must be at least 2")
    if float(s.get("min_volatility_bps", 0)) < 0:
        raise ValueError("strategy.min_volatility_bps must be nonnegative")
    searchable = {"min_volatility_bps", "trailing_stop", "stop_loss", "take_profit", "min_hold_bars",
                  "cooldown_bars", "max_hold_bars", "exit_bps"}
    for key, values in s.get("search", {}).items():
        if key not in searchable or not isinstance(values, list) or not values:
            raise ValueError(f"strategy.search.{key} must be a non-empty list of a searchable key")
        for value in values:
            if key == "stop_loss" and not 0 < value < 1:
                raise ValueError("strategy.search.stop_loss values must be within (0, 1)")
            if key == "trailing_stop" and not 0 <= value < 1:
                raise ValueError("strategy.search.trailing_stop values must be within [0, 1)")
            if key == "min_volatility_bps" and value < 0:
                raise ValueError("strategy.search.min_volatility_bps values must be nonnegative")
    if not 0 <= float(s.get("limit_offset_bps", 0)) < 100:
        raise ValueError("strategy.limit_offset_bps must be within [0, 100)")
    if not 0 <= float(s.get("trailing_stop", 0)) < 1:
        raise ValueError("strategy.trailing_stop must be within [0, 1)")
    for key in ("min_hold_bars", "cooldown_bars"):
        if int(s.get(key, 0)) < 0:
            raise ValueError(f"strategy.{key} must be nonnegative")
    if int(s.get("min_hold_bars", 0)) > s["max_hold_bars"]:
        raise ValueError("strategy.min_hold_bars cannot exceed max_hold_bars")
    from .policy import settings as policy_settings

    policy_settings(cfg)  # Fails fast on unsafe policy bounds.
    return cfg


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()
