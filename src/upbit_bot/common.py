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
    if t.get("lr_schedule", "constant") not in ("constant", "cosine"):
        raise ValueError("training.lr_schedule must be 'constant' or 'cosine'")
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
