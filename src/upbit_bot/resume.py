"""What a training checkpoint depends on, and why it cannot be resumed.

Only settings that change the meaning of saved weights belong to the identity: model shape,
features, optimizer/sampling settings, the data split and the exact candles. Settings that are
safe to change between runs are deliberately excluded:

* ``training.epochs`` / ``patience`` - extend or shorten training.
* ``training.batch_size`` / ``workers`` - throughput only (batch also shrinks on GPU OOM).
* ``model.ensemble`` - members are trained independently; finished members are reused.
* ``strategy`` / ``gate`` / ``runtime`` / ``policy`` / ``llm`` / ``macro`` / ``hmm`` - evaluated
  after training and never stored in weights.

This module imports no torch so the GUI can check a checkpoint before launching training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import read_json, save_json, utcnow

VERSION = 2
INFO_FILE = "resume.json"
MISMATCH_FILE = "resume-mismatch.json"
MODEL_KEYS = ("lookback", "horizon", "hidden", "layers", "dropout", "feature_version", "attention")
MODEL_DEFAULTS = {"feature_version": 1, "attention": False}
TRAINING_KEYS = (
    "seed",
    "learning_rate",
    "weight_decay",
    "samples_per_epoch",
    "stride",
    "validation_days",
    "test_days",
    "lr_schedule",
    "warmup_epochs",
)
TRAINING_DEFAULTS = {"lr_schedule": "constant", "warmup_epochs": 0}
LEGACY = "legacy"


class ResumeMismatch(ValueError):
    """Raised when a checkpoint was produced for different data or settings."""

    def __init__(self, changes):
        self.changes = changes
        lines = "\n".join(f"  - {describe(c)}" for c in changes)
        super().__init__(
            "Data/config changed since checkpoint; cannot resume:\n"
            f"{lines}\nStart a fresh run without --resume (data is kept)."
        )


def config_identity(cfg):
    m, t = cfg["model"], cfg["training"]
    return {
        "model": {k: m.get(k, MODEL_DEFAULTS.get(k)) for k in MODEL_KEYS},
        "training": {k: t.get(k, TRAINING_DEFAULTS.get(k)) for k in TRAINING_KEYS},
    }


def identity(cfg, markets, snapshots):
    return {
        "version": VERSION,
        "config": config_identity(cfg),
        "markets": list(markets),
        "data": {m: snapshots[m] for m in markets},
    }


def _normal(value):
    """1 and 1.0 (or 0 and 0.0) written differently in TOML are the same setting."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normal(v) for v in value]
    return value


def fingerprint(ident):
    return hashlib.sha256(json.dumps(_normal(ident), sort_keys=True).encode()).hexdigest()


def config_changes(old, new):
    """[(section, key, old, new)] for differing training-relevant settings."""
    changes = []
    for section in ("model", "training"):
        before, after = old.get(section, {}), new.get(section, {})
        for key in sorted(set(before) | set(after)):
            if _normal(before.get(key)) != _normal(after.get(key)):
                changes.append((section, key, before.get(key), after.get(key)))
    return changes


def changes(old, new):
    """Every reason ``old`` (checkpoint) cannot be resumed with ``new`` (current)."""
    if not isinstance(old, dict) or old.get("version") != VERSION:
        return [(LEGACY, None, None, None)]
    out = config_changes(old["config"], new["config"])
    if old["markets"] != new["markets"]:
        out.append(("data", "markets", old["markets"], new["markets"]))
    for market in new["markets"]:
        a, b = old["data"].get(market), new["data"].get(market)
        if a != b:
            out.append(("data", market, a and a.get("last"), b and b.get("last")))
    return out


def describe(change):
    section, key, old, new = change
    if section == LEGACY:
        return "checkpoint was created by an older version of this program"
    if section == "data" and key == "markets":
        return f"market universe changed: {old} -> {new}"
    if section == "data":
        return f"saved candles for {key} changed (last candle {old} -> {new})"
    return f"{section}.{key}: {old!r} -> {new!r}"


def write_info(out, ident):
    save_json(Path(out) / INFO_FILE, {"identity": ident, "created_at": utcnow().isoformat()})


def read_info(out):
    path = Path(out) / INFO_FILE
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def has_checkpoint(out):
    return any(Path(out).glob("checkpoint*.pt"))
