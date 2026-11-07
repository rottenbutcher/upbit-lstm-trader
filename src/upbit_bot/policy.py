"""Daily trading policy: bounded parameter overrides on top of the backtested LSTM strategy.

The policy file is produced once a day (normally by the LLM pipeline in ``llm_policy.py``) and is
re-validated by the runtime on *every* 5-minute cycle. Nothing in the file is trusted:

* Risk sizing (position size, total exposure, stop distance, daily loss limit) can only be
  tightened relative to the values the model was evaluated with. The LLM cannot add risk.
* Other parameters move inside configured multiplier bands around the evaluated values.
* Each daily update may move a parameter at most ``max_step`` of its allowed range away from the
  previous valid policy, so one bad answer cannot swing the strategy to an extreme.
* A missing, malformed or expired policy falls back to the evaluated baseline (or blocks new
  entries if ``on_stale = "no_new_entries"``). ``max_drawdown`` is never policy-controlled because
  it latches a halt that only an operator may reset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from .common import read_json, save_json, utcnow

SCHEMA = "policy-v1"
FILENAME = "policy.json"
REGIMES = ("risk_on", "neutral", "risk_off")

DEFAULT_SETTINGS = {
    "enabled": True,
    "valid_hours": 36.0,
    "on_stale": "baseline",
    "max_step": 0.5,
    "threshold_multiplier": [0.8, 3.0],
    "position_multiplier": [0.0, 1.0],
    "exposure_multiplier": [0.0, 1.0],
    "stop_loss_multiplier": [0.5, 1.0],
    "take_profit_multiplier": [0.5, 1.5],
    "hold_multiplier": [0.5, 2.0],
    "daily_loss_multiplier": [0.5, 1.0],
    "exit_bps": [-20.0, 30.0],
    "uncertainty_penalty": [0.0, 2.0],
}

# Multipliers that scale risk. Their upper bound may never exceed 1 (no added risk).
RISK_BANDS = ("position_multiplier", "exposure_multiplier", "stop_loss_multiplier", "daily_loss_multiplier")


def settings(cfg):
    s = copy.deepcopy(DEFAULT_SETTINGS)
    s.update(cfg.get("policy", {}))
    return validate_settings(s)


def validate_settings(s):
    if s["on_stale"] not in ("baseline", "no_new_entries"):
        raise ValueError("policy.on_stale must be 'baseline' or 'no_new_entries'")
    if not 0 < float(s["valid_hours"]) <= 24 * 7:
        raise ValueError("policy.valid_hours must be within (0, 168]")
    if not 0 < float(s["max_step"]) <= 1:
        raise ValueError("policy.max_step must be within (0, 1]")
    for key in (*RISK_BANDS, "threshold_multiplier", "take_profit_multiplier", "hold_multiplier"):
        lo, hi = map(float, s[key])
        if not (0 <= lo <= hi and math.isfinite(hi)):
            raise ValueError(f"policy.{key} must be [low, high] with 0 <= low <= high")
    for key in RISK_BANDS:
        if float(s[key][1]) > 1:
            raise ValueError(f"policy.{key} upper bound must be <= 1: the policy may only reduce risk")
    if float(s["threshold_multiplier"][0]) <= 0:
        raise ValueError("policy.threshold_multiplier lower bound must be positive")
    for key in ("exit_bps", "uncertainty_penalty"):
        lo, hi = map(float, s[key])
        if not (lo <= hi and math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"policy.{key} must be a finite [low, high]")
    if float(s["uncertainty_penalty"][0]) < 0:
        raise ValueError("policy.uncertainty_penalty must be nonnegative")
    return s


def bounds(base, threshold, s):
    """Absolute [low, high] for every tunable parameter, derived from the evaluated baseline."""

    def band(key, value):
        lo, hi = s[key]
        return [value * lo, value * hi]

    hold_lo, hold_hi = band("hold_multiplier", base["max_hold_bars"])
    return {
        "threshold_bps": band("threshold_multiplier", threshold),
        "position_fraction": band("position_multiplier", base["position_fraction"]),
        "max_exposure": band("exposure_multiplier", base["max_exposure"]),
        "stop_loss": band("stop_loss_multiplier", base["stop_loss"]),
        "take_profit": band("take_profit_multiplier", base["take_profit"]),
        "max_hold_bars": [max(1, math.ceil(hold_lo)), max(1, math.floor(hold_hi))],
        "daily_loss_limit": band("daily_loss_multiplier", base["daily_loss_limit"]),
        "exit_bps": list(map(float, s["exit_bps"])),
        "uncertainty_penalty": list(map(float, s["uncertainty_penalty"])),
    }


def baseline_params(base, threshold):
    return {
        "allow_new_entries": True,
        "threshold_bps": float(threshold),
        "position_fraction": float(base["position_fraction"]),
        "max_exposure": float(base["max_exposure"]),
        "stop_loss": float(base["stop_loss"]),
        "take_profit": float(base["take_profit"]),
        "max_hold_bars": int(base["max_hold_bars"]),
        "daily_loss_limit": float(base["daily_loss_limit"]),
        "exit_bps": float(base["exit_bps"]),
        "uncertainty_penalty": 0.0,
        "blocked_markets": [],
    }


def _number(value):
    if isinstance(value, bool):
        raise ValueError("boolean where number expected")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("nonfinite number")
    return x


def sanitize(proposal, base, threshold, s, markets, previous=None):
    """Clamp a proposal into bounds and limit the step from ``previous``.

    Returns (params, notes). Unknown keys are ignored; invalid values keep the previous value.
    """
    limits = bounds(base, threshold, s)
    start = baseline_params(base, threshold)
    if previous:
        for key in start:
            if key in previous:
                start[key] = previous[key]
    params, notes = copy.deepcopy(start), []
    proposal = proposal if isinstance(proposal, dict) else {}
    for key, (lo, hi) in limits.items():
        prior = min(max(float(start[key]), lo), hi)
        if key not in proposal:
            params[key] = prior
            continue
        try:
            value = _number(proposal[key])
        except (TypeError, ValueError):
            notes.append(f"{key}: invalid value ignored")
            params[key] = prior
            continue
        clamped = min(max(value, lo), hi)
        if clamped != value:
            notes.append(f"{key}: {value:g} clamped to [{lo:g}, {hi:g}]")
        step = float(s["max_step"]) * (hi - lo)
        stepped = min(max(clamped, prior - step), prior + step)
        if abs(stepped - clamped) > 1e-12:
            notes.append(f"{key}: change limited to {stepped:g} (max daily step)")
        params[key] = stepped
    params["max_hold_bars"] = round(params["max_hold_bars"])
    params["max_hold_bars"] = int(min(max(params["max_hold_bars"], limits["max_hold_bars"][0]),
                                      limits["max_hold_bars"][1]))
    # Exposure must hold at least one position; otherwise entries are effectively off.
    params["position_fraction"] = min(params["position_fraction"], params["max_exposure"])
    allow = proposal.get("allow_new_entries", start["allow_new_entries"])
    params["allow_new_entries"] = allow if isinstance(allow, bool) else False
    if not isinstance(allow, bool):
        notes.append("allow_new_entries: invalid value, entries disabled")
    blocked = proposal.get("blocked_markets", start["blocked_markets"])
    if not isinstance(blocked, list):
        blocked = []
    params["blocked_markets"] = sorted({m for m in blocked if isinstance(m, str) and m in markets})
    return params, notes


def path(state_dir):
    return Path(state_dir) / FILENAME


def digest(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def write(state_dir, params, meta, now=None):
    now = now or utcnow()
    doc = {
        "schema": SCHEMA,
        "created_at": now.isoformat(),
        "params": params,
        "params_sha256": digest(params),
        **meta,
    }
    save_json(path(state_dir), doc)
    history = Path(state_dir) / "policy-history"
    history.mkdir(parents=True, exist_ok=True)
    save_json(history / f"{now.strftime('%Y%m%dT%H%M%SZ')}.json", doc)
    return doc


def load(state_dir):
    p = path(state_dir)
    if not p.exists():
        return None
    try:
        doc = read_json(p)
    except (OSError, ValueError):
        return {"invalid": True}
    return doc


def previous_params(state_dir, s, now=None):
    """Last policy params usable as the step anchor (only while still valid)."""
    doc = load(state_dir)
    if not doc or doc.get("invalid") or doc.get("schema") != SCHEMA:
        return None
    try:
        created = datetime.fromisoformat(doc["created_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if (now or utcnow()) - created > timedelta(hours=float(s["valid_hours"])):
        return None
    return doc.get("params")


def effective(cfg, manifest, state_dir, now=None):
    """Strategy dict, threshold, uncertainty penalty and a status record for this cycle.

    Every value is re-clamped here, so a hand-edited or corrupted file still cannot exceed bounds.
    """
    now = now or utcnow()
    base, threshold = manifest["strategy"], float(manifest["threshold_bps"])
    s = settings(cfg)
    params, info = baseline_params(base, threshold), {"source": "baseline"}
    doc = load(state_dir) if s["enabled"] else None
    stale = False
    if s["enabled"]:
        if doc is None:
            info["reason"] = "no policy file"
            stale = True
        elif doc.get("invalid") or doc.get("schema") != SCHEMA or not isinstance(doc.get("params"), dict):
            info["reason"] = "invalid policy file"
            stale = True
        else:
            try:
                created = datetime.fromisoformat(doc["created_at"])
                age = (now - created).total_seconds() / 3600
            except (KeyError, TypeError, ValueError):
                age = float("inf")
            if not (-0.1 <= age <= float(s["valid_hours"])):
                info["reason"] = f"policy expired ({age:.1f} h)"
                stale = True
            else:
                # max_step=1 here: step limits apply when the policy is created, not on reload.
                reload = {**s, "max_step": 1.0}
                params, notes = sanitize(doc["params"], base, threshold, reload, manifest["markets"])
                info = {
                    "source": doc.get("source", "unknown"),
                    "created_at": doc["created_at"],
                    "regime": doc.get("regime"),
                    "params_sha256": doc.get("params_sha256"),
                }
                if notes:
                    info["reclamped"] = notes
    else:
        info["reason"] = "policy disabled"
    if stale and s["on_stale"] == "no_new_entries":
        params["allow_new_entries"] = False
    info["stale"] = stale
    strategy = copy.deepcopy(base)
    for key in ("position_fraction", "max_exposure", "stop_loss", "take_profit", "max_hold_bars",
                "daily_loss_limit", "exit_bps"):
        strategy[key] = params[key]
    return strategy, params, info
