"""Train every candidate architecture on the same data/splits and rank them.

Each candidate gets its own output directory (a full, deployable bundle) so the winner can be
packaged directly. Ranking uses the validation period only; the test period is reported but
never used for choosing, otherwise it stops being an honest holdout.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from .common import read_json, save_json, utcnow

LOG = logging.getLogger(__name__)
DEFAULT_CANDIDATES = ("lstm", "gru", "tcn", "transformer", "mixed")
MIXED = ["lstm", "tcn", "transformer"]


def candidate_config(cfg, name):
    c = copy.deepcopy(cfg)
    c["model"]["architecture"] = MIXED if name == "mixed" else name
    if name == "mixed":
        c["model"]["ensemble"] = max(int(c["model"].get("ensemble", 1)), len(MIXED))
    return c


def summary(name, out):
    report, manifest = read_json(out / "report.json"), read_json(out / "manifest.json")
    v, t, f = report["validation"], report["test"], report["forecast"]
    return {
        "candidate": name,
        "architectures": manifest.get("architectures"),
        "directory": str(out),
        "approved_for_live": report["approved_for_live"],
        "threshold_bps": report["threshold_bps"],
        "val_rank_ic": f["val"].get("rank_ic"),
        "val_sharpe": v["sharpe_daily"],
        "val_return_pct": v["return_pct"],
        "val_profit_factor": v["profit_factor"],
        "val_trades": v["trades"],
        "test_rank_ic": f["test"].get("rank_ic"),
        "test_sharpe": t["sharpe_daily"],
        "test_return_pct": t["return_pct"],
        "test_max_drawdown_pct": t["max_drawdown_pct"],
        "training_minutes": manifest["training_seconds"] / 60,
    }


def compare(cfg, data="data", output="artifacts/compare", candidates=DEFAULT_CANDIDATES, allow_cpu=False):
    from .train import train

    root = Path(output)
    rows, failures = [], {}
    for name in candidates:
        out = root / name
        LOG.info("=== candidate %s -> %s ===", name, out)
        try:
            train(candidate_config(cfg, name), data, out, resume=False, allow_cpu=allow_cpu)
            rows.append(summary(name, out))
        except Exception as exc:  # one failing candidate must not lose the others
            LOG.exception("candidate %s failed", name)
            failures[name] = f"{type(exc).__name__}: {exc}"
    # Validation-only ranking: risk-adjusted trading result first, forecast skill second.
    rows.sort(key=lambda r: (r["val_sharpe"], r["val_rank_ic"] or 0), reverse=True)
    result = {
        "created_at": utcnow().isoformat(),
        "ranking_basis": "validation sharpe, then validation rank IC (test not used for selection)",
        "ranking": rows,
        "failures": failures,
        "recommended": next((r["candidate"] for r in rows if r["approved_for_live"]), None),
    }
    save_json(root / "comparison.json", result)
    for r in rows:
        LOG.info(
            "%-12s val sharpe=%6.2f IC=%.4f | test sharpe=%6.2f ret=%6.2f%% | approved=%s",
            r["candidate"], r["val_sharpe"], r["val_rank_ic"] or 0, r["test_sharpe"],
            r["test_return_pct"], r["approved_for_live"],
        )
    return result
