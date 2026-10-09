"""Evaluation on shared candles and timestamps, independent of PyTorch."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .backtest import passed, simulate


def candle_snapshot(frame):
    digest = hashlib.sha256()
    digest.update(frame.index.asi8.astype("<i8").tobytes())
    digest.update(frame[["open", "high", "low", "close", "volume", "value"]].to_numpy(dtype="<f8").tobytes())
    return {
        "first": frame.index[0].isoformat(),
        "last": frame.index[-1].isoformat(),
        "rows": len(frame),
        "sha256": digest.hexdigest(),
    }


def forecast_metrics(y, pred, majority_up):
    actual, predicted = y > 0, pred > 0
    recalls = [
        float(np.mean(predicted[actual == label] == label))
        for label in (False, True)
        if np.any(actual == label)
    ]
    return {
        "samples": len(y),
        "direction_accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
        "mae_bps": float(np.mean(np.abs(y - pred))),
        "zero_baseline_mae_bps": float(np.mean(np.abs(y))),
        "always_up_accuracy": float(np.mean(actual)),
        "train_majority_accuracy": float(np.mean(actual == majority_up)),
        "flat_target_fraction": float(np.mean(y == 0)),
    }


def evaluate_forecasts(cfg, frames, splits, labels, forecasts, markets, out, model_name):
    metrics, predictions, sim_frames = {}, {}, {}
    for name in ("val", "test"):
        per_coin, rows = {}, []
        predictions[name], sim_frames[name] = {}, {}
        all_y, all_pred, all_majority = [], [], []
        for i, market in enumerate(markets):
            idx, frame = splits[i][name], frames[i]
            y, pred = labels[i][idx], forecasts[i][idx]
            if not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite {model_name} forecast for {market}")
            majority = np.full(len(idx), np.mean(labels[i][splits[i]["train"]] > 0) > 0.5)
            per_coin[market] = forecast_metrics(y, pred, majority)
            all_y.append(y)
            all_pred.append(pred)
            all_majority.append(majority)
            predictions[name][market] = pd.Series(pred, index=frame.index[idx])
            sim_frames[name][market] = frame.iloc[idx[0] : idx[-1] + 2]
            rows.append(
                pd.DataFrame(
                    {"market": market, "time": frame.index[idx], "actual_bps": y, "prediction_bps": pred}
                )
            )
        metrics[name] = forecast_metrics(
            np.concatenate(all_y), np.concatenate(all_pred), np.concatenate(all_majority)
        )
        metrics[name]["per_market"] = per_coin
        pd.concat(rows, ignore_index=True).to_csv(out / f"{model_name}_{name}_predictions.csv", index=False)
    results = []
    for threshold in cfg["strategy"]["entry_thresholds_bps"]:
        result = simulate(
            sim_frames["val"],
            predictions["val"],
            cfg["strategy"],
            threshold,
            cfg["runtime"]["capital_krw"],
            max_order=cfg["runtime"]["max_order_krw"],
        )
        results.append((threshold, result))
    candidates = [(th, r) for th, r in results if passed(r, cfg["gate"])]
    threshold, validation = max(candidates or results, key=lambda item: item[1]["metrics"]["return_pct"])
    test = simulate(
        sim_frames["test"],
        predictions["test"],
        cfg["strategy"],
        threshold,
        cfg["runtime"]["capital_krw"],
        max_order=cfg["runtime"]["max_order_krw"],
    )
    for name, result in (("val", validation), ("test", test)):
        pd.DataFrame(result["equity"], columns=["time", "equity_krw"]).to_csv(
            out / f"{model_name}_{name}_equity.csv", index=False
        )
        pd.DataFrame(result["trades"]).to_csv(out / f"{model_name}_{name}_trades.csv", index=False)
    return {
        "forecast": metrics,
        "validation": validation["metrics"],
        "test": test["metrics"],
        "threshold_bps": threshold,
        "threshold_search": [{"bps": th, **r["metrics"]} for th, r in results],
    }
