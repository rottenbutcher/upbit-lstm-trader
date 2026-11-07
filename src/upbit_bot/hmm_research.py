from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_json, save_json, utcnow
from .data import CandleStore
from .features import WARMUP, feature_frame, model_input, split_indices
from .hmm_model import HMM_FEATURES, fit_forecaster, settings
from .research import candle_snapshot, evaluate_forecasts

LOG = logging.getLogger(__name__)


def load_dataset(cfg, directory, manifest=None):
    universe = manifest["universe"] if manifest else read_json(Path(directory) / "universe.json")
    markets = universe["markets"]
    # Same feature definitions as the LSTM being compared (older bundles use version 1).
    version = int(manifest.get("feature_version", 1)) if manifest else int(cfg["model"].get("feature_version", 1))
    frames, features, snapshots = [], [], {}
    store = CandleStore(directory)
    try:
        for market in markets:
            raw = store.frame(market)
            expected = manifest.get("data_snapshot", {}).get(market) if manifest else None
            if expected:
                raw = raw.loc[pd.Timestamp(expected["first"]) : pd.Timestamp(expected["last"])]
            elif manifest:
                raw = raw.loc[: pd.Timestamp(manifest["data_end"])]
            if raw.empty:
                raise ValueError(f"No saved candles for {market}; download data first")
            snapshots[market] = candle_snapshot(raw)
            if expected and snapshots[market] != expected:
                raise ValueError(
                    f"{market}: data differs from the LSTM training snapshot; restore original data"
                )
            frame, feat = feature_frame(raw, version)
            if (frame.index[-1] - frame.index[0]).days < cfg["data"]["minimum_days"]:
                raise ValueError(f"{market}: fewer than minimum_days")
            frames.append(frame)
            features.append(feat)
    finally:
        store.close()
    end = min(f.index[-1] for f in frames)
    if manifest:
        if end < pd.Timestamp(manifest["data_end"]):
            raise ValueError("Saved candles do not reach the LSTM data_end")
        train_end, val_end = pd.Timestamp(manifest["train_end"]), pd.Timestamp(manifest["validation_end"])
    else:
        if (max(f.index[-1] for f in frames) - end).total_seconds() > 3600:
            raise ValueError("One market is stale; update data before training")
        val_end = end - pd.Timedelta(days=cfg["training"]["test_days"])
        train_end = val_end - pd.Timedelta(days=cfg["training"]["validation_days"])
    splits, labels = [], []
    horizon = cfg["model"]["horizon"]
    for frame, feat in zip(frames, features):
        split = split_indices(
            frame, feat, cfg["model"]["lookback"], horizon, train_end, val_end, warmup=WARMUP[version]
        )
        if any(len(indexes) < 50 for indexes in split.values()):
            raise ValueError("Insufficient history in a time split (minimum 50 samples)")
        splits.append(split)
        labels.append(
            (np.log(frame["open"].shift(-horizon - 1) / frame["open"].shift(-1)) * 10000).to_numpy(np.float32)
        )
    return markets, frames, features, splits, labels, snapshots, train_end, val_end


def lstm_predictions(bundle, features, splits):
    m = bundle.manifest
    outputs = []
    for coin, feat in enumerate(features):
        LOG.info("Re-evaluating frozen LSTM: %s", m["markets"][coin])
        values = model_input(feat.to_numpy(), m["scaler"], coin, len(features))
        result = np.full(len(values), np.nan)
        indexes = np.concatenate([splits[coin][name] for name in ("val", "test")])
        for start in range(0, len(indexes), 128):
            batch = indexes[start : start + 128]
            x = np.stack([values[i - m["lookback"] + 1 : i + 1] for i in batch])
            result[batch] = bundle.session.run(None, {"features": x})[0].reshape(-1)
        outputs.append(result)
    return outputs


def train_hmm(cfg, directory="data", output="artifacts/hmm", compare_lstm=None):
    started = time.monotonic()
    out = Path(output)
    if (out / "manifest.json").exists() or (out / "model.onnx").exists():
        raise ValueError("Choose a separate HMM output directory; this directory contains an LSTM bundle")
    options = settings(cfg)
    bundle, manifest = None, None
    if compare_lstm:
        from .runtime import Bundle

        bundle = Bundle(compare_lstm)
        manifest = bundle.manifest
        # Frozen LSTM target, dates, costs and risk budget also govern HMM evaluation.
        cfg = copy.deepcopy(manifest["config"])
        if cfg["model"]["horizon"] != manifest["horizon"] or cfg["model"]["lookback"] != manifest["lookback"]:
            raise ValueError("Inconsistent LSTM manifest/config")
    cfg = copy.deepcopy(cfg)
    cfg["hmm"] = options
    markets, frames, feats, splits, labels, snapshots, train_end, val_end = load_dataset(
        cfg, directory, manifest
    )
    out.mkdir(parents=True, exist_ok=True)
    models, forecasts = {}, []
    for i, market in enumerate(markets):
        LOG.info("Fitting HMM for %s (%d/%d)", market, i + 1, len(markets))
        model, pred = fit_forecaster(
            feats[i][HMM_FEATURES].to_numpy(dtype=float),
            np.asarray(frames[i].index < train_end),
            splits[i]["train"],
            labels[i],
            options,
            cfg["training"]["seed"] + i * 1000,
        )
        models[market] = model
        forecasts.append(pred)
    results = {"HMM": evaluate_forecasts(cfg, frames, splits, labels, forecasts, markets, out, "hmm")}
    if bundle:
        pred = lstm_predictions(bundle, feats, splits)
        results["LSTM"] = evaluate_forecasts(cfg, frames, splits, labels, pred, markets, out, "lstm")
    notes = [
        "상승은 실제/예측 수익률 > 0, 비상승은 <= 0입니다. 잠재 상태 자체의 정답률은 아닙니다.",
        "HMM 상태 수와 초기값 선택은 테스트 성과로 조정하지 않습니다. 매수 기준은 검증 구간에서만 선택합니다.",
        "현재 상위 종목 선정에는 생존/선정 편향이 있고, 다음 봉 시가 체결과 고정 거래비용은 근사입니다.",
        "30분 예측을 5분마다 평가하므로 표본이 겹칩니다. 표본 수를 독립 시행 수로 해석할 수 없습니다.",
        "같은 테스트 결과를 보고 반복 튜닝하면 새 미사용 기간에서 다시 검증해야 합니다.",
        "HMM은 연구·비교용입니다. Oracle 배포 패키지는 LSTM 모델을 사용합니다.",
    ]
    if manifest and not manifest.get("data_snapshot"):
        notes.append(
            "이전 LSTM에는 데이터 해시가 없습니다. 저장된 날짜 안의 현재 데이터로 두 모델을 "
            "다시 평가했으며, LSTM 학습 당시 원본과의 동일성은 확인할 수 없습니다."
        )
    if not bundle:
        notes.append(
            "LSTM 학습 결과가 없어 HMM과 단순 기준만 표시합니다. TRAIN.cmd 완료 후 HMM.cmd를 다시 실행하세요."
        )
    record = {
        "schema": "hmm-research-v1",
        "created_at": utcnow().isoformat(),
        "markets": markets,
        "train_end": train_end.isoformat(),
        "validation_end": val_end.isoformat(),
        "data_end": min(f.index[-1] for f in frames).isoformat(),
        "data_snapshot": snapshots,
        "lstm_model_sha256": manifest["model_sha256"] if manifest else None,
        "snapshot_verified_against_lstm": bool(manifest and manifest.get("data_snapshot")),
        "horizon_bars": cfg["model"]["horizon"],
        "settings": options,
        "evaluation_strategy": cfg["strategy"],
        "evaluation_budget": cfg["runtime"],
        "results": results,
        "limitations": notes,
        "elapsed_seconds": time.monotonic() - started,
    }
    save_json(
        out / "hmm-model.json",
        {
            "schema": "hmm-forecast-v1",
            "markets": models,
            "horizon_bars": cfg["model"]["horizon"],
            "train_end": train_end.isoformat(),
            "data_snapshot": snapshots,
        },
    )
    save_json(out / "comparison.json", record)
    from .report import write_comparison

    write_comparison(record, out / "comparison.html")
    LOG.info(
        "HMM research complete in %.1f minutes: %s", record["elapsed_seconds"] / 60, out / "comparison.html"
    )
    return record
