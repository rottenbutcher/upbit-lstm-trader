from __future__ import annotations

import numpy as np
import pandas as pd

from .data import regularize

FEATURES_V1 = [
    "ret1",
    "ret3",
    "ret12",
    "ret48",
    "range",
    "body",
    "vol12",
    "vol48",
    "ma12",
    "ma48",
    "rsi14",
    "volume_z",
    "value_z",
    "observed",
    "hour_sin",
    "hour_cos",
]

# v2 adds daily-scale context (288 five-minute bars = 24 hours), intrabar volatility,
# range position and weekday seasonality. Every column remains a finite causal rolling window.
FEATURES_V2 = [
    *FEATURES_V1,
    "ret6",
    "ret24",
    "ret288",
    "vol288",
    "ma288",
    "park12",
    "stoch48",
    "dd48",
    "value_z288",
    "dow_sin",
    "dow_cos",
]

FEATURE_SETS = {1: FEATURES_V1, 2: FEATURES_V2}
# Bars before the first model input whose features need history (longest rolling window).
WARMUP = {1: 48, 2: 288}
LATEST_VERSION = 2
FEATURES = FEATURE_SETS[LATEST_VERSION]


def feature_list(version):
    if version not in FEATURE_SETS:
        raise ValueError(f"Unsupported feature version {version}")
    return FEATURE_SETS[version]


def feature_frame(raw, version=LATEST_VERSION):
    columns = feature_list(version)
    df = regularize(raw)
    c = df["close"]
    logc = np.log(c)
    out = pd.DataFrame(index=df.index)
    for n in (1, 3, 12, 48):
        out[f"ret{n}"] = logc.diff(n)
    out["range"] = np.log(df["high"] / df["low"])
    out["body"] = np.log(c / df["open"])
    for n in (12, 48):
        out[f"vol{n}"] = out["ret1"].rolling(n).std()
        out[f"ma{n}"] = c / c.rolling(n).mean() - 1
    change = c.diff()
    up = change.clip(lower=0).rolling(14).mean()
    down = (-change.clip(upper=0)).rolling(14).mean()
    out["rsi14"] = (up / (up + down).replace(0, np.nan)).fillna(0.5)
    for col in ("volume", "value"):
        x = np.log1p(df[col])
        out[f"{col}_z"] = (x - x.rolling(48).mean()) / x.rolling(48).std().clip(lower=1e-6)
    out["observed"] = df["observed"]
    minutes = df.index.hour * 60 + df.index.minute
    out["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    out["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    if version >= 2:
        for n in (6, 24, 288):
            out[f"ret{n}"] = logc.diff(n)
        out["vol288"] = out["ret1"].rolling(288).std()
        out["ma288"] = c / c.rolling(288).mean() - 1
        # Parkinson high-low volatility estimator over one hour.
        out["park12"] = np.sqrt((out["range"] ** 2).rolling(12).mean() / (4 * np.log(2)))
        low48, high48 = df["low"].rolling(48).min(), df["high"].rolling(48).max()
        out["stoch48"] = ((c - low48) / (high48 - low48).replace(0, np.nan)).fillna(0.5)
        out["dd48"] = np.log(c / high48)
        x = np.log1p(df["value"])
        out["value_z288"] = (x - x.rolling(288).mean()) / x.rolling(288).std().clip(lower=1e-6)
        day = df.index.dayofweek + minutes / 1440
        out["dow_sin"] = np.sin(2 * np.pi * day / 7)
        out["dow_cos"] = np.cos(2 * np.pi * day / 7)
    return df, out[columns].replace([np.inf, -np.inf], np.nan).astype("float32")


def normalize(values, scaler):
    return np.clip(
        (values - np.asarray(scaler["mean"], np.float32)) / np.asarray(scaler["std"], np.float32), -10, 10
    ).astype(np.float32)


def model_input(features, scaler, coin_index, coin_count):
    x = normalize(features, scaler)
    coins = np.zeros((len(x), coin_count), dtype=np.float32)
    coins[:, coin_index] = 1
    return np.concatenate([x, coins], axis=1)


def split_indices(frame, features, lookback, horizon, train_end, val_end, stride=1, warmup=48):
    n = len(frame)
    idx = np.arange(lookback + warmup - 1, n - horizon - 1, stride, dtype=np.int64)
    if not len(idx):
        return {k: idx.copy() for k in ("train", "val", "test")}
    # Features and next-open entry/exit must exist. Train labels end strictly before the next split.
    finite = np.isfinite(features.to_numpy()).all(axis=1)
    valid = pd.Series(finite).rolling(lookback).min().fillna(0).to_numpy().astype(bool)
    good = valid[idx] & (frame["observed"].to_numpy()[idx] > 0)
    idx = idx[good]
    times, label_times = frame.index[idx], frame.index[idx + horizon + 1]
    return {
        "train": idx[label_times < train_end],
        "val": idx[(times >= train_end) & (label_times < val_end)],
        "test": idx[times >= val_end],
    }
