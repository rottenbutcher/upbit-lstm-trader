from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .backtest import buy_hold, passed, simulate
from .common import read_json, save_json, sha256, utcnow
from .data import CandleStore
from .features import FEATURES, feature_frame, model_input, split_indices
from .model import ReturnLSTM, export_onnx

LOG = logging.getLogger(__name__)


class Windows(Dataset):
    """Only the current batch materializes windows; arrays are memory mapped."""

    def __init__(self, arrays, labels, references, lookback):
        self.arrays, self.labels, self.refs, self.lookback = arrays, labels, references, lookback

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        coin, end = self.refs[i]
        x = self.arrays[coin][end - self.lookback + 1 : end + 1]
        return np.array(x, dtype=np.float32), np.float32(self.labels[coin][end])


def predict(model, dataset, device, batch_size):
    out = []
    model.eval()
    with torch.inference_mode():
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def prepare(cfg, directory, output):
    universe = read_json(Path(directory) / "universe.json")
    store = CandleStore(directory)
    frames, features = [], []
    try:
        for market in universe["markets"]:
            raw = store.frame(market)
            if raw.empty:
                raise ValueError(f"No candles for {market}; run download first")
            frame, feat = feature_frame(raw)
            if (frame.index[-1] - frame.index[0]).days < cfg["data"]["minimum_days"]:
                raise ValueError(
                    f"{market} has fewer than minimum_days; review universe, do not fabricate history"
                )
            frames.append(frame)
            features.append(feat)
    finally:
        store.close()
    end = min(f.index[-1] for f in frames)
    if any((max(f.index[-1] for f in frames) - f.index[-1]).total_seconds() > 3600 for f in frames):
        raise ValueError("One market is stale; update data before training")
    val_end = end - pd.Timedelta(days=cfg["training"]["test_days"])
    train_end = val_end - pd.Timedelta(days=cfg["training"]["validation_days"])
    # Streaming sufficient statistics, trained strictly before train_end.
    count, sums, squares = 0, np.zeros(len(FEATURES)), np.zeros(len(FEATURES))
    splits = []
    for frame, feat in zip(frames, features):
        values = feat.loc[feat.index < train_end].dropna().to_numpy(dtype=np.float64)
        count += len(values)
        sums += values.sum(axis=0)
        squares += (values**2).sum(axis=0)
        split = split_indices(
            frame, feat, cfg["model"]["lookback"], cfg["model"]["horizon"], train_end, val_end
        )
        if any(len(split[k]) < 50 for k in split):
            raise ValueError(
                "Insufficient history in a time split; longer history or explicit smaller split needed"
            )
        splits.append(split)
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-12))
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "features": FEATURES}
    cache = Path(output) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    arrays, labels = [], []
    horizon = cfg["model"]["horizon"]
    for i, (frame, feat) in enumerate(zip(frames, features)):
        x = model_input(feat.to_numpy(), scaler, i, len(frames))
        path = cache / f"coin-{i}.npy"
        np.save(path, x)
        arrays.append(np.load(path, mmap_mode="r"))
        # Decision at close[t], approximate entry at open[t+1], exit at open[t+horizon+1].
        labels.append(
            (np.log(frame["open"].shift(-horizon - 1) / frame["open"].shift(-1)) * 10000).to_numpy(np.float32)
        )
    return universe, frames, splits, scaler, arrays, labels, train_end, val_end


def references(splits, name, stride=1):
    return np.concatenate(
        [
            np.column_stack([np.full(len(s[name][::stride]), i), s[name][::stride]])
            for i, s in enumerate(splits)
        ]
    ).astype(np.int64)


def train(cfg, directory="data", output="artifacts", resume=False, allow_cpu=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA GPU not found. Check NVIDIA driver/PyTorch. Use --allow-cpu only intentionally."
        )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    t, mc = cfg["training"], cfg["model"]
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    universe, frames, splits, scaler, arrays, labels, train_end, val_end = prepare(cfg, directory, out)
    inputs = len(FEATURES) + len(frames)
    model = ReturnLSTM(inputs, **{k: mc[k] for k in ("hidden", "layers", "dropout")}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_refs = references(splits, "train", t["stride"])
    val_refs = references(splits, "val")
    val_small = val_refs[np.linspace(0, len(val_refs) - 1, min(8192, len(val_refs))).astype(int)]
    val_ds = Windows(arrays, labels, val_small, mc["lookback"])
    val_y = np.array([labels[c][i] for c, i in val_small])
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "cfg": cfg,
                "universe": universe,
                "scaler": scaler,
                "last": [f.index[-1].isoformat() for f in frames],
                "counts": [len(f) for f in frames],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    checkpoint = out / "checkpoint.pt"
    best, bad, first_epoch, history, best_state = float("inf"), 0, 0, [], None
    if resume and checkpoint.exists():
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
        if ck["fingerprint"] != fingerprint:
            raise ValueError("Data/config changed since checkpoint. Start a fresh run without --resume.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        amp.load_state_dict(ck["amp"])
        best, bad, first_epoch, history, best_state = (
            ck[k] for k in ("best", "bad", "epoch", "history", "best_state")
        )
        torch.set_rng_state(ck["rng"].cpu())
        if device.type == "cuda" and ck.get("cuda_rng"):
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])
    batch = t["batch_size"]
    for epoch in range(first_epoch, t["epochs"]):
        if bad >= t["patience"]:
            break
        rng = np.random.default_rng(t["seed"] + epoch)
        cap = t["samples_per_epoch"] or len(train_refs)
        selected = train_refs[rng.choice(len(train_refs), min(cap, len(train_refs)), replace=False)]
        ds = Windows(arrays, labels, selected, mc["lookback"])
        # OOM: restart this epoch from its starting weights/optimizer with half batch size.
        start_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        opt_state = copy.deepcopy(optimizer.state_dict())
        amp_state = copy.deepcopy(amp.state_dict())
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        while True:
            model.train()
            losses = []
            try:
                for x, y in DataLoader(
                    ds,
                    batch_size=batch,
                    shuffle=False,
                    num_workers=t["workers"],
                    pin_memory=device.type == "cuda",
                ):
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                    ):
                        pred = model(x)
                        loss = torch.nn.functional.huber_loss(
                            pred.float() / 100, y.clamp(-1000, 1000) / 100, delta=1
                        )
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    amp.scale(loss).backward()
                    amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    amp.step(optimizer)
                    amp.update()
                    losses.append(float(loss.detach()))
                break
            except torch.cuda.OutOfMemoryError:
                if batch <= 16:
                    raise
                batch //= 2
                optimizer.zero_grad(set_to_none=True)
                x = y = pred = loss = None
                torch.cuda.empty_cache()
                model.load_state_dict(start_state)
                optimizer.load_state_dict(opt_state)
                amp = torch.amp.GradScaler("cuda", enabled=True)
                amp.load_state_dict(amp_state)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                LOG.warning("GPU memory limit: restart epoch with batch=%d", batch)
        val_pred = predict(model, val_ds, device, batch)
        score = float(np.mean(np.abs(val_pred - val_y)))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_mae_bps": score,
                "elapsed_seconds": time.monotonic() - started,
                "batch_size": batch,
            }
        )
        LOG.info(
            "epoch %d/%d train=%.5f validation MAE=%.3f bps elapsed=%.1f min",
            epoch + 1,
            t["epochs"],
            np.mean(losses),
            score,
            (time.monotonic() - started) / 60,
        )
        if score < best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        temp = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "fingerprint": fingerprint,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "amp": amp.state_dict(),
                "best": best,
                "bad": bad,
                "epoch": epoch + 1,
                "history": history,
                "best_state": best_state,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
            temp,
        )
        temp.replace(checkpoint)
        save_json(out / "training_history.json", history)
    if best_state is None:
        raise ValueError("No completed training epoch")
    model.load_state_dict(best_state)
    predictions, sim_frames, forecast = {}, {}, {}
    for name in ("val", "test"):
        refs = references(splits, name)
        ds = Windows(arrays, labels, refs, mc["lookback"])
        pred = predict(model, ds, device, batch)
        y = np.array([labels[c][i] for c, i in refs])
        forecast[name] = {
            "mae_bps": float(np.mean(np.abs(pred - y))),
            "zero_baseline_mae_bps": float(np.mean(np.abs(y))),
            "direction_accuracy": float(np.mean((pred > 0) == (y > 0))),
        }
        predictions[name], sim_frames[name] = {}, {}
        for i, market in enumerate(universe["markets"]):
            indexes = refs[refs[:, 0] == i, 1]
            predictions[name][market] = pd.Series(pred[refs[:, 0] == i], index=frames[i].index[indexes])
            sim_frames[name][market] = frames[i].loc[
                frames[i].index[indexes[0]] : frames[i].index[min(indexes[-1] + 1, len(frames[i]) - 1)]
            ]
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
    # Three sequential validation subperiods; this is NOT walk-forward retraining.
    block_metrics = []
    boundaries = pd.date_range(train_end, val_end, periods=4)
    for a, b in pairwise(boundaries):
        block_pred = {m: p.loc[(p.index >= a) & (p.index < b)] for m, p in predictions["val"].items()}
        block_metrics.append(
            simulate(
                sim_frames["val"],
                block_pred,
                cfg["strategy"],
                threshold,
                cfg["runtime"]["capital_krw"],
                max_order=cfg["runtime"]["max_order_krw"],
            )["metrics"]
        )
    approved = bool(
        candidates and passed(test, cfg["gate"]) and sum(m["return_pct"] > 0 for m in block_metrics) >= 2
    )
    test_baseline = buy_hold(
        sim_frames["test"], predictions["test"], cfg["runtime"]["capital_krw"], cfg["strategy"]
    )
    save_json(
        out / "report.json",
        {
            "validation": validation["metrics"],
            "test": test["metrics"],
            "validation_blocks": block_metrics,
            "forecast": forecast,
            "buy_hold_test": test_baseline,
            "cash_return_pct": 0,
            "threshold_search": [{"bps": th, **r["metrics"]} for th, r in results],
            "approved_for_live": approved,
            "threshold_bps": threshold,
            "limitations": [
                "Current top-5 selection has survivorship/selection bias",
                "No historical order books",
                "Next-open approximate fills, fixed costs; no live fill guarantee",
                "Validation blocks share one trained model",
                "Test gate is prespecified; repeated experimentation on this test invalidates holdout claims",
            ],
        },
    )
    for name, result in (("validation", validation), ("test", test)):
        pd.DataFrame(result["equity"], columns=["time", "equity_krw"]).to_csv(
            out / f"{name}_equity.csv", index=False
        )
        pd.DataFrame(result["trades"]).to_csv(out / f"{name}_trades.csv", index=False)
    parity_error = export_onnx(model, inputs, mc["lookback"], out / "model.onnx")
    manifest = {
        "schema": 1,
        "created_at": utcnow().isoformat(),
        "model_sha256": sha256(out / "model.onnx"),
        "report_sha256": sha256(out / "report.json"),
        "features": FEATURES,
        "markets": universe["markets"],
        "universe": universe,
        "lookback": mc["lookback"],
        "horizon": mc["horizon"],
        "scaler": scaler,
        "train_end": train_end.isoformat(),
        "validation_end": val_end.isoformat(),
        "data_end": min(f.index[-1] for f in frames).isoformat(),
        "threshold_bps": threshold,
        "approved_for_live": approved,
        "strategy": cfg["strategy"],
        "gate": cfg["gate"],
        "onnx_max_error_bps": parity_error,
        "training_seconds": time.monotonic() - started,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "config": cfg,
    }
    save_json(out / "manifest.json", manifest)
    from .report import write_report

    write_report(read_json(out / "report.json"), out / "report.html")
    LOG.info("Complete. Live gate=%s. Review %s", approved, out / "report.json")
    return manifest
