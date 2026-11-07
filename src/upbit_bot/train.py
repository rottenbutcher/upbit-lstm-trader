from __future__ import annotations

import copy
import logging
import math
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
from .features import WARMUP, feature_frame, feature_list, model_input, split_indices
from .model import Ensemble, ReturnLSTM, export_onnx
from .research import candle_snapshot
from .resume import MISMATCH_FILE, ResumeMismatch, changes, identity, write_info
from .resume import fingerprint as identity_fingerprint

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
            y = model(x.to(device))
            if isinstance(y, tuple):
                y = y[0]
            out.append(y.float().cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def model_settings(cfg):
    """Model options with defaults that keep older config files valid."""
    mc = cfg["model"]
    return {
        "feature_version": int(mc.get("feature_version", 1)),
        "attention": bool(mc.get("attention", False)),
        "ensemble": int(mc.get("ensemble", 1)),
    }


def prepare(cfg, directory, output):
    version = model_settings(cfg)["feature_version"]
    columns = feature_list(version)
    universe = read_json(Path(directory) / "universe.json")
    universe["data_snapshot"] = {}
    store = CandleStore(directory)
    frames, features = [], []
    try:
        for market in universe["markets"]:
            raw = store.frame(market)
            if raw.empty:
                raise ValueError(f"No candles for {market}; run download first")
            universe["data_snapshot"][market] = candle_snapshot(raw)
            frame, feat = feature_frame(raw, version)
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
    count, sums, squares = 0, np.zeros(len(columns)), np.zeros(len(columns))
    splits = []
    for frame, feat in zip(frames, features):
        values = feat.loc[feat.index < train_end].dropna().to_numpy(dtype=np.float64)
        count += len(values)
        sums += values.sum(axis=0)
        squares += (values**2).sum(axis=0)
        split = split_indices(
            frame,
            feat,
            cfg["model"]["lookback"],
            cfg["model"]["horizon"],
            train_end,
            val_end,
            warmup=WARMUP[version],
        )
        if any(len(split[k]) < 50 for k in split):
            raise ValueError(
                "Insufficient history in a time split; longer history or explicit smaller split needed"
            )
        splits.append(split)
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-12))
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "features": columns}
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


def lr_factor(epoch, total, warmup):
    """Linear warmup then cosine decay to 5% of the base learning rate (per epoch)."""
    if warmup > 0 and epoch < warmup:
        return 0.3 + 0.7 * epoch / warmup
    span = max(total - warmup, 1)
    progress = min(max(epoch - warmup, 0) / span, 1.0)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))


def fit_member(member, cfg, ctx, device, out, resume, started):
    """Train one ensemble member with early stopping; checkpoint after every epoch."""
    t, mc, ms = cfg["training"], cfg["model"], model_settings(cfg)
    seed = t["seed"] + member * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ReturnLSTM(
        ctx["inputs"], **{k: mc[k] for k in ("hidden", "layers", "dropout")}, attention=ms["attention"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    schedule = t.get("lr_schedule", "constant")
    warmup = int(t.get("warmup_epochs", 0))

    def set_lr(epoch):
        factor = lr_factor(epoch, t["epochs"], warmup) if schedule == "cosine" else 1.0
        for group in optimizer.param_groups:
            group["lr"] = t["learning_rate"] * factor

    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_refs, val_ds, val_y = ctx["train_refs"], ctx["val_ds"], ctx["val_y"]
    arrays, labels = ctx["arrays"], ctx["labels"]
    fingerprint = f"{ctx['fingerprint']}:member{member}"
    checkpoint = checkpoint_path(out, member)
    best, bad, first_epoch, history, best_state = float("inf"), 0, 0, [], None
    if resume and checkpoint.exists():
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
        if ck.get("fingerprint") != fingerprint:  # check_resume() reports details first
            raise ResumeMismatch(changes(ck.get("identity"), ctx["identity"]) or [("legacy", None, None, None)])
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        amp.load_state_dict(ck["amp"])
        best, bad, first_epoch, history, best_state = (
            ck[k] for k in ("best", "bad", "epoch", "history", "best_state")
        )
        torch.set_rng_state(ck["rng"].cpu())
        if device.type == "cuda" and ck.get("cuda_rng"):
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])
        if ck.get("complete"):
            LOG.info("Ensemble member %d already complete; reusing checkpoint", member + 1)
            ctx["progress"](member, history)
            return best_state, history
        LOG.info("Ensemble member %d: resuming after epoch %d", member + 1, first_epoch)
    batch = ctx["batch"]
    for epoch in range(first_epoch, t["epochs"]):
        if bad >= t["patience"]:
            break
        set_lr(epoch)
        rng = np.random.default_rng(seed + epoch)
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
        ctx["batch"] = batch
        val_pred = predict(model, val_ds, device, batch)
        score = float(np.mean(np.abs(val_pred - val_y)))
        history.append(
            {
                "member": member + 1,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": float(np.mean(losses)),
                "validation_mae_bps": score,
                "elapsed_seconds": time.monotonic() - started,
                "batch_size": batch,
            }
        )
        LOG.info(
            "member %d/%d epoch %d/%d lr=%.2e train=%.5f validation MAE=%.3f bps elapsed=%.1f min",
            member + 1,
            ctx["members"],
            epoch + 1,
            t["epochs"],
            optimizer.param_groups[0]["lr"],
            np.mean(losses),
            score,
            (time.monotonic() - started) / 60,
        )
        if score < best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        save_checkpoint(
            checkpoint, fingerprint, ctx["identity"], model, optimizer, amp, best, bad, epoch + 1, history, best_state
        )
        ctx["progress"](member, history)
    if best_state is None:
        raise ValueError("No completed training epoch")
    save_checkpoint(
        checkpoint,
        fingerprint,
        ctx["identity"],
        model,
        optimizer,
        amp,
        best,
        bad,
        len(history),
        history,
        best_state,
        True,
    )
    return best_state, history


def checkpoint_path(out, member):
    return Path(out) / ("checkpoint.pt" if member == 0 else f"checkpoint-{member}.pt")


def check_resume(out, ident, members):
    """Fail before any training if an existing checkpoint belongs to other data/settings.

    Every reason is collected and also written to resume-mismatch.json for the GUI.
    """
    found = []
    for member in range(members):
        path = checkpoint_path(out, member)
        if path.exists():
            ck = torch.load(path, map_location="cpu", weights_only=True)
            for change in changes(ck.get("identity"), ident):
                if change not in found:
                    found.append(change)
    if found:
        save_json(out / MISMATCH_FILE, {"changes": [list(c) for c in found], "at": utcnow().isoformat()})
        raise ResumeMismatch(found)


def archive_checkpoints(out):
    """A fresh run keeps old checkpoints (never deletes) but moves them out of the resume path."""
    old = sorted(Path(out).glob("checkpoint*.pt"))
    if not old:
        return None
    target = Path(out) / "old-checkpoints" / utcnow().strftime("%Y%m%dT%H%M%SZ")
    target.mkdir(parents=True, exist_ok=True)
    for path in old:
        path.replace(target / path.name)
    LOG.info("Fresh run: previous checkpoints moved to %s", target)
    return target


def save_checkpoint(
    path, fingerprint, ident, model, optimizer, amp, best, bad, epoch, history, best_state, done=False
):
    temp = path.with_suffix(".tmp")
    torch.save(
        {
            "fingerprint": fingerprint,
            "identity": ident,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "amp": amp.state_dict(),
            "best": best,
            "bad": bad,
            "epoch": epoch,
            "history": history,
            "best_state": best_state,
            "complete": done,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temp,
    )
    temp.replace(path)


def train(cfg, directory="data", output="artifacts", resume=False, allow_cpu=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA GPU not found. Check NVIDIA driver/PyTorch. Use --allow-cpu only intentionally."
        )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    t, mc, ms = cfg["training"], cfg["model"], model_settings(cfg)
    if ms["ensemble"] < 1:
        raise ValueError("model.ensemble must be at least 1")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    (out / MISMATCH_FILE).unlink(missing_ok=True)
    universe, frames, splits, scaler, arrays, labels, train_end, val_end = prepare(cfg, directory, out)
    ident = identity(cfg, universe["markets"], universe["data_snapshot"])
    if resume:
        check_resume(out, ident, ms["ensemble"])
    else:
        archive_checkpoints(out)
    write_info(out, ident)
    inputs = len(scaler["features"]) + len(frames)
    train_refs = references(splits, "train", t["stride"])
    val_refs = references(splits, "val")
    val_small = val_refs[np.linspace(0, len(val_refs) - 1, min(8192, len(val_refs))).astype(int)]
    progress_path = out / "progress.json"
    progress_path.unlink(missing_ok=True)

    def progress(member, member_history):
        save_json(
            progress_path,
            {
                "member": member + 1,
                "members": ms["ensemble"],
                "epochs": t["epochs"],
                "history": history + member_history,
                "updated_at": utcnow().isoformat(),
            },
        )

    ctx = {
        "inputs": inputs,
        "train_refs": train_refs,
        "val_ds": Windows(arrays, labels, val_small, mc["lookback"]),
        "val_y": np.array([labels[c][i] for c, i in val_small]),
        "arrays": arrays,
        "labels": labels,
        "fingerprint": identity_fingerprint(ident),
        "identity": ident,
        "batch": t["batch_size"],
        "members": ms["ensemble"],
        "progress": progress,
    }
    states, history = [], []
    for member in range(ms["ensemble"]):
        state, member_history = fit_member(member, cfg, ctx, device, out, resume, started)
        states.append(state)
        history.extend(member_history)
        save_json(out / "training_history.json", history)
    members = []
    for state in states:
        net = ReturnLSTM(
            inputs, **{k: mc[k] for k in ("hidden", "layers", "dropout")}, attention=ms["attention"]
        )
        net.load_state_dict(state)
        members.append(net)
    model = Ensemble(members).to(device)
    batch = ctx["batch"]
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
            # Information coefficient: correlation of forecast and realized return.
            "ic": float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 and np.std(pred) > 1e-9 else 0.0,
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
        "schema": 2,
        "feature_version": ms["feature_version"],
        "warmup": WARMUP[ms["feature_version"]],
        "ensemble": ms["ensemble"],
        "attention": ms["attention"],
        "created_at": utcnow().isoformat(),
        "model_sha256": sha256(out / "model.onnx"),
        "report_sha256": sha256(out / "report.json"),
        "features": scaler["features"],
        "markets": universe["markets"],
        "universe": universe,
        "lookback": mc["lookback"],
        "data_snapshot": universe["data_snapshot"],
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
