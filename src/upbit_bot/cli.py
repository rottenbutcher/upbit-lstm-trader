from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

from filelock import FileLock

from .common import load_config


def adoption(cfg, args):
    from .api import Upbit
    from .common import utcnow
    from .runtime import Bundle
    from .state import State

    if not args.confirm:
        raise ValueError("Use --confirm to transfer this exact existing quantity into bot management")
    bundle = Bundle(args.artifacts)
    if args.market not in bundle.manifest["markets"]:
        raise ValueError("Market not in model universe")
    if not math.isfinite(args.quantity) or args.quantity <= 0:
        raise ValueError("Quantity must be positive")
    api = Upbit(os.environ.get("UPBIT_ACCESS_KEY", ""), os.environ.get("UPBIT_SECRET_KEY", ""))
    directory = Path(args.state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "bot.lock"), timeout=0):
        state = State(directory / "live.sqlite", cfg["runtime"]["capital_krw"], "live")
        try:
            if state.pending() or args.market in state.positions():
                raise ValueError("Pending orders or existing position: cannot adopt")
            balances = {a["currency"]: float(a["balance"]) for a in api.accounts()}
            if balances.get(args.market.split("-")[1], 0) < args.quantity:
                raise ValueError("Not enough available exchange inventory")
            price = float(api.tickers([args.market])[0]["trade_price"])
            value = price * args.quantity
            if value > cfg["runtime"]["capital_krw"] * bundle.manifest["strategy"]["position_fraction"]:
                raise ValueError("Adoption exceeds per-market allocation")
            cash = state.get("cash") - value
            if cash < 0 or balances.get("KRW", 0) < cash:
                raise ValueError("Remaining cash budget is not backed by available KRW")
            with state.db:
                state.db.execute(
                    "INSERT INTO positions VALUES (?,?,?,?,?)",
                    (args.market, args.quantity, price, value, int(utcnow().timestamp() // 300) - 1),
                )
                state.set("cash", cash)
        finally:
            state.close()


def main():
    parser = argparse.ArgumentParser(description="Upbit local LSTM research + Oracle inference")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("download")
    p.add_argument("--data", default="data")
    p = sub.add_parser("train")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="artifacts")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p = sub.add_parser("train-hmm", help="CPU HMM research and optional frozen LSTM comparison")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="artifacts/hmm")
    p.add_argument("--compare-lstm", metavar="DIRECTORY")
    p = sub.add_parser("package")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--destination", default="dist/oracle-upload.zip")
    p = sub.add_parser("run")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--state-dir", default="state")
    p.add_argument("--mode", choices=["paper", "live"], default=os.environ.get("MODE", "paper"))
    p.add_argument("--once", action="store_true")
    for name in ("status", "reset-errors", "resume-risk"):
        p = sub.add_parser(name)
        p.add_argument("--state-dir", default="state")
        if name != "status":
            p.add_argument("--mode", choices=["paper", "live"], default="paper")
            p.add_argument("--confirm", action="store_true")
    p = sub.add_parser("adopt")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--state-dir", default="state")
    p.add_argument("--market", required=True)
    p.add_argument("--quantity", type=float, required=True)
    p.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.command == "doctor":
        import importlib.util
        import platform

        result = {"python": platform.python_version(), "architecture": platform.machine()}
        if importlib.util.find_spec("torch"):
            import torch

            result.update(torch=torch.__version__, cuda=torch.cuda.is_available())
            if torch.cuda.is_available():
                result.update(
                    gpu=torch.cuda.get_device_name(),
                    vram_gb=torch.cuda.get_device_properties(0).total_memory / 2**30,
                )
        print(json.dumps(result, indent=2))
    elif args.command == "download":
        from .api import Upbit
        from .data import download

        download(Upbit(interval=cfg["data"]["request_interval"]), cfg, args.data)
    elif args.command == "train":
        from .train import train

        train(cfg, args.data, args.output, args.resume, args.allow_cpu)
    elif args.command == "package":
        from .package import package

        print(package(artifacts=args.artifacts, destination=args.destination))
    elif args.command == "train-hmm":
        from .hmm_research import train_hmm

        train_hmm(cfg, args.data, args.output, args.compare_lstm)
    elif args.command == "run":
        from .runtime import run

        run(cfg, args.artifacts, args.state_dir, args.mode, args.once)
    elif args.command == "status":
        from .runtime import status

        print(json.dumps(status(args.state_dir), indent=2, ensure_ascii=False))
    elif args.command in ("reset-errors", "resume-risk"):
        from .state import State

        if not args.confirm:
            raise ValueError("Inspect logs and pass --confirm after correcting the cause")
        directory = Path(args.state_dir)
        if not (directory / f"{args.mode}.sqlite").exists():
            raise ValueError("State does not exist")
        with FileLock(str(directory / "bot.lock"), timeout=0):
            state = State(directory / f"{args.mode}.sqlite", cfg["runtime"]["capital_krw"], args.mode)
            try:
                with state.db:
                    if args.command == "reset-errors":
                        state.set("errors", 0)
                    else:
                        if state.pending() or state.positions():
                            raise ValueError(
                                "Resolve orders and close positions before resetting risk baseline"
                            )
                        state.set("risk_halt", False)
                        state.set("daily_halt", False)
                        state.set("peak", state.get("cash"))
                        state.set("day_start", state.get("cash"))
            finally:
                state.close()
    elif args.command == "adopt":
        adoption(cfg, args)


if __name__ == "__main__":
    main()
