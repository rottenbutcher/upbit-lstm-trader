from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from .api import healthy_market
from .common import read_json, save_json, utcnow

LOG = logging.getLogger(__name__)
COLS = ["open", "high", "low", "close", "volume", "value"]


def candle_frame(rows):
    if not rows:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(rows).rename(
        columns={
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
            "candle_acc_trade_price": "value",
        }
    )
    df.index = pd.to_datetime(df["candle_date_time_utc"], utc=True)
    df.index.name = "time"
    return df[COLS].astype(float).sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]


def select_universe(api, cfg, directory):
    path = Path(directory) / "universe.json"
    if path.exists():
        return read_json(path)
    markets = [r["market"] for r in api.markets() if r["market"].startswith("KRW-") and healthy_market(r)]
    ranked = sorted(api.tickers(markets), key=lambda r: r["acc_trade_price_24h"], reverse=True)
    selected = ranked[: cfg["data"]["top_n"]]
    if len(selected) != cfg["data"]["top_n"]:
        raise ValueError("Not enough healthy KRW markets")
    result = {
        "selected_at": utcnow().isoformat(),
        "basis": "24h KRW traded value; current-universe bias",
        "markets": [r["market"] for r in selected],
        "ranking": [{"market": r["market"], "value_24h": r["acc_trade_price_24h"]} for r in selected],
    }
    save_json(path, result)
    return result


class CandleStore:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(Path(directory) / "candles.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS candles (market TEXT, time TEXT, open REAL, high REAL, "
            "low REAL, close REAL, volume REAL, value REAL, PRIMARY KEY(market,time))"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS cursors (market TEXT PRIMARY KEY, before TEXT, done INT)")

    def put(self, market, frame, before=None, done=False):
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)",
                [(market, t.isoformat(), *map(float, row)) for t, row in zip(frame.index, frame.to_numpy())],
            )
            if before is not None:
                self.db.execute("INSERT OR REPLACE INTO cursors VALUES (?,?,?)", (market, before, int(done)))

    def frame(self, market):
        df = pd.read_sql_query(
            "SELECT time,open,high,low,close,volume,value FROM candles WHERE market=? ORDER BY time",
            self.db,
            params=(market,),
        )
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time")

    def close(self):
        self.db.close()


def iso(ts):
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def download(api, cfg, directory="data"):
    universe = select_universe(api, cfg, directory)
    store = CandleStore(directory)
    start = pd.Timestamp(cfg["data"]["start"], tz="UTC")
    end = pd.Timestamp.now(tz="UTC").floor("5min")
    coverage = {}
    try:
        for market in universe["markets"]:
            cursor = store.db.execute("SELECT before,done FROM cursors WHERE market=?", (market,)).fetchone()
            before = pd.Timestamp(cursor[0]) if cursor else end
            done = bool(cursor and cursor[1])
            pages = 0
            while not done:
                frame = candle_frame(api.candles(market, iso(before)))
                if frame.empty:
                    store.put(market, frame, iso(before), True)
                    break
                oldest = frame.index.min()
                if oldest >= before:
                    raise RuntimeError("Pagination made no progress")
                done = oldest <= start
                store.put(market, frame.loc[(frame.index >= start) & (frame.index < end)], iso(oldest), done)
                before = oldest
                pages += 1
                if pages % 100 == 0:
                    LOG.info("%s: %s, %s pages saved; safe to resume", market, before, pages)
            # Bring the latest end up to date, without changing the historical resume cursor.
            latest = store.db.execute("SELECT MAX(time) FROM candles WHERE market=?", (market,)).fetchone()[0]
            if latest:
                before = end
                while before > pd.Timestamp(latest):
                    frame = candle_frame(api.candles(market, iso(before)))
                    if frame.empty:
                        break
                    store.put(market, frame.loc[(frame.index > pd.Timestamp(latest)) & (frame.index < end)])
                    if frame.index.min() >= before:
                        raise RuntimeError("Update pagination made no progress")
                    before = frame.index.min()
            first, last, count = store.db.execute(
                "SELECT MIN(time),MAX(time),COUNT(*) FROM candles WHERE market=?", (market,)
            ).fetchone()
            coverage[market] = {"first": first, "last": last, "observed_candles": count}
            LOG.info("%s coverage: %s", market, coverage[market])
    finally:
        store.close()
    save_json(Path(directory) / "coverage.json", coverage)
    return coverage


def regularize(frame):
    """Missing no-trade candles use last close only; never backfill pre-listing data."""
    if frame.empty:
        return frame
    df = frame.sort_index().asfreq("5min")
    observed = df["close"].notna()
    df["close"] = df["close"].ffill()
    for col in ["open", "high", "low"]:
        df[col] = df[col].fillna(df["close"])
    df[["volume", "value"]] = df[["volume", "value"]].fillna(0)
    df["observed"] = observed.astype(float)
    if (df["close"] <= 0).any() or (df["low"] <= 0).any():
        raise ValueError("Nonpositive candle price")
    return df
