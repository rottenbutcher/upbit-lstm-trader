from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path


class State:
    """Durable intent log and bot-owned portfolio, one SQLite transaction per fill."""

    def __init__(self, path, capital, mode):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS positions (market TEXT PRIMARY KEY, qty REAL, entry REAL, cost REAL, bar INT);
          CREATE TABLE IF NOT EXISTS intents (id TEXT PRIMARY KEY, market TEXT, side TEXT, bar INT,
            payload TEXT, status TEXT, response TEXT);
          CREATE TABLE IF NOT EXISTS cycles (bar INT PRIMARY KEY);
        """)
        if self.get("mode") is None:
            with self.db:
                self.set("mode", mode)
                self.set("capital", capital)
                self.set("cash", capital)
                self.set("peak", capital)
        if self.get("mode") != mode or self.get("capital") != capital:
            self.db.close()
            raise ValueError(
                "State mode/capital mismatch. Never delete live state; use its original settings."
            )

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))

    def positions(self):
        return {r["market"]: dict(r) for r in self.db.execute("SELECT * FROM positions")}

    def claim_cycle(self, bar):
        with self.db:
            try:
                self.db.execute("INSERT INTO cycles VALUES (?)", (bar,))
                return True
            except sqlite3.IntegrityError:
                return False

    def intent(self, identifier, market, side, bar, payload):
        with self.db:
            self.db.execute(
                "INSERT INTO intents VALUES (?,?,?,?,?,?,?)",
                (identifier, market, side, bar, json.dumps(payload), "pending", "{}"),
            )

    def pending(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM intents WHERE status='pending' ORDER BY bar")]

    def apply_fill(self, identifier, qty, funds, fee, response):
        if not all(math.isfinite(x) for x in (qty, funds, fee)) or min(qty, funds, fee) < 0:
            raise ValueError("Negative fill values")
        with self.db:
            row = self.db.execute("SELECT * FROM intents WHERE id=?", (identifier,)).fetchone()
            if not row or row["status"] != "pending":
                return
            cash, positions = self.get("cash"), self.positions()
            market = row["market"]
            if qty > 0:
                if funds <= 0:
                    raise ValueError("Nonzero volume without funds")
                p = positions.get(market, {"qty": 0, "cost": 0, "bar": row["bar"]})
                if row["side"] == "buy":
                    cash -= funds + fee
                    total_qty, cost = p["qty"] + qty, p["cost"] + funds + fee
                    if cash < -1e-4:
                        raise ValueError("Fill exceeds bot cash budget")
                    entry = (p.get("entry", 0) * p["qty"] + funds) / total_qty
                    self.db.execute(
                        "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?)",
                        (market, total_qty, entry, cost, p["bar"]),
                    )
                else:
                    if qty > p["qty"] + max(1e-10, p["qty"] * 1e-8):
                        raise ValueError("Fill exceeds bot-owned inventory")
                    cash += funds - fee
                    remaining = max(0, p["qty"] - qty)
                    if remaining < 1e-12:
                        self.db.execute("DELETE FROM positions WHERE market=?", (market,))
                    else:
                        self.db.execute(
                            "UPDATE positions SET qty=?,cost=? WHERE market=?",
                            (remaining, p["cost"] * remaining / p["qty"], market),
                        )
            self.set("cash", cash)
            self.db.execute(
                "UPDATE intents SET status='done',response=? WHERE id=?", (json.dumps(response), identifier)
            )

    def close(self):
        self.db.close()
