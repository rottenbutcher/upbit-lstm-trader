"""Daily macro / official-document snapshot for the LLM policy step.

Every source is optional and fail-soft: a failure is recorded in ``errors`` and the rest of the
snapshot is still produced. Text from feeds is untrusted third-party content; it is trimmed,
stripped of markup and passed to the LLM strictly as data.
"""

from __future__ import annotations

import csv
import html
import io
import itertools
import json
import logging
import math
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import requests

from .common import utcnow

LOG = logging.getLogger(__name__)
USER_AGENT = "upbit-lstm-trader/1.0 (daily macro snapshot)"
MAX_BYTES = 2_000_000

# FRED public CSV downloads need no API key. Series IDs are official FRED identifiers.
DEFAULT_SERIES = {
    "DFF": "Effective federal funds rate (%)",
    "DGS2": "US 2-year Treasury yield (%)",
    "DGS10": "US 10-year Treasury yield (%)",
    "T10Y2Y": "10y-2y Treasury spread (pp)",
    "T10YIE": "10-year breakeven inflation (%)",
    "VIXCLS": "CBOE VIX",
    "DTWEXBGS": "Broad trade-weighted US dollar index",
    "DEXKOUS": "KRW per USD",
    "BAMLH0A0HYM2": "US high-yield OAS (%)",
    "DCOILWTICO": "WTI crude (USD/bbl)",
    "SP500": "S&P 500",
    "NASDAQCOM": "NASDAQ Composite",
}

# Official press-release feeds (central banks, regulators, statistics agencies).
DEFAULT_FEEDS = {
    "Federal Reserve press releases": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB press releases": "https://www.ecb.europa.eu/rss/press.html",
    "US SEC press releases": "https://www.sec.gov/news/pressreleases.rss",
    "US BLS latest releases": "https://www.bls.gov/feed/bls_latest.rss",
}


def get(session, url, timeout=20):
    response = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT}, stream=True)
    response.raise_for_status()
    content = response.raw.read(MAX_BYTES + 1, decode_content=True)
    if len(content) > MAX_BYTES:
        raise ValueError("response too large")
    return content


def clean(text, limit=400):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def parse_fred_csv(content):
    """FRED CSV: header row then ``date,value`` with '.' for missing observations."""
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    out = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        try:
            value = float(row[1])
            date = datetime.fromisoformat(row[0]).date()
        except ValueError:
            continue
        if math.isfinite(value):
            out.append((date, value))
    return out


def summarize_series(observations):
    if not observations:
        raise ValueError("no observations")
    last_date, last = observations[-1]

    def change(days):
        cutoff = last_date - timedelta(days=days)
        prior = [v for d, v in observations if d <= cutoff]
        return None if not prior else round(last - prior[-1], 6)

    def pct(days):
        cutoff = last_date - timedelta(days=days)
        prior = [v for d, v in observations if d <= cutoff]
        return None if not prior or prior[-1] == 0 else round((last / prior[-1] - 1) * 100, 3)

    return {
        "date": last_date.isoformat(),
        "value": last,
        "change_1w": change(7),
        "change_1m": change(30),
        "pct_change_1w": pct(7),
        "pct_change_1m": pct(30),
    }


def fred(session, series, now):
    start = (now - timedelta(days=90)).date().isoformat()
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
    return summarize_series(parse_fred_csv(get(session, url)))


def parse_feed(content, now, days=7, limit=8):
    """RSS 2.0 or Atom. Returns recent items newest first."""
    root = ET.fromstring(content)
    items = []
    atom = "{http://www.w3.org/2005/Atom}"
    for node in root.iter():
        tag = node.tag
        if tag not in ("item", f"{atom}entry"):
            continue

        def text(*names, node=node):
            for name in names:
                child = node.find(name)
                if child is not None and (child.text or child.get("href")):
                    return child.text or child.get("href")
            return ""

        raw_date = text("pubDate", f"{atom}updated", f"{atom}published", "{http://purl.org/dc/elements/1.1/}date")
        published = None
        if raw_date:
            try:
                published = parsedate_to_datetime(raw_date)
            except (TypeError, ValueError):
                try:
                    published = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                except ValueError:
                    published = None
        if published is not None:
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
            if now - published > timedelta(days=days):
                continue
        items.append(
            {
                "title": clean(text("title", f"{atom}title"), 200),
                "published": published.isoformat() if published else None,
                "summary": clean(text("description", f"{atom}summary", f"{atom}content"), 400),
                "link": clean(text("link", f"{atom}link"), 300),
            }
        )
    items.sort(key=lambda x: x["published"] or "", reverse=True)
    return items[:limit]


def fear_greed(session):
    data = json.loads(get(session, "https://api.alternative.me/fng/?limit=8"))["data"]
    return [
        {
            "date": datetime.fromtimestamp(int(d["timestamp"]), UTC).date().isoformat(),
            "value": int(d["value"]),
            "label": clean(d.get("value_classification", ""), 40),
        }
        for d in data
    ]


def crypto_global(session):
    data = json.loads(get(session, "https://api.coingecko.com/api/v3/global"))["data"]
    return {
        "btc_dominance_pct": round(float(data["market_cap_percentage"]["btc"]), 3),
        "eth_dominance_pct": round(float(data["market_cap_percentage"].get("eth", 0)), 3),
        "total_market_cap_change_24h_pct": round(float(data["market_cap_change_percentage_24h_usd"]), 3),
    }


def upbit_daily(api, markets, days=30):
    """Daily trend/volatility per traded market from Upbit's public daily candles."""
    out = {}
    for market in markets:
        rows = api.request("GET", "/v1/candles/days", {"market": market, "count": days})
        closes = [float(r["trade_price"]) for r in sorted(rows, key=lambda r: r["candle_date_time_utc"])]
        if len(closes) < 8:
            raise ValueError(f"{market}: insufficient daily candles")
        rets = [math.log(b / a) for a, b in itertools.pairwise(closes)]
        mean = sum(rets) / len(rets)
        vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1))
        out[market] = {
            "close": closes[-1],
            "return_1d_pct": round((closes[-1] / closes[-2] - 1) * 100, 3),
            "return_7d_pct": round((closes[-1] / closes[-8] - 1) * 100, 3),
            "return_30d_pct": round((closes[-1] / closes[0] - 1) * 100, 3),
            "daily_vol_pct": round(vol * 100, 3),
            "above_20d_mean": closes[-1] > sum(closes[-20:]) / len(closes[-20:]),
        }
    return out


def collect(cfg, api, markets, session=None, now=None):
    """Snapshot dict. Never raises for an individual source failure."""
    now = now or utcnow()
    session = session or requests.Session()
    macro_cfg = cfg.get("macro", {})
    series = macro_cfg.get("fred_series", DEFAULT_SERIES)
    feeds = macro_cfg.get("feeds", DEFAULT_FEEDS)
    snapshot = {"collected_at": now.isoformat(), "fred": {}, "official_releases": {}, "errors": []}

    def attempt(name, fn):
        try:
            return fn()
        except Exception as exc:
            LOG.warning("macro source %s failed: %s", name, exc)
            snapshot["errors"].append(f"{name}: {type(exc).__name__}: {clean(str(exc), 160)}")
            return None

    for sid, label in series.items():
        result = attempt(f"FRED {sid}", lambda sid=sid: fred(session, sid, now))
        if result:
            snapshot["fred"][sid] = {"label": label, **result}
    for label, url in feeds.items():
        result = attempt(label, lambda url=url: parse_feed(get(session, url), now))
        if result is not None:
            snapshot["official_releases"][label] = result
    if macro_cfg.get("crypto_sentiment", True):
        snapshot["crypto_fear_greed"] = attempt("Fear & Greed", lambda: fear_greed(session))
        snapshot["crypto_global"] = attempt("CoinGecko global", lambda: crypto_global(session))
    snapshot["upbit_daily"] = attempt("Upbit daily candles", lambda: upbit_daily(api, markets))
    ok = len(snapshot["fred"]) + len(snapshot["official_releases"]) + bool(snapshot.get("upbit_daily"))
    snapshot["source_count"] = ok
    return snapshot
