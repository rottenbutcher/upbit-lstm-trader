from __future__ import annotations

import hashlib
import time
import uuid
from urllib.parse import unquote, urlencode

import jwt
import requests


class APIError(RuntimeError):
    def __init__(self, status, name):
        self.status, self.name = status, name
        super().__init__(f"Upbit HTTP {status}: {name}")


class Upbit:
    """Only GET is retried. Unknown POST outcomes must be reconciled by identifier."""

    base = "https://api.upbit.com"

    def __init__(self, access="", secret="", interval=0.14, session=None):
        self.access, self.secret = access, secret
        self.session = session or requests.Session()
        self.interval, self.last = interval, 0.0

    def headers(self, params):
        if not self.access or not self.secret:
            raise RuntimeError("Missing UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY")
        payload = {"access_key": self.access, "nonce": str(uuid.uuid4())}
        if params:
            query = unquote(urlencode(params, doseq=True))
            payload.update(query_hash=hashlib.sha512(query.encode()).hexdigest(), query_hash_alg="SHA512")
        return {"Authorization": "Bearer " + jwt.encode(payload, self.secret, algorithm="HS512")}

    def request(self, method, path, params=None, private=False):
        params = params or {}
        attempts = 5 if method == "GET" else 1
        for attempt in range(attempts):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            headers = self.headers(params) if private else {}
            try:
                response = self.session.request(
                    method,
                    self.base + path,
                    headers=headers,
                    timeout=(5, 20),
                    **({"params": params} if method == "GET" else {"json": params}),
                )
            except (requests.Timeout, requests.ConnectionError):
                if method != "GET" or attempt == attempts - 1:
                    raise
                time.sleep(min(2**attempt, 16))
                continue
            if response.status_code == 418:
                raise APIError(418, "IP blocked; stop and respect Retry-After")
            if (
                (response.status_code == 429 or response.status_code >= 500)
                and method == "GET"
                and attempt < attempts - 1
            ):
                delay = float(response.headers.get("Retry-After", min(2**attempt, 30)))
                time.sleep(min(max(delay, 1), 60))
                continue
            if not response.ok:
                try:
                    name = response.json().get("error", {}).get("name", "http_error")
                except ValueError:
                    name = "http_error"
                raise APIError(response.status_code, name)
            return response.json()
        raise RuntimeError("GET retry budget exhausted")

    def markets(self):
        return self.request("GET", "/v1/market/all", {"is_details": "true"})

    def tickers(self, markets):
        out = []
        for i in range(0, len(markets), 50):
            out.extend(self.request("GET", "/v1/ticker", {"markets": ",".join(markets[i : i + 50])}))
        return out

    def candles(self, market, to, count=200):
        return self.request("GET", "/v1/candles/minutes/5", {"market": market, "to": to, "count": count})

    def orderbook(self, market):
        return self.request("GET", "/v1/orderbook", {"markets": market})[0]

    def accounts(self):
        return self.request("GET", "/v1/accounts", private=True)

    def chance(self, market):
        return self.request("GET", "/v1/orders/chance", {"market": market}, private=True)

    def order(self, identifier):
        return self.request("GET", "/v1/order", {"identifier": identifier}, private=True)

    def submit(self, payload):
        return self.request("POST", "/v1/orders", payload, private=True)


def healthy_market(row):
    event = row.get("market_event", {})
    return row.get("market_warning", "NONE") == "NONE" and not event.get("warning", False)
