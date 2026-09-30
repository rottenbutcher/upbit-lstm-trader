import hashlib
from types import SimpleNamespace
from urllib.parse import unquote, urlencode

import jwt
import pytest
import requests

from upbit_bot.api import APIError, Upbit


def test_auth_query_hash_and_nonce():
    api = Upbit("access", "a" * 64, interval=0)
    params = {"market": "KRW-BTC", "states[]": ["wait", "watch"]}
    first = jwt.decode(api.headers(params)["Authorization"][7:], "a" * 64, algorithms=["HS512"])
    second = jwt.decode(api.headers(params)["Authorization"][7:], "a" * 64, algorithms=["HS512"])
    assert first["query_hash"] == hashlib.sha512(unquote(urlencode(params, doseq=True)).encode()).hexdigest()
    assert first["nonce"] != second["nonce"]


def test_order_timeout_is_not_retried():
    class Session:
        calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            raise requests.Timeout()

    session = Session()
    api = Upbit("a", "s" * 64, session=session, interval=0)
    with pytest.raises(requests.Timeout):
        api.submit({"market": "KRW-BTC"})
    assert session.calls == 1


def test_rate_limit_get_retries(monkeypatch):
    monkeypatch.setattr("upbit_bot.api.time.sleep", lambda _: None)
    responses = [
        SimpleNamespace(status_code=429, headers={"Retry-After": "1"}, ok=False),
        SimpleNamespace(status_code=200, headers={}, ok=True, json=lambda: ["ok"]),
    ]
    api = Upbit(session=SimpleNamespace(request=lambda *a, **k: responses.pop(0)), interval=0)
    assert api.markets() == ["ok"]


def test_ip_block_no_retry():
    api = Upbit(
        session=SimpleNamespace(request=lambda *a, **k: SimpleNamespace(status_code=418, headers={})),
        interval=0,
    )
    with pytest.raises(APIError, match="IP blocked"):
        api.markets()
