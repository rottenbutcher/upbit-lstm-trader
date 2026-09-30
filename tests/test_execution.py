import copy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from upbit_bot.api import APIError
from upbit_bot.common import load_config
from upbit_bot.runtime import process_cycle, reconcile, risk_state, seconds_to_next
from upbit_bot.state import State

NOW = datetime(2026, 9, 30, 0, 5, 15, tzinfo=UTC)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def setup(tmp_path, config, monkeypatch):
    monkeypatch.setattr("upbit_bot.runtime.utcnow", lambda: NOW)
    state = State(tmp_path / "test.sqlite", 100000, "paper")
    manifest = {
        "strategy": config["strategy"],
        "data_end": "2026-09-29T00:00:00+00:00",
        "approved_for_live": True,
        "threshold_bps": 20,
        "markets": ["KRW-BTC"],
        "model_sha256": "abc",
    }
    bundle = SimpleNamespace(
        manifest=manifest,
        report={"approved_for_live": True},
        infer=lambda api, end: ({"KRW-BTC": 100}, {"KRW-BTC": 10000}),
    )

    class Fake:
        submits = 0

        def markets(self):
            return [{"market": "KRW-BTC"}]

        def orderbook(self, market):
            return {
                "timestamp": NOW.timestamp() * 1000,
                "orderbook_units": [{"ask_price": 10001, "bid_price": 10000, "ask_size": 20, "bid_size": 20}],
            }

        def submit(self, payload):
            self.submits += 1
            raise AssertionError("paper must never submit")

    yield state, bundle, Fake(), tmp_path
    state.close()


def test_paper_no_private_api_and_same_bar_idempotency(setup, config):
    state, bundle, api, path = setup
    first = process_cycle(config, bundle, api, state, "paper", path, NOW)
    positions = state.positions()
    assert first["status"] == "ok"
    assert positions["KRW-BTC"]["qty"] > 0
    assert state.get("cash") == pytest.approx(89995)
    second = process_cycle(config, bundle, api, state, "paper", path, NOW)
    assert second["status"] == "already_processed"
    assert state.positions() == positions
    assert api.submits == 0


def test_stop_file_prevents_orders(setup, config):
    state, bundle, api, path = setup
    (path / "STOP").touch()
    assert process_cycle(config, bundle, api, state, "paper", path, NOW)["status"] == "stopped"
    assert state.positions() == {}


def test_loss_limit_latches_and_next_day_resets_daily_only(setup, config):
    state, *_ = setup
    assert not risk_state(state, 100000, "2026-09-30", config["strategy"])
    assert risk_state(state, 96000, "2026-09-30", config["strategy"])
    assert risk_state(state, 101000, "2026-09-30", config["strategy"])
    assert not risk_state(state, 101000, "2026-10-01", config["strategy"])
    assert risk_state(state, 88000, "2026-10-01", config["strategy"])
    assert risk_state(state, 110000, "2026-10-02", config["strategy"])


def test_fill_applied_once_across_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    state = State(path, 100000, "live")
    state.intent("id", "KRW-BTC", "buy", 1, {})
    state.close()
    state = State(path, 100000, "live")
    state.apply_fill("id", 1, 10000, 5, {})
    state.apply_fill("id", 1, 10000, 5, {})
    assert state.get("cash") == 89995
    assert state.positions()["KRW-BTC"]["qty"] == 1
    state.intent("sell", "KRW-BTC", "sell", 2, {})
    state.apply_fill("sell", 0.5, 5100, 2.55, {})
    assert state.positions()["KRW-BTC"]["qty"] == 0.5
    assert state.positions()["KRW-BTC"]["cost"] == 5002.5
    assert state.get("cash") == pytest.approx(95092.45)
    state.close()


def test_unknown_order_never_retries(setup):
    state, *_ = setup
    state.intent("unknown", "KRW-BTC", "buy", 1, {})
    api = SimpleNamespace(order=lambda identifier: (_ for _ in ()).throw(APIError(404, "order_not_found")))
    with pytest.raises(RuntimeError, match="do not resend"):
        reconcile(api, state)
    assert len(state.pending()) == 1


def test_terminal_cancel_with_partial_fill_reconciles(setup):
    state, *_ = setup
    state.intent("partial", "KRW-BTC", "buy", 1, {})
    order = {
        "market": "KRW-BTC",
        "side": "bid",
        "state": "cancel",
        "executed_volume": "0.2",
        "paid_fee": "1",
        "trades": [{"funds": "2000", "volume": "0.2"}],
    }
    assert reconcile(SimpleNamespace(order=lambda _: order), state)
    assert state.positions()["KRW-BTC"]["qty"] == 0.2
    assert state.get("cash") == 97999


def test_incomplete_fill_details_fail_closed(setup):
    state, *_ = setup
    state.intent("partial", "KRW-BTC", "buy", 1, {})
    order = {"market": "KRW-BTC", "side": "bid", "state": "done", "executed_volume": "1", "trades": []}
    with pytest.raises(ValueError, match="incomplete"):
        reconcile(SimpleNamespace(order=lambda _: order), state)
    assert state.get("cash") == 100000
    assert len(state.pending()) == 1


def test_schedule_aligns_boundary():
    assert seconds_to_next(301, 15) == 314
    assert seconds_to_next(599, 15) == 16


def test_failed_gate_blocks_live_buys_before_private_calls(setup, config):
    state, bundle, api, path = setup
    bundle.manifest["approved_for_live"] = False
    result = process_cycle(config, bundle, api, state, "live", path, NOW)
    assert not result["buy_allowed"]
    assert not state.positions()


def test_insufficient_budget_does_not_create_order(setup, config):
    state, bundle, api, path = setup
    with state.db:
        state.set("cash", 1000)
        state.set("peak", 1000)
    process_cycle(config, bundle, api, state, "paper", path, NOW)
    assert not state.positions()
    assert not state.pending()


def test_mode_mismatch_rejected(tmp_path):
    path = tmp_path / "state.sqlite"
    State(path, 100000, "paper").close()
    with pytest.raises(ValueError, match="mismatch"):
        State(path, 100000, "live")


def test_wide_spread_skips_order(setup, config):
    state, bundle, api, path = setup
    original = api.orderbook

    def wide(market):
        book = copy.deepcopy(original(market))
        book["orderbook_units"][0]["ask_price"] = 12000
        return book

    api.orderbook = wide
    process_cycle(config, bundle, api, state, "paper", path, NOW)
    assert not state.positions()


def test_live_post_timeout_then_restart_reconciles_without_duplicate(setup, config):
    import requests

    state, bundle, api, path = setup
    payloads = []
    api.chance = lambda _: {
        "market": {
            "state": "active",
            "bid_types": ["price"],
            "ask_types": ["market"],
            "bid": {"min_total": "5000"},
            "ask": {"min_total": "5000"},
            "max_total": "100000000",
        },
        "bid_fee": "0.0005",
        "ask_fee": "0.0005",
        "bid_account": {"balance": "100000"},
        "ask_account": {"balance": "0"},
    }

    def timed_out(payload):
        payloads.append(payload)
        raise requests.Timeout("accepted but reply lost")

    api.submit = timed_out
    with pytest.raises(requests.Timeout):
        process_cycle(config, bundle, api, state, "live", path, NOW)
    assert len(state.pending()) == 1 and len(payloads) == 1
    api.order = lambda identifier: {
        "market": "KRW-BTC",
        "side": "bid",
        "state": "done",
        "executed_volume": "1",
        "paid_fee": "5",
        "trades": [{"funds": "10000", "volume": "1"}],
    }
    result = process_cycle(config, bundle, api, state, "live", path, NOW)
    assert result["status"] == "already_processed"
    assert len(payloads) == 1
    assert state.positions()["KRW-BTC"]["qty"] == 1
    assert state.get("cash") == 89995


def test_stale_model_allows_exit_but_blocks_buy(setup, config):
    state, bundle, api, path = setup
    bundle.manifest["data_end"] = "2025-01-01T00:00:00Z"
    result = process_cycle(config, bundle, api, state, "live", path, NOW)
    assert not result["buy_allowed"]
    assert not state.positions()
