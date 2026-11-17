import pytest

import upbit_bot.common as common

_load = common.load_config


@pytest.fixture(autouse=True)
def market_entries_by_default(monkeypatch):
    """Most tests pin the market-order path; limit-entry tests opt in explicitly."""

    def load(path="config.toml"):
        cfg = _load(path)
        cfg["strategy"]["entry_order"] = "market"
        return cfg

    monkeypatch.setattr(common, "load_config", load)
    for module in ("test_backtest", "test_execution", "test_policy", "test_strategy_rules", "test_pipeline",
                   "test_architectures"):
        mod = __import__("sys").modules.get(module)
        if mod is not None and hasattr(mod, "load_config"):
            monkeypatch.setattr(mod, "load_config", load)
