import itertools

import numpy as np
import pandas as pd
import pytest
from test_features import candles

from upbit_bot.common import load_config, read_json, save_json
from upbit_bot.data import CandleStore
from upbit_bot.hmm_model import fit_forecaster, forward_filter, settings, training_blocks
from upbit_bot.hmm_research import load_dataset, train_hmm
from upbit_bot.research import candle_snapshot, forecast_metrics


def test_forward_filter_matches_enumerated_paths_and_cannot_see_future():
    p = dict(
        startprob=[0.7, 0.3],
        transmat=[[0.8, 0.2], [0.1, 0.9]],
        means=[[-1.0], [1.0]],
        variances=[[0.5], [2.0]],
    )
    x = np.array([[-1.0], [0.0], [1.0], [-0.2]])
    actual = forward_filter(x, p)
    for t in range(len(x)):
        probability = np.zeros(2)
        for path in itertools.product(range(2), repeat=t + 1):
            weight = p["startprob"][path[0]]
            for j, state in enumerate(path):
                variance = p["variances"][state][0]
                weight *= np.exp(-0.5 * (x[j, 0] - p["means"][state][0]) ** 2 / variance)
                weight /= np.sqrt(2 * np.pi * variance)
                if j:
                    weight *= p["transmat"][path[j - 1]][state]
            probability[path[-1]] += weight
        np.testing.assert_allclose(actual[t], probability / probability.sum(), atol=1e-12)
    changed = x.copy()
    changed[2:] = 1000
    np.testing.assert_array_equal(actual[:2], forward_filter(changed, p)[:2])
    broken = np.concatenate([x[:2], [[np.nan]], x])
    np.testing.assert_allclose(forward_filter(broken, p)[3:], actual)


def test_training_blocks_preserve_adjacency_and_cap():
    x = np.arange(1000.0)[:, None]
    mask = np.arange(1000) < 900
    mask[100:200] = False
    selected, lengths = training_blocks(x, mask, 120, 40)
    assert 0 < len(selected) <= 120 and sum(lengths) == len(selected)
    offset = 0
    for length in lengths:
        sequence = selected[offset : offset + length, 0]
        assert np.all(np.diff(sequence) == 1)
        assert np.all(mask[sequence.astype(int)])
        offset += length
    assert selected[0, 0] == 0 and selected[-1, 0] == 899


def test_hmm_fit_and_calibration_ignore_held_out_prices_and_labels():
    pytest.importorskip("hmmlearn")
    rng = np.random.default_rng(12)
    x = rng.normal(size=(600, 4))
    y = rng.normal(size=600)
    train_mask = np.arange(600) < 400
    indexes = np.arange(20, 390)
    options = settings({"hmm": {"iterations": 5, "restarts": 1, "max_fit_rows": 300, "block_size": 100}})
    first, pred = fit_forecaster(x, train_mask, indexes, y, options, 42)
    x[400:] *= 100
    y[390:] += 50000
    second, altered = fit_forecaster(x, train_mask, indexes, y, options, 42)
    assert first == second
    np.testing.assert_array_equal(pred[:400], altered[:400])


def test_metrics_show_majority_baseline_and_class_imbalance():
    y = np.array([1, 2, 3, -1.0])
    m = forecast_metrics(y, np.ones(4), np.ones(4, dtype=bool))
    assert m["direction_accuracy"] == 0.75
    assert m["balanced_accuracy"] == 0.5
    assert m["train_majority_accuracy"] == 0.75
    assert forecast_metrics(np.zeros(4), np.zeros(4), np.zeros(4))["balanced_accuracy"] is None


@pytest.fixture
def history(tmp_path):
    cfg = load_config()
    cfg["data"]["minimum_days"] = 5
    cfg["model"]["lookback"] = 16
    cfg["training"].update(validation_days=1, test_days=1)
    cfg["hmm"].update(iterations=4, restarts=1, max_fit_rows=400, block_size=100)
    directory = tmp_path / "data"
    universe = {"markets": ["KRW-BTC", "KRW-ETH"]}
    save_json(directory / "universe.json", universe)
    store = CandleStore(directory)
    for market in universe["markets"]:
        store.put(market, candles(2400))
    store.close()
    return cfg, directory, universe


def test_snapshot_freezes_lstm_dates_and_rejects_changed_candles(history):
    cfg, directory, universe = history
    dataset = load_dataset(cfg, directory)
    manifest = {
        "universe": universe,
        "data_snapshot": dataset[5],
        "train_end": dataset[6].isoformat(),
        "validation_end": dataset[7].isoformat(),
        "data_end": candles(2400).index[-1].isoformat(),
    }
    store = CandleStore(directory)
    for market in universe["markets"]:
        store.put(market, candles(2600).iloc[2400:])
    unchanged = load_dataset(cfg, directory, manifest)
    assert dataset[5] == unchanged[5]
    for a, b in zip(dataset[3], unchanged[3]):
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
    raw = candles(2400).iloc[100:101].copy()
    raw["close"] *= 2
    store.put("KRW-BTC", raw)
    store.close()
    with pytest.raises(ValueError, match="differs from the LSTM"):
        load_dataset(cfg, directory, manifest)


def test_hmm_standalone_report_and_saved_parameters(history, tmp_path):
    pytest.importorskip("hmmlearn")
    cfg, directory, _ = history
    out = tmp_path / "hmm"
    result = train_hmm(cfg, directory, out)
    assert list(result["results"]) == ["HMM"]
    assert not (out / "manifest.json").exists()
    assert "균형 정확도" in (out / "comparison.html").read_text(encoding="utf-8")
    params = read_json(out / "hmm-model.json")["markets"]["KRW-BTC"]
    assert params["fit_rows"] <= cfg["hmm"]["max_fit_rows"]
    assert np.isfinite(params["state_returns_bps"]).all()
    predictions = pd.read_csv(out / "hmm_test_predictions.csv")
    assert len(predictions) == result["results"]["HMM"]["forecast"]["test"]["samples"]
    with pytest.raises(ValueError, match="separate HMM output"):
        save_json(out / "manifest.json", {})
        train_hmm(cfg, directory, out)


def test_snapshot_changes_when_a_candle_changes():
    raw = candles(100)
    before = candle_snapshot(raw)
    raw.iloc[1, 0] *= 2
    assert before["sha256"] != candle_snapshot(raw)["sha256"]


@pytest.mark.parametrize("options", [{"states": 1}, {"max_fit_rows": -1}, {"shrinkage": float("nan")}])
def test_invalid_hmm_options_fail_early(options):
    with pytest.raises(ValueError):
        settings({"hmm": options})
