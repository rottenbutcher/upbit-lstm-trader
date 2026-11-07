import tomllib
from pathlib import Path

import pytest

from upbit_bot.gui import FIELDS, job_steps, parse_field, progress_text, set_toml, validate_text

CONFIG = Path(__file__).resolve().parents[1] / "config.toml"


def test_set_toml_preserves_comments_and_other_values():
    text = CONFIG.read_text(encoding="utf-8")
    edited = set_toml(text, "model", "ensemble", 5)
    edited = set_toml(edited, "strategy", "entry_thresholds_bps", [25.0, 50.0])
    edited = set_toml(edited, "training", "lr_schedule", "constant")
    edited = set_toml(edited, "model", "attention", False)
    cfg = tomllib.loads(edited)
    assert cfg["model"]["ensemble"] == 5 and cfg["model"]["attention"] is False
    assert cfg["strategy"]["entry_thresholds_bps"] == [25.0, 50.0]
    assert cfg["training"]["lr_schedule"] == "constant"
    assert "# Independently seeded members" in edited
    original = tomllib.loads(text)
    assert cfg["policy"] == original["policy"] and cfg["data"] == original["data"]
    validate_text(edited)


def test_set_toml_adds_missing_key_to_its_section():
    cfg = tomllib.loads(set_toml("[model]\nhidden = 64\n\n[training]\nseed = 1\n", "model", "ensemble", 2))
    assert cfg == {"model": {"hidden": 64, "ensemble": 2}, "training": {"seed": 1}}


def test_every_gui_field_exists_in_config():
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    for section, key, *_ in FIELDS:
        assert key in cfg[section], (section, key)


def test_validation_rejects_bad_settings():
    text = set_toml(CONFIG.read_text(encoding="utf-8"), "strategy", "position_fraction", 0.9)
    with pytest.raises(ValueError):
        validate_text(text)
    with pytest.raises(ValueError):
        parse_field((1, 2), "3")
    assert parse_field("floats", "20, 30 45") == [20.0, 30.0, 45.0]


def test_job_steps_match_command_line_launchers(tmp_path):
    assert job_steps("fresh", CONFIG)[0][6:] == ["train"]
    steps = job_steps("train", CONFIG, allow_cpu=True)
    assert [s[6:] for s in steps] == [
        ["download"], ["train", "--allow-cpu"], ["package"]]
    assert job_steps("hmm", CONFIG, has_lstm=True)[0][-2:] == ["--compare-lstm", "artifacts"]
    path = tmp_path / "progress.json"
    path.write_text('{"members": 3, "epochs": 10, "history": [{"member": 2, "epoch": 5, '
                    '"validation_mae_bps": 12.5}]}')
    fraction, text = progress_text(path)
    assert fraction == pytest.approx(0.5) and "2/3" in text and "12.50" in text
    assert progress_text(tmp_path / "missing.json") == (None, None)


def test_resume_plan_and_restore(tmp_path):
    import copy

    from upbit_bot.common import load_config
    from upbit_bot.gui import describe_change, restore_checkpoint_settings, resume_plan
    from upbit_bot.resume import identity, write_info

    cfg = load_config(CONFIG)
    assert resume_plan(tmp_path, cfg)["kind"] == "none"
    (tmp_path / "checkpoint.pt").write_bytes(b"x")
    assert resume_plan(tmp_path, cfg)["kind"] == "legacy"  # checkpoint without resume.json
    write_info(tmp_path, identity(cfg, ["KRW-BTC"], {"KRW-BTC": {}}))
    assert resume_plan(tmp_path, cfg)["kind"] == "ok"
    safe = copy.deepcopy(cfg)
    safe["training"]["epochs"] = 40
    safe["model"]["ensemble"] = 5
    safe["strategy"]["stop_loss"] = 0.01
    assert resume_plan(tmp_path, safe)["kind"] == "ok"
    preset = copy.deepcopy(cfg)
    preset["model"]["hidden"] = 32
    preset["training"]["samples_per_epoch"] = 20000
    plan = resume_plan(tmp_path, preset)
    assert plan["kind"] == "mismatch" and len(plan["changes"]) == 2
    assert "LSTM 크기: 64 → 32" in [describe_change(c) for c in plan["changes"]]
    text = set_toml(CONFIG.read_text(encoding="utf-8"), "model", "hidden", 32)
    text = set_toml(text, "training", "epochs", 40)
    restored = tomllib.loads(restore_checkpoint_settings(text, plan["changes"]))
    assert restored["model"]["hidden"] == 64 and isinstance(restored["model"]["hidden"], int)
    assert restored["training"]["epochs"] == 40  # unrelated edits are kept


def test_chart_and_result_helpers(tmp_path):
    from upbit_bot.gui import (
        eta_text,
        failure_hint,
        load_equity,
        nice_ticks,
        remaining_epochs,
        report_cards,
        training_series,
    )

    ticks = nice_ticks(12.3, 47.9)
    assert ticks[0] <= 12.3 and ticks[-1] >= 47.9 and 3 <= len(ticks) <= 8
    assert len(nice_ticks(5, 5)) >= 2 and nice_ticks(float("nan"), 1) == [0.0, 1.0]
    history = [
        {"member": 1, "epoch": 1, "validation_mae_bps": 30.0, "train_loss": 0.4},
        {"member": 1, "epoch": 2, "validation_mae_bps": 28.0, "train_loss": 0.3},
        {"member": 2, "epoch": 1, "validation_mae_bps": 29.0, "train_loss": float("nan")},
    ]
    series = training_series(history)
    assert [s["points"] for s in series["val"]] == [[(1, 30.0), (2, 28.0)], [(1, 29.0)]]
    assert series["loss"][1]["points"] == [] and series["val"][0]["color"] != series["val"][1]["color"]
    assert remaining_epochs({"members": 3, "epochs": 10, "history": history}) == 19
    assert eta_text([], 5) == "—" and eta_text([60, 60], 10) == "최대 약 10분"
    csv_path = tmp_path / "equity.csv"
    rows = "\n".join(f"2026-01-01T00:{i % 60:02d}:00+00:00,{100000 + i}" for i in range(3000))
    csv_path.write_text("time,equity_krw\n" + rows + "\n")
    points = load_equity(csv_path, max_points=500)
    assert len(points) <= 501 and points[-1][1] == 102999
    assert load_equity(tmp_path / "missing.csv") == []
    report = {
        "test": {"return_pct": 4.0, "max_drawdown_pct": 3.0, "sharpe_daily": 1.2, "trades": 50,
                 "win_rate": 0.52, "profit_factor": 1.3},
        "validation": {"return_pct": -1.0},
        "buy_hold_test": {"return_pct": 6.0},
        "forecast": {"test": {"direction_accuracy": 0.51, "ic": 0.02}},
        "approved_for_live": False,
        "threshold_bps": 30,
    }
    cards = {c[0]: c for c in report_cards(report)}
    assert cards["테스트 수익률"][1:] == ("+4.00%", "단순 보유 +6.00%", "good")
    assert cards["보유 대비"][1] == "-2.00%" and cards["보유 대비"][3] == "bad"
    assert cards["실거래 승인"][1] == "미승인" and report_cards(None) == []
    assert "CPU" in failure_hint("RuntimeError: CUDA GPU not found. Check")
    assert failure_hint("all good") is None


def test_epoch_axis_uses_whole_numbers():
    from upbit_bot.gui import x_ticks

    labels = [x for x, _ in x_ticks(1, 6, integer=True)]
    assert labels and all(float(x).is_integer() and 1 <= x <= 6 for x in labels)
    assert [x for x, _ in x_ticks(0, 1, integer=True)] in ([0, 1], [0.0, 1.0])
    assert len(x_ticks(0.0, 100.0)) == 5
