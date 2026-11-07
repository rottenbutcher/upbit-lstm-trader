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
