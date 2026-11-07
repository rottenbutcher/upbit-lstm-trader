import copy
import json

import pytest
from test_features import candles

from upbit_bot.common import load_config, save_json
from upbit_bot.data import CandleStore
from upbit_bot.resume import (
    INFO_FILE,
    MISMATCH_FILE,
    ResumeMismatch,
    changes,
    config_identity,
    fingerprint,
    identity,
)

SNAP = {"KRW-BTC": {"first": "a", "last": "b", "rows": 1, "sha256": "x"}}


def test_only_weight_relevant_settings_matter():
    cfg = load_config()
    base = identity(cfg, ["KRW-BTC"], SNAP)
    safe = copy.deepcopy(cfg)
    safe["training"].update(epochs=99, patience=9, batch_size=64, workers=2)
    safe["model"]["ensemble"] = 7
    safe["strategy"]["stop_loss"] = 0.01
    safe["runtime"]["capital_krw"] = 5_000_000.0
    safe["llm"]["model"] = "other"
    assert fingerprint(identity(safe, ["KRW-BTC"], SNAP)) == fingerprint(base)
    # 0 and 0.0 written differently in TOML are the same value.
    same = copy.deepcopy(cfg)
    same["model"]["dropout"] = float(cfg["model"]["dropout"])
    same["training"]["seed"] = float(cfg["training"]["seed"])
    assert fingerprint(identity(same, ["KRW-BTC"], SNAP)) == fingerprint(base)
    changed = copy.deepcopy(cfg)
    changed["model"]["hidden"] = 32
    assert changes(base, identity(changed, ["KRW-BTC"], SNAP)) == [("model", "hidden", 64, 32)]
    newer = {"KRW-BTC": {**SNAP["KRW-BTC"], "last": "c", "sha256": "y"}}
    assert changes(base, identity(cfg, ["KRW-BTC"], newer)) == [("data", "KRW-BTC", "b", "c")]
    assert changes(None, base)[0][0] == "legacy"
    assert set(config_identity(cfg)["training"]) >= {"seed", "learning_rate", "lr_schedule"}


@pytest.fixture
def tiny(tmp_path):
    pytest.importorskip("torch")
    cfg = copy.deepcopy(load_config())
    cfg["data"]["minimum_days"] = 5
    cfg["model"].update(lookback=16, hidden=8, layers=1, dropout=0, ensemble=1, attention=True)
    cfg["training"].update(epochs=1, batch_size=32, samples_per_epoch=64, validation_days=1, test_days=1)
    directory = tmp_path / "data"
    save_json(directory / "universe.json", {"markets": ["KRW-BTC", "KRW-ETH"], "selected_at": "synthetic"})
    store = CandleStore(directory)
    for market in ("KRW-BTC", "KRW-ETH"):
        store.put(market, candles(2400))
    store.close()
    return cfg, directory, tmp_path / "artifacts"


def test_resume_survives_safe_edits_and_explains_real_changes(tiny):
    from upbit_bot.train import train

    cfg, directory, out = tiny
    first = train(cfg, directory, out, allow_cpu=True)
    assert (out / INFO_FILE).exists()
    # The reported failure: GUI edits outside model/training used to block --resume.
    edited = copy.deepcopy(cfg)
    edited["training"]["epochs"] = 2
    edited["model"]["ensemble"] = 2  # member 1 reused, member 2 trained
    edited["strategy"]["take_profit"] = 0.05
    edited["policy"]["max_step"] = 0.3
    resumed = train(edited, directory, out, resume=True, allow_cpu=True)
    assert resumed["ensemble"] == 2 and (out / "checkpoint-1.pt").exists()
    history = json.loads((out / "training_history.json").read_text())
    # Member 1 finished in the first run (1 epoch) and was reused, not retrained.
    assert [h["member"] for h in history] == [1, 2, 2]
    assert first["model_sha256"] != resumed["model_sha256"]
    changed = copy.deepcopy(edited)
    changed["model"]["hidden"] = 16
    with pytest.raises(ResumeMismatch) as info:
        train(changed, directory, out, resume=True, allow_cpu=True)
    assert ("model", "hidden", 8, 16) in info.value.changes
    assert "model.hidden" in str(info.value)
    record = json.loads((out / MISMATCH_FILE).read_text())
    assert ["model", "hidden", 8, 16] in record["changes"]
    # A fresh run never deletes old checkpoints; it moves them aside and clears the mismatch.
    train(changed, directory, out, allow_cpu=True)
    assert list((out / "old-checkpoints").glob("*/checkpoint.pt"))
    assert not (out / MISMATCH_FILE).exists()
    train(changed, directory, out, resume=True, allow_cpu=True)


def test_legacy_checkpoint_is_reported_not_crashed(tiny):
    import torch

    from upbit_bot.train import train

    cfg, directory, out = tiny
    out.mkdir(parents=True)
    torch.save({"fingerprint": "old-format"}, out / "checkpoint.pt")
    with pytest.raises(ResumeMismatch) as info:
        train(cfg, directory, out, resume=True, allow_cpu=True)
    assert info.value.changes[0][0] == "legacy"


def test_cli_exit_code_for_mismatch(tiny, monkeypatch, capsys):
    from upbit_bot import cli

    def boom(*args, **kwargs):
        raise ResumeMismatch([("model", "hidden", 8, 16)])

    monkeypatch.setattr("upbit_bot.train.train", boom)
    monkeypatch.setattr("sys.argv", ["upbit-bot", "train", "--resume", "--allow-cpu"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 3
    assert "model.hidden" in capsys.readouterr().err
