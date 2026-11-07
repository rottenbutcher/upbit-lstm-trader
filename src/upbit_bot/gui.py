"""Training dashboard: settings, presets, job runner, live learning curves and backtest results.

Standard library only (tkinter + Canvas charts). Every job runs the normal CLI in a subprocess,
so the GUI and TRAIN.cmd / train.sh produce identical artifacts. Comments in config.toml are
preserved: only the value on an edited ``key = value`` line is replaced.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import webbrowser
from datetime import UTC, datetime
from pathlib import Path

from .resume import (
    LEGACY,
    MISMATCH_FILE,
    config_changes,
    config_identity,
    has_checkpoint,
    read_info,
)

# (section, key, label, kind, help). kind: int, float, bool, str, floats (list), or a tuple of choices.
FIELDS = [
    (
        "data",
        "top_n",
        "종목 수",
        "int",
        "처음 데이터 수집 때 거래대금 상위 N개로 고정됩니다. 바꾸려면 data 폴더를 새로 만드세요.",
    ),
    ("data", "start", "수집 시작일", "str", "YYYY-MM-DD. 오래될수록 학습량과 다운로드 시간이 늘어납니다."),
    ("data", "minimum_days", "최소 이력(일)", "int", "이보다 짧은 종목이 있으면 학습을 멈춥니다."),
    (
        "model",
        "feature_version",
        "피처 버전",
        (1, 2),
        "2 = 24시간 문맥, 변동성, 요일 피처를 추가합니다 (권장).",
    ),
    ("model", "lookback", "입력 길이(5분봉)", "int", "96 = 8시간. 길수록 느리고 메모리를 더 씁니다."),
    ("model", "horizon", "예측 구간(5분봉)", "int", "6 = 30분 뒤 수익률을 예측합니다."),
    (
        "model",
        "hidden",
        "LSTM 크기",
        "int",
        "64 권장. 클수록 표현력이 늘지만 과적합과 메모리 사용이 늘어납니다.",
    ),
    ("model", "layers", "LSTM 층 수", "int", "1~3."),
    ("model", "dropout", "드롭아웃", "float", "0~0.5. 과적합을 줄입니다."),
    ("model", "attention", "어텐션 풀링", "bool", "시퀀스 전체를 가중 요약해 마지막 상태와 함께 사용합니다."),
    (
        "model",
        "ensemble",
        "앙상블 모델 수",
        "int",
        "독립 학습 모델 수. 3이면 학습 시간이 약 3배입니다. 늘리면 완료된 모델은 재사용됩니다.",
    ),
    (
        "training",
        "epochs",
        "최대 epoch",
        "int",
        "조기 종료로 더 일찍 끝날 수 있습니다. 바꿔도 이어서 학습할 수 있습니다.",
    ),
    (
        "training",
        "patience",
        "조기 종료 patience",
        "int",
        "검증 오차가 이 횟수만큼 개선되지 않으면 멈춥니다.",
    ),
    ("training", "batch_size", "배치 크기", "int", "GPU 메모리가 부족하면 자동으로 절반으로 줄입니다."),
    ("training", "learning_rate", "학습률", "float", "AdamW 기본 학습률."),
    (
        "training",
        "lr_schedule",
        "학습률 스케줄",
        ("cosine", "constant"),
        "cosine = 워밍업 후 점진 감소 (권장).",
    ),
    ("training", "warmup_epochs", "워밍업 epoch", "int", "cosine 스케줄의 초기 워밍업 기간."),
    ("training", "weight_decay", "가중치 감쇠", "float", "L2 정규화 강도."),
    ("training", "samples_per_epoch", "epoch당 샘플 수", "int", "0 = 전체. 클수록 epoch가 길어집니다."),
    ("training", "validation_days", "검증 기간(일)", "int", "매수 기준 선택과 조기 종료에 사용합니다."),
    ("training", "test_days", "테스트 기간(일)", "int", "최종 승인 판정에만 쓰는 미사용 구간입니다."),
    ("training", "seed", "랜덤 시드", "int", "같은 시드와 같은 데이터면 같은 결과가 나옵니다."),
    (
        "strategy",
        "entry_thresholds_bps",
        "매수 기준 후보(bps)",
        "floats",
        "검증 구간에서 이 중 하나를 고릅니다. 예: 20, 30, 45",
    ),
    (
        "strategy",
        "stop_loss",
        "손절 비율",
        "float",
        "0.02 = 2%. LLM 정책은 이 값보다 느슨하게 할 수 없습니다.",
    ),
    ("strategy", "take_profit", "익절 비율", "float", "0.04 = 4%."),
    (
        "strategy",
        "position_fraction",
        "종목당 비중",
        "float",
        "자본 대비 비율. LLM은 이 값 이하로만 조정합니다.",
    ),
    ("strategy", "max_exposure", "최대 총 노출", "float", "자본 대비 동시 보유 한도."),
    ("strategy", "max_hold_bars", "최대 보유(5분봉)", "int", "12 = 1시간."),
    ("strategy", "daily_loss_limit", "일일 손실 한도", "float", "넘으면 당일 신규 매수를 중단합니다."),
    ("strategy", "max_drawdown", "최대 낙폭 한도", "float", "넘으면 운영자가 해제할 때까지 정지합니다."),
    ("gate", "min_trades", "승인 최소 거래 수", "int", "검증과 테스트 모두 충족해야 실거래가 승인됩니다."),
    ("gate", "min_profit_factor", "승인 최소 PF", "float", "총이익 / 총손실."),
    ("runtime", "capital_krw", "운용 자본(KRW)", "float", "실거래 시 서버 설정과 같아야 합니다."),
]
TAB_TITLES = {
    "data": "데이터",
    "model": "모델",
    "training": "학습",
    "strategy": "전략",
    "gate": "승인·자본",
    "runtime": "승인·자본",
}
LABELS = {(section, key): label for section, key, label, *_ in FIELDS}
DEFAULTS = {
    "feature_version": 1,
    "attention": False,
    "ensemble": 1,
    "lr_schedule": "constant",
    "warmup_epochs": 0,
}

PRESETS = {
    "빠른 점검": (
        "몇 분 · 동작 확인",
        {
            ("model", "ensemble"): 1,
            ("training", "epochs"): 2,
            ("training", "samples_per_epoch"): 20000,
            ("model", "hidden"): 32,
        },
    ),
    "균형": (
        "권장 · 앙상블 3",
        {
            ("model", "ensemble"): 3,
            ("training", "epochs"): 12,
            ("training", "samples_per_epoch"): 120000,
            ("model", "hidden"): 64,
            ("model", "attention"): True,
            ("model", "feature_version"): 2,
        },
    ),
    "고성능": (
        "오래 걸림 · 앙상블 5",
        {
            ("model", "ensemble"): 5,
            ("training", "epochs"): 20,
            ("training", "samples_per_epoch"): 200000,
            ("training", "patience"): 4,
            ("model", "hidden"): 64,
        },
    ),
}

# Palette: GitHub-dark inspired, readable contrast on every surface.
BG, PANEL, CARD, FIELD, BORDER = "#0d1117", "#161b22", "#1c2129", "#0d1117", "#30363d"
TEXT, MUTED, ACCENT, ACCENT_HI = "#e6edf3", "#8b949e", "#2f81f7", "#58a6ff"
GOOD, BAD, WARN = "#3fb950", "#f85149", "#d29922"
SERIES = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff", "#ff7b72", "#39c5cf", "#e3b341"]
EXIT_MISMATCH = 3


# ----------------------------------------------------------------------------- config editing


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, float):
        return repr(value)
    return str(int(value))


def set_toml(text, section, key, value):
    """Replace (or append) ``key = value`` inside ``[section]`` keeping comments and layout."""
    lines = text.splitlines(keepends=True)
    current, insert_at, header = None, None, re.compile(r"^\s*\[([^\]]+)\]\s*(#.*)?$")
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#\n]*?)(\s*#.*)?(\r?\n)?$")
    for i, line in enumerate(lines):
        h = header.match(line)
        if h:
            if current == section and insert_at is None:
                insert_at = i
            current = h.group(1).strip()
            continue
        if current == section:
            m = pattern.match(line)
            if m:
                lines[i] = f"{m.group(1)}{toml_value(value)}{m.group(3) or ''}{m.group(4) or ''}"
                return "".join(lines)
    if current == section:
        insert_at = len(lines)
    if insert_at is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"\n[{section}]\n")
        insert_at = len(lines)
    lines.insert(insert_at, f"{key} = {toml_value(value)}\n")
    return "".join(lines)


def parse_field(kind, raw):
    raw = raw.strip()
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        return raw in ("1", "True", "true")
    if kind == "floats":
        values = [float(x) for x in re.split(r"[,\s]+", raw.strip("[]")) if x]
        if not values:
            raise ValueError("at least one value")
        return values
    if isinstance(kind, tuple):
        value = type(kind[0])(raw)
        if value not in kind:
            raise ValueError(f"one of {kind}")
        return value
    return raw


def display_value(kind, value):
    if kind == "floats":
        return ", ".join(f"{v:g}" for v in value)
    return value


def validate_text(text):
    """Write to a temp file and run the same validation as training (raises ValueError)."""
    from .common import load_config

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as f:
        f.write(text)
        name = f.name
    try:
        return load_config(name)
    finally:
        os.unlink(name)


# ----------------------------------------------------------------------------- jobs and resume


def job_steps(job, config, allow_cpu=False, has_lstm=False):
    """CLI argv lists for each action (identical to the command-line launchers)."""
    cli = [sys.executable, "-u", "-m", "upbit_bot.cli", "--config", str(config)]
    cpu = ["--allow-cpu"] if allow_cpu else []
    steps = {
        "train": [[*cli, "download"], [*cli, "train", *cpu], [*cli, "package"]],
        "fresh": [[*cli, "train", *cpu], [*cli, "package"]],
        "resume": [[*cli, "train", "--resume", *cpu], [*cli, "package"]],
        "download": [[*cli, "download"]],
        "hmm": [[*cli, "train-hmm", *(["--compare-lstm", "artifacts"] if has_lstm else [])]],
        "package": [[*cli, "package"]],
        "doctor": [[*cli, "doctor"]],
    }
    if job not in steps:
        raise ValueError(job)
    return steps[job]


def resume_plan(artifacts, cfg):
    """What "이어서 학습" can do with the current (possibly unsaved) settings.

    kind: none (no checkpoint), legacy (older program version), ok, mismatch (with changes).
    Data changes are only detectable after loading candles; training reports those (exit code 3).
    """
    if not has_checkpoint(artifacts):
        return {"kind": "none", "changes": []}
    info = read_info(artifacts)
    if not info or "identity" not in info or "config" not in info["identity"]:
        return {"kind": "legacy", "changes": [(LEGACY, None, None, None)]}
    found = config_changes(info["identity"]["config"], config_identity(cfg))
    return {"kind": "mismatch" if found else "ok", "changes": found}


def restore_checkpoint_settings(text, found):
    """Put the checkpoint's values back for the given config changes (other edits are kept)."""
    for section, key, old, _ in found:
        if section in ("model", "training") and old is not None:
            text = set_toml(text, section, key, old)
    return text


def describe_change(change):
    section, key, old, new = change
    if section == LEGACY:
        return "이전 버전 프로그램으로 만든 체크포인트입니다"
    if section == "data" and key == "markets":
        return "학습 종목 구성이 바뀌었습니다"
    if section == "data":
        return f"{key} 캔들 데이터가 바뀌었습니다 (마지막 봉 {old} → {new})"
    label = LABELS.get((section, key), f"{section}.{key}")
    return f"{label}: {old} → {new}"


def read_mismatch(artifacts):
    try:
        data = json.loads((Path(artifacts) / MISMATCH_FILE).read_text(encoding="utf-8"))
        return [tuple(c) for c in data["changes"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def failure_hint(log_text):
    """Plain-language next step for common failures (None when unknown)."""
    rules = [
        (
            "CUDA GPU not found",
            "GPU를 찾지 못했습니다. 'CPU 모드'를 켜고 다시 실행하거나 그래픽 드라이버를 확인하세요.",
        ),
        ("run download first", "저장된 데이터가 없습니다. '데이터 갱신' 또는 '학습 시작'을 먼저 실행하세요."),
        ("One market is stale", "일부 종목 데이터가 오래됐습니다. '데이터 갱신'을 먼저 실행하세요."),
        (
            "fewer than minimum_days",
            "이력이 짧은 종목이 있습니다. '최소 이력(일)'을 줄이거나 종목 구성을 확인하세요.",
        ),
        (
            "Insufficient history in a time split",
            "검증/테스트 기간에 비해 데이터가 부족합니다. 기간을 줄여 보세요.",
        ),
        ("OutOfMemoryError", "메모리가 부족합니다. 배치 크기나 LSTM 크기를 줄여 보세요."),
        ("No completed training epoch", "학습이 한 epoch도 끝나지 않았습니다. 로그의 첫 오류를 확인하세요."),
        ("Timeout", "네트워크 응답이 늦습니다. 잠시 후 다시 실행하면 이어서 받습니다."),
        ("ConnectionError", "네트워크 연결을 확인한 뒤 다시 실행하면 이어서 받습니다."),
    ]
    lowered = log_text.lower()
    for needle, hint in rules:
        if needle.lower() in lowered:
            return hint
    return None


# ----------------------------------------------------------------------------- progress and results


def progress_text(path):
    try:
        p = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    history = p.get("history") or []
    last = history[-1] if history else {}
    members, epochs = max(p.get("members", 1), 1), max(p.get("epochs", 1), 1)
    member = last.get("member", p.get("member", 1))
    epoch = last.get("epoch", 0)
    fraction = ((member - 1) * epochs + epoch) / (members * epochs)
    text = f"모델 {member}/{members} · epoch {epoch}/{epochs}"
    if "validation_mae_bps" in last:
        text += f" · 검증 MAE {last['validation_mae_bps']:.2f} bps"
    return min(fraction, 1.0), text


def remaining_epochs(progress):
    history = progress.get("history") or []
    members, epochs = max(progress.get("members", 1), 1), max(progress.get("epochs", 1), 1)
    last = history[-1] if history else {"member": 1, "epoch": 0}
    return max(0, (members - last.get("member", 1)) * epochs + epochs - last.get("epoch", 0))


def eta_text(intervals, remaining):
    """Upper-bound remaining time from observed seconds per epoch (early stopping may end sooner)."""
    if not intervals or remaining <= 0:
        return "—"
    seconds = sum(intervals[-5:]) / len(intervals[-5:]) * remaining
    if seconds < 90:
        return f"최대 약 {seconds:.0f}초"
    if seconds < 5400:
        return f"최대 약 {seconds / 60:.0f}분"
    return f"최대 약 {seconds / 3600:.1f}시간"


def training_series(history):
    """Per-member learning curves: {"val": [...], "loss": [...]}, each {name, points, color}."""
    members = {}
    for h in history or []:
        members.setdefault(int(h.get("member", 1)), []).append(h)
    out = {"val": [], "loss": []}
    for i, (member, rows) in enumerate(sorted(members.items())):
        color = SERIES[i % len(SERIES)]
        name = f"모델 {member}"
        val = [(r["epoch"], r["validation_mae_bps"]) for r in rows if _finite(r.get("validation_mae_bps"))]
        loss = [(r["epoch"], r["train_loss"]) for r in rows if _finite(r.get("train_loss"))]
        out["val"].append({"name": name, "points": val, "color": color})
        out["loss"].append({"name": name, "points": loss, "color": color})
    return out


def best_mae(history):
    values = [h["validation_mae_bps"] for h in history or [] if _finite(h.get("validation_mae_bps"))]
    return min(values) if values else None


def load_equity(path, max_points=900):
    """[(unix_seconds, equity)] downsampled for drawing; [] when missing."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [
                (datetime.fromisoformat(r["time"]).timestamp(), float(r["equity_krw"]))
                for r in csv.DictReader(f)
                if r.get("time") and r.get("equity_krw")
            ]
    except (OSError, ValueError, KeyError):
        return []
    if len(rows) > max_points:
        step = math.ceil(len(rows) / max_points)
        rows = rows[::step] + ([rows[-1]] if (len(rows) - 1) % step else [])
    return rows


def report_cards(report):
    """[(title, value, subtitle, tone)] for the results grid; tone in good/bad/neutral."""
    if not report:
        return []
    test, val = report.get("test", {}), report.get("validation", {})
    forecast = report.get("forecast", {}).get("test", {})
    hold = report.get("buy_hold_test", {}).get("return_pct")

    def pct(x, sign=True):
        return "—" if x is None else (f"{x:+.2f}%" if sign else f"{x:.2f}%")

    def tone(x):
        if x is None or abs(x) < 1e-9:
            return "neutral"
        return "good" if x > 0 else "bad"

    ret = test.get("return_pct")
    trades = test.get("trades", 0)
    beat = None if ret is None or hold is None else ret - hold
    approved = bool(report.get("approved_for_live"))
    ic = forecast.get("ic")
    acc = forecast.get("direction_accuracy")
    return [
        ("테스트 수익률", pct(ret), f"단순 보유 {pct(hold)}", tone(ret)),
        ("보유 대비", pct(beat), "테스트 구간 초과 성과", tone(beat)),
        ("최대 낙폭", pct(test.get("max_drawdown_pct"), False), "테스트 구간", "neutral"),
        (
            "Sharpe (일간)",
            "—" if test.get("sharpe_daily") is None or not trades else f"{test['sharpe_daily']:.2f}",
            "연율화",
            tone(test.get("sharpe_daily") if trades else None),
        ),
        ("거래 수", f"{test.get('trades', 0):,}", f"승률 {test.get('win_rate', 0) * 100:.1f}%", "neutral"),
        (
            "Profit Factor",
            "—" if test.get("profit_factor") is None or not trades else f"{test['profit_factor']:.2f}",
            "총이익 / 총손실",
            tone(None if test.get("profit_factor") is None or not trades else test["profit_factor"] - 1),
        ),
        (
            "방향 정확도",
            "—" if acc is None else f"{acc * 100:.1f}%",
            "IC —" if ic is None else f"IC {ic:+.3f}",
            tone(None if acc is None else acc - 0.5),
        ),
        ("검증 수익률", pct(val.get("return_pct")), "매수 기준 선택 구간", tone(val.get("return_pct"))),
        (
            "실거래 승인",
            "승인" if approved else "미승인",
            f"매수 기준 {report.get('threshold_bps', 0):g} bps",
            "good" if approved else "bad",
        ),
    ]


def nice_ticks(lo, hi, count=5):
    """Round axis ticks covering [lo, hi]."""
    if not (_finite(lo) and _finite(hi)):
        return [0.0, 1.0]
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-12:
        pad = abs(lo) * 0.05 or 1.0
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.floor(lo / step) * step
    ticks, value = [], start
    while value < hi + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks if len(ticks) >= 2 else [start, start + step]


def x_ticks(x0, x1, integer=False):
    """[(x, anchor)] labels: whole numbers for epochs, five even steps otherwise."""
    if integer:
        values = [t for t in nice_ticks(x0, x1, 6) if x0 - 1e-9 <= t <= x1 + 1e-9 and float(t).is_integer()]
        values = values or [x0]
    else:
        values = [x0 + (x1 - x0) * i / 4 for i in range(5)]
    out = []
    for i, x in enumerate(values):
        edge_left = abs(x - x0) < 1e-9 and len(values) > 1
        edge_right = abs(x - x1) < 1e-9 and len(values) > 1 and i > 0
        out.append((x, "nw" if edge_left else ("ne" if edge_right else "n")))
    return out


def fmt_epoch(v):
    return f"{v:.0f}"


def fmt_number(v):
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:.0f}k"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.3g}"


def fmt_krw(v):
    return f"{v / 10000:,.1f}만" if abs(v) < 1e8 else f"{v / 1e8:,.2f}억"


def fmt_date(v):
    return datetime.fromtimestamp(v, UTC).strftime("%m-%d")


def blend(color, background, amount):
    a = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(background[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * amount + y * (1 - amount)):02x}" for x, y in zip(a, b))


def _finite(x):
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


# ----------------------------------------------------------------------------- widgets


class LineChart:
    """Responsive line chart on a Canvas with grid, axes, legend and optional area fill."""

    def __init__(
        self,
        parent,
        title,
        x_fmt=fmt_number,
        y_fmt=fmt_number,
        empty="학습을 시작하면 그래프가 표시됩니다",
        x_int=False,
    ):
        import tkinter as tk

        self.canvas = tk.Canvas(parent, bg=CARD, highlightthickness=0, height=200)
        self.title, self.x_fmt, self.y_fmt, self.empty, self.x_int = title, x_fmt, y_fmt, empty, x_int
        self.series, self.refs = [], []
        self.canvas.bind("<Configure>", lambda _: self.draw())

    def set(self, series, refs=()):
        self.series, self.refs = [s for s in series if s["points"]], list(refs)
        self.draw()

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 80 or h < 80:
            return
        c.create_text(14, 12, text=self.title, anchor="nw", fill=TEXT, font=FONTS["h3"])
        points = [p for s in self.series for p in s["points"]]
        if not points:
            c.create_text(w / 2, h / 2 + 8, text=self.empty, fill=MUTED, font=FONTS["small"])
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points] + [y for y, _ in self.refs]
        x0, x1 = min(xs), max(xs)
        if x1 - x0 < 1e-12:
            x0, x1 = x0 - 1, x1 + 1
        ticks = nice_ticks(min(ys), max(ys))
        y0, y1 = ticks[0], ticks[-1]
        left, right, top, bottom = 62, w - 18, 44, h - 30

        def px(x):
            return left + (x - x0) / (x1 - x0) * (right - left)

        def py(y):
            return bottom - (y - y0) / (y1 - y0) * (bottom - top)

        for t in ticks:
            y = py(t)
            c.create_line(left, y, right, y, fill=BORDER, dash=(2, 4))
            c.create_text(left - 8, y, text=self.y_fmt(t), anchor="e", fill=MUTED, font=FONTS["tiny"])
        for x, anchor in x_ticks(x0, x1, self.x_int):
            c.create_text(
                px(x), bottom + 8, text=self.x_fmt(x), anchor=anchor, fill=MUTED, font=FONTS["tiny"]
            )
        for value, label in self.refs:
            y = py(value)
            c.create_line(left, y, right, y, fill=MUTED, dash=(6, 4))
            c.create_text(right - 4, y - 4, text=label, anchor="se", fill=MUTED, font=FONTS["tiny"])
        for s in self.series:
            coords = [v for x, y in s["points"] for v in (px(x), py(y))]
            if s.get("fill") and len(s["points"]) > 1:
                c.create_polygon(
                    px(s["points"][0][0]),
                    bottom,
                    *coords,
                    px(s["points"][-1][0]),
                    bottom,
                    fill=blend(s["color"], CARD, 0.18),
                    outline="",
                )
            if len(coords) >= 4:
                c.create_line(*coords, fill=s["color"], width=2.2, capstyle="round", joinstyle="round")
            lx, ly = coords[-2], coords[-1]
            c.create_oval(lx - 3.5, ly - 3.5, lx + 3.5, ly + 3.5, fill=s["color"], outline=CARD, width=2)
        x = right
        for s in reversed(self.series):
            label_id = c.create_text(x, 14, text=s["name"], anchor="ne", fill=MUTED, font=FONTS["tiny"])
            bbox = c.bbox(label_id)
            c.create_rectangle(bbox[0] - 14, 17, bbox[0] - 6, 25, fill=s["color"], outline="")
            x = bbox[0] - 22


FONTS = {}


def setup_fonts(root):
    import tkinter.font as tkfont

    family = (
        "Malgun Gothic" if sys.platform == "win32" else tkfont.nametofont("TkDefaultFont").actual()["family"]
    )
    mono = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"
    FONTS.update(
        title=(family, 17, "bold"),
        h2=(family, 12, "bold"),
        h3=(family, 10, "bold"),
        body=(family, 10),
        small=(family, 9),
        tiny=(family, 8),
        metric=(family, 18, "bold"),
        mono=(mono, 9),
    )
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        tkfont.nametofont(name).configure(family=family, size=10)


def setup_style(root):
    from tkinter import ttk

    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=BG)
    style.configure(
        ".",
        background=PANEL,
        foreground=TEXT,
        fieldbackground=FIELD,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        troughcolor=FIELD,
        focuscolor=ACCENT,
        selectbackground=ACCENT,
        selectforeground="#ffffff",
        font=FONTS["body"],
    )
    style.configure("TFrame", background=PANEL)
    style.configure("Bg.TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD)
    style.configure("TLabel", background=PANEL, foreground=TEXT)
    style.configure("Card.TLabel", background=CARD, foreground=TEXT)
    style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=FONTS["small"])
    style.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=FONTS["small"])
    style.configure("H2.TLabel", background=PANEL, foreground=TEXT, font=FONTS["h2"])
    style.configure("CardH2.TLabel", background=CARD, foreground=TEXT, font=FONTS["h2"])
    style.configure(
        "TButton", background=CARD, foreground=TEXT, bordercolor=BORDER, padding=(12, 7), focusthickness=0
    )
    style.map(
        "TButton",
        background=[("disabled", PANEL), ("pressed", BORDER), ("active", "#262c36")],
        foreground=[("disabled", "#5c636c")],
    )
    style.configure(
        "Accent.TButton",
        background=ACCENT,
        foreground="#ffffff",
        bordercolor=ACCENT,
        font=FONTS["h3"],
        padding=(16, 8),
    )
    style.map(
        "Accent.TButton",
        background=[("disabled", "#1f3b63"), ("pressed", "#1f6feb"), ("active", ACCENT_HI)],
        foreground=[("disabled", "#8aa4c8")],
    )
    style.configure("Danger.TButton", background=CARD, foreground=BAD, bordercolor="#5a2a2a")
    style.map(
        "Danger.TButton",
        background=[("disabled", PANEL), ("active", "#3a1f22")],
        foreground=[("disabled", "#5c636c")],
    )
    style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT, padding=5)
    style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])
    style.configure(
        "TCombobox", fieldbackground=FIELD, background=CARD, foreground=TEXT, arrowcolor=TEXT, padding=4
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", FIELD)],
        foreground=[("readonly", TEXT)],
        bordercolor=[("focus", ACCENT)],
    )
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.font", FONTS["body"])
    for name, bg in (("TCheckbutton", PANEL), ("Card.TCheckbutton", CARD)):
        style.configure(
            name,
            background=bg,
            foreground=TEXT,
            indicatorbackground=FIELD,
            indicatorforeground=ACCENT_HI,
            indicatorcolor=FIELD,
        )
        style.map(name, background=[("active", bg)], indicatorcolor=[("selected", ACCENT)])
    style.configure(
        "TNotebook",
        background=PANEL,
        borderwidth=1,
        bordercolor=BORDER,
        lightcolor=CARD,
        darkcolor=CARD,
        tabmargins=(0, 4, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=PANEL,
        foreground=MUTED,
        padding=(14, 7),
        borderwidth=0,
        bordercolor=BORDER,
        lightcolor=PANEL,
        darkcolor=PANEL,
        focuscolor=PANEL,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", CARD), ("active", "#1f252e")],
        foreground=[("selected", TEXT), ("active", TEXT)],
        lightcolor=[("selected", ACCENT)],
        expand=[("selected", (0, 0, 0, 0))],
    )
    style.configure("Card.TNotebook", background=CARD)
    style.configure("Card.TNotebook.Tab", background=CARD, foreground=MUTED)
    style.map("Card.TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", TEXT)])
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=FIELD,
        background=ACCENT,
        bordercolor=FIELD,
        lightcolor=ACCENT,
        darkcolor=ACCENT,
        thickness=10,
    )
    style.configure(
        "Vertical.TScrollbar", background=CARD, troughcolor=PANEL, arrowcolor=MUTED, bordercolor=PANEL
    )


class Chip:
    """Rounded-looking status pill."""

    def __init__(self, parent, text="", tone="neutral"):
        import tkinter as tk

        self.label = tk.Label(parent, text=text, padx=10, pady=3, font=FONTS["small"], bd=0)
        self.set(text, tone)

    def set(self, text, tone="neutral"):
        colors = {"good": GOOD, "bad": BAD, "warn": WARN, "accent": ACCENT_HI, "neutral": MUTED}
        color = colors.get(tone, MUTED)
        self.label.configure(text=f"●  {text}", fg=color, bg=blend(color, BG, 0.16))


def ask_choice(root, title, message, options):
    """Modal dialog with one button per option. Returns the chosen index or None."""
    import tkinter as tk
    from tkinter import ttk

    win = tk.Toplevel(root, bg=PANEL)
    win.title(title)
    win.transient(root)
    win.resizable(False, False)
    result = {"value": None}
    frame = ttk.Frame(win, padding=22)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=title, style="H2.TLabel").pack(anchor="w")
    ttk.Label(frame, text=message, wraplength=520, justify="left").pack(anchor="w", pady=(10, 18))
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")

    def choose(i):
        result["value"] = i
        win.destroy()

    ttk.Button(buttons, text="취소", command=win.destroy).pack(side="right")
    for i, label in reversed(list(enumerate(options))):
        ttk.Button(
            buttons,
            text=label,
            style="Accent.TButton" if i == 0 else "TButton",
            command=lambda i=i: choose(i),
        ).pack(side="right", padx=(0, 8))
    win.update_idletasks()
    x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
    y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 3
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    win.grab_set()
    root.wait_window(win)
    return result["value"]


# ----------------------------------------------------------------------------- application


class App:
    def __init__(self, root, config_path):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.root = tk, ttk, root
        self.config_path = Path(config_path).resolve()
        self.project = self.config_path.parent
        self.artifacts = self.project / "artifacts"
        self.process, self.events = None, queue.Queue()
        self.vars, self.saved, self.job_started, self.stopping = {}, {}, 0.0, False
        self.epoch_marks, self.last_epoch_key, self.current_job = [], None, None
        setup_fonts(root)
        setup_style(root)
        root.title("Upbit LSTM Trainer")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1440, sw - 60)}x{min(900, sh - 80)}")
        root.minsize(1080, 680)
        self.build_header()
        body = ttk.Frame(root, style="Bg.TFrame", padding=(14, 0, 14, 14))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self.build_settings(body)
        self.build_main(body)
        self.load()
        self.refresh_results()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(200, self.poll)
        threading.Thread(target=self.detect_environment, daemon=True).start()

    # ---- layout

    def build_header(self):
        tk, ttk = self.tk, self.ttk
        header = tk.Frame(self.root, bg=BG, padx=20, pady=14)
        header.pack(fill="x")
        mark = tk.Canvas(header, width=34, height=34, bg=BG, highlightthickness=0)
        mark.create_rectangle(2, 2, 32, 32, fill=blend(ACCENT, BG, 0.25), outline="")
        mark.create_line(7, 24, 13, 17, 19, 21, 27, 9, fill=ACCENT_HI, width=2.5, capstyle="round")
        mark.pack(side="left", padx=(0, 12))
        titles = tk.Frame(header, bg=BG)
        titles.pack(side="left")
        tk.Label(titles, text="Upbit LSTM Trainer", bg=BG, fg=TEXT, font=FONTS["title"]).pack(anchor="w")
        tk.Label(
            titles,
            text="설정 → 학습 → 백테스트 → 서버 배포 파일까지 한 곳에서",
            bg=BG,
            fg=MUTED,
            font=FONTS["small"],
        ).pack(anchor="w")
        right = tk.Frame(header, bg=BG)
        right.pack(side="right")
        self.status_chip = Chip(right, "대기 중")
        self.status_chip.label.pack(side="right")
        self.env_chip = Chip(right, "환경 확인 중…")
        self.env_chip.label.pack(side="right", padx=8)
        ttk.Frame(self.root, style="Bg.TFrame", height=1).pack(fill="x")

    def build_settings(self, body):
        tk, ttk = self.tk, self.ttk
        panel = ttk.Frame(body, padding=16)
        panel.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        ttk.Label(panel, text="프리셋", style="H2.TLabel").pack(anchor="w")
        presets = ttk.Frame(panel)
        presets.pack(fill="x", pady=(8, 14))
        self.preset_cards = {}
        for i, (name, (sub, _)) in enumerate(PRESETS.items()):
            card = tk.Frame(
                presets,
                bg=CARD,
                highlightthickness=1,
                highlightbackground=BORDER,
                cursor="hand2",
                padx=10,
                pady=8,
            )
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            presets.columnconfigure(i, weight=1)
            title = tk.Label(card, text=name, bg=CARD, fg=TEXT, font=FONTS["h3"], cursor="hand2")
            title.pack(anchor="w")
            subtitle = tk.Label(card, text=sub, bg=CARD, fg=MUTED, font=FONTS["tiny"], cursor="hand2")
            subtitle.pack(anchor="w")
            for widget in (card, title, subtitle):
                widget.bind("<Button-1>", lambda _e, n=name: self.apply_preset(n))
                widget.bind("<Enter>", lambda _e, c=card: c.configure(highlightbackground=ACCENT))
                widget.bind("<Leave>", lambda _e, c=card, n=name: self.style_preset(n))
            self.preset_cards[name] = card
        head = ttk.Frame(panel)
        head.pack(fill="x")
        ttk.Label(head, text="설정", style="H2.TLabel").pack(side="left")
        self.dirty_label = ttk.Label(head, text="", style="Muted.TLabel")
        self.dirty_label.pack(side="right")
        tabs = ttk.Notebook(panel)
        frames = {}
        for section, key, label, kind, help_text in FIELDS:
            title = TAB_TITLES[section]
            if title not in frames:
                frame = ttk.Frame(tabs, style="Card.TFrame", padding=(14, 12))
                frame.columnconfigure(1, weight=1)
                tabs.add(frame, text=title)
                frames[title] = frame
            frame = frames[title]
            row = frame.grid_size()[1]
            ttk.Label(frame, text=label, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=4, padx=(0, 12)
            )
            if kind == "bool":
                var = tk.BooleanVar()
                widget = ttk.Checkbutton(frame, variable=var, style="Card.TCheckbutton")
            elif isinstance(kind, tuple):
                var = tk.StringVar()
                widget = ttk.Combobox(
                    frame, textvariable=var, values=[str(k) for k in kind], state="readonly", width=14
                )
            else:
                var = tk.StringVar()
                widget = ttk.Entry(frame, textvariable=var, width=16)
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            for event in ("<FocusIn>", "<Enter>"):
                widget.bind(event, lambda _e, t=label, h=help_text: self.show_help(t, h))
            var.trace_add("write", lambda *_: self.update_dirty())
            self.vars[(section, key)] = (var, kind)
        help_box = tk.Frame(panel, bg=blend(ACCENT, PANEL, 0.08), padx=12, pady=10)

        self.help_title = tk.Label(help_box, text="도움말", bg=help_box["bg"], fg=ACCENT_HI, font=FONTS["h3"])
        self.help_title.pack(anchor="w")
        self.help_text = tk.Label(
            help_box,
            text="항목에 마우스를 올리면 설명이 표시됩니다.",
            bg=help_box["bg"],
            fg=TEXT,
            font=FONTS["small"],
            wraplength=330,
            justify="left",
            height=2,
            anchor="nw",
        )
        self.help_text.pack(anchor="w", fill="x")
        actions = ttk.Frame(panel)
        ttk.Button(actions, text="설정 저장", style="Accent.TButton", command=self.save).pack(side="left")
        ttk.Button(actions, text="되돌리기", command=self.load).pack(side="left", padx=8)
        actions.pack(side="bottom", fill="x")
        help_box.pack(side="bottom", fill="x", pady=12)
        tabs.pack(fill="both", expand=True, pady=(8, 0))

    def build_main(self, body):
        tk, ttk = self.tk, self.ttk
        main = ttk.Frame(body, style="Bg.TFrame")
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        run = ttk.Frame(main, padding=(16, 14))
        run.grid(row=0, column=0, sticky="ew")
        self.run_buttons = []
        for text, job, style in (
            ("▶  학습 시작", "train", "Accent.TButton"),
            ("이어서 학습", "resume", "TButton"),
            ("데이터 갱신", "download", "TButton"),
            ("HMM 비교", "hmm", "TButton"),
            ("환경 확인", "doctor", "TButton"),
        ):
            b = ttk.Button(run, text=text, style=style, command=lambda job=job: self.start(job))
            b.pack(side="left", padx=(0, 8))
            self.run_buttons.append(b)
        self.stop_button = ttk.Button(
            run, text="■  중지", style="Danger.TButton", command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="right")
        self.allow_cpu = tk.BooleanVar(value=False)
        ttk.Checkbutton(run, text="CPU 모드", variable=self.allow_cpu).pack(side="right", padx=14)
        prog = ttk.Frame(main, padding=(16, 12))
        prog.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        prog.columnconfigure(0, weight=1)
        top = ttk.Frame(prog)
        top.grid(row=0, column=0, sticky="ew")
        self.phase_label = ttk.Label(top, text="준비됨", style="H2.TLabel")
        self.phase_label.pack(side="left")
        self.percent_label = ttk.Label(top, text="", style="Muted.TLabel")
        self.percent_label.pack(side="right")
        self.progress = ttk.Progressbar(prog, maximum=1.0)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(8, 10))
        stats = ttk.Frame(prog)
        stats.grid(row=2, column=0, sticky="ew")
        self.stats = {}
        for i, (key, title) in enumerate(
            (("model", "모델"), ("epoch", "Epoch"), ("best", "최고 검증 MAE"), ("eta", "남은 시간"))
        ):
            stats.columnconfigure(i, weight=1)
            cell = ttk.Frame(stats)
            cell.grid(row=0, column=i, sticky="w")
            ttk.Label(cell, text=title, style="Muted.TLabel").pack(anchor="w")
            value = ttk.Label(cell, text="—", font=FONTS["h2"])
            value.pack(anchor="w")
            self.stats[key] = value
        views = ttk.Notebook(main)
        views.grid(row=2, column=0, sticky="nsew")
        curves = ttk.Frame(views, style="Card.TFrame", padding=10)
        curves.columnconfigure((0, 1), weight=1, uniform="c")
        curves.rowconfigure(0, weight=1)
        self.val_chart = LineChart(curves, "검증 MAE (bps) · 낮을수록 좋음", x_fmt=fmt_epoch, x_int=True)
        self.val_chart.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.loss_chart = LineChart(curves, "학습 손실 (Huber)", x_fmt=fmt_epoch, x_int=True)
        self.loss_chart.canvas.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        views.add(curves, text="학습 곡선")
        equity = ttk.Frame(views, style="Card.TFrame", padding=10)
        equity.columnconfigure(0, weight=1)
        equity.rowconfigure(0, weight=1)
        self.equity_chart = LineChart(
            equity,
            "백테스트 자산 (KRW)",
            x_fmt=fmt_date,
            y_fmt=fmt_krw,
            empty="학습이 끝나면 검증·테스트 구간 자산 곡선이 표시됩니다",
        )
        self.equity_chart.canvas.grid(row=0, column=0, sticky="nsew")
        views.add(equity, text="백테스트 자산")
        results = ttk.Frame(views, style="Card.TFrame", padding=14)
        self.cards_frame = ttk.Frame(results, style="Card.TFrame")
        self.cards_frame.pack(fill="both", expand=True)
        bottom = ttk.Frame(results, style="Card.TFrame")
        bottom.pack(fill="x", pady=(10, 0))
        self.report_note = ttk.Label(bottom, text="", style="CardMuted.TLabel")
        self.report_note.pack(side="left")
        ttk.Button(bottom, text="전체 보고서 열기 ↗", command=self.open_report).pack(side="right")
        views.add(results, text="결과 요약")
        logs = ttk.Frame(views, style="Card.TFrame", padding=10)
        self.log = tk.Text(
            logs,
            wrap="word",
            font=FONTS["mono"],
            bg=FIELD,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        scroll = ttk.Scrollbar(logs, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        for tag, color in (
            ("error", BAD),
            ("warn", WARN),
            ("epoch", ACCENT_HI),
            ("cmd", MUTED),
            ("ok", GOOD),
        ):
            self.log.tag_configure(tag, foreground=color)
        views.add(logs, text="로그")
        self.views, self.logs_tab = views, logs

    # ---- settings

    def show_help(self, title, text):
        self.help_title.configure(text=title)
        self.help_text.configure(text=text)

    def current_values(self):
        return {k: str(var.get()) for k, (var, _) in self.vars.items()}

    def update_dirty(self):
        dirty = bool(self.saved) and self.current_values() != self.saved
        self.dirty_label.configure(
            text="● 저장되지 않은 변경" if dirty else "저장됨", foreground=WARN if dirty else MUTED
        )
        for name in self.preset_cards:
            self.style_preset(name)

    def style_preset(self, name):
        _, values = PRESETS[name]
        active = all(str(self.vars[k][0].get()) == str(v) for k, v in values.items())
        self.preset_cards[name].configure(highlightbackground=ACCENT if active else BORDER)

    def load(self):
        with open(self.config_path, "rb") as f:
            cfg = tomllib.load(f)
        for (section, key), (var, kind) in self.vars.items():
            value = cfg.get(section, {}).get(key, DEFAULTS.get(key, ""))
            var.set(display_value(kind, value))
        self.saved = self.current_values()
        self.update_dirty()

    def edited_text(self):
        text = self.config_path.read_text(encoding="utf-8")
        for (section, key), (var, kind) in self.vars.items():
            try:
                value = parse_field(kind, str(var.get()))
            except ValueError as exc:
                raise ValueError(f"'{LABELS[(section, key)]}' 값이 올바르지 않습니다: {var.get()}") from exc
            text = set_toml(text, section, key, value)
        return text, validate_text(text)

    def write_config(self, text):
        if text == self.config_path.read_text(encoding="utf-8"):
            self.saved = self.current_values()
            self.update_dirty()
            return
        shutil.copy2(self.config_path, self.config_path.with_suffix(".toml.bak"))
        self.config_path.write_text(text, encoding="utf-8")
        self.load()

    def save(self):
        from tkinter import messagebox

        try:
            text, _ = self.edited_text()
        except Exception as exc:  # shown to the user
            messagebox.showerror("설정 오류", str(exc))
            return False
        self.write_config(text)
        return True

    def apply_preset(self, name):
        for (section, key), value in PRESETS[name][1].items():
            var, kind = self.vars[(section, key)]
            var.set(display_value(kind, value))
        self.show_help(
            f"프리셋: {name}", f"{PRESETS[name][0]}. '설정 저장' 또는 '학습 시작'을 누르면 반영됩니다."
        )

    # ---- jobs

    def start(self, job):
        from tkinter import messagebox

        if self.process:
            return
        if job == "resume":
            return self.start_resume()
        if job in ("train", "fresh", "package"):
            try:
                text, cfg = self.edited_text()
            except Exception as exc:
                messagebox.showerror("설정 오류", str(exc))
                return None
            plan = resume_plan(self.artifacts, cfg) if job == "train" else {"kind": "none"}
            if plan["kind"] in ("mismatch", "legacy") and not messagebox.askyesno(
                "처음부터 학습",
                "남아 있는 체크포인트는 현재 설정과 달라 이어서 할 수 없습니다.\n"
                "처음부터 학습하고, 이전 체크포인트는 artifacts/old-checkpoints 폴더로 옮겨 보관합니다.\n계속할까요?",
            ):
                return None
            if plan["kind"] == "ok":
                choice = ask_choice(
                    self.root,
                    "이전 학습이 남아 있습니다",
                    "같은 설정으로 학습하던 체크포인트가 있습니다.\n\n"
                    "이어서 학습하면 완료된 epoch와 모델을 재사용해 시간을 아낍니다. "
                    "처음부터 하면 데이터를 새로 받고 이전 체크포인트는 "
                    "artifacts/old-checkpoints 폴더로 옮겨 보관합니다.",
                    ["이어서 학습", "처음부터 새로 학습"],
                )
                if choice is None:
                    return None
                if choice == 0:
                    self.write_config(text)
                    return self.run_job("resume")
            elif (
                plan["kind"] == "none"
                and job == "train"
                and (self.artifacts / "manifest.json").exists()
                and not messagebox.askyesno(
                    "확인",
                    "기존 모델과 보고서를 새 결과로 덮어씁니다. 필요하면 artifacts 폴더를 먼저 보관하세요.\n계속할까요?",
                )
            ):
                return None
            self.write_config(text)
        return self.run_job(job)

    def start_resume(self):
        from tkinter import messagebox

        try:
            text, cfg = self.edited_text()
        except Exception as exc:
            messagebox.showerror("설정 오류", str(exc))
            return None
        plan = resume_plan(self.artifacts, cfg)
        if plan["kind"] == "ok":
            self.write_config(text)  # e.g. more epochs or ensemble members: still resumable
            return self.run_job("resume")
        if plan["kind"] == "none":
            choice = ask_choice(
                self.root,
                "이어서 할 학습이 없습니다",
                "저장된 체크포인트가 없습니다. 저장된 데이터로 새로 학습할까요?",
                ["새로 학습 (데이터 유지)"],
            )
            if choice == 0:
                self.write_config(text)
                return self.run_job("fresh")
            return None
        return self.offer_recovery(text, plan["changes"])

    def offer_recovery(self, text, found):
        reasons = "\n".join(f"  •  {describe_change(c)}" for c in found[:8])
        config_only = [c for c in found if c[0] in ("model", "training")]
        options = (
            ["체크포인트 설정으로 되돌리고 이어서 학습"]
            if config_only and len(config_only) == len(found)
            else []
        )
        options.append("현재 설정으로 처음부터 학습 (데이터 유지)")
        choice = ask_choice(
            self.root,
            "이어서 학습할 수 없습니다",
            f"체크포인트를 만들 때와 다음이 다릅니다.\n\n{reasons}\n\n"
            "처음부터 학습하면 이전 체크포인트는 artifacts/old-checkpoints 폴더로 옮겨 보관합니다.",
            options,
        )
        if choice is None:
            return None
        if options[choice].startswith("체크포인트"):
            self.write_config(restore_checkpoint_settings(text, config_only))
            return self.run_job("resume")
        self.write_config(text)
        return self.run_job("fresh")

    def run_job(self, job):
        steps = job_steps(
            job, self.config_path, self.allow_cpu.get(), (self.artifacts / "manifest.json").exists()
        )
        for b in self.run_buttons:
            b.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.log.delete("1.0", "end")
        self.stopping, self.current_job = False, job
        self.job_started = time.time()
        self.epoch_marks, self.last_epoch_key = [], None
        self.progress.configure(value=0)
        names = {
            "train": "데이터 수집 + 학습",
            "fresh": "새로 학습",
            "resume": "이어서 학습",
            "download": "데이터 갱신",
            "hmm": "HMM 비교",
            "package": "패키지",
            "doctor": "환경 확인",
        }
        self.phase_label.configure(text=names[job])
        self.status_chip.set("실행 중", "accent")
        if job in ("download", "hmm", "doctor"):
            self.views.select(self.logs_tab)
        threading.Thread(target=self.worker, args=(steps,), daemon=True).start()
        return True

    def worker(self, steps):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for argv in steps:
            step = argv[6]
            self.events.put(("step", step))
            self.events.put(("log", "$ " + " ".join(argv[6:]) + "\n"))
            try:
                self.process = subprocess.Popen(
                    argv,
                    cwd=self.project,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=flags,
                )
            except OSError as exc:
                self.events.put(("done", ("failed", f"실행할 수 없습니다: {exc}")))
                return
            for line in self.process.stdout:
                self.events.put(("log", line))
            code = self.process.wait()
            if self.stopping:
                self.events.put(("done", ("stopped", "중지됨 · '이어서 학습'으로 이어갈 수 있습니다")))
                return
            if code == EXIT_MISMATCH and step == "train":
                self.events.put(("done", ("mismatch", "이어서 학습할 수 없습니다")))
                return
            if code != 0:
                self.events.put(("done", ("failed", f"실패 (종료 코드 {code})")))
                return
        self.events.put(("done", ("ok", "완료")))

    def append_log(self, line):
        lowered = line.lower()
        tag = None
        if line.startswith("$ "):
            tag = "cmd"
        elif (
            "traceback" in lowered
            or " error" in lowered
            or lowered.startswith(("error", "valueerror", "runtimeerror"))
        ):
            tag = "error"
        elif "warning" in lowered:
            tag = "warn"
        elif "epoch" in lowered and "validation" in lowered:
            tag = "epoch"
        elif "complete" in lowered or "done" in lowered:
            tag = "ok"
        self.log.insert("end", line, tag)
        self.log.see("end")

    def finish(self, status, message):
        from tkinter import messagebox

        self.process = None
        for b in self.run_buttons:
            b.configure(state="normal")
        self.stop_button.configure(state="disabled")
        tones = {"ok": "good", "stopped": "warn", "failed": "bad", "mismatch": "warn"}
        self.status_chip.set(
            {"ok": "완료", "stopped": "중지됨", "failed": "실패", "mismatch": "확인 필요"}[status],
            tones[status],
        )
        self.phase_label.configure(text=message)
        self.append_log(f"\n== {message} ==\n")
        self.refresh_results()
        if status == "ok" and self.current_job in ("train", "fresh", "resume"):
            self.progress.configure(value=1.0)
            self.percent_label.configure(text="100%")
            self.views.select(2)
        elif status == "failed":
            self.views.select(self.logs_tab)
            hint = failure_hint(self.log.get("1.0", "end"))
            messagebox.showerror(
                "작업 실패",
                (hint or "로그 탭의 오류 메시지를 확인하세요.")
                + "\n\n학습 도중 실패했다면 '이어서 학습'으로 완료된 epoch부터 이어갈 수 있습니다.",
            )
        elif status == "mismatch":
            text = self.config_path.read_text(encoding="utf-8")
            self.offer_recovery(text, read_mismatch(self.artifacts) or [(LEGACY, None, None, None)])

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                elif kind == "env":
                    self.env_chip.set(*payload)
                    if payload[1] == "warn" and payload[0].startswith("CPU"):
                        self.allow_cpu.set(True)
                elif kind == "step":
                    labels = {
                        "download": "데이터 수집 중",
                        "train": "학습 중",
                        "package": "배포 파일 생성 중",
                        "train-hmm": "HMM 학습 중",
                        "doctor": "환경 확인 중",
                    }
                    self.phase_label.configure(text=labels.get(payload, payload))
                else:
                    self.finish(*payload)
        except queue.Empty:
            pass
        if self.process:
            self.update_progress()
        self.root.after(700, self.poll)

    def update_progress(self):
        path = self.artifacts / "progress.json"
        try:
            if path.stat().st_mtime < self.job_started - 1:
                return
            progress = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        history = progress.get("history") or []
        fraction, _ = progress_text(path)
        if fraction is not None:
            self.progress.configure(value=fraction)
            self.percent_label.configure(text=f"{fraction * 100:.0f}%")
        last = history[-1] if history else {}
        key = (last.get("member"), last.get("epoch"))
        if key != self.last_epoch_key:
            # progress.json is written at the end of each epoch: time between updates = epoch time.
            self.epoch_marks = [*self.epoch_marks, time.time()][-20:]
            self.last_epoch_key = key
            self.stats["model"].configure(text=f"{last.get('member', '—')} / {progress.get('members', '—')}")
            self.stats["epoch"].configure(text=f"{last.get('epoch', '—')} / {progress.get('epochs', '—')}")
            best = best_mae(history)
            self.stats["best"].configure(text="—" if best is None else f"{best:.2f} bps")
            intervals = [b - a for a, b in zip(self.epoch_marks, self.epoch_marks[1:])]
            self.stats["eta"].configure(text=eta_text(intervals, remaining_epochs(progress)))
            series = training_series(history)
            self.val_chart.set(series["val"])
            self.loss_chart.set(series["loss"])

    def refresh_results(self):
        try:
            history = json.loads((self.artifacts / "training_history.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            history = []
        series = training_series(history)
        self.val_chart.set(series["val"])
        self.loss_chart.set(series["loss"])
        if history and not self.process:
            best = best_mae(history)
            self.stats["best"].configure(text="—" if best is None else f"{best:.2f} bps")
        try:
            report = json.loads((self.artifacts / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = None
        equity = []
        val_points = load_equity(self.artifacts / "validation_equity.csv")
        test_points = load_equity(self.artifacts / "test_equity.csv")
        if val_points:
            equity.append({"name": "검증 구간", "points": val_points, "color": SERIES[1], "fill": True})
        if test_points:
            equity.append({"name": "테스트 구간", "points": test_points, "color": SERIES[0], "fill": True})
        start = (val_points or test_points or [(0, None)])[0][1]
        self.equity_chart.set(equity, [(start, "시작 자본")] if start else [])
        self.render_cards(report)

    def render_cards(self, report):
        tk = self.tk
        for child in self.cards_frame.winfo_children():
            child.destroy()
        cards = report_cards(report)
        if not cards:
            tk.Label(
                self.cards_frame,
                text="아직 결과가 없습니다. 학습이 끝나면 핵심 지표가 여기에 표시됩니다.",
                bg=CARD,
                fg=MUTED,
                font=FONTS["body"],
            ).pack(pady=40)
            self.report_note.configure(text="")
            return
        colors = {"good": GOOD, "bad": BAD, "neutral": TEXT}
        for i, (title, value, sub, tone) in enumerate(cards):
            cell = tk.Frame(
                self.cards_frame, bg=PANEL, padx=14, pady=12, highlightthickness=1, highlightbackground=BORDER
            )
            cell.grid(row=i // 3, column=i % 3, sticky="nsew", padx=5, pady=5)
            self.cards_frame.columnconfigure(i % 3, weight=1, uniform="cards")
            tk.Label(cell, text=title, bg=PANEL, fg=MUTED, font=FONTS["small"]).pack(anchor="w")
            tk.Label(cell, text=value, bg=PANEL, fg=colors[tone], font=FONTS["metric"]).pack(
                anchor="w", pady=(2, 0)
            )
            tk.Label(cell, text=sub, bg=PANEL, fg=MUTED, font=FONTS["tiny"]).pack(anchor="w")
        try:
            stamp = datetime.fromtimestamp((self.artifacts / "report.json").stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
        except OSError:
            stamp = "—"
        self.report_note.configure(
            text=f"artifacts/report.json · {stamp} · 과거 성과는 미래 수익을 보장하지 않습니다"
        )

    def detect_environment(self):
        argv = job_steps("doctor", self.config_path)[0]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            out = subprocess.run(
                argv,
                cwd=self.project,
                capture_output=True,
                text=True,
                timeout=180,
                creationflags=flags,
                encoding="utf-8",
                errors="replace",
            )
            info = json.loads(out.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            self.events.put(("env", ("환경 확인 실패", "warn")))
            return
        if info.get("cuda"):
            text, tone = f"GPU · {info.get('gpu', 'CUDA')}", "good"
        elif "torch" in info:
            text, tone = "CPU 전용 · 'CPU 모드'로 학습", "warn"
        else:
            text, tone = "PyTorch 미설치", "bad"
        self.events.put(("env", (text, tone)))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stopping = True
            self.process.terminate()

    def open_report(self):
        for name in ("report.html", "hmm/comparison.html"):
            path = self.artifacts / name
            if path.exists():
                webbrowser.open(path.as_uri())
                return
        self.show_help("보고서", "아직 보고서가 없습니다. 학습을 먼저 실행하세요.")

    def close(self):
        from tkinter import messagebox

        if self.process and not messagebox.askyesno(
            "종료",
            "작업이 실행 중입니다. 중지하고 닫을까요?\n학습은 나중에 '이어서 학습'으로 이어갈 수 있습니다.",
        ):
            return
        self.stop()
        self.root.destroy()


def enable_dpi_awareness():
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def launch(config="config.toml"):
    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit(
            "tkinter is unavailable in this Python. On Ubuntu: sudo apt-get install python3-tk"
        ) from exc
    enable_dpi_awareness()
    root = tk.Tk()
    App(root, config)
    root.mainloop()
