"""Training control panel: edit config.toml with explanations, then download/train/package.

Uses only the standard library (tkinter). Each job runs the normal CLI in a subprocess, so the
GUI and TRAIN.cmd / train.sh produce identical artifacts. Comments in config.toml are preserved:
only the value on the edited ``key = value`` line is replaced.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib
import webbrowser
from pathlib import Path

# (section, key, label, kind, help). kind: int, float, bool, str, floats (list), or a tuple of choices.
FIELDS = [
    ("data", "top_n", "종목 수 (거래대금 상위)", "int", "처음 데이터 수집 때 고정됩니다. 바꾸려면 data 폴더를 새로 만드세요."),
    ("data", "start", "수집 시작일", "str", "YYYY-MM-DD. 오래될수록 학습량과 다운로드 시간이 늘어납니다."),
    ("data", "minimum_days", "최소 이력(일)", "int", "이보다 짧은 종목이 있으면 학습을 멈춥니다."),
    ("model", "feature_version", "피처 버전", (1, 2), "2 = 일봉 수준 문맥, 변동성, 요일 피처 추가 (권장)."),
    ("model", "lookback", "입력 길이(5분봉)", "int", "96 = 8시간. 길수록 느리고 VRAM을 더 씁니다."),
    ("model", "horizon", "예측 구간(5분봉)", "int", "6 = 30분 뒤 수익률을 예측합니다."),
    ("model", "hidden", "LSTM hidden 크기", "int", "64 권장(8GB). 128은 과적합/메모리 주의."),
    ("model", "layers", "LSTM 층 수", "int", "1~3."),
    ("model", "dropout", "드롭아웃", "float", "0~0.5. 과적합 방지."),
    ("model", "attention", "어텐션 풀링", "bool", "전체 시퀀스를 가중 요약해 마지막 상태와 함께 사용합니다."),
    ("model", "ensemble", "앙상블 모델 수", "int", "독립 학습 모델의 평균. 3이면 학습 시간 약 3배, 예측 안정성/불확실성 추정 향상."),
    ("training", "epochs", "최대 epoch", "int", "조기 종료(patience)로 더 일찍 끝날 수 있습니다."),
    ("training", "patience", "조기 종료 patience", "int", "검증 오차가 이 횟수만큼 개선되지 않으면 멈춥니다."),
    ("training", "batch_size", "배치 크기", "int", "GPU 메모리 부족 시 자동으로 절반으로 줄입니다."),
    ("training", "learning_rate", "학습률", "float", "AdamW 기본 학습률."),
    ("training", "lr_schedule", "학습률 스케줄", ("cosine", "constant"), "cosine = 워밍업 후 점진 감소 (권장)."),
    ("training", "warmup_epochs", "워밍업 epoch", "int", "cosine 스케줄의 초기 워밍업 기간."),
    ("training", "weight_decay", "가중치 감쇠", "float", "L2 정규화 강도."),
    ("training", "samples_per_epoch", "epoch당 샘플 수", "int", "0 = 전체. 클수록 epoch가 길어집니다."),
    ("training", "validation_days", "검증 기간(일)", "int", "매수 기준 선택과 조기 종료에 사용."),
    ("training", "test_days", "테스트 기간(일)", "int", "최종 승인 판정에만 사용하는 미사용 구간."),
    ("training", "seed", "랜덤 시드", "int", "같은 시드 + 같은 데이터 = 재현 가능한 결과."),
    ("strategy", "entry_thresholds_bps", "매수 기준 후보(bps)", "floats", "검증 구간에서 이 중 하나를 고릅니다. 예: 20, 30, 45"),
    ("strategy", "stop_loss", "손절 비율", "float", "0.02 = 2%. LLM 정책은 이 값보다 느슨하게 할 수 없습니다."),
    ("strategy", "take_profit", "익절 비율", "float", "0.04 = 4%."),
    ("strategy", "position_fraction", "종목당 비중", "float", "자본 대비. LLM은 이 값 이하로만 조정."),
    ("strategy", "max_exposure", "최대 총 노출", "float", "자본 대비 동시 보유 한도."),
    ("strategy", "max_hold_bars", "최대 보유(5분봉)", "int", "12 = 1시간."),
    ("strategy", "daily_loss_limit", "일일 손실 한도", "float", "초과 시 당일 신규 매수 중단."),
    ("strategy", "max_drawdown", "최대 낙폭 한도", "float", "초과 시 운영자가 해제할 때까지 정지."),
    ("gate", "min_trades", "승인 최소 거래 수", "int", "검증/테스트 모두 충족해야 실거래 승인."),
    ("gate", "min_profit_factor", "승인 최소 PF", "float", "총이익/총손실."),
    ("runtime", "capital_krw", "운용 자본(KRW)", "float", "실거래 시 서버 설정과 같아야 합니다."),
]

PRESETS = {
    "빠른 점검 (수 분)": {
        ("model", "ensemble"): 1, ("training", "epochs"): 2, ("training", "samples_per_epoch"): 20000,
        ("model", "hidden"): 32,
    },
    "균형 (권장)": {
        ("model", "ensemble"): 3, ("training", "epochs"): 12, ("training", "samples_per_epoch"): 120000,
        ("model", "hidden"): 64, ("model", "attention"): True, ("model", "feature_version"): 2,
    },
    "고성능 (오래 걸림)": {
        ("model", "ensemble"): 5, ("training", "epochs"): 20, ("training", "samples_per_epoch"): 200000,
        ("training", "patience"): 4, ("model", "hidden"): 64,
    },
}


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


def job_steps(job, config, allow_cpu=False, has_lstm=False):
    """CLI argv lists for each button (identical to the command-line launchers)."""
    cli = [sys.executable, "-u", "-m", "upbit_bot.cli", "--config", str(config)]
    cpu = ["--allow-cpu"] if allow_cpu else []
    if job == "train":
        return [[*cli, "download"], [*cli, "train", *cpu], [*cli, "package"]]
    if job == "resume":
        return [[*cli, "train", "--resume", *cpu], [*cli, "package"]]
    if job == "download":
        return [[*cli, "download"]]
    if job == "hmm":
        return [[*cli, "train-hmm", *(["--compare-lstm", "artifacts"] if has_lstm else [])]]
    if job == "package":
        return [[*cli, "package"]]
    if job == "doctor":
        return [[*cli, "doctor"]]
    raise ValueError(job)


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


class App:
    def __init__(self, root, config_path):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.root = tk, ttk, root
        self.config_path = Path(config_path).resolve()
        self.project = self.config_path.parent
        self.process, self.events = None, queue.Queue()
        self.vars = {}
        root.title("Upbit LSTM 학습 도구")
        root.geometry("980x760")
        root.minsize(820, 600)
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="프리셋:").pack(side="left")
        self.preset = ttk.Combobox(top, values=list(PRESETS), state="readonly", width=22)
        self.preset.pack(side="left", padx=4)
        ttk.Button(top, text="적용", command=self.apply_preset).pack(side="left")
        ttk.Button(top, text="설정 저장", command=self.save).pack(side="left", padx=(16, 4))
        ttk.Button(top, text="다시 불러오기", command=self.load).pack(side="left")
        self.status = ttk.Label(top, text="", foreground="#555")
        self.status.pack(side="right")
        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=False, padx=8)
        titles = {"data": "데이터", "model": "모델", "training": "학습", "strategy": "전략", "gate": "승인/자본",
                  "runtime": "승인/자본"}
        frames = {}
        for section, key, label, kind, help_text in FIELDS:
            title = titles[section]
            if title not in frames:
                frame = ttk.Frame(tabs, padding=10)
                frame.columnconfigure(2, weight=1)
                tabs.add(frame, text=title)
                frames[title] = frame
            frame = frames[title]
            row = frame.grid_size()[1]
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            if kind == "bool":
                var = tk.BooleanVar()
                widget = ttk.Checkbutton(frame, variable=var)
            elif isinstance(kind, tuple):
                var = tk.StringVar()
                widget = ttk.Combobox(frame, textvariable=var, values=[str(k) for k in kind],
                                      state="readonly", width=16)
            else:
                var = tk.StringVar()
                widget = ttk.Entry(frame, textvariable=var, width=18)
            widget.grid(row=row, column=1, sticky="w", padx=8)
            ttk.Label(frame, text=help_text, foreground="#666", wraplength=560).grid(
                row=row, column=2, sticky="w")
            self.vars[(section, key)] = (var, kind)
        run = ttk.LabelFrame(root, text="실행", padding=8)
        run.pack(fill="x", padx=8, pady=6)
        self.allow_cpu = tk.BooleanVar(value=False)
        buttons = [
            ("① 데이터 받기 + 학습 + 패키지", "train"),
            ("이어서 학습", "resume"),
            ("데이터만 갱신", "download"),
            ("HMM 비교", "hmm"),
            ("GPU 확인", "doctor"),
        ]
        self.run_buttons = []
        for text, job in buttons:
            b = ttk.Button(run, text=text, command=lambda job=job: self.start(job))
            b.pack(side="left", padx=3)
            self.run_buttons.append(b)
        ttk.Checkbutton(run, text="GPU 없이(CPU)", variable=self.allow_cpu).pack(side="left", padx=8)
        self.stop_button = ttk.Button(run, text="중지", command=self.stop, state="disabled")
        self.stop_button.pack(side="right")
        ttk.Button(run, text="결과 보고서 열기", command=self.open_report).pack(side="right", padx=4)
        bar = ttk.Frame(root, padding=(8, 0))
        bar.pack(fill="x")
        self.progress = ttk.Progressbar(bar, maximum=1.0)
        self.progress.pack(fill="x", side="left", expand=True)
        self.progress_label = ttk.Label(bar, text="", width=48)
        self.progress_label.pack(side="left", padx=8)
        self.log = tk.Text(root, height=14, wrap="word", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=8, pady=8)
        self.load()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(200, self.poll)

    def load(self):
        with open(self.config_path, "rb") as f:
            cfg = tomllib.load(f)
        for (section, key), (var, kind) in self.vars.items():
            value = cfg.get(section, {}).get(key)
            if value is None:
                value = {"feature_version": 1, "attention": False, "ensemble": 1,
                         "lr_schedule": "constant", "warmup_epochs": 0}.get(key, "")
            if kind == "floats":
                value = ", ".join(f"{v:g}" for v in value)
            var.set(value)
        self.status.config(text=f"불러옴: {self.config_path.name}")

    def collect(self):
        text = self.config_path.read_text(encoding="utf-8")
        for (section, key), (var, kind) in self.vars.items():
            try:
                value = parse_field(kind, str(var.get()))
            except ValueError as exc:
                raise ValueError(f"[{section}] {key}: {exc}") from exc
            text = set_toml(text, section, key, value)
        validate_text(text)
        return text

    def save(self):
        from tkinter import messagebox

        try:
            text = self.collect()
        except Exception as exc:
            messagebox.showerror("설정 오류", str(exc))
            return False
        backup = self.config_path.with_suffix(".toml.bak")
        shutil.copy2(self.config_path, backup)
        self.config_path.write_text(text, encoding="utf-8")
        self.status.config(text=f"저장됨 (이전 설정: {backup.name})")
        return True

    def apply_preset(self):
        for (section, key), value in PRESETS.get(self.preset.get(), {}).items():
            var, _ = self.vars[(section, key)]
            var.set(value)
        self.status.config(text="프리셋 적용됨 — '설정 저장'을 눌러야 반영됩니다")

    def start(self, job):
        from tkinter import messagebox

        if self.process:
            return
        if job in ("train", "resume", "package") and not self.save():
            return
        if job == "train" and (self.project / "artifacts" / "manifest.json").exists() and not messagebox.askyesno(
            "확인", "기존 artifacts의 모델/보고서를 덮어씁니다. 필요하면 먼저 다른 폴더에 보관하세요.\n계속할까요?"
        ):
            return
        steps = job_steps(job, self.config_path, self.allow_cpu.get(),
                          (self.project / "artifacts" / "manifest.json").exists())
        for b in self.run_buttons:
            b.config(state="disabled")
        self.stop_button.config(state="normal")
        self.log.delete("1.0", "end")
        threading.Thread(target=self.worker, args=(steps,), daemon=True).start()

    def worker(self, steps):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for argv in steps:
            self.events.put(("log", "\n$ " + " ".join(argv[3:]) + "\n"))
            try:
                self.process = subprocess.Popen(
                    argv, cwd=self.project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace", env=env, creationflags=flags,
                )
            except OSError as exc:
                self.events.put(("done", f"실행 실패: {exc}"))
                return
            for line in self.process.stdout:
                self.events.put(("log", line))
            code = self.process.wait()
            if code != 0:
                self.events.put(("done", f"실패 (종료 코드 {code}). 위 로그를 확인하세요. 학습 중단은 '이어서 학습'으로 재개됩니다."))
                return
        self.events.put(("done", "완료"))

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload)
                    self.log.see("end")
                else:
                    self.process = None
                    for b in self.run_buttons:
                        b.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.progress_label.config(text=payload)
                    self.log.insert("end", f"\n== {payload} ==\n")
                    self.log.see("end")
        except queue.Empty:
            pass
        if self.process:
            fraction, text = progress_text(self.project / "artifacts" / "progress.json")
            if fraction is not None:
                self.progress["value"] = fraction
                self.progress_label.config(text=text)
        self.root.after(500, self.poll)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def open_report(self):
        for name in ("artifacts/report.html", "artifacts/hmm/comparison.html"):
            path = self.project / name
            if path.exists():
                webbrowser.open(path.as_uri())
                return
        self.status.config(text="아직 보고서가 없습니다")

    def close(self):
        from tkinter import messagebox

        if self.process and not messagebox.askyesno("종료", "작업이 실행 중입니다. 중지하고 닫을까요?"):
            return
        self.stop()
        self.root.destroy()


def launch(config="config.toml"):
    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit("tkinter is unavailable in this Python. On Ubuntu: sudo apt-get install python3-tk") from exc
    root = tk.Tk()
    App(root, config)
    root.mainloop()
