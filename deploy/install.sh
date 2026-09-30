#!/usr/bin/env bash
set -euo pipefail
if [[ $EUID -ne 0 ]]; then echo 'Run: sudo bash deploy/install.sh'; exit 1; fi
source /etc/os-release
if [[ "${ID}" != ubuntu || "${VERSION_ID}" != 24.04 ]]; then
  echo 'This installer targets Ubuntu 24.04 LTS (ARM64 or AMD64).'; exit 1
fi
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test -f "$SOURCE_DIR/artifacts/model.onnx" || { echo 'Trained model missing. Upload oracle-upload.zip first.'; exit 1; }
apt-get update
apt-get install -y python3 python3-venv python3-pip rsync unzip ca-certificates
id upbitbot >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin upbitbot
install -d -m 755 /opt/upbit-lstm/releases
install -d -o upbitbot -g upbitbot -m 700 /var/lib/upbit-lstm
RELEASE_DIR="/opt/upbit-lstm/releases/$(date -u +%Y%m%dT%H%M%S)-$$"
install -d -m 755 "$RELEASE_DIR"
rsync -a --exclude='.venv' --exclude='.git' --exclude='.env' --exclude='state' --exclude='data' --exclude='.tools' "$SOURCE_DIR/" "$RELEASE_DIR/"
python3 -m venv "$RELEASE_DIR/.venv"
"$RELEASE_DIR/.venv/bin/python" -m pip install --upgrade pip
"$RELEASE_DIR/.venv/bin/python" -m pip install -r "$RELEASE_DIR/requirements-runtime.txt" "$RELEASE_DIR"
cd "$RELEASE_DIR"
"$RELEASE_DIR/.venv/bin/python" -c 'from upbit_bot.runtime import Bundle; Bundle("artifacts"); print("Model and report integrity verified")'
if [[ ! -f /etc/upbit-lstm.env ]]; then
  install -o root -g upbitbot -m 640 "$RELEASE_DIR/.env.example" /etc/upbit-lstm.env
fi
# Refuse an in-place model swap with open live inventory or unresolved orders.
systemctl stop upbit-lstm.timer 2>/dev/null || true
systemctl stop upbit-lstm.service 2>/dev/null || true
python3 - <<'PY'
import sqlite3
from pathlib import Path
p = Path('/var/lib/upbit-lstm/live.sqlite')
if p.exists():
    db = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    assert db.execute('SELECT COUNT(*) FROM positions').fetchone()[0] == 0, 'Live inventory is open; do not replace model.'
    assert db.execute("SELECT COUNT(*) FROM intents WHERE status='pending'").fetchone()[0] == 0, 'Resolve pending orders first.'
PY
ln -sfn "$RELEASE_DIR" /opt/upbit-lstm/current
install -m 644 deploy/upbit-lstm.service /etc/systemd/system/upbit-lstm.service
install -m 644 deploy/upbit-lstm.timer /etc/systemd/system/upbit-lstm.timer
systemctl daemon-reload
systemctl enable --now upbit-lstm.timer
systemctl list-timers upbit-lstm.timer --no-pager
echo 'Installed. Paper trading is the default. Logs: sudo journalctl -u upbit-lstm.service -f'
