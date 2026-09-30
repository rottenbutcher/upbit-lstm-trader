"""Load only expected systemd environment keys for manual server commands."""

import os
import sys
from pathlib import Path

from upbit_bot.cli import main

for line in Path("/etc/upbit-lstm.env").read_text().splitlines():
    key, sep, value = line.partition("=")
    if sep and key in {"MODE", "LIVE_ACK", "UPBIT_ACCESS_KEY", "UPBIT_SECRET_KEY"}:
        os.environ[key] = value

if len(sys.argv) < 2:
    sys.argv.append("status")
sys.argv.extend(["--state-dir", "/var/lib/upbit-lstm"])
main()
