"""Load only expected systemd environment keys for manual server commands."""

import os
import sys
from pathlib import Path

from upbit_bot.cli import main

ALLOWED = {
    "/etc/upbit-lstm.env": {"MODE", "LIVE_ACK", "UPBIT_ACCESS_KEY", "UPBIT_SECRET_KEY"},
    "/etc/upbit-lstm-llm.env": {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"},
}
for name, keys in ALLOWED.items():
    path = Path(name)
    if not path.exists():
        continue
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key in keys:
            os.environ[key] = value

if len(sys.argv) < 2:
    sys.argv.append("status")
sys.argv.extend(["--state-dir", "/var/lib/upbit-lstm"])
main()
