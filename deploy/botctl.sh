#!/usr/bin/env bash
set -euo pipefail
cd /opt/upbit-lstm/current
exec sudo -u upbitbot .venv/bin/python deploy/with_env.py "$@"
