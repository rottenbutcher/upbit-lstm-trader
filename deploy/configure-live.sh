#!/usr/bin/env bash
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
cd /opt/upbit-lstm/current
.venv/bin/python - <<'PY'
from upbit_bot.runtime import Bundle
from upbit_bot.common import load_config, utcnow
from datetime import datetime
b = Bundle('artifacts'); c = load_config()
assert b.manifest['approved_for_live'] and b.report['approved_for_live'], 'Backtest gate failed: keep paper mode.'
age = (utcnow() - datetime.fromisoformat(b.manifest['data_end'])).total_seconds() / 86400
assert 0 <= age <= c['runtime']['model_max_age_days'], 'Model is stale; retrain first.'
print('Bot capital (KRW):', c['runtime']['capital_krw'])
print('Existing coins are excluded until explicitly adopted. Real losses are possible.')
PY
read -r -p 'Type LIVE to enable real orders with this capital: ' ACK
[[ "$ACK" == LIVE ]] || { echo 'Unchanged.'; exit 0; }
read -r -s -p 'Upbit Access Key: ' UPBIT_ACCESS_KEY; echo
read -r -s -p 'Upbit Secret Key: ' UPBIT_SECRET_KEY; echo
export UPBIT_ACCESS_KEY UPBIT_SECRET_KEY
.venv/bin/python - <<'PY'
import os, re
from upbit_bot.api import Upbit
from upbit_bot.runtime import Bundle
for k in ('UPBIT_ACCESS_KEY', 'UPBIT_SECRET_KEY'):
    assert re.fullmatch(r'[A-Za-z0-9_-]+', os.environ[k]), 'Unexpected key format'
a = Upbit(os.environ['UPBIT_ACCESS_KEY'], os.environ['UPBIT_SECRET_KEY'])
a.accounts()
for m in Bundle('artifacts').manifest['markets']:
    a.chance(m)
print('Read authentication, IP allowlist and order-query permission verified. No order submitted.')
PY
umask 077
TEMP_ENV="$(mktemp /etc/upbit-lstm.env.XXXXXX)"
trap 'rm -f "$TEMP_ENV"' EXIT
printf 'MODE=live\nUPBIT_ACCESS_KEY=%s\nUPBIT_SECRET_KEY=%s\nLIVE_ACK=I_UNDERSTAND_REAL_ORDERS\n' "$UPBIT_ACCESS_KEY" "$UPBIT_SECRET_KEY" > "$TEMP_ENV"
chown root:upbitbot "$TEMP_ENV"
chmod 640 "$TEMP_ENV"
mv "$TEMP_ENV" /etc/upbit-lstm.env
unset UPBIT_ACCESS_KEY UPBIT_SECRET_KEY
systemctl restart upbit-lstm.timer
echo 'Real-order mode enabled for the next scheduled cycle. STOP: sudo touch /var/lib/upbit-lstm/STOP'
