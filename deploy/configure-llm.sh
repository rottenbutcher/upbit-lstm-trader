#!/usr/bin/env bash
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
cd /opt/upbit-lstm/current
echo 'LLM provider/model are set in config.toml [llm]. This stores only the API key.'
PROVIDER="$(.venv/bin/python -c 'from upbit_bot.common import load_config; from upbit_bot.llm_policy import llm_settings; s=llm_settings(load_config()); print(s["api_key_env"])')"
read -r -s -p "${PROVIDER} value: " LLM_KEY; echo
[[ -n "$LLM_KEY" && "$LLM_KEY" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo 'Unexpected key format; unchanged.'; exit 1; }
umask 077
TEMP_ENV="$(mktemp /etc/upbit-lstm-llm.env.XXXXXX)"
trap 'rm -f "$TEMP_ENV"' EXIT
printf '%s=%s\n' "$PROVIDER" "$LLM_KEY" > "$TEMP_ENV"
chown root:upbitbot "$TEMP_ENV"
chmod 640 "$TEMP_ENV"
mv "$TEMP_ENV" /etc/upbit-lstm-llm.env
unset LLM_KEY
echo 'Testing: collecting data and asking the LLM (dry run, nothing saved)...'
# with_env.py reads the key from the file, so it never appears on a command line.
if sudo -u upbitbot .venv/bin/python deploy/with_env.py policy-update --artifacts artifacts --dry-run; then
  systemctl enable --now upbit-lstm-policy.timer
  systemctl start upbit-lstm-policy.service || true
  echo 'Daily policy enabled. Check: sudo bash deploy/botctl.sh policy-show'
else
  echo 'Dry run failed. Fix the error above; the bot keeps using baseline parameters.'
  exit 1
fi
