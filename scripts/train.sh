#!/usr/bin/env bash
# Local Linux x86_64 training. Run as your regular user, not with sudo.
set -euo pipefail

usage() {
  cat <<'TEXT'
Usage: bash scripts/train.sh [train|resume|hmm|check|gui]
  train   Install Python/CUDA PyTorch, check GPU, download, train and package (default)
  resume  Resume LSTM on unchanged saved data/config; no download
  hmm     CPU HMM research; compare with saved LSTM when available
  check   Install environment and run a small CUDA LSTM forward/backward check only
  gui     Install environment and open the training settings window
Targets Linux x86_64 (Ubuntu 22.04/24.04). NVIDIA driver is required for LSTM.
TEXT
}
die() { printf '%s\n' "$*" >&2; exit 1; }
mode="${1:-train}"
if [[ "$mode" == --help || "$mode" == -h ]]; then usage; exit 0; fi
[[ $# -le 1 ]] || { usage; exit 2; }
case "$mode" in train|resume|hmm|check|gui) ;; *) usage; exit 2 ;; esac
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Use a Linux x86_64 training desktop.'
[[ $EUID -ne 0 ]] || die 'Run this script as your normal user, without sudo.'

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ "$mode" != hmm ]]; then
  command -v nvidia-smi >/dev/null || die 'nvidia-smi missing: install the NVIDIA driver and reboot first.'
  nvidia-smi || die 'NVIDIA driver is not working. Fix the driver before training.'
fi

tools_dir="$project_dir/.tools"
uv_path="$tools_dir/uv"
python_path="$project_dir/.venv/bin/python"
mkdir -p "$tools_dir"
if [[ ! -x "$uv_path" ]]; then
  for tool in curl tar sha256sum; do
    command -v "$tool" >/dev/null || die "Missing $tool. On Ubuntu: sudo apt-get install curl ca-certificates tar coreutils"
  done
  uv_version=0.12.21
  archive=uv-x86_64-unknown-linux-gnu.tar.gz
  base_url="https://github.com/astral-sh/uv/releases/download/$uv_version"
  curl --fail --location --retry 3 --connect-timeout 15 --max-time 600 \
    "$base_url/$archive" --output "$tools_dir/$archive"
  curl --fail --location --retry 3 --connect-timeout 15 --max-time 600 \
    "$base_url/$archive.sha256" --output "$tools_dir/$archive.sha256"
  read -r expected_hash _ < "$tools_dir/$archive.sha256"
  [[ "$expected_hash" =~ ^[0-9a-fA-F]{64}$ ]] || die 'Invalid bootstrap checksum format'
  (cd "$tools_dir"; printf '%s  %s\n' "$expected_hash" "$archive" | sha256sum --check --status) \
    || die 'Bootstrap SHA256 mismatch'
  tar -xzf "$tools_dir/$archive" -C "$tools_dir" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
fi
export UV_PYTHON_INSTALL_DIR="$tools_dir/python"
export UV_CACHE_DIR="$tools_dir/cache"
if [[ -d .venv && ! -x "$python_path" ]]; then
  die 'This .venv is not a Linux environment. Use a fresh source folder and copy only data/ and artifacts/.'
fi
if [[ ! -x "$python_path" ]]; then "$uv_path" venv --python 3.12 .venv; fi
"$python_path" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 environment"'

if [[ "$mode" == hmm ]]; then
  "$uv_path" pip install --python "$python_path" -r requirements-runtime.txt -e '.[hmm]'
  hmm_args=(-m upbit_bot.cli train-hmm)
  if [[ -f artifacts/manifest.json ]]; then
    printf '%s\n' 'Comparing with saved LSTM dates and candles; no download or GPU training.'
    hmm_args+=(--compare-lstm artifacts)
  else
    printf '%s\n' 'HMM-only research. After LSTM training, run this command again for comparison.'
    "$python_path" -m upbit_bot.cli download
  fi
  "$python_path" "${hmm_args[@]}"
  printf 'DONE: %s/artifacts/hmm/comparison.html\n' "$project_dir"
  exit 0
fi

"$uv_path" pip install --python "$python_path" --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1+cu128'
"$uv_path" pip install --python "$python_path" -r requirements-runtime.txt 'onnx==1.21.0' -e .
if [[ "$mode" == gui ]]; then
  "$uv_path" pip install --python "$python_path" -e '.[hmm]'
  exec "$python_path" -m upbit_bot.cli gui
fi
"$python_path" -m upbit_bot.cli doctor
# is_available alone cannot catch an unsupported GPU kernel or a broken cuDNN install.
"$python_path" - <<'PY'
import torch
from upbit_bot.model import ReturnLSTM

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: check NVIDIA driver, CUDA_VISIBLE_DEVICES and PyTorch CUDA build")
model = ReturnLSTM(32, hidden=64, layers=2, dropout=0.15, attention=True).cuda().train()
x = torch.randn(4, 96, 32, device="cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    prediction = model(x)
    loss = (prediction.float() / 100).square().mean()
loss.backward()
torch.cuda.synchronize()
if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
    raise SystemExit("CUDA LSTM check returned nonfinite values")
print("CUDA LSTM forward/backward passed:", torch.cuda.get_device_name(0))
PY
if [[ "$mode" == check ]]; then exit 0; fi
train_args=(-m upbit_bot.cli train)
if [[ "$mode" == resume ]]; then
  train_args+=(--resume)
else
  "$python_path" -m upbit_bot.cli download
fi
"$python_path" "${train_args[@]}"
"$python_path" -m upbit_bot.cli package
printf 'DONE: %s/artifacts/report.html\nUpload: %s/dist/oracle-upload.zip\n' "$project_dir" "$project_dir"
