#!/usr/bin/env bash
set -euo pipefail

# Setup script for MOSS-Image on Google Cloud TPU (Colab TPU v6e-1 / TPU VM) with TorchTPU.
# TorchTPU 0.2.0 provides Linux wheels for Python 3.11–3.14.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 14), "TorchTPU 0.2.0 requires Python 3.11–3.14"'

"${PYTHON_BIN}" -m pip install --upgrade pip
torch_install=()
if "${PYTHON_BIN}" -c 'import torch; raise SystemExit(torch.version.cuda is None)' 2>/dev/null; then
  torch_install+=(--force-reinstall)
fi
"${PYTHON_BIN}" -m pip install "${torch_install[@]}" 'torch>=2.11,<=2.15.0' --index-url https://download.pytorch.org/whl/cpu
"${PYTHON_BIN}" -m pip install 'torch_tpu==0.2.0'
"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"

cat <<'EOF'
TorchTPU environment setup complete.

Recommended persistent XLA compilation cache environment variables:
  export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
  export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"

Quick probe:
  python3 tpu/run_tpu.py --mode probe
EOF
