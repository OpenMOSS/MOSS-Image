#!/usr/bin/env bash
# Create an isolated Python environment; model download is a separate step.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  echo "This installer requires an Apple silicon Mac and a native arm64 terminal." >&2
  exit 1
fi
python_bin=${MOSS_PYTHON:-python3}
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required; 3.12 is recommended"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[ui]' -c constraints-mac.txt
.venv/bin/python -c 'import torch; assert torch.backends.mps.is_available(), "PyTorch cannot access MPS. Check macOS and the Python architecture."; print("MPS available. Next: download the SFT bundle using the macOS guide.")'
