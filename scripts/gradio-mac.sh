#!/usr/bin/env bash
# Gradio UI on an Apple-silicon Mac, using the bundles downloaded into models/.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False
# Use native fp16 DiT arithmetic on M1/M2; set bfloat16 to compare.
export MOSS_MPS_DIT_DTYPE="${MOSS_MPS_DIT_DTYPE:-float16}"
if [[ ! -x "${repo_dir}/.venv/bin/python" ]]; then
  echo "Run bash scripts/setup-mac.sh first." >&2
  exit 1
fi
cd "$repo_dir"
exec "${repo_dir}/.venv/bin/python" "${repo_dir}/gradio_app.py" \
  --model-root "${repo_dir}/models" \
  --model "${repo_dir}/models/MOSS-Image-SFT" \
  --local-files-only "$@"
