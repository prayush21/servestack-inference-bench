#!/usr/bin/env bash
# Re-create deploy/helm/llm-serving/files/triton/{config.pbtxt,model.py,helpers.py}
# from NVIDIA's TensorRT-LLM repo at the pinned tag, then apply our patch.
#
#   scripts/vendor_triton_llmapi.sh          # rewrite the vendored files
#   scripts/vendor_triton_llmapi.sh --check  # fail if vendored files drift from upstream+patch
set -euo pipefail

TRTLLM_REF="${TRTLLM_REF:-376f7e1bd8ed543f75014309e3fd4b237e9b0e73}"  # tag v1.2.1
BASE="https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/${TRTLLM_REF}/triton_backend/all_models/llmapi/tensorrt_llm"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/deploy/helm/llm-serving/files/triton"
PATCH="${DEST}/llmapi-ignore-eos.patch"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/b"
curl -fsSL "${BASE}/config.pbtxt" -o "$work/b/config.pbtxt"
curl -fsSL "${BASE}/1/model.py" -o "$work/b/model.py"
curl -fsSL "${BASE}/1/helpers.py" -o "$work/b/helpers.py"
(cd "$work/b" && patch -p1 --quiet < "$PATCH")

if [[ "${1:-}" == "--check" ]]; then
  for f in config.pbtxt model.py helpers.py; do
    diff -u "$work/b/$f" "$DEST/$f"
  done
  echo "vendored Triton LLM-API files match upstream ${TRTLLM_REF} + patch"
else
  cp "$work/b/config.pbtxt" "$work/b/model.py" "$work/b/helpers.py" "$DEST/"
  echo "wrote $DEST from upstream ${TRTLLM_REF} + patch"
fi
