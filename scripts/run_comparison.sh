#!/usr/bin/env bash
# Full sequential comparison on one GPU: cache -> vLLM -> uninstall -> Triton -> uninstall -> report.
#   NAMESPACE=llm scripts/run_comparison.sh [RUN_DIR]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_DIR="${1:-results/$(date -u +%Y%m%dT%H%M%SZ)}"
export NAMESPACE="${NAMESPACE:-llm}"

kubectl get pvc llm-model-cache --namespace "$NAMESPACE" >/dev/null 2>&1 || scripts/prepare_model_cache.sh
scripts/run_benchmark.sh vllm "$RUN_DIR"      # uninstalls itself on exit (KEEP=1 to keep)
scripts/run_benchmark.sh triton "$RUN_DIR"
llmbench report "$RUN_DIR" --out "reports/$(basename "$RUN_DIR")"
echo "Report: reports/$(basename "$RUN_DIR")/summary.md"
