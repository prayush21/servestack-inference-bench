#!/usr/bin/env bash
# Create the shared model-cache PVC and run the one-shot download Job, rendered from the
# same chart values the backends use (so model id/revision cannot drift).
#   NAMESPACE=llm scripts/prepare_model_cache.sh [extra helm --set/-f args]
set -euo pipefail
NAMESPACE="${NAMESPACE:-llm}"
CHART="$(cd "$(dirname "$0")/.." && pwd)/deploy/helm/llm-serving"

kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"
helm template model-cache "$CHART" --namespace "$NAMESPACE" \
  --show-only templates/model-cache.yaml \
  --set modelCache.create=true --set modelDownload.enabled=true "$@" \
  | kubectl apply --namespace "$NAMESPACE" -f -

echo "Waiting for the download Job (first pull of the vLLM image can take several minutes)..."
kubectl wait --namespace "$NAMESPACE" --for=condition=complete --timeout=30m job/model-cache-model-download
kubectl logs --namespace "$NAMESPACE" job/model-cache-model-download --tail=5
