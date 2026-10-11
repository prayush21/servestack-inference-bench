#!/usr/bin/env bash
# Remove everything this project created in the cluster. Pass --keep-cache to keep the model PVC.
set -euo pipefail
NAMESPACE="${NAMESPACE:-llm}"
for r in vllm triton; do
  helm uninstall "$r" --namespace "$NAMESPACE" 2>/dev/null && echo "uninstalled $r" || true
done
kubectl delete job model-cache-model-download --namespace "$NAMESPACE" --ignore-not-found
if [[ "${1:-}" != "--keep-cache" ]]; then
  kubectl delete pvc llm-model-cache --namespace "$NAMESPACE" --ignore-not-found
  kubectl delete namespace "$NAMESPACE" --ignore-not-found
fi
echo "Cluster resources removed. Delete the cluster / GPU node pool itself to stop billing (see README)."
