#!/usr/bin/env bash
# Deploy ONE backend, record exactly what ran, benchmark it, collect logs, and (optionally) tear it down.
#
#   scripts/run_benchmark.sh <vllm|triton> [RUN_DIR]
#
# Env: NAMESPACE (default llm), CLIENT (incluster|local, default incluster), LOCAL_PORT (default 8000),
#      CONFIG (default configs/benchmark.yaml),
#      LEVELS (e.g. "1,4" to override concurrency), KEEP=1 to leave the release running,
#      HELM_ARGS (extra args for helm, e.g. "--set nodeSelector.cloud\.google\.com/gke-accelerator=nvidia-l4").
set -euo pipefail

BACKEND="${1:?usage: run_benchmark.sh <vllm|triton> [RUN_DIR]}"
RUN_DIR="${2:-results/$(date -u +%Y%m%dT%H%M%SZ)}"
NAMESPACE="${NAMESPACE:-llm}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
CONFIG="${CONFIG:-configs/benchmark.yaml}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHART="$ROOT/deploy/helm/llm-serving"
LLMBENCH="${LLMBENCH:-llmbench}"
cd "$ROOT"   # RUN_DIR and CONFIG are relative to the repo root

case "$BACKEND" in
  vllm) OTHER=triton ;;
  triton) OTHER=vllm ;;
  *) echo "backend must be vllm or triton" >&2; exit 2 ;;
esac

# One GPU: never run both backends at once (the second pod would sit Pending, or worse, share the GPU).
if helm status "$OTHER" --namespace "$NAMESPACE" >/dev/null 2>&1; then
  echo "Release '$OTHER' is still installed in '$NAMESPACE'. Run: helm uninstall $OTHER -n $NAMESPACE" >&2
  exit 1
fi
kubectl get pvc llm-model-cache --namespace "$NAMESPACE" >/dev/null 2>&1 || {
  echo "Model cache PVC missing. Run: NAMESPACE=$NAMESPACE scripts/prepare_model_cache.sh" >&2; exit 1; }

OUT="$RUN_DIR/$BACKEND"
ART="$OUT/k8s"
mkdir -p "$ART"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$ART/commands.log" >&2; }

log "helm upgrade --install $BACKEND $CHART -n $NAMESPACE -f values-$BACKEND.yaml ${HELM_ARGS:-}"
# shellcheck disable=SC2086
helm upgrade --install "$BACKEND" "$CHART" --namespace "$NAMESPACE" \
  -f "$CHART/values-$BACKEND.yaml" ${HELM_ARGS:-}

log "waiting for rollout (model load + CUDA graph capture; first image pull can take 10+ min)"
kubectl rollout status "deployment/$BACKEND" --namespace "$NAMESPACE" --timeout=45m

POD="$(kubectl get pod --namespace "$NAMESPACE" -l "llm-serving/backend=$BACKEND,app.kubernetes.io/instance=$BACKEND" \
  -o jsonpath='{.items[0].metadata.name}')"
NODE="$(kubectl get pod "$POD" --namespace "$NAMESPACE" -o jsonpath='{.spec.nodeName}')"
log "pod=$POD node=$NODE"

# --- provenance: exactly what is running -----------------------------------------------------
helm get values "$BACKEND" --namespace "$NAMESPACE" --all -o json > "$ART/helm-values.json"
helm get manifest "$BACKEND" --namespace "$NAMESPACE" > "$ART/manifest.yaml"
kubectl get pod "$POD" --namespace "$NAMESPACE" -o json > "$ART/pod.json"
kubectl get node "$NODE" -o json > "$ART/node.json"
python3 - "$ART" <<'EOF'
import json, sys, pathlib
art = pathlib.Path(sys.argv[1])
pod = json.loads((art / "pod.json").read_text())
images = [{"name": s["name"], "image": s["image"], "imageID": s.get("imageID")}
          for s in pod["status"].get("initContainerStatuses", []) + pod["status"]["containerStatuses"]]
(art / "pod-images.json").write_text(json.dumps(images, indent=2))
node = json.loads((art / "node.json").read_text())
labels = node["metadata"].get("labels", {})
keep = {k: v for k, v in labels.items() if any(s in k for s in ("instance-type", "accelerator", "nvidia.com", "zone", "gke-nodepool"))}
summary = {"name": node["metadata"]["name"], "labels": keep,
           "kubelet": node["status"]["nodeInfo"]["kubeletVersion"],
           "kernel": node["status"]["nodeInfo"]["kernelVersion"],
           "os": node["status"]["nodeInfo"]["osImage"],
           "containerRuntime": node["status"]["nodeInfo"]["containerRuntimeVersion"],
           "allocatable": node["status"]["allocatable"]}
(art / "node-summary.json").write_text(json.dumps(summary, indent=2))
EOF
kubectl exec "$POD" --namespace "$NAMESPACE" -- nvidia-smi > "$ART/nvidia-smi.txt" 2>&1 || log "nvidia-smi not available in pod"
kubectl version -o json > "$ART/kubectl-version.json" 2>/dev/null || true

# --- benchmark ---------------------------------------------------------------------------------
# CLIENT=incluster (default): load generator runs in a pod and calls the ClusterIP service directly.
# CLIENT=local: load generator runs here through `kubectl port-forward` (simpler, but the tunnel adds
#               latency and can bottleneck at high concurrency; fine for smoke tests).
CLIENT="${CLIENT:-incluster}"
EXEC="kubectl --namespace $NAMESPACE exec $POD --"
SMI_QUERY="nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
$EXEC nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader \
  > "$ART/gpu-info.txt" 2>/dev/null || echo "unavailable" > "$ART/gpu-info.txt"
LEVEL_ARGS=()
[[ -n "${LEVELS:-}" ]] && LEVEL_ARGS=(--levels "$LEVELS")

PF_PID="" ; GW_PID=""
cleanup() {
  [[ -n "$PF_PID" ]] && kill "$PF_PID" 2>/dev/null || true
  [[ -n "$GW_PID" ]] && kill "$GW_PID" 2>/dev/null || true
  kubectl logs "$POD" --namespace "$NAMESPACE" --all-containers > "$ART/server.log" 2>&1 || true
  kubectl delete pod llmbench-client --namespace "$NAMESPACE" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  if [[ "${KEEP:-0}" != "1" ]]; then
    log "helm uninstall $BACKEND -n $NAMESPACE"
    helm uninstall "$BACKEND" --namespace "$NAMESPACE" || true
  fi
}
trap cleanup EXIT

if [[ "$CLIENT" == "local" ]]; then
  kubectl port-forward --namespace "$NAMESPACE" "svc/$BACKEND" "$LOCAL_PORT:8000" > "$ART/port-forward.log" 2>&1 &
  PF_PID=$!
  URL="http://localhost:$LOCAL_PORT"
  for _ in $(seq 1 60); do "$LLMBENCH" health --backend "$BACKEND" --url "$URL" >/dev/null 2>&1 && break; sleep 2; done
  "$LLMBENCH" health --backend "$BACKEND" --url "$URL"
  BENCH_ARGS=(bench --backend "$BACKEND" --url "$URL" --config "$CONFIG" --out "$RUN_DIR"
    --gpu-cmd "$EXEC $SMI_QUERY" --gpu-info-cmd "cat $ART/gpu-info.txt"
    --meta "helm_values=$ART/helm-values.json" --meta "pod_images=$ART/pod-images.json"
    --meta "node=$ART/node-summary.json" --meta "nvidia_smi=$ART/nvidia-smi.txt"
    --meta "client_mode=$ART/client-mode.txt" ${LEVEL_ARGS[@]+"${LEVEL_ARGS[@]}"})
  echo "local (kubectl port-forward)" > "$ART/client-mode.txt"
  log "$LLMBENCH ${BENCH_ARGS[*]}"
  "$LLMBENCH" "${BENCH_ARGS[@]}"
elif [[ "$CLIENT" == "incluster" ]]; then
  log "starting in-cluster client pod"
  kubectl delete pod llmbench-client --namespace "$NAMESPACE" --ignore-not-found >/dev/null
  helm template "$BACKEND" "$CHART" --namespace "$NAMESPACE" --show-only templates/loadgen-pod.yaml \
    --set loadgen.enabled=true -f "$CHART/values-$BACKEND.yaml" ${HELM_ARGS:-} \
    | kubectl apply --namespace "$NAMESPACE" -f - >/dev/null
  kubectl wait --namespace "$NAMESPACE" --for=condition=Ready pod/llmbench-client --timeout=10m
  CEXEC=(kubectl --namespace "$NAMESPACE" exec -i llmbench-client --)   # -i: stdin for tar
  echo "in-cluster pod -> http://$BACKEND.$NAMESPACE.svc:8000" > "$ART/client-mode.txt"
  # Ship the client code + configs + provenance files into the pod (needs no registry).
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$ROOT" -czf - pyproject.toml README.md llmbench configs \
    | "${CEXEC[@]}" sh -c 'mkdir -p /work/meta && tar -xzf - -C /work'   # pod is fresh each run
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$ART" -czf - helm-values.json pod-images.json node-summary.json nvidia-smi.txt gpu-info.txt client-mode.txt \
    | "${CEXEC[@]}" sh -c 'tar -xzf - -C /work/meta'
  "${CEXEC[@]}" pip install --quiet --disable-pip-version-check /work > "$ART/client-pip.log" 2>&1
  # GPU sampling runs here (it needs `kubectl exec`); the report attributes samples by wall clock.
  "$LLMBENCH" gpu-watch --cmd "$EXEC $SMI_QUERY" --out "$OUT/gpu.jsonl" --interval 1 &
  GW_PID=$!
  URL="http://$BACKEND.$NAMESPACE.svc:8000"
  "${CEXEC[@]}" llmbench health --backend "$BACKEND" --url "$URL"
  BENCH="cd /work && llmbench bench --backend $BACKEND --url $URL --config $CONFIG --out /work/out \
    --gpu-info-cmd 'cat /work/meta/gpu-info.txt' \
    --meta helm_values=/work/meta/helm-values.json --meta pod_images=/work/meta/pod-images.json \
    --meta node=/work/meta/node-summary.json --meta nvidia_smi=/work/meta/nvidia-smi.txt \
    --meta client_mode=/work/meta/client-mode.txt ${LEVEL_ARGS[*]:-}"
  log "in pod: $BENCH"
  "${CEXEC[@]}" sh -c "$BENCH"
  kill "$GW_PID" 2>/dev/null || true; GW_PID=""
  "${CEXEC[@]}" tar -C "/work/out/$BACKEND" -czf - . | tar -xzf - -C "$OUT"
else
  echo "CLIENT must be incluster or local" >&2; exit 2
fi
log "raw results in $OUT"
