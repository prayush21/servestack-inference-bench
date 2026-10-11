# Verification record

_As of 2026-10-07. Machine: Apple M2 MacBook Air (8 GB), macOS 27, **no NVIDIA GPU**, no cloud resources provisioned._

## Actually run and verified

| What | How | Result |
|---|---|---|
| Version pins exist | GitHub releases API, Docker Hub API, nvcr.io registry API | vLLM v0.30.0 and Triton `26.07-trtllm-python-py3` tags exist; digests recorded. TRT-LLM in 26.07 = 1.2.1 (release notes). 26.08/26.09 ship no TRT-LLM image |
| Model availability / support | HF API at pinned SHA; vLLM v0.30.0 and TRT-LLM v1.2.1 supported-model docs | Ungated, Apache-2.0, `Qwen2ForCausalLM` listed by both |
| LLM-API arg names used in `model.yaml` | Read `tensorrt_llm/llmapi/llm_args.py` @ v1.2.1 | `kv_cache_config.max_gpu_total_bytes`, `enable_block_reuse`, `cuda_graph_config`, `max_num_tokens`, `max_seq_len`, `enable_chunked_prefill`, `revision`, `dtype` all exist |
| vLLM flags used | Read `vllm/engine/arg_utils.py` and protocol files @ v0.30.0 | `--kv-cache-memory-bytes`, `--no-enable-prefix-caching`, `--generation-config`, `ignore_eos`/`min_tokens` all exist |
| Streaming semantics of TRT-LLM | Read `executor/result.py` and Triton `model.py` @ v1.2.1 | `text` is cumulative (`text_diff` is the delta), and the model streams `text` |
| Sampling defaults | Read `sampling_params.py` @ v1.2.1 + Qwen `generation_config.json` | TRT-LLM uses generation_config only for stop tokens. Qwen ships rep. penalty 1.1 etc. |
| Chat template | Compared with `tokenizer_config.json` at pinned SHA | Matches. No BOS token |
| Prompt token counts | Model's own `tokenizer.json` (pinned SHA) | 37–302 tokens, mean 128.5 |
| Vendored Triton files | `scripts/vendor_triton_llmapi.sh --check` (re-downloads upstream, applies patch, diffs) | Identical. Python syntax OK |
| Unit/integration tests | `pytest -q` | **93 passed** (~5 s) |
| Tests actually catch bugs | Deliberately broke Triton cumulative parsing and warm-up exclusion | 4 tests failed as expected, then reverted |
| Helm chart | `helm lint` for both backends, `helm template` + schema negative tests | Pass. Schema rejects bad backend/revision/GPU count/unknown keys |
| Manifests vs. a real API server | Local k3s v1.35 (Colima): `kubectl apply --dry-run=server` | vLLM Deployment/Service, Triton ConfigMap/Deployment/Service, cache PVC/Job all accepted |
| GPU scheduling semantics | `helm install` of each backend on the GPU-less k3s node | Pods stay `Pending` with `Insufficient nvidia.com/gpu`. They never land on a non-GPU node |
| One-GPU guard | `scripts/run_benchmark.sh triton` while `vllm` installed | Refuses with a clear message |
| Triton model-repo staging | busybox pod running the chart's exact init command against the rendered ConfigMap | Produces the upstream layout as real files (no `..data` symlinks); patched input present |
| In-cluster client path | On k3s: client pod from the chart, code shipped via `tar`/`exec -i`, `pip install`, benchmark run in-pod against the **fake** server, results copied back, local `gpu-watch` | Works end to end. Found and fixed 2 real bugs (`exec` missing `-i`; `--cmd` clobbering the CLI subcommand) |
| Real-socket client path | `llmbench fake-server` + `generate`/`bench`/`report` over HTTP | Works. Found and fixed a FastAPI annotation bug in the fake server's HTTP wrapper |
| Report rendering | Generated plot from synthetic data, inspected visually; palette validator | Fixed label collisions and title/legend overlap. Palette passes all colorblind checks. Synthetic runs are stamped SYNTHETIC |

Local side effects were cleaned up:
- **Colima:** stopped.
- **Test namespaces:** deleted.
- **kubeconfig:** restored to its original (empty) state.
- **Scratch results:** kept in the session scratchpad only.

## Not verified (needs an NVIDIA GPU)

- vLLM and Triton actually loading the model, capturing CUDA graphs, and passing probes within the configured timeouts.
- The patched Triton `model.py`/`helpers.py` running in the 26.07 container (e.g. that a NumPy-typed `ignore_eos` is accepted by `SamplingParams`).
- The exact Triton `generate_stream` JSON (`"length"` vs `["length"]`, `finish_reason` while streaming). The client tolerates both forms.
- `generate` (non-streaming) on a decoupled Triton model. The benchmark only uses streaming.
- That `HF_HUB_OFFLINE=1` + `revision` resolves the cached snapshot in both containers.
- `nvidia-smi` being available inside both pods.
- 8 GiB KV + CUDA graphs + `chunkedPrefill: true` fitting and working on an L4.
- GKE/GPU-Operator commands in the README (shown, not run, because they create billable resources).
- The model-download Job (needs the vLLM image pull plus a 3 GB download; skipped on the 8 GB laptop).
- **All performance numbers.** The report is PENDING.

## Commands to finish (on a cluster with 1× NVIDIA GPU)

```bash
# 0) once: client tooling
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu   # expect 1

# 1) full comparison: cache → vLLM → Triton → report (≈ 40–60 min incl. image pulls; estimate)
NAMESPACE=llm scripts/run_comparison.sh results/$(date -u +%Y%m%d)-l4

# 2) inspect
cat reports/$(date -u +%Y%m%d)-l4/summary.md
open reports/$(date -u +%Y%m%d)-l4/comparison.png

# 3) fill docs/FINDINGS.md §1 from summary.md and commit results/ + reports/

# 4) teardown (then delete the GPU node pool / cluster to stop billing)
scripts/teardown.sh
```

If a step fails, see README → Troubleshooting. `KEEP=1 scripts/run_benchmark.sh <backend>` leaves the release running for debugging, and `CLIENT=local` falls back to `kubectl port-forward`.
