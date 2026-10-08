# Implementation plan

_Written 2026-10-07, before implementation._

## 0. Environment inspection

| Item | Finding | Consequence |
|---|---|---|
| Machine | Apple M2 MacBook Air, 8 GB RAM, macOS 27 | **No NVIDIA GPU.** No CUDA, so neither server can run locally. |
| Container runtime | Docker CLI 29.8 + Colima (stopped) | Can run a local CPU-only `kind` cluster to validate manifests. It can't schedule GPU pods. |
| Kubernetes tooling | kubectl 1.37, helm 4.3, kind 0.33, no kube contexts | Helm lint/template and server-side dry-run are possible. Nothing is deployed. |
| Cloud | No `gcloud`/`aws`/`az` CLI, no cloud env vars | No paid resources were provisioned, and none will be without explicit approval. |
| Repository | Empty directory | Greenfield. |

## 1. Version pins (verified against GitHub releases, Docker Hub and nvcr.io on 2026-10-07)

| Component | Pin | Why |
|---|---|---|
| vLLM | `vllm/vllm-openai:v0.30.0` (`sha256:8a69ffad…4b90`) | v0.31.0 is 2 days old. v0.30.0 has been out 2 weeks with no hotfix. Image is built on CUDA 13.0.3. |
| Triton + TRT-LLM | `nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3` (`sha256:b097871d…7007`) | Triton 2.71.0 with **TensorRT-LLM 1.2.1** (latest stable TRT-LLM). The 26.08/26.09 release notes state the TRT-LLM image was **not** published, so 26.07 is the newest available. |
| TRT-LLM Triton model files | `triton_backend/all_models/llmapi/` @ TRT-LLM `v1.2.1` (`376f7e1b`) | NVIDIA's LLM API (PyTorch backend) model for Triton. No engine build. |
| Kubernetes | Tested against API v1.37 (client) | Uses only stable `apps/v1`, `v1`, `batch/v1` APIs. |
| NVIDIA device plugin / GPU Operator | v0.20.1 / v26.7.1 (documented, not installed) | Exposes `nvidia.com/gpu`. |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | See below. |

### Model choice
- Qwen2.5-1.5B-Instruct is **Apache-2.0 and not gated**, so no HF token is needed and anyone can reproduce the runs.
- Its architecture (`Qwen2ForCausalLM`) appears in the supported-model lists of both vLLM v0.30.0 and TRT-LLM v1.2.1.
- About 3.1 GB of BF16 weights, which leaves most of a 24 GB GPU free for KV cache and batching.
- Large enough to produce coherent text for a demo, small enough for one cheap GPU.

### Hardware choice
- Target: **1× NVIDIA L4 24 GB** (Ada, SM89), e.g. GKE `g2-standard-8` or similar. It's one of the least expensive datacenter GPUs supported by both stacks.
- A10G (AWS `g5.xlarge`) or a local RTX 30/40-series also work.
- **T4 is excluded** because TRT-LLM 1.2.1's support matrix lists Ampere and newer only.
- Both images are CUDA 13 builds, so the node driver must support CUDA 13 (R580+).

## 2. Design decisions for a fair comparison

1. **Same bytes in, same tokens out.**
   - Prompts are stored **pre-templated** with the Qwen chat template and sent as raw text: vLLM gets `/v1/completions`, Triton gets `text_input`. Both servers tokenize the identical string.
   - `ignore_eos=true` + fixed `max_tokens` forces every request to generate exactly N tokens.
2. **Same KV-cache budget.**
   - vLLM's `--gpu-memory-utilization` and TRT-LLM's `free_gpu_memory_fraction` mean different things.
   - Both are pinned to an identical byte budget instead: vLLM `--kv-cache-memory-bytes`, TRT-LLM `kv_cache_config.max_gpu_total_bytes`.
3. **Prefix caching off on both.** The fixed prompt set repeats, and caching would turn the test into a cache benchmark.
4. **Greedy decoding on both**, same seed, BF16 (`dtype: auto` → model's bf16), same max sequence length, same max concurrent sequences, CUDA graphs on for both.
5. **The upstream Triton LLM-API model does not forward `ignore_eos`.**
   - I vendor NVIDIA's three files at the pinned tag and apply a 3-line documented patch to whitelist `ignore_eos` and `min_tokens`.
   - The patch is kept as a separate `.patch` file so the diff from upstream is auditable.
6. **Streaming on both** (SSE). TTFT = time from request send to first non-empty text chunk, measured client-side the same way for both.
7. **Token counting:** vLLM reports `usage.completion_tokens`. Triton's LLM-API model returns no token count, so for `finish_reason == "length"` the count equals `max_tokens` by definition. The source of each count is recorded per request.

## 3. Work breakdown
1. `llmbench` Python package: unified request/response schema, vLLM + Triton async clients, CLI, tiny FastAPI gateway.
2. Load generator: closed-loop concurrency, warm-up, fixed prompts, raw JSONL output, optional GPU sampler (`nvidia-smi` via `kubectl exec`).
3. Report generator: TTFT, output tok/s, e2e p50/p95, req/s, failures, GPU memory → CSV/JSON/Markdown + PNG plots.
4. Helm chart (`deploy/helm/llm-serving`) with one shared `engine:` block rendered into each backend's native config. Also a model pre-download Job and a shared PVC.
5. Tests (pytest, no GPU): payload mapping, stream parsing (incl. Triton's cumulative text), metrics math, config validation, Helm rendering.
6. Docs: README, tutorial, findings report (results **pending**), demo script.

## 4. What cannot be validated without a GPU
- Either server actually starting: model load, CUDA graph capture, KV allocation, probe timings.
- Whether the patched Triton model.py loads in the 26.07 container (Python syntax and patch application are checked; runtime is not).
- The exact SSE wire format of Triton's `generate_stream` for this model. The client is written against the documented format and is tolerant of cumulative vs. delta chunks.
- `nvidia-smi` being present inside both pods (normally injected by the NVIDIA container toolkit).
- Whether the 8 GiB KV budget + CUDA graphs fit on an L4 without OOM. The math says yes; runtime unverified.
- **All performance numbers.**
