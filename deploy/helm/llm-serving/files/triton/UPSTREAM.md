# Vendored Triton LLM-API model files

| File | Source |
|---|---|
| `config.pbtxt` | `triton_backend/all_models/llmapi/tensorrt_llm/config.pbtxt` |
| `model.py` | `triton_backend/all_models/llmapi/tensorrt_llm/1/model.py` (unmodified) |
| `helpers.py` | `triton_backend/all_models/llmapi/tensorrt_llm/1/helpers.py` |

- **Upstream:** [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) tag `v1.2.1`, commit `376f7e1bd8ed543f75014309e3fd4b237e9b0e73`. This is the TRT-LLM version inside `nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3`.
- **License:** the upstream files keep their original NVIDIA copyright headers. The TensorRT-LLM repository is Apache-2.0.
- `model.yaml` is **not** vendored. The Helm chart renders it from `values.yaml` (`templates/configmap-triton.yaml`).

## Why patch?

- Upstream `helpers.py` forwards only a whitelist of `SamplingParams` fields. `ignore_eos` and `min_tokens` are not on it.
- Without `ignore_eos`, output length depends on when the model emits EOS. Output length then differs from vLLM's, which makes tokens/s and latency incomparable.
- [`llmapi-ignore-eos.patch`](llmapi-ignore-eos.patch) adds both fields to the whitelist and declares the matching optional inputs in `config.pbtxt`. No other behavior changes.

Regenerate or verify the files with:

```bash
scripts/vendor_triton_llmapi.sh --check
```
