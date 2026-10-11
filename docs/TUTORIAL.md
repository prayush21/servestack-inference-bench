# Tutorial: reproduce one inference request on each backend

Goal: send **the same prompt with the same settings** to vLLM and to Triton + TensorRT-LLM, see exactly what goes over the wire, and confirm which backend answered. Allow about 20 minutes with a GPU cluster ready (most of it is image pull and model load).

> Output snippets below show the **shape** of each response, with generated text elided (`…`). This tutorial
> has not yet been run on a GPU (see [VERIFICATION.md](VERIFICATION.md)), so no real outputs are quoted.

## 0. Prerequisites

- A cluster with one NVIDIA GPU node and a default StorageClass (README §1).
- `kubectl`, `helm`, and this repo with `pip install -e .`.

```bash
export NAMESPACE=llm
scripts/prepare_model_cache.sh      # downloads Qwen2.5-1.5B-Instruct @ 989aa79… into a PVC (once)
```

## 1. Build the prompt

Both servers get a **raw string**. vLLM uses `/v1/completions`, not chat; Triton uses `text_input`. So the Qwen chat template is applied client-side, and both tokenize identical text:

```bash
python - <<'EOF'
from llmbench.prompts import qwen_chat
print(repr(qwen_chat("Explain what a KV cache is in one paragraph.")))
EOF
# '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain what a KV cache is in one paragraph.<|im_end|>\n<|im_start|>assistant\n'
```

That string matches the model's own `chat_template` for one user turn (checked against `tokenizer_config.json` at the pinned revision). Qwen2.5 adds no BOS token, so there are no hidden special-token differences.

## 2. vLLM

```bash
helm install vllm deploy/helm/llm-serving -n llm -f deploy/helm/llm-serving/values-vllm.yaml
kubectl -n llm rollout status deploy/vllm --timeout=45m
kubectl -n llm port-forward svc/vllm 8000:8000 &
```

Raw request (OpenAI-compatible completions, streamed):

```bash
PROMPT=$(python -c 'from llmbench.prompts import qwen_chat; import json; print(json.dumps(qwen_chat("Explain what a KV cache is in one paragraph.")))')
curl -sN http://localhost:8000/v1/completions -H 'content-type: application/json' -d "{
  \"model\": \"qwen2.5-1.5b-instruct\", \"prompt\": $PROMPT,
  \"max_tokens\": 64, \"temperature\": 0, \"seed\": 0, \"ignore_eos\": true,
  \"stream\": true, \"stream_options\": {\"include_usage\": true}}"
```

Expected shape: one SSE event per token with a **delta**, then a usage event, then `[DONE]`:

```
data: {"id":"cmpl-…","object":"text_completion","choices":[{"index":0,"text":"A","finish_reason":null,…}],…}
data: {"id":"cmpl-…","choices":[{"index":0,"text":" KV","finish_reason":null,…}],…}
…
data: {"id":"cmpl-…","choices":[{"index":0,"text":"…","finish_reason":"length",…}],…}
data: {"id":"cmpl-…","choices":[],"usage":{"prompt_tokens":…,"completion_tokens":64,…}}
data: [DONE]
```

Same request through the unified client:

```bash
llmbench generate --backend vllm --url http://localhost:8000 --max-tokens 64 --ignore-eos \
  "Explain what a KV cache is in one paragraph."
```
```json
{ "backend": "vllm", "ok": true, "text": "…", "finish_reason": "length",
  "ttft_s": …, "e2e_s": …, "output_tokens": 64, "output_tokens_source": "server_usage",
  "stream_mode": "delta", … }
```

Stop vLLM before starting Triton (one GPU):

```bash
kill %1; helm uninstall vllm -n llm
```

## 3. Triton + TensorRT-LLM (LLM API)

```bash
helm install triton deploy/helm/llm-serving -n llm -f deploy/helm/llm-serving/values-triton.yaml
kubectl -n llm rollout status deploy/triton --timeout=45m
kubectl -n llm port-forward svc/triton 8000:8000 &
llmbench info --backend triton --url http://localhost:8000 | grep ignore_eos   # → "ignore_eos_supported": true
```

Under the hood, the pod runs `mpirun -n 1 tritonserver --model-repository=/models`. The single model `tensorrt_llm` is NVIDIA's LLM-API Python model, which builds `tensorrt_llm.LLM(**model.yaml)` with `backend: pytorch`, so there's no engine build. Check the rendered config with:

```bash
kubectl -n llm get configmap triton-model-repo -o jsonpath='{.data.model\.yaml}'
```

Raw request (Triton HTTP *generate* extension, streamed):

```bash
curl -sN http://localhost:8000/v2/models/tensorrt_llm/generate_stream -H 'content-type: application/json' -d "{
  \"text_input\": $PROMPT, \"streaming\": true,
  \"sampling_param_max_tokens\": 64, \"sampling_param_top_k\": 1, \"sampling_param_seed\": 0,
  \"sampling_param_ignore_eos\": true, \"sampling_param_exclude_input_from_output\": true,
  \"return_finish_reason\": true}"
```

Note the differences from vLLM:

| | vLLM | Triton LLM-API model |
|---|---|---|
| Greedy | `temperature: 0` | `sampling_param_top_k: 1` (no temperature sent) |
| Output length | `max_tokens` + `ignore_eos` | `sampling_param_max_tokens` + `sampling_param_ignore_eos` (**needs this repo's 3-line patch**) |
| Prompt echo | never | `sampling_param_exclude_input_from_output: true` must be explicit (upstream default is `false`) |
| Stream chunks | delta text | **cumulative** text (each event repeats everything so far) |
| Token count | `usage.completion_tokens` | not returned; `finish_reason: "length"` ⇒ exactly `max_tokens` |

Expected shape: one SSE event per step, each carrying the full text so far:

```
data: {"model_name":"tensorrt_llm","model_version":"1","text_output":"A", "finish_reason":…, …}
data: {"model_name":"tensorrt_llm","model_version":"1","text_output":"A KV", …}
…
data: {"model_name":"tensorrt_llm","model_version":"1","text_output":"A KV cache … ", "finish_reason":"length", …}
```

The exact JSON encoding of 1-element outputs (`"length"` vs `["length"]`) wasn't verifiable without a GPU, and the client accepts both. Through the unified client:

```bash
llmbench generate --backend triton --url http://localhost:8000 --max-tokens 64 --ignore-eos \
  "Explain what a KV cache is in one paragraph."
```
```json
{ "backend": "triton", "ok": true, "text": "…", "finish_reason": "length",
  "ttft_s": …, "e2e_s": …, "output_tokens": 64, "output_tokens_source": "finish_reason_length",
  "stream_mode": "cumulative", … }
```

`stream_mode: "cumulative"` is detected from the first two chunks, not assumed, so the text isn't double-counted.

## 4. Same request via the gateway (optional)

```bash
pip install -e '.[gateway]'
LLMBENCH_TRITON_URL=http://localhost:8000 LLMBENCH_DEFAULT_BACKEND=triton llmbench gateway --port 9000 &
curl -si localhost:9000/v1/generate -H 'content-type: application/json' \
  -d '{"prompt": "Explain what a KV cache is in one paragraph.", "max_tokens": 64}' | grep -i x-llm-backend
# x-llm-backend: triton
```

## 5. Clean up

```bash
kill %1 %2 2>/dev/null; helm uninstall triton -n llm
scripts/teardown.sh        # also removes the model cache PVC and namespace
```

## No GPU? Rehearse the client side

```bash
llmbench fake-server --port 18000 &          # FAKE server speaking both wire formats
llmbench generate --backend triton --url http://localhost:18000 --max-tokens 5 "hi"
```

This exercises parsing and timing only. The output is placeholder tokens (` tok0 tok1 …`), and any report built from it is stamped SYNTHETIC.
