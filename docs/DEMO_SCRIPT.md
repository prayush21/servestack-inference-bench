# Demo script (3–5 minutes)

**Audience:** engineers or hiring managers. **Setup before recording:** model cache prepared, vLLM deployed and ready, a terminal at the repo root, and `reports/<run>/` generated if a GPU run exists.

If no GPU run exists yet, use the **[no-GPU variant](#no-gpu-variant)** and say so on camera. Never present fake-server numbers as results.

---

### 0:00–0:30 · The question

> "Two popular ways to serve an LLM on NVIDIA GPUs are vLLM and Triton with TensorRT-LLM. I wanted a comparison
> where the *only* variable is the serving stack: same GPU, same model revision, same prompts, same sampling, same
> KV-cache budget. And I wanted all of it deployable on Kubernetes."

Show the README architecture diagram.

### 0:30–1:30 · One chart, one engine config

```bash
sed -n '/^engine:/,/^$/p' deploy/helm/llm-serving/values.yaml
helm template vllm deploy/helm/llm-serving -f deploy/helm/llm-serving/values-vllm.yaml | grep -A16 'args:'
helm template triton deploy/helm/llm-serving -f deploy/helm/llm-serving/values-triton.yaml | sed -n '/model.yaml/,/triton_config/p'
```

> "One `engine:` block is rendered into vLLM flags and into TensorRT-LLM's `model.yaml`. A test fails if they
> ever disagree. Triton uses NVIDIA's current LLM API path with the PyTorch backend, so there's no engine build step."

Point out: GPU limit 1, `Recreate` strategy, startup probe, images pinned by digest.

### 1:30–2:30 · One request, two backends

```bash
llmbench generate --backend vllm --url http://localhost:8000 --max-tokens 48 --ignore-eos "What is a KV cache?" \
  | python -m json.tool | grep -E '"(backend|ttft_s|e2e_s|output_tokens|output_tokens_source|stream_mode)"'
```

> "Same request object, and the result says which backend served it. vLLM streams deltas and reports token usage.
> Triton's LLM-API model streams *cumulative* text and doesn't report a count. The client handles both and records
> where each number came from."

(If time allows, show the Triton curl from the tutorial and the cumulative chunks.)

### 2:30–3:30 · The fairness problems I had to solve

Open [docs/FINDINGS.md §2](FINDINGS.md#2-what-was-established-without-a-gpu) and hit three points:

1. Triton's model didn't forward `ignore_eos`, so I added a 3-line patch and kept the diff auditable.
2. vLLM silently applies Qwen's `repetition_penalty 1.1` defaults, so I disabled them.
3. The two engines' memory knobs mean different things, so both get an identical 8 GiB KV-cache budget.

### 3:30–4:30 · Results

```bash
open reports/<run>/comparison.png        # or show summary.md side-by-side table
```

Read the real numbers from `summary.md`: TTFT at c=1, throughput at c=64, failures, and the automatic caveats section. Tie any gap to a hypothesis in FINDINGS §4, and state the limitations: one run, one GPU type, closed-loop load.

### 4:30–5:00 · Reproducibility

```bash
NAMESPACE=llm scripts/run_comparison.sh results/<run>
pytest -q
```

> "One command reproduces the run. Every request is in raw JSONL with image digests, model SHA and GPU driver
> recorded, and 93 tests run without a GPU."

---

## No-GPU variant

Replace the 1:30 and 3:30 segments with:

```bash
llmbench fake-server --port 18000 &
llmbench generate --backend triton --url http://localhost:18000 --max-tokens 5 "hi" | python -m json.tool
llmbench report results --out reports/latest && cat reports/latest/summary.md     # shows PENDING
```

> "This is a fake server that speaks both wire formats, so you can see the client handle cumulative versus delta
> streaming. It's not a benchmark. The real results are pending GPU time, and the report says PENDING rather than
> showing made-up numbers."
