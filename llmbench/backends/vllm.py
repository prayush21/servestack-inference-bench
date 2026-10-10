"""vLLM via its OpenAI-compatible /v1/completions endpoint."""

from __future__ import annotations

import json
from typing import Any

from llmbench.backends.base import Backend, BackendError, StreamAccumulator, normalize_finish_reason
from llmbench.schema import GenerationRequest


class VLLMAccumulator(StreamAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.stream_mode = "delta"  # OpenAI streaming always sends deltas

    def on_event(self, data: str, now: float) -> None:
        if data.strip() == "[DONE]":
            return
        obj = json.loads(data)
        if "error" in obj:
            raise BackendError(str(obj["error"]))
        for choice in obj.get("choices") or []:
            piece = choice.get("text") or ""
            if piece:
                self.chunks += 1
                self.text += piece
                self._mark_text(now)
            if choice.get("finish_reason") is not None:
                self.finish_reason = normalize_finish_reason(choice["finish_reason"])
        usage = obj.get("usage")
        if usage and usage.get("completion_tokens") is not None:
            self.server_output_tokens = int(usage["completion_tokens"])


class VLLMBackend(Backend):
    name = "vllm"

    def build_payload(self, req: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            # vLLM treats temperature 0 as greedy.
            "temperature": req.temperature,
            "top_p": req.top_p,
            "ignore_eos": req.ignore_eos,
            "stream": req.stream,
        }
        if req.top_k is not None and not req.greedy:
            payload["top_k"] = req.top_k
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.stream:
            # Final chunk carries usage.completion_tokens: a server-side token count.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def endpoint(self, stream: bool) -> str:
        return "/v1/completions"

    def new_accumulator(self) -> StreamAccumulator:
        return VLLMAccumulator()

    def parse_unary(self, body: dict[str, Any]) -> StreamAccumulator:
        acc = VLLMAccumulator()
        acc.stream_mode = None
        acc.on_event(json.dumps(body), 0.0)
        return acc

    async def health(self) -> bool:
        try:
            r = await self.client.get(self.base_url + "/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def server_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": self.name, "base_url": self.base_url}
        for key, path in (("version", "/version"), ("models", "/v1/models")):
            try:
                r = await self.client.get(self.base_url + path, timeout=5.0)
                info[key] = r.json() if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as e:  # informational only
                info[key] = f"unavailable: {e}"
        info["ignore_eos_supported"] = True
        return info
