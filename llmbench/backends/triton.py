"""Triton + TensorRT-LLM (LLM API Python model) via Triton's HTTP generate extension."""

from __future__ import annotations

import json
from typing import Any

from llmbench.backends.base import Backend, BackendError, StreamAccumulator, normalize_finish_reason
from llmbench.schema import GenerationRequest

TRITON_MODEL = "tensorrt_llm"


def _scalar(value: Any) -> Any:
    """Triton's generate extension may return a 1-element tensor as a scalar or a list."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


class TritonAccumulator(StreamAccumulator):
    """TensorRT-LLM's LLM API puts the *cumulative* text in `CompletionOutput.text`, and the
    upstream Triton model.py streams that field. We still detect the mode from the first two
    non-empty chunks instead of assuming it, so a future switch to deltas won't corrupt text."""

    def __init__(self, mode: str = "auto") -> None:
        super().__init__()
        if mode not in ("auto", "cumulative", "delta"):
            raise ValueError(f"invalid stream mode {mode!r}")
        self.stream_mode = None if mode == "auto" else mode
        self._first_piece: str | None = None

    def on_event(self, data: str, now: float) -> None:
        obj = json.loads(data)
        if "error" in obj:
            raise BackendError(str(obj["error"]))
        piece = _scalar(obj.get("text_output"))
        if isinstance(piece, str) and piece:
            self.chunks += 1
            self._apply(piece)
            self._mark_text(now)
        fr = normalize_finish_reason(obj.get("finish_reason"))
        if fr and fr not in ("none", "null"):
            self.finish_reason = fr

    def _apply(self, piece: str) -> None:
        if self.stream_mode is None:
            if self._first_piece is None:
                self._first_piece = piece
                self.text = piece
                return
            self.stream_mode = "cumulative" if piece.startswith(self._first_piece) else "delta"
        if self.stream_mode == "cumulative":
            self.text = piece
        else:
            self.text += piece


class TritonBackend(Backend):
    name = "triton"

    def __init__(self, *args: Any, stream_mode: str = "auto", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stream_mode = stream_mode

    def build_payload(self, req: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text_input": req.prompt,
            "streaming": req.stream,
            "sampling_param_max_tokens": req.max_tokens,
            # Upstream helpers.py defaults this to False; be explicit so only new text returns.
            "sampling_param_exclude_input_from_output": True,
            "return_finish_reason": True,
        }
        if req.greedy:
            # TRT-LLM greedy == top_k 1. Temperature 0 is not sent: its handling differs by version.
            payload["sampling_param_top_k"] = 1
        else:
            payload["sampling_param_temperature"] = req.temperature
            payload["sampling_param_top_p"] = req.top_p
            if req.top_k is not None:
                payload["sampling_param_top_k"] = req.top_k
        if req.seed is not None:
            payload["sampling_param_seed"] = req.seed
        if req.ignore_eos:
            # Requires the patched helpers.py/config.pbtxt in deploy/helm/llm-serving/files/triton.
            payload["sampling_param_ignore_eos"] = True
        return payload

    def endpoint(self, stream: bool) -> str:
        return f"/v2/models/{TRITON_MODEL}/" + ("generate_stream" if stream else "generate")

    def new_accumulator(self) -> StreamAccumulator:
        return TritonAccumulator(self.stream_mode)

    def parse_unary(self, body: dict[str, Any]) -> StreamAccumulator:
        acc = TritonAccumulator("cumulative")
        acc.stream_mode = None
        acc.on_event(json.dumps(body), 0.0)
        return acc

    async def health(self) -> bool:
        try:
            r = await self.client.get(f"{self.base_url}/v2/models/{TRITON_MODEL}/ready", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def server_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": self.name, "base_url": self.base_url}
        try:
            r = await self.client.get(self.base_url + "/v2", timeout=5.0)
            info["server"] = r.json() if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            info["server"] = f"unavailable: {e}"
        try:
            r = await self.client.get(f"{self.base_url}/v2/models/{TRITON_MODEL}/config", timeout=5.0)
            cfg = r.json() if r.status_code == 200 else {}
            inputs = {i.get("name") for i in cfg.get("input", [])}
            info["ignore_eos_supported"] = "sampling_param_ignore_eos" in inputs
            info["model_config_inputs"] = sorted(n for n in inputs if n)
        except Exception as e:
            info["ignore_eos_supported"] = None
            info["model_config_error"] = str(e)
        return info
