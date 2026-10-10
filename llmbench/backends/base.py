"""Shared request/timing flow. Subclasses only describe payloads and how to read responses."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx

from llmbench.schema import GenerationRequest, GenerationResult
from llmbench.sse import aiter_sse_data

TokenCounter = Callable[[str], int]


class BackendError(RuntimeError):
    """The server reported an error inside an otherwise successful HTTP response."""


def normalize_finish_reason(value: Any) -> str | None:
    """'length', ['length'], 'FinishReason.LENGTH' -> 'length'."""
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    s = str(value).strip().lower()
    if s.startswith("finishreason."):
        s = s.split(".", 1)[1]
    return s or None


class StreamAccumulator(ABC):
    """Collects one streamed response. `on_event` gets each SSE data payload and the
    perf_counter() timestamp at which it arrived."""

    def __init__(self) -> None:
        self.text = ""
        self.first_token_at: float | None = None
        self.chunks = 0
        self.finish_reason: str | None = None
        self.server_output_tokens: int | None = None
        self.stream_mode: str | None = None

    @abstractmethod
    def on_event(self, data: str, now: float) -> None: ...

    def _mark_text(self, now: float) -> None:
        if self.text and self.first_token_at is None:
            self.first_token_at = now


class Backend(ABC):
    name: str = "base"

    def __init__(
        self,
        base_url: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        request_timeout_s: float = 300.0,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout_s = request_timeout_s
        self.token_counter = token_counter
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_s, connect=10.0, pool=None),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> "Backend":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- backend-specific pieces -------------------------------------------------------
    @abstractmethod
    def build_payload(self, req: GenerationRequest) -> dict[str, Any]: ...

    @abstractmethod
    def endpoint(self, stream: bool) -> str: ...

    @abstractmethod
    def new_accumulator(self) -> StreamAccumulator: ...

    @abstractmethod
    def parse_unary(self, body: dict[str, Any]) -> StreamAccumulator:
        """Parse a non-streaming response into the same accumulator shape."""

    @abstractmethod
    async def health(self) -> bool: ...

    @abstractmethod
    async def server_info(self) -> dict[str, Any]: ...

    # --- shared flow ---------------------------------------------------------------------
    def _count_tokens(self, req: GenerationRequest, acc: StreamAccumulator) -> tuple[int | None, str]:
        if acc.server_output_tokens is not None:
            return acc.server_output_tokens, "server_usage"
        if acc.finish_reason == "length":
            # The engine stopped because it produced exactly max_tokens tokens.
            return req.max_tokens, "finish_reason_length"
        if self.token_counter is not None:
            return self.token_counter(acc.text), "tokenizer"
        return None, "unknown"

    def _result(self, req: GenerationRequest, acc: StreamAccumulator, t0: float, t_end: float,
                status: int) -> GenerationResult:
        tokens, source = self._count_tokens(req, acc)
        ok = bool(acc.text)
        return GenerationResult(
            backend=self.name,
            model=self.model,
            ok=ok,
            text=acc.text,
            error=None if ok else "empty response",
            status_code=status,
            finish_reason=acc.finish_reason,
            ttft_s=(acc.first_token_at - t0) if acc.first_token_at is not None else None,
            e2e_s=t_end - t0,
            output_tokens=tokens,
            output_tokens_source=source,
            stream_chunks=acc.chunks,
            stream_mode=acc.stream_mode,
        )

    def _failure(self, error: str, t0: float, status: int | None = None) -> GenerationResult:
        return GenerationResult(backend=self.name, model=self.model, ok=False, error=error,
                                status_code=status, e2e_s=time.perf_counter() - t0)

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        req.validate()
        payload = self.build_payload(req)
        url = self.base_url + self.endpoint(req.stream)
        t0 = time.perf_counter()
        try:
            if req.stream:
                async with self.client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode(errors="replace")
                        return self._failure(f"HTTP {resp.status_code}: {body[:500]}", t0, resp.status_code)
                    acc = self.new_accumulator()
                    async for data in aiter_sse_data(resp.aiter_lines()):
                        acc.on_event(data, time.perf_counter())
                    return self._result(req, acc, t0, time.perf_counter(), resp.status_code)
            resp = await self.client.post(url, json=payload)
            t_end = time.perf_counter()
            if resp.status_code != 200:
                return self._failure(f"HTTP {resp.status_code}: {resp.text[:500]}", t0, resp.status_code)
            acc = self.parse_unary(resp.json())
            # No streaming: the first token is only observable when the whole body arrives.
            acc.first_token_at = None
            return self._result(req, acc, t0, t_end, resp.status_code)
        except BackendError as e:
            return self._failure(f"server error: {e}", t0, 200)
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return self._failure(f"{type(e).__name__}: {e}", t0)
