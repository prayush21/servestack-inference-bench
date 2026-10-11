"""A FAKE server that mimics the vLLM and Triton wire formats, for tests and GPU-less smoke runs.

It generates placeholder tokens with artificial delays. Its numbers mean nothing about real
serving performance; it identifies itself as FAKE in /version and /v2 so any run made against
it is labeled as such in meta.json and in reports.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

FAKE_VERSION = "FAKE-0.0.0 (synthetic test server, not a benchmark)"


@dataclass
class FakeEngine:
    ttft_s: float = 0.002
    itl_s: float = 0.0005
    fail_every: int = 0          # make every Nth request fail with HTTP 500 (0 = never)
    triton_has_ignore_eos: bool = True
    _requests: int = 0
    _active: int = 0

    def _tokens(self, n: int) -> list[str]:
        return [f" tok{i}" for i in range(n)]

    async def _delay(self, base: float) -> None:
        # Mild slowdown with concurrency, so curves are not flat in smoke runs.
        await asyncio.sleep(base * (1 + self._active / 16))

    async def vllm_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        self._active += 1
        try:
            await self._delay(self.ttft_s)
            toks = self._tokens(int(body.get("max_tokens", 16)))
            for i, t in enumerate(toks):
                if i:
                    await self._delay(self.itl_s)
                last = i == len(toks) - 1
                chunk = {"choices": [{"index": 0, "text": t, "finish_reason": "length" if last else None}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            if body.get("stream_options", {}).get("include_usage"):
                yield f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': len(toks)}})}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            self._active -= 1

    async def triton_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Cumulative text in every chunk, exactly like TRT-LLM's LLM API CompletionOutput.text."""
        self._active += 1
        try:
            await self._delay(self.ttft_s)
            toks = self._tokens(int(body.get("sampling_param_max_tokens", 16)))
            text = ""
            for i, t in enumerate(toks):
                if i:
                    await self._delay(self.itl_s)
                text += t
                last = i == len(toks) - 1
                chunk = {"model_name": "tensorrt_llm", "model_version": "1", "text_output": text,
                         "finish_reason": ["length" if last else "None"]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
        finally:
            self._active -= 1

    async def handle(self, request: httpx.Request) -> httpx.Response:
        """httpx.MockTransport handler."""
        path = request.url.path
        if request.method == "GET":
            status, payload = self.get(path)
            return httpx.Response(status, json=payload)
        body = json.loads(request.content or b"{}")
        self._requests += 1
        if self.fail_every and self._requests % self.fail_every == 0:
            return httpx.Response(500, json={"error": "fake failure"})
        if path == "/v1/completions":
            if body.get("stream"):
                return httpx.Response(200, content=self.vllm_stream(body),
                                      headers={"content-type": "text/event-stream"})
            n = int(body.get("max_tokens", 16))
            return httpx.Response(200, json={"choices": [{"text": "".join(self._tokens(n)), "finish_reason": "length"}],
                                             "usage": {"completion_tokens": n}})
        if path == "/v2/models/tensorrt_llm/generate_stream":
            if "sampling_param_ignore_eos" in body and not self.triton_has_ignore_eos:
                return httpx.Response(400, json={"error": "unexpected inference input 'sampling_param_ignore_eos'"})
            return httpx.Response(200, content=self.triton_stream(body), headers={"content-type": "text/event-stream"})
        if path == "/v2/models/tensorrt_llm/generate":
            n = int(body.get("sampling_param_max_tokens", 16))
            return httpx.Response(200, json={"text_output": "".join(self._tokens(n)), "finish_reason": ["length"]})
        return httpx.Response(404, json={"error": f"no route {path}"})

    def get(self, path: str) -> tuple[int, Any]:
        routes: dict[str, Any] = {
            "/health": {},
            "/version": {"version": FAKE_VERSION},
            "/v1/models": {"data": [{"id": "fake"}]},
            "/v2": {"name": "triton", "version": FAKE_VERSION},
            "/v2/health/live": {},
            "/v2/models/tensorrt_llm/ready": {},
            "/v2/models/tensorrt_llm/config": {"input": [
                {"name": "text_input"}, {"name": "sampling_param_max_tokens"},
                *([{"name": "sampling_param_ignore_eos"}] if self.triton_has_ignore_eos else [])]},
        }
        if path in routes:
            return 200, routes[path]
        return 404, {"error": "not found"}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


def fake_client(engine: FakeEngine | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=(engine or FakeEngine()).transport(), base_url="http://fake")


def create_fake_app(engine: FakeEngine | None = None):
    """Serve the fake over real HTTP (both wire formats on one port) for CLI smoke tests."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    engine = engine or FakeEngine()

    async def route(request):
        upstream = httpx.Request(request.method, f"http://fake{request.url.path}", content=await request.body())
        resp = await engine.handle(upstream)
        if resp.headers.get("content-type") == "text/event-stream":
            return StreamingResponse(resp.aiter_raw(), media_type="text/event-stream")
        return JSONResponse(json.loads(await resp.aread()), status_code=resp.status_code)

    return Starlette(routes=[Route("/{path:path}", route, methods=["GET", "POST"])])
