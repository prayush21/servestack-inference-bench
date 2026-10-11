"""A thin HTTP gateway: one request format for both backends, and every response says which
backend served it (JSON field `backend` and header `X-LLM-Backend`).

Configure with env vars (only configured backends are routable):
  LLMBENCH_VLLM_URL=http://localhost:8000  LLMBENCH_TRITON_URL=http://localhost:8001
  LLMBENCH_DEFAULT_BACKEND=vllm  LLMBENCH_MODEL=qwen2.5-1.5b-instruct
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from llmbench.backends import DEFAULT_MODEL, Backend, make_backend
from llmbench.prompts import qwen_chat
from llmbench.schema import GenerationRequest


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1)
    backend: Literal["vllm", "triton"] | None = None
    max_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    seed: int | None = Field(0, ge=0)
    ignore_eos: bool = False
    stream: bool = True  # stream upstream so TTFT is measured; the gateway itself replies once
    apply_chat_template: bool = True


def backends_from_env(env: dict[str, str] | None = None) -> dict[str, Backend]:
    env = dict(os.environ if env is None else env)
    model = env.get("LLMBENCH_MODEL", DEFAULT_MODEL)
    out: dict[str, Backend] = {}
    for name in ("vllm", "triton"):
        url = env.get(f"LLMBENCH_{name.upper()}_URL")
        if url:
            out[name] = make_backend(name, url, model)
    return out


def create_app(backends: dict[str, Backend] | None = None, default_backend: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not app.state.backends:
            app.state.backends = backends_from_env()
        yield
        for b in app.state.backends.values():
            await b.aclose()

    app = FastAPI(title="llmbench gateway", version="0.1.0", lifespan=lifespan)
    app.state.backends = backends or {}
    app.state.default_backend = default_backend or os.environ.get("LLMBENCH_DEFAULT_BACKEND", "vllm")

    @app.get("/backends")
    async def list_backends() -> dict:
        return {"default": app.state.default_backend,
                "backends": {n: {"url": b.base_url, "ready": await b.health()}
                             for n, b in app.state.backends.items()}}

    @app.post("/v1/generate")
    async def generate(body: GenerateBody, response: Response) -> dict:
        name = body.backend or app.state.default_backend
        backend = app.state.backends.get(name)
        if backend is None:
            raise HTTPException(400, f"backend {name!r} is not configured; have {sorted(app.state.backends)}")
        prompt = qwen_chat(body.prompt) if body.apply_chat_template else body.prompt
        req = GenerationRequest(prompt=prompt, max_tokens=body.max_tokens, temperature=body.temperature,
                                top_p=body.top_p, seed=body.seed, ignore_eos=body.ignore_eos, stream=body.stream)
        result = await backend.generate(req)
        response.headers["X-LLM-Backend"] = result.backend
        if not result.ok:
            response.status_code = 502
        return result.to_dict()

    return app


app = create_app()
