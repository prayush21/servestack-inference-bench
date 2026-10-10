from __future__ import annotations

from typing import Any

from llmbench.backends.base import Backend, BackendError
from llmbench.backends.triton import TritonBackend
from llmbench.backends.vllm import VLLMBackend

BACKENDS: dict[str, type[Backend]] = {"vllm": VLLMBackend, "triton": TritonBackend}

DEFAULT_MODEL = "qwen2.5-1.5b-instruct"


def make_backend(name: str, base_url: str, model: str = DEFAULT_MODEL, **kwargs: Any) -> Backend:
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None
    return cls(base_url, model, **kwargs)


__all__ = ["Backend", "BackendError", "BACKENDS", "DEFAULT_MODEL", "TritonBackend", "VLLMBackend", "make_backend"]
