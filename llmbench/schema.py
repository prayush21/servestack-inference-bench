"""Backend-neutral request/response types. Every backend client speaks these."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    """One text-completion request, independent of the serving backend.

    `prompt` is sent verbatim: chat templating is the caller's job (the benchmark prompt set
    is stored pre-templated), so both backends tokenize exactly the same string.
    `temperature == 0` means greedy decoding; each backend maps that to its own convention.
    """

    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = 0
    ignore_eos: bool = False
    stream: bool = True

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 or None")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0 or None")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationRequest":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown request fields: {sorted(unknown)}")
        req = cls(**d)
        req.validate()
        return req


@dataclass
class GenerationResult:
    """What came back, plus client-side timings. `backend` says who served it."""

    backend: str
    model: str
    ok: bool
    text: str = ""
    error: str | None = None
    status_code: int | None = None
    finish_reason: str | None = None
    # Seconds, measured with time.perf_counter() on the client.
    ttft_s: float | None = None
    e2e_s: float | None = None
    output_tokens: int | None = None
    # "server_usage" | "finish_reason_length" | "tokenizer" | "unknown"
    output_tokens_source: str = "unknown"
    stream_chunks: int = 0
    # Triton streams cumulative text; recorded so the parsing decision is auditable.
    stream_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
