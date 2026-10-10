"""Benchmark configuration: load, validate, and describe exactly what will be sent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llmbench.schema import GenerationRequest


@dataclass(frozen=True)
class Prompt:
    id: str
    prompt: str
    category: str = ""


@dataclass(frozen=True)
class SamplingConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = 0
    ignore_eos: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str
    model_revision: str
    served_model_name: str
    prompts_file: str
    sampling: SamplingConfig
    concurrency: list[int]
    min_requests_per_level: int = 64
    rounds_per_level: int = 4
    warmup_rounds: int = 1
    min_warmup_requests: int = 4
    request_timeout_s: float = 300.0
    gpu_sample_interval_s: float = 1.0
    source_path: str | None = field(default=None, compare=False)

    def requests_for(self, concurrency: int) -> int:
        """Measured requests at a level: enough rounds that every worker runs several times."""
        return max(self.min_requests_per_level, concurrency * self.rounds_per_level)

    def warmups_for(self, concurrency: int) -> int:
        return max(self.min_warmup_requests, concurrency * self.warmup_rounds)

    def request_for(self, prompt: Prompt) -> GenerationRequest:
        s = self.sampling
        return GenerationRequest(prompt=prompt.prompt, max_tokens=s.max_tokens, temperature=s.temperature,
                                 top_p=s.top_p, top_k=s.top_k, seed=s.seed, ignore_eos=s.ignore_eos,
                                 stream=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigError(ValueError):
    pass


_REQUIRED = ("model_id", "model_revision", "served_model_name", "prompts_file", "sampling", "concurrency")


def validate_config(raw: dict[str, Any]) -> list[str]:
    """Return human-readable problems; empty list means valid."""
    errors: list[str] = []
    for key in _REQUIRED:
        if key not in raw:
            errors.append(f"missing required key: {key}")
    known = set(BenchmarkConfig.__dataclass_fields__) - {"source_path"}
    for key in sorted(set(raw) - known):
        errors.append(f"unknown key: {key}")
    rev = raw.get("model_revision")
    if rev is not None and not (isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)):
        errors.append("model_revision must be a full 40-char commit SHA")
    conc = raw.get("concurrency")
    if conc is not None:
        if not isinstance(conc, list) or not conc or not all(isinstance(c, int) and c >= 1 for c in conc):
            errors.append("concurrency must be a non-empty list of positive integers")
        elif len(set(conc)) != len(conc):
            errors.append("concurrency levels must be unique")
    sampling = raw.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, dict):
            errors.append("sampling must be a mapping")
        else:
            unknown = set(sampling) - set(SamplingConfig.__dataclass_fields__)
            if unknown:
                errors.append(f"unknown sampling keys: {sorted(unknown)}")
            else:
                try:
                    s = SamplingConfig(**sampling)
                    GenerationRequest(prompt="x", max_tokens=s.max_tokens, temperature=s.temperature, top_p=s.top_p,
                                      top_k=s.top_k, seed=s.seed, ignore_eos=s.ignore_eos).validate()
                except (TypeError, ValueError) as e:
                    errors.append(f"invalid sampling: {e}")
    for key in ("min_requests_per_level", "rounds_per_level", "min_warmup_requests"):
        if key in raw and not (isinstance(raw[key], int) and raw[key] >= 1):
            errors.append(f"{key} must be a positive integer")
    if "warmup_rounds" in raw and not (isinstance(raw["warmup_rounds"], int) and raw["warmup_rounds"] >= 0):
        errors.append("warmup_rounds must be an integer >= 0")
    for key in ("request_timeout_s", "gpu_sample_interval_s"):
        if key in raw and not (isinstance(raw[key], (int, float)) and raw[key] > 0):
            errors.append(f"{key} must be > 0")
    return errors


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at top level")
    errors = validate_config(raw)
    if errors:
        raise ConfigError(f"{path}: " + "; ".join(errors))
    raw = dict(raw)
    raw["sampling"] = SamplingConfig(**raw["sampling"])
    return BenchmarkConfig(**raw, source_path=str(path))


def load_prompts(path: str | Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if "id" not in obj or "prompt" not in obj:
            raise ConfigError(f"{path}:{n}: each line needs 'id' and 'prompt'")
        if obj["id"] in seen:
            raise ConfigError(f"{path}:{n}: duplicate prompt id {obj['id']!r}")
        seen.add(obj["id"])
        prompts.append(Prompt(id=obj["id"], prompt=obj["prompt"], category=obj.get("category", "")))
    if not prompts:
        raise ConfigError(f"{path}: no prompts")
    return prompts


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
