"""Closed-loop load generator.

For each concurrency level C: run warm-up requests (recorded, but tagged phase=warmup and
excluded from metrics), then a measured batch where exactly C workers each keep one request
in flight until the batch is exhausted. Prompts are used in a fixed order, so every backend
sees the same sequence of requests.
"""

from __future__ import annotations

import asyncio
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import llmbench
from llmbench.backends import Backend
from llmbench.config import BenchmarkConfig, Prompt, sha256_file
from llmbench.gpu import GpuSampler, run_command


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def run_phase(backend: Backend, cfg: BenchmarkConfig, prompts: list[Prompt], concurrency: int,
                    n_requests: int, phase: str, save_text: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    indices = iter(range(n_requests))
    t0 = time.perf_counter()

    async def worker(worker_id: int) -> None:
        for i in indices:  # shared iterator: each index is taken by exactly one worker
            prompt = prompts[i % len(prompts)]
            req = cfg.request_for(prompt)
            wall_start = time.time()
            start = time.perf_counter()
            res = await backend.generate(req)
            end = time.perf_counter()
            rec = {
                "backend": backend.name,
                "concurrency": concurrency,
                "phase": phase,
                "request_index": i,
                "prompt_id": prompt.id,
                "prompt_category": prompt.category,
                "worker": worker_id,
                "max_tokens": req.max_tokens,
                "wall_start": wall_start,
                "start_offset_s": start - t0,
                "end_offset_s": end - t0,
                "ok": res.ok,
                "error": res.error,
                "status_code": res.status_code,
                "finish_reason": res.finish_reason,
                "ttft_s": res.ttft_s,
                "e2e_s": res.e2e_s,
                "output_tokens": res.output_tokens,
                "output_tokens_source": res.output_tokens_source,
                "stream_chunks": res.stream_chunks,
                "stream_mode": res.stream_mode,
                "text_chars": len(res.text),
            }
            if save_text:
                rec["text"] = res.text
            records.append(rec)

    await asyncio.gather(*(worker(w) for w in range(concurrency)))
    records.sort(key=lambda r: r["request_index"])
    return records


async def run_benchmark(
    backend: Backend,
    cfg: BenchmarkConfig,
    prompts: list[Prompt],
    out_dir: str | Path,
    *,
    levels: list[int] | None = None,
    gpu_cmd: str | None = None,
    gpu_info_cmd: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    cooldown_s: float = 2.0,
    save_text: bool = True,
    log=print,
) -> Path:
    out = Path(out_dir) / backend.name
    out.mkdir(parents=True, exist_ok=True)
    levels = levels or cfg.concurrency

    if not await backend.health():
        raise RuntimeError(f"{backend.name} at {backend.base_url} is not healthy/ready")
    info = await backend.server_info()
    if cfg.sampling.ignore_eos and info.get("ignore_eos_supported") is False:
        raise RuntimeError(
            "ignore_eos=true but the Triton model does not declare sampling_param_ignore_eos; "
            "deploy the patched model repository (deploy/helm/llm-serving/files/triton)")

    meta: dict[str, Any] = {
        "status": "running",
        "llmbench_version": llmbench.__version__,
        "started_utc": utc_now(),
        "argv": sys.argv,
        "python": sys.version,
        "client_platform": platform.platform(),
        "backend": backend.name,
        "base_url": backend.base_url,
        "served_model": backend.model,
        "levels": levels,
        "config": cfg.to_dict(),
        "prompts_sha256": sha256_file(cfg.prompts_file),
        "num_prompts": len(prompts),
        "server_info": info,
        "gpu_info": await run_command(gpu_info_cmd) if gpu_info_cmd else None,
        "extra": extra_meta or {},
    }
    meta_path = out / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    sampler = GpuSampler(gpu_cmd, cfg.gpu_sample_interval_s) if gpu_cmd else None
    if sampler:
        sampler.start()
    requests_path = out / "requests.jsonl"
    completed = False
    try:
        with requests_path.open("w") as f:
            for c in levels:
                for phase, n in (("warmup", cfg.warmups_for(c)), ("measure", cfg.requests_for(c))):
                    if n == 0:
                        continue
                    if sampler:
                        sampler.label = {"concurrency": c, "phase": phase}
                    log(f"[{backend.name}] concurrency={c} {phase}: {n} requests")
                    started = time.perf_counter()
                    recs = await run_phase(backend, cfg, prompts, c, n, phase, save_text)
                    failed = sum(not r["ok"] for r in recs)
                    log(f"[{backend.name}]   done in {time.perf_counter() - started:.1f}s, failures={failed}")
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                    f.flush()
                if sampler:
                    sampler.label = {"concurrency": c, "phase": "cooldown"}
                await asyncio.sleep(cooldown_s)
        completed = True
    finally:
        if sampler:
            await sampler.stop()
            with (out / "gpu.jsonl").open("w") as g:
                for s in sampler.samples:
                    g.write(json.dumps(s) + "\n")
            meta["gpu_sampler_errors"] = sampler.errors
            meta["gpu_samples"] = len(sampler.samples)
        meta["finished_utc"] = utc_now()
        meta["status"] = "complete" if completed else "interrupted"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return out
