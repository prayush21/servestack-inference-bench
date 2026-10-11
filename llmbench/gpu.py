"""Optional GPU sampling via any command that prints nvidia-smi CSV.

The default query is meant to run inside the serving pod, e.g.
  kubectl exec -n llm deploy/vllm -- nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
"""

from __future__ import annotations

import asyncio
import shlex
import time
from typing import Any

NVIDIA_SMI_QUERY = "--query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
NVIDIA_SMI_INFO_QUERY = "--query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader"


def parse_smi_sample(output: str) -> dict[str, float] | None:
    """Parse the first GPU line of `memory.used, memory.total, utilization.gpu` (MiB, MiB, %)."""
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            used, total, util = (float(p) for p in parts[:3])
        except ValueError:
            continue  # e.g. "[N/A]"
        return {"memory_used_mib": used, "memory_total_mib": total, "utilization_pct": util}
    return None


class GpuSampler:
    """Runs `command` every `interval_s` seconds in the background; tags samples with a label
    (the current concurrency level / phase) so the report can attribute them."""

    def __init__(self, command: str, interval_s: float = 1.0) -> None:
        self.argv = shlex.split(command)
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.errors = 0
        self.label: dict[str, Any] = {}
        self._task: asyncio.Task[None] | None = None

    async def sample_once(self) -> dict[str, Any] | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=max(5.0, self.interval_s * 5))
        except (OSError, asyncio.TimeoutError):
            self.errors += 1
            return None
        parsed = parse_smi_sample(out.decode(errors="replace")) if proc.returncode == 0 else None
        if parsed is None:
            self.errors += 1
            return None
        sample = {"t": time.time(), **self.label, **parsed}
        self.samples.append(sample)
        return sample

    async def _loop(self) -> None:
        while True:
            started = time.perf_counter()
            await self.sample_once()
            await asyncio.sleep(max(0.0, self.interval_s - (time.perf_counter() - started)))

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


async def run_command(command: str, timeout_s: float = 15.0) -> str | None:
    """Run a one-off informational command (e.g. GPU name/driver); None on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(command), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except (OSError, asyncio.TimeoutError):
        return None
    return out.decode(errors="replace").strip() if proc.returncode == 0 else None
