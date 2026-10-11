"""Metric definitions. Pure functions over the raw per-request records written by loadgen.

Definitions (all from the client's point of view):
- TTFT: request sent -> first non-empty text chunk received.
- E2E latency: request sent -> stream closed.
- TPOT (time per output token): (E2E - TTFT) / (output_tokens - 1), i.e. mean inter-token
  time after the first token.
- Window: first measured request start -> last measured request end, at one concurrency level.
- Request throughput: successful requests / window.
- Output token throughput: sum of output tokens of successful requests / window.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default 'linear' method)."""
    if not 0 <= p <= 100:
        raise ValueError("p must be within [0, 100]")
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def mean(values: Iterable[float]) -> float | None:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def tpot_s(rec: dict[str, Any]) -> float | None:
    n, e2e, ttft = rec.get("output_tokens"), rec.get("e2e_s"), rec.get("ttft_s")
    if not rec.get("ok") or n is None or n < 2 or e2e is None or ttft is None:
        return None
    return (e2e - ttft) / (n - 1)


def _ms(x: float | None) -> float | None:
    return None if x is None else x * 1000.0


def summarize_level(records: Sequence[dict[str, Any]],
                    gpu_samples: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """Summarize the measured requests of one (backend, concurrency) group."""
    if not records:
        raise ValueError("no records to summarize")
    ok = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok")]
    window = max(r["end_offset_s"] for r in records) - min(r["start_offset_s"] for r in records)
    counted = [r for r in ok if r.get("output_tokens") is not None]
    total_out = sum(r["output_tokens"] for r in counted)
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    e2es = [r["e2e_s"] for r in ok if r.get("e2e_s") is not None]
    tpots = [t for t in (tpot_s(r) for r in ok) if t is not None]
    # With ignore_eos every request should produce exactly max_tokens; anything else means the
    # comparison is not like-for-like and must be visible in the report.
    length_mismatch = sum(1 for r in counted if r.get("max_tokens") is not None
                          and r["output_tokens"] != r["max_tokens"])
    mem = [s["memory_used_mib"] for s in gpu_samples if s.get("memory_used_mib") is not None]
    util = [s["utilization_pct"] for s in gpu_samples if s.get("utilization_pct") is not None]
    total_mem = [s["memory_total_mib"] for s in gpu_samples if s.get("memory_total_mib") is not None]
    first = records[0]
    return {
        "backend": first["backend"],
        "concurrency": first["concurrency"],
        "requests": len(records),
        "successes": len(ok),
        "failures": len(failed),
        "failure_rate": len(failed) / len(records),
        "error_examples": [e for e, _ in Counter(r.get("error") for r in failed).most_common(3)],
        "window_s": window,
        "request_throughput_rps": len(ok) / window if window > 0 else None,
        "output_tokens_total": total_out,
        "output_throughput_tok_s": total_out / window if window > 0 else None,
        "ttft_p50_ms": _ms(percentile(ttfts, 50)),
        "ttft_p95_ms": _ms(percentile(ttfts, 95)),
        "e2e_p50_ms": _ms(percentile(e2es, 50)),
        "e2e_p95_ms": _ms(percentile(e2es, 95)),
        "tpot_p50_ms": _ms(percentile(tpots, 50)),
        "tpot_p95_ms": _ms(percentile(tpots, 95)),
        "output_tokens_uncounted": len(ok) - len(counted),
        "output_length_mismatches": length_mismatch,
        "token_count_sources": dict(Counter(r.get("output_tokens_source") for r in ok)),
        "gpu_memory_used_peak_mib": max(mem) if mem else None,
        "gpu_memory_total_mib": max(total_mem) if total_mem else None,
        "gpu_utilization_mean_pct": mean(util),
        "gpu_samples": len(gpu_samples),
    }


def summarize(records: Iterable[dict[str, Any]],
              gpu_samples: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    """One summary row per (backend, concurrency), measured phase only, sorted."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("phase") == "measure":
            groups[(r["backend"], r["concurrency"])].append(r)
    # GPU samples are either labeled by the in-process sampler (concurrency/phase), or, when
    # sampled by a separate `llmbench gpu-watch` process, attributed by wall-clock time to the
    # measured window of each concurrency level.
    windows: dict[int, tuple[float, float]] = {}
    for (_, c), recs in groups.items():
        starts = [r["wall_start"] for r in recs if r.get("wall_start") is not None]
        ends = [r["wall_start"] + r["end_offset_s"] - r["start_offset_s"] for r in recs
                if r.get("wall_start") is not None]
        if starts:
            lo, hi = windows.get(c, (min(starts), max(ends)))
            windows[c] = (min(lo, min(starts)), max(hi, max(ends)))
    gpu_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in gpu_samples:
        if "phase" in s:
            if s["phase"] == "measure":
                gpu_groups[s.get("concurrency")].append(s)
        elif s.get("t") is not None:
            for c, (lo, hi) in windows.items():
                if lo <= s["t"] <= hi:
                    gpu_groups[c].append(s)
    return [summarize_level(recs, gpu_groups.get(c, ())) for (b, c), recs in sorted(groups.items())]


def relative_change(new: float | None, base: float | None) -> float | None:
    """(new - base) / base, or None if undefined."""
    if new is None or base is None or base == 0:
        return None
    return (new - base) / base


def output_agreement(records_a: Iterable[dict[str, Any]], records_b: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compare greedy outputs per prompt between two backends (first measured, successful
    concurrency-1 sample per prompt). Exact agreement is not expected (different kernels and
    accumulation order), but large divergence would hint at a sampling/config mismatch."""
    def first_text(recs: Iterable[dict[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for r in recs:
            if r.get("phase") == "measure" and r.get("concurrency") == 1 and r.get("ok") and "text" in r:
                out.setdefault(r["prompt_id"], r["text"])
        return out

    a, b = first_text(records_a), first_text(records_b)
    common = sorted(set(a) & set(b))
    if not common:
        return {"prompts_compared": 0}
    exact = sum(a[k] == b[k] for k in common)
    prefix_fracs = []
    for k in common:
        x, y = a[k], b[k]
        n = 0
        for cx, cy in zip(x, y):
            if cx != cy:
                break
            n += 1
        prefix_fracs.append(n / max(len(x), len(y), 1))
    return {
        "prompts_compared": len(common),
        "exact_match": exact,
        "exact_match_rate": exact / len(common),
        "mean_common_prefix_frac": sum(prefix_fracs) / len(prefix_fracs),
    }
