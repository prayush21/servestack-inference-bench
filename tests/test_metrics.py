"""Result calculations, checked against hand-computed values."""

from __future__ import annotations

import pytest

from llmbench.metrics import output_agreement, percentile, relative_change, summarize, summarize_level, tpot_s


def rec(i, *, backend="vllm", c=2, phase="measure", ok=True, start=0.0, end=1.0, ttft=0.1, tokens=10,
        max_tokens=10, prompt="p0", text="x", source="server_usage"):
    return {"backend": backend, "concurrency": c, "phase": phase, "request_index": i, "ok": ok,
            "start_offset_s": start, "end_offset_s": end, "ttft_s": ttft if ok else None,
            "e2e_s": end - start, "output_tokens": tokens if ok else None, "max_tokens": max_tokens,
            "prompt_id": prompt, "text": text, "output_tokens_source": source if ok else "unknown",
            "error": None if ok else "HTTP 500: boom"}


@pytest.mark.parametrize("p,expected", [(0, 1.0), (50, 2.5), (95, 3.85), (100, 4.0), (25, 1.75)])
def test_percentile_linear_interpolation_matches_numpy(p, expected):
    # numpy.percentile([1, 2, 3, 4], p) gives these values.
    assert percentile([4, 1, 3, 2], p) == pytest.approx(expected)


def test_percentile_edge_cases():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([None, 2.0, float("nan")], 50) == 2.0
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_tpot():
    assert tpot_s(rec(0, start=0, end=1.0, ttft=0.1, tokens=10)) == pytest.approx(0.1)
    assert tpot_s(rec(0, tokens=1)) is None
    assert tpot_s(rec(0, ok=False)) is None


def test_summarize_level_throughput_latency_and_failures():
    records = [
        rec(0, start=0.0, end=1.0, ttft=0.1),
        rec(1, start=0.0, end=2.0, ttft=0.3),
        rec(2, start=1.0, end=3.0, ttft=0.2),
        rec(3, start=2.0, end=4.0, ok=False),   # failure still extends the window
    ]
    s = summarize_level(records, [{"memory_used_mib": 100.0, "memory_total_mib": 24000.0, "utilization_pct": 50.0},
                                  {"memory_used_mib": 300.0, "memory_total_mib": 24000.0, "utilization_pct": 70.0}])
    assert s["requests"] == 4 and s["successes"] == 3 and s["failures"] == 1
    assert s["failure_rate"] == 0.25
    assert s["window_s"] == 4.0
    assert s["request_throughput_rps"] == pytest.approx(3 / 4)
    assert s["output_tokens_total"] == 30
    assert s["output_throughput_tok_s"] == pytest.approx(30 / 4)
    assert s["ttft_p50_ms"] == pytest.approx(200.0)
    assert s["ttft_p95_ms"] == pytest.approx(290.0)            # 0.2 + 0.95*... linear between 0.2 and 0.3
    assert s["e2e_p50_ms"] == pytest.approx(2000.0)
    assert s["e2e_p95_ms"] == pytest.approx(2000.0)
    assert s["gpu_memory_used_peak_mib"] == 300.0 and s["gpu_utilization_mean_pct"] == 60.0
    assert s["error_examples"] == ["HTTP 500: boom"]
    assert s["output_length_mismatches"] == 0


def test_length_mismatch_and_uncounted_are_reported():
    records = [rec(0, tokens=7, max_tokens=10), rec(1), {**rec(2), "output_tokens": None}]
    s = summarize_level(records)
    assert s["output_length_mismatches"] == 1
    assert s["output_tokens_uncounted"] == 1
    assert s["output_tokens_total"] == 17


def test_summarize_excludes_warmup_and_groups_by_backend_and_concurrency():
    records = [rec(0, phase="warmup", end=100.0), rec(0, c=1), rec(1, c=1), rec(0, backend="triton", c=1),
               rec(0, c=4)]
    gpu = [{"concurrency": 1, "phase": "measure", "memory_used_mib": 5.0},
           {"concurrency": 1, "phase": "warmup", "memory_used_mib": 999.0}]
    rows = summarize(records, gpu)
    assert [(r["backend"], r["concurrency"], r["requests"]) for r in rows] == \
        [("triton", 1, 1), ("vllm", 1, 2), ("vllm", 4, 1)]
    assert all(r["window_s"] <= 1.0 for r in rows)            # the 100 s warm-up is excluded
    vllm_c1 = rows[1]
    assert vllm_c1["gpu_memory_used_peak_mib"] == 5.0          # warm-up GPU sample excluded


def test_relative_change():
    assert relative_change(110, 100) == pytest.approx(0.10)
    assert relative_change(None, 100) is None and relative_change(1, 0) is None


def test_output_agreement():
    a = [rec(0, c=1, prompt="p0", text="hello world"), rec(1, c=1, prompt="p1", text="same")]
    b = [rec(0, c=1, backend="triton", prompt="p0", text="hello there"),
         rec(1, c=1, backend="triton", prompt="p1", text="same")]
    out = output_agreement(a, b)
    assert out["prompts_compared"] == 2 and out["exact_match"] == 1
    assert out["mean_common_prefix_frac"] == pytest.approx((6 / 11 + 1.0) / 2)
    assert output_agreement([], b) == {"prompts_compared": 0}


def test_unlabeled_gpu_samples_are_attributed_by_wall_time():
    def r(i, c, wall):
        return {**rec(i, c=c, start=0.0, end=2.0), "wall_start": wall}
    records = [r(0, 1, 100.0), r(1, 1, 102.0), r(0, 4, 200.0)]       # c=1 window 100..104, c=4 200..202
    gpu = [{"t": 101.0, "memory_used_mib": 10.0}, {"t": 150.0, "memory_used_mib": 999.0},
           {"t": 201.5, "memory_used_mib": 40.0}]
    rows = {row["concurrency"]: row for row in summarize(records, gpu)}
    assert rows[1]["gpu_memory_used_peak_mib"] == 10.0 and rows[1]["gpu_samples"] == 1
    assert rows[4]["gpu_memory_used_peak_mib"] == 40.0             # the idle-gap sample is ignored
