"""Load generator -> raw files -> report, against the FAKE server (no GPU). Also the gateway."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmbench.backends import make_backend
from llmbench.config import load_config, load_prompts
from llmbench.fake import FakeEngine, fake_client
from llmbench.gateway import create_app
from llmbench.loadgen import run_benchmark
from llmbench.report import generate_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.chdir(ROOT)
    c = load_config(ROOT / "configs" / "benchmark.yaml")
    # Keep the test fast: tiny outputs, two levels.
    from dataclasses import replace
    return replace(c, sampling=replace(c.sampling, max_tokens=8), min_requests_per_level=6,
                   rounds_per_level=2, min_warmup_requests=2)


def bench(backend_name, cfg, out, engine=None, levels=(1, 3), **kw):
    async def go():
        async with fake_client(engine) as client:
            b = make_backend(backend_name, "http://fake", cfg.served_model_name, client=client)
            return await run_benchmark(b, cfg, load_prompts(cfg.prompts_file), out, levels=list(levels),
                                       cooldown_s=0, log=lambda m: None, **kw)
    return asyncio.run(go())


def test_loadgen_writes_reproducible_raw_results(cfg, tmp_path):
    out = bench("triton", cfg, tmp_path, gpu_cmd="printf '1000, 24000, 55\\n'", gpu_info_cmd="echo FAKE-GPU")
    recs = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    meta = json.loads((out / "meta.json").read_text())
    # Plan: c=1 -> 2 warm-up + 6 measured; c=3 -> 3 warm-up + 6 measured.
    count = {(r["concurrency"], r["phase"]) for r in recs}
    assert count == {(1, "warmup"), (1, "measure"), (3, "warmup"), (3, "measure")}
    assert sum(r["phase"] == "measure" for r in recs) == 12
    assert all(r["ok"] and r["output_tokens"] == 8 and r["backend"] == "triton" for r in recs)
    prompts = load_prompts(cfg.prompts_file)
    measured_c1 = [r for r in recs if r["concurrency"] == 1 and r["phase"] == "measure"]
    assert [r["prompt_id"] for r in measured_c1] == [p.id for p in prompts[:6]]   # fixed order
    assert meta["status"] == "complete" and meta["gpu_info"] == "FAKE-GPU"
    assert meta["config"]["sampling"]["ignore_eos"] is True and len(meta["prompts_sha256"]) == 64
    assert "FAKE" in json.dumps(meta["server_info"])
    gpu = [json.loads(line) for line in (out / "gpu.jsonl").read_text().splitlines()]
    assert gpu and gpu[0]["memory_used_mib"] == 1000.0


def test_concurrency_is_actually_bounded(cfg, tmp_path):
    engine = FakeEngine(ttft_s=0.01)
    peak = 0
    original = engine.triton_stream

    async def tracking(body):
        nonlocal peak
        async for chunk in original(body):
            peak = max(peak, engine._active)
            yield chunk
    engine.triton_stream = tracking
    bench("triton", cfg, tmp_path, engine=engine, levels=(3,))
    assert peak == 3


def test_refuses_unpatched_triton_when_ignore_eos(cfg, tmp_path):
    with pytest.raises(RuntimeError, match="sampling_param_ignore_eos"):
        bench("triton", cfg, tmp_path, engine=FakeEngine(triton_has_ignore_eos=False))


def test_failures_are_recorded_not_dropped(cfg, tmp_path):
    out = bench("vllm", cfg, tmp_path, engine=FakeEngine(fail_every=4), levels=(1,))
    recs = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    assert sum(not r["ok"] for r in recs) == 2 and all(r["error"] for r in recs if not r["ok"])


def test_report_end_to_end(cfg, tmp_path):
    run_dir = tmp_path / "run"
    bench("vllm", cfg, run_dir)
    bench("triton", cfg, run_dir)
    md = generate_report([run_dir], tmp_path / "report")
    text = md.read_text()
    assert "Side by side" in text and "Greedy output agreement" in text
    rows = list(csv.DictReader((tmp_path / "report" / "summary.csv").open()))
    assert {(r["backend"], r["concurrency"]) for r in rows} == {("vllm", "1"), ("vllm", "3"),
                                                                ("triton", "1"), ("triton", "3")}
    assert all(float(r["output_throughput_tok_s"]) > 0 for r in rows)
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["output_agreement"]["exact_match_rate"] == 1.0
    assert (tmp_path / "report" / "comparison.png").stat().st_size > 10_000
    assert "FAKE" in text and "SYNTHETIC RESULTS" in text   # synthetic runs are visibly labeled


def test_report_pending_when_no_results(tmp_path):
    md = generate_report([tmp_path], tmp_path / "out")
    assert "PENDING" in md.read_text()
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["status"] == "pending"


def test_gateway_routes_and_labels_backend():
    engine = FakeEngine()
    backends = {n: make_backend(n, "http://fake", "m", client=fake_client(engine)) for n in ("vllm", "triton")}
    with TestClient(create_app(backends, default_backend="vllm")) as client:
        r = client.post("/v1/generate", json={"prompt": "hello", "max_tokens": 4, "backend": "triton"})
        assert r.status_code == 200 and r.headers["X-LLM-Backend"] == "triton"
        assert r.json()["backend"] == "triton" and r.json()["output_tokens"] == 4
        r = client.post("/v1/generate", json={"prompt": "hello", "max_tokens": 4})
        assert r.json()["backend"] == "vllm"                      # default backend
        assert client.post("/v1/generate", json={"prompt": ""}).status_code == 422
        assert client.post("/v1/generate", json={"prompt": "x", "backend": "tgi"}).status_code == 422
        listing = client.get("/backends").json()
        assert listing["backends"]["triton"]["ready"] is True


def test_gateway_unconfigured_backend_is_400():
    backends = {"vllm": make_backend("vllm", "http://fake", "m", client=fake_client())}
    with TestClient(create_app(backends)) as client:
        r = client.post("/v1/generate", json={"prompt": "x", "backend": "triton"})
        assert r.status_code == 400 and "not configured" in r.json()["detail"]


def test_fake_server_over_real_asgi_http():
    """Guards the HTTP wrapper itself (MockTransport tests bypass it)."""
    from llmbench.fake import create_fake_app

    with TestClient(create_fake_app()) as client:
        assert client.get("/health").status_code == 200
        r = client.post("/v2/models/tensorrt_llm/generate_stream",
                        json={"text_input": "x", "sampling_param_max_tokens": 3})
        assert r.status_code == 200 and r.text.count("data: ") == 3
        r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 2, "stream": False})
        assert r.json()["usage"]["completion_tokens"] == 2
