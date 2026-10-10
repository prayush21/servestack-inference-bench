"""Client behavior: payload mapping, stream parsing, timing, token counting, failure handling."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llmbench.backends import TritonBackend, VLLMBackend, make_backend
from llmbench.backends.base import normalize_finish_reason
from llmbench.backends.triton import TritonAccumulator
from llmbench.backends.vllm import VLLMAccumulator
from llmbench.fake import FAKE_VERSION, FakeEngine, fake_client
from llmbench.schema import GenerationRequest
from llmbench.sse import iter_sse_data

run = asyncio.run


# --- request schema -------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"prompt": ""}, {"prompt": "x", "max_tokens": 0}, {"prompt": "x", "temperature": -1},
    {"prompt": "x", "top_p": 0}, {"prompt": "x", "top_p": 1.5}, {"prompt": "x", "top_k": 0},
    {"prompt": "x", "seed": -1},
])
def test_request_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        GenerationRequest(**kwargs).validate()


def test_request_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown request fields"):
        GenerationRequest.from_dict({"prompt": "hi", "max_new_tokens": 5})


def test_make_backend_unknown_name():
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("sglang", "http://x")


# --- payload mapping: same request, backend-native fields ------------------------------------

GREEDY = GenerationRequest(prompt="P", max_tokens=64, temperature=0.0, seed=7, ignore_eos=True, stream=True)
SAMPLED = GenerationRequest(prompt="P", max_tokens=8, temperature=0.8, top_p=0.9, top_k=40, seed=None, stream=False)


def test_vllm_payload_greedy():
    p = VLLMBackend("http://h", "m").build_payload(GREEDY)
    assert p == {"model": "m", "prompt": "P", "max_tokens": 64, "temperature": 0.0, "top_p": 1.0,
                 "ignore_eos": True, "stream": True, "seed": 7, "stream_options": {"include_usage": True}}


def test_vllm_payload_sampled_non_stream():
    p = VLLMBackend("http://h", "m").build_payload(SAMPLED)
    assert p["temperature"] == 0.8 and p["top_p"] == 0.9 and p["top_k"] == 40
    assert "seed" not in p and "stream_options" not in p and p["stream"] is False


def test_triton_payload_greedy_maps_to_top_k_1():
    p = TritonBackend("http://h", "m").build_payload(GREEDY)
    assert p == {"text_input": "P", "streaming": True, "sampling_param_max_tokens": 64,
                 "sampling_param_exclude_input_from_output": True, "return_finish_reason": True,
                 "sampling_param_top_k": 1, "sampling_param_seed": 7, "sampling_param_ignore_eos": True}
    assert "sampling_param_temperature" not in p


def test_triton_payload_sampled():
    p = TritonBackend("http://h", "m").build_payload(SAMPLED)
    assert p["sampling_param_temperature"] == 0.8 and p["sampling_param_top_p"] == 0.9
    assert p["sampling_param_top_k"] == 40 and "sampling_param_ignore_eos" not in p
    assert p["streaming"] is False


def test_endpoints():
    assert VLLMBackend("http://h/", "m").endpoint(True) == "/v1/completions"
    t = TritonBackend("http://h", "m")
    assert t.endpoint(True) == "/v2/models/tensorrt_llm/generate_stream"
    assert t.endpoint(False) == "/v2/models/tensorrt_llm/generate"


# --- SSE + accumulators ---------------------------------------------------------------------

def test_sse_parsing_handles_comments_multiline_and_trailing_event():
    lines = [": keep-alive", "data: {\"a\": 1}", "", "event: x", "data: line1", "data: line2", "",
             "data:no-space", "", "data: [DONE]"]
    assert list(iter_sse_data(lines)) == ['{"a": 1}', "line1\nline2", "no-space", "[DONE]"]


def test_vllm_accumulator_deltas_usage_and_ttft():
    acc = VLLMAccumulator()
    acc.on_event(json.dumps({"choices": [{"text": "", "finish_reason": None}]}), 1.0)  # empty: not TTFT
    acc.on_event(json.dumps({"choices": [{"text": "Hel", "finish_reason": None}]}), 2.0)
    acc.on_event(json.dumps({"choices": [{"text": "lo", "finish_reason": "length"}]}), 3.0)
    acc.on_event(json.dumps({"choices": [], "usage": {"completion_tokens": 2}}), 3.1)
    acc.on_event("[DONE]", 3.2)
    assert (acc.text, acc.first_token_at, acc.chunks, acc.finish_reason, acc.server_output_tokens) == \
        ("Hello", 2.0, 2, "length", 2)


@pytest.mark.parametrize("chunks,mode,expected", [
    (["The", "The cat", "The cat sat"], "cumulative", "The cat sat"),       # what TRT-LLM LLM API sends
    (["The", " cat", " sat"], "delta", "The cat sat"),
    (["Only"], None, "Only"),                                              # undetermined, still correct
])
def test_triton_accumulator_detects_cumulative_vs_delta(chunks, mode, expected):
    acc = TritonAccumulator()
    for i, c in enumerate(chunks):
        acc.on_event(json.dumps({"text_output": c, "finish_reason": ["None"]}), float(i))
    assert acc.text == expected and acc.stream_mode == mode and acc.first_token_at == 0.0


def test_triton_accumulator_accepts_list_tensors_and_finish_reason():
    acc = TritonAccumulator()
    acc.on_event(json.dumps({"text_output": ["abc"], "finish_reason": ["length"]}), 0.5)
    assert acc.text == "abc" and acc.finish_reason == "length"


@pytest.mark.parametrize("raw,norm", [("length", "length"), (["stop"], "stop"),
                                      ("FinishReason.LENGTH", "length"), (None, None), ([], None)])
def test_normalize_finish_reason(raw, norm):
    assert normalize_finish_reason(raw) == norm


# --- end-to-end over HTTP (fake server via httpx.MockTransport) ------------------------------

@pytest.mark.parametrize("name", ["vllm", "triton"])
def test_generate_stream_reports_backend_timing_and_tokens(name):
    async def go():
        async with fake_client() as client:
            b = make_backend(name, "http://fake", "m", client=client)
            return await b.generate(GREEDY.__class__(prompt="hi", max_tokens=5, ignore_eos=True))
    r = run(go())
    assert r.ok and r.backend == name
    assert r.text == " tok0 tok1 tok2 tok3 tok4"   # Triton's cumulative chunks are not double-counted
    assert r.output_tokens == 5
    assert r.output_tokens_source == ("server_usage" if name == "vllm" else "finish_reason_length")
    assert 0 < r.ttft_s <= r.e2e_s
    assert r.finish_reason == "length"


@pytest.mark.parametrize("name", ["vllm", "triton"])
def test_generate_non_stream_has_no_ttft(name):
    async def go():
        async with fake_client() as client:
            return await make_backend(name, "http://fake", "m", client=client).generate(
                GenerationRequest(prompt="hi", max_tokens=3, stream=False))
    r = run(go())
    assert r.ok and r.ttft_s is None and r.output_tokens == 3


def _client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("name", ["vllm", "triton"])
def test_http_error_becomes_failed_result(name):
    async def go():
        async with _client_with(lambda req: httpx.Response(503, text="overloaded")) as c:
            return await make_backend(name, "http://x", "m", client=c).generate(GenerationRequest(prompt="hi"))
    r = run(go())
    assert not r.ok and r.status_code == 503 and "overloaded" in r.error and r.backend == name


def test_connection_error_becomes_failed_result():
    def boom(req):
        raise httpx.ConnectError("refused", request=req)

    async def go():
        async with _client_with(boom) as c:
            return await make_backend("vllm", "http://x", "m", client=c).generate(GenerationRequest(prompt="hi"))
    r = run(go())
    assert not r.ok and "ConnectError" in r.error and r.e2e_s is not None


def test_error_event_mid_stream_is_a_failure():
    body = b'data: {"text_output": "par"}\n\ndata: {"error": "CUDA out of memory"}\n\n'

    async def go():
        async with _client_with(lambda req: httpx.Response(200, content=body)) as c:
            return await make_backend("triton", "http://x", "m", client=c).generate(GenerationRequest(prompt="hi"))
    r = run(go())
    assert not r.ok and "CUDA out of memory" in r.error


def test_empty_stream_is_a_failure():
    async def go():
        async with _client_with(lambda req: httpx.Response(200, content=b"data: [DONE]\n\n")) as c:
            return await make_backend("vllm", "http://x", "m", client=c).generate(GenerationRequest(prompt="hi"))
    r = run(go())
    assert not r.ok and r.error == "empty response"


def test_token_count_falls_back_to_tokenizer_then_unknown():
    body = b'data: {"text_output": "a b c", "finish_reason": ["stop"]}\n\n'

    async def go(counter):
        async with _client_with(lambda req: httpx.Response(200, content=body)) as c:
            b = make_backend("triton", "http://x", "m", client=c, token_counter=counter)
            return await b.generate(GenerationRequest(prompt="hi", max_tokens=10))
    with_tok = run(go(lambda s: len(s.split())))
    assert (with_tok.output_tokens, with_tok.output_tokens_source) == (3, "tokenizer")
    without = run(go(None))
    assert (without.output_tokens, without.output_tokens_source) == (None, "unknown")


@pytest.mark.parametrize("name", ["vllm", "triton"])
def test_health_and_server_info(name):
    async def go():
        async with fake_client() as client:
            b = make_backend(name, "http://fake", "m", client=client)
            return await b.health(), await b.server_info()
    healthy, info = run(go())
    assert healthy and info["ignore_eos_supported"] is True
    assert FAKE_VERSION in json.dumps(info)


def test_triton_info_detects_unpatched_model():
    async def go():
        async with fake_client(FakeEngine(triton_has_ignore_eos=False)) as client:
            return await make_backend("triton", "http://fake", "m", client=client).server_info()
    assert run(go())["ignore_eos_supported"] is False
