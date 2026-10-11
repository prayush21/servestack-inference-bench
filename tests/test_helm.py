"""Deployment configuration: render the chart and check what matters for a fair, safe deploy.
Skipped automatically when the `helm` binary is not installed."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "llm-serving"
pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")


def render(backend: str, *extra: str) -> dict[str, dict]:
    out = subprocess.run(["helm", "template", backend, str(CHART), "-f", str(CHART / f"values-{backend}.yaml"), *extra],
                         capture_output=True, text=True, check=True).stdout
    docs = [d for d in yaml.safe_load_all(out) if d]
    return {d["kind"]: d for d in docs}


def container(objs: dict) -> dict:
    return objs["Deployment"]["spec"]["template"]["spec"]["containers"][0]


def flag(args: list[str], name: str) -> str:
    for a in args:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    raise KeyError(name)


@pytest.fixture(scope="module")
def vllm():
    return render("vllm")


@pytest.fixture(scope="module")
def triton():
    return render("triton")


@pytest.fixture(scope="module")
def trt_model_yaml(triton):
    return yaml.safe_load(triton["ConfigMap"]["data"]["model.yaml"])


@pytest.mark.parametrize("backend", ["vllm", "triton"])
def test_gpu_resources_probes_and_strategy(backend, request):
    objs = request.getfixturevalue(backend)
    c = container(objs)
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert c["resources"]["requests"]["nvidia.com/gpu"] == 1
    assert "memory" in c["resources"]["limits"] and "cpu" in c["resources"]["limits"]
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert c[probe]["httpGet"]["port"] == "http"
    ready = "/health" if backend == "vllm" else "/v2/models/tensorrt_llm/ready"
    assert c["readinessProbe"]["httpGet"]["path"] == ready
    assert objs["Deployment"]["spec"]["strategy"]["type"] == "Recreate"
    assert objs["Service"]["spec"]["type"] == "ClusterIP"
    assert re.search(r"@sha256:[0-9a-f]{64}$", c["image"]), "image must be pinned by digest"
    mounts = {m["mountPath"] for m in c["volumeMounts"]}
    assert "/dev/shm" in mounts and "/cache" in mounts
    env = {e["name"]: e for e in c["env"]}
    assert env["HF_HUB_OFFLINE"]["value"] == "1"
    assert "Secret" not in objs


def test_shared_engine_settings_are_identical(vllm, trt_model_yaml):
    args = container(vllm)["args"]
    m = trt_model_yaml
    assert args[0] == m["model"]
    assert flag(args, "revision") == m["revision"]
    assert int(flag(args, "kv-cache-memory-bytes")) == m["kv_cache_config"]["max_gpu_total_bytes"]
    assert int(flag(args, "max-num-seqs")) == m["max_batch_size"]
    assert int(flag(args, "max-model-len")) == m["max_seq_len"]
    assert int(flag(args, "max-num-batched-tokens")) == m["max_num_tokens"]
    assert flag(args, "dtype") == m["dtype"]
    assert "--no-enable-prefix-caching" in args and m["kv_cache_config"]["enable_block_reuse"] is False
    assert "--enforce-eager" not in args and m["cuda_graph_config"]["max_batch_size"] == m["max_batch_size"]
    assert flag(args, "generation-config") == "vllm"


def test_triton_uses_llm_api_pytorch_path_with_streaming(trt_model_yaml, triton):
    assert trt_model_yaml["backend"] == "pytorch"           # LLM API, no TensorRT engine build
    assert trt_model_yaml["triton_config"] == {"max_batch_size": 0, "decoupled": True,
                                               "cancellation_check_period_ms": 100}
    c = container(triton)
    assert c["command"] == ["mpirun"] and "tritonserver" in c["args"]
    data = triton["ConfigMap"]["data"]
    assert 'name: "sampling_param_ignore_eos"' in data["config.pbtxt"]
    assert "'ignore_eos'," in data["helpers.py"]
    init = triton["Deployment"]["spec"]["template"]["spec"]["initContainers"][0]
    assert "cp -rL" in init["command"][-1]
    items = {i["path"] for v in triton["Deployment"]["spec"]["template"]["spec"]["volumes"]
             if "configMap" in v for i in v["configMap"]["items"]}
    assert items == {"tensorrt_llm/config.pbtxt", "tensorrt_llm/1/model.py",
                     "tensorrt_llm/1/helpers.py", "tensorrt_llm/1/model.yaml"}


def test_overrides_flow_to_both_backends():
    v = render("vllm", "--set", "engine.kvCacheBytes=4294967296", "--set", "engine.prefixCaching=true")
    t = render("triton", "--set", "engine.kvCacheBytes=4294967296", "--set", "engine.prefixCaching=true")
    assert "--kv-cache-memory-bytes=4294967296" in container(v)["args"]
    assert "--enable-prefix-caching" in container(v)["args"]
    m = yaml.safe_load(t["ConfigMap"]["data"]["model.yaml"])
    assert m["kv_cache_config"]["max_gpu_total_bytes"] == 4294967296
    assert m["kv_cache_config"]["enable_block_reuse"] is True


def test_hf_token_comes_only_from_a_secret_reference():
    c = container(render("vllm", "--set", "hfToken.secretName=hf-token"))
    env = {e["name"]: e for e in c["env"]}
    ref = env["HF_TOKEN"]["valueFrom"]["secretKeyRef"]
    assert ref == {"name": "hf-token", "key": "token", "optional": True}
    assert "value" not in env["HF_TOKEN"]


def test_model_cache_rendered_only_on_request():
    assert "PersistentVolumeClaim" not in render("vllm")
    objs = render("vllm", "--set", "modelCache.create=true", "--set", "modelDownload.enabled=true")
    assert objs["PersistentVolumeClaim"]["metadata"]["name"] == "llm-model-cache"
    job = objs["Job"]["spec"]["template"]["spec"]
    assert "989aa7980e4cf806f80c7fef2b1adb7bc71aa306" in job["containers"][0]["args"][0]
    assert job["tolerations"][0]["key"] == "nvidia.com/gpu"


@pytest.mark.parametrize("bad", [["--set", "backend=sglang"], ["--set", "model.revision=main"],
                                 ["--set", "engine.maxNumSeqs=0"], ["--set", "resources.limits.nvidia\\.com/gpu=2"],
                                 ["--set", "engine.typo=1"]])
def test_schema_rejects_invalid_values(bad):
    proc = subprocess.run(["helm", "template", "x", str(CHART), *bad], capture_output=True, text=True)
    assert proc.returncode != 0 and "schema" in proc.stderr.lower()


def test_lint():
    for b in ("vllm", "triton"):
        subprocess.run(["helm", "lint", str(CHART), "-f", str(CHART / f"values-{b}.yaml")],
                       check=True, capture_output=True)


def test_no_credentials_committed():
    pattern = re.compile(r"hf_[A-Za-z0-9]{30,}|nvapi-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}")
    skip = {".venv", ".git", "__pycache__", ".pytest_cache"}
    for p in ROOT.rglob("*"):
        if p.is_file() and not skip.intersection(p.parts) and p.suffix not in {".png"}:
            assert not pattern.search(p.read_text(errors="ignore")), f"credential-like string in {p}"
