"""Benchmark configuration and prompt-set validation."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from llmbench.config import ConfigError, load_config, load_prompts, validate_config
from llmbench.prompts import qwen_chat

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark.yaml"
VALUES = ROOT / "deploy" / "helm" / "llm-serving" / "values.yaml"


@pytest.fixture
def raw():
    return yaml.safe_load(CONFIG.read_text())


def test_shipped_config_is_valid(monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config(CONFIG)
    assert cfg.sampling.ignore_eos is True and cfg.sampling.temperature == 0.0
    assert len(load_prompts(cfg.prompts_file)) == 24


def test_config_matches_helm_values(raw):
    """The benchmark and the deployment must describe the same model and limits."""
    values = yaml.safe_load(VALUES.read_text())
    assert raw["model_id"] == values["model"]["id"]
    assert raw["model_revision"] == values["model"]["revision"]
    assert raw["served_model_name"] == values["model"]["servedName"]
    assert max(raw["concurrency"]) <= values["engine"]["maxNumSeqs"]


@pytest.mark.parametrize("mutate,message", [
    (lambda r: r.pop("concurrency"), "missing required key: concurrency"),
    (lambda r: r.update(model_revision="main"), "full 40-char commit SHA"),
    (lambda r: r.update(concurrency=[1, 1]), "unique"),
    (lambda r: r.update(concurrency=[0]), "positive integers"),
    (lambda r: r.update(typo_key=1), "unknown key: typo_key"),
    (lambda r: r["sampling"].update(max_new_tokens=5), "unknown sampling keys"),
    (lambda r: r["sampling"].update(top_p=0), "invalid sampling"),
    (lambda r: r.update(request_timeout_s=0), "request_timeout_s must be > 0"),
    (lambda r: r.update(warmup_rounds=-1), "warmup_rounds"),
])
def test_invalid_configs_are_rejected(raw, mutate, message):
    bad = copy.deepcopy(raw)
    mutate(bad)
    errors = validate_config(bad)
    assert any(message in e for e in errors), errors


def test_load_config_raises(tmp_path, raw):
    raw["model_revision"] = "main"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(p)


def test_request_plan(monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config(CONFIG)
    assert cfg.requests_for(1) == 64 and cfg.requests_for(64) == 256
    assert cfg.warmups_for(1) == 4 and cfg.warmups_for(32) == 32


def test_prompts_file_is_up_to_date_and_templated():
    """configs/prompts.jsonl must be exactly what scripts/build_prompts.py generates."""
    out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'scripts'); "
                          "import build_prompts; sys.stdout.write(build_prompts.render())"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out == (ROOT / "configs" / "prompts.jsonl").read_text()
    rows = [json.loads(line) for line in out.splitlines()]
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["prompt"] == qwen_chat(r["user"])
        assert r["prompt"].endswith("<|im_start|>assistant\n")


def test_prompt_loader_rejects_duplicates(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"id": "a", "prompt": "x"}\n{"id": "a", "prompt": "y"}\n')
    with pytest.raises(ConfigError, match="duplicate"):
        load_prompts(p)
