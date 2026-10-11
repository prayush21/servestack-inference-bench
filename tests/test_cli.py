"""CLI wiring (argument parsing and dispatch), without network."""

from __future__ import annotations

import json
from pathlib import Path

from llmbench import cli

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_watch_cmd_flag_does_not_clobber_subcommand():
    args = cli.build_parser().parse_args(["gpu-watch", "--cmd", "nvidia-smi -q", "--out", "x.jsonl"])
    assert args.command == "gpu-watch" and args.cmd == "nvidia-smi -q"


def test_validate_config_command(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert cli.main(["validate-config", "--config", "configs/benchmark.yaml"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] and [p["concurrency"] for p in out["plan"]] == [1, 4, 16, 32, 64]


def test_report_command_pending(tmp_path, capsys):
    assert cli.main(["report", str(tmp_path), "--out", str(tmp_path / "r")]) == 0
    assert "PENDING" in (tmp_path / "r" / "summary.md").read_text()


def test_gpu_watch_writes_samples(tmp_path, monkeypatch):
    import asyncio

    out = tmp_path / "gpu.jsonl"
    args = cli.build_parser().parse_args(["gpu-watch", "--cmd", "printf '1, 2, 3\\n'", "--out", str(out),
                                          "--interval", "0.05"])

    async def run_briefly():
        task = asyncio.create_task(cli._gpu_watch(args))
        await asyncio.sleep(0.4)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    asyncio.run(run_briefly())
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and rows[0]["memory_used_mib"] == 1.0 and "t" in rows[0]
