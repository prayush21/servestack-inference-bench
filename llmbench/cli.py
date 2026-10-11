"""`llmbench` command line: generate | health | info | bench | report | validate-config | gateway |
gpu-watch | fake-server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from llmbench.backends import BACKENDS, DEFAULT_MODEL, make_backend
from llmbench.config import ConfigError, load_config, load_prompts
from llmbench.prompts import qwen_chat
from llmbench.schema import GenerationRequest


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    p.add_argument("--url", required=True, help="e.g. http://localhost:8000 (kubectl port-forward)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="served model name (vLLM only; Triton uses tensorrt_llm)")


async def _generate(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.raw else qwen_chat(args.prompt)
    req = GenerationRequest(prompt=prompt, max_tokens=args.max_tokens, temperature=args.temperature,
                            top_p=args.top_p, seed=args.seed, ignore_eos=args.ignore_eos,
                            stream=not args.no_stream)
    async with make_backend(args.backend, args.url, args.model) as backend:
        res = await backend.generate(req)
    print(json.dumps(res.to_dict(), indent=2))
    return 0 if res.ok else 1


async def _health(args: argparse.Namespace) -> int:
    async with make_backend(args.backend, args.url, args.model) as backend:
        ok = await backend.health()
    print(json.dumps({"backend": args.backend, "url": args.url, "ready": ok}))
    return 0 if ok else 1


async def _info(args: argparse.Namespace) -> int:
    async with make_backend(args.backend, args.url, args.model) as backend:
        print(json.dumps(await backend.server_info(), indent=2))
    return 0


def _parse_meta(items: list[str]) -> dict:
    meta = {}
    for item in items:
        key, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--meta expects key=path, got {item!r}")
        text = Path(path).read_text()
        try:
            meta[key] = json.loads(text)
        except json.JSONDecodeError:
            meta[key] = text
    return meta


async def _bench(args: argparse.Namespace) -> int:
    from llmbench.loadgen import run_benchmark

    cfg = load_config(args.config)
    prompts = load_prompts(cfg.prompts_file)
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    model = args.model or cfg.served_model_name
    async with make_backend(args.backend, args.url, model, request_timeout_s=cfg.request_timeout_s) as backend:
        out = await run_benchmark(backend, cfg, prompts, args.out, levels=levels, gpu_cmd=args.gpu_cmd,
                                  gpu_info_cmd=args.gpu_info_cmd, extra_meta=_parse_meta(args.meta),
                                  cooldown_s=args.cooldown, save_text=not args.no_save_text,
                                  log=lambda m: print(m, file=sys.stderr))
    print(out)
    return 0


def _report(args: argparse.Namespace) -> int:
    from llmbench.report import generate_report

    print(generate_report(args.results, args.out))
    return 0


def _validate(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
        prompts = load_prompts(cfg.prompts_file)
    except (ConfigError, FileNotFoundError) as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    plan = [{"concurrency": c, "warmup": cfg.warmups_for(c), "measured": cfg.requests_for(c)} for c in cfg.concurrency]
    print(json.dumps({"valid": True, "prompts": len(prompts), "plan": plan}, indent=2))
    return 0


def _require_gateway_extra() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        raise SystemExit("this command needs the optional HTTP extra: pip install '.[gateway]'") from None


def _gateway(args: argparse.Namespace) -> int:
    _require_gateway_extra()
    import uvicorn

    uvicorn.run("llmbench.gateway:app", host=args.host, port=args.port)
    return 0


async def _gpu_watch(args: argparse.Namespace) -> int:
    """Sample GPU stats into a JSONL file until interrupted (used when the load generator runs
    in-cluster and cannot exec into the serving pod itself)."""
    from llmbench.gpu import GpuSampler

    sampler = GpuSampler(args.cmd, args.interval)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        try:
            while True:
                started = asyncio.get_running_loop().time()
                sample = await sampler.sample_once()
                if sample:
                    f.write(json.dumps(sample) + "\n")
                    f.flush()
                await asyncio.sleep(max(0.0, args.interval - (asyncio.get_running_loop().time() - started)))
        except asyncio.CancelledError:
            pass
    return 0


def _fake_server(args: argparse.Namespace) -> int:
    _require_gateway_extra()
    import uvicorn

    from llmbench.fake import FakeEngine, create_fake_app

    print("Starting a FAKE backend (synthetic tokens, NOT a benchmark target)", file=sys.stderr)
    engine = FakeEngine(ttft_s=args.ttft_ms / 1000, itl_s=args.itl_ms / 1000)
    uvicorn.run(create_fake_app(engine), host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llmbench", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="send one request and print the result as JSON")
    _add_backend_args(g)
    g.add_argument("prompt")
    g.add_argument("--raw", action="store_true", help="send the prompt verbatim (no chat template)")
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--ignore-eos", action="store_true")
    g.add_argument("--no-stream", action="store_true")

    for name, help_ in (("health", "check readiness"), ("info", "print server version/config info")):
        h = sub.add_parser(name, help=help_)
        _add_backend_args(h)

    b = sub.add_parser("bench", help="run the load generator against one backend")
    b.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    b.add_argument("--url", required=True)
    b.add_argument("--model", default=None, help="defaults to served_model_name from the config")
    b.add_argument("--config", default="configs/benchmark.yaml")
    b.add_argument("--out", required=True, help="run directory, e.g. results/2026-10-08_l4")
    b.add_argument("--levels", help="override concurrency levels, e.g. 1,4")
    b.add_argument("--gpu-cmd", help="command printing `memory.used,memory.total,utilization.gpu` CSV")
    b.add_argument("--gpu-info-cmd", help="command printing GPU name/driver (recorded in meta.json)")
    b.add_argument("--meta", action="append", default=[], metavar="KEY=PATH",
                   help="attach a file (JSON or text) to meta.json under extra.KEY")
    b.add_argument("--cooldown", type=float, default=2.0)
    b.add_argument("--no-save-text", action="store_true")

    r = sub.add_parser("report", help="summarize results into CSV/JSON/Markdown/PNG")
    r.add_argument("results", nargs="+", help="run directories containing <backend>/requests.jsonl")
    r.add_argument("--out", default="reports/latest")

    v = sub.add_parser("validate-config", help="validate a benchmark config and show the request plan")
    v.add_argument("--config", default="configs/benchmark.yaml")

    gw = sub.add_parser("gateway", help="run the HTTP gateway (pip install '.[gateway]')")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=9000)
    gwatch = sub.add_parser("gpu-watch", help="sample nvidia-smi into JSONL until interrupted")
    gwatch.add_argument("--cmd", required=True, help="command printing memory.used,memory.total,utilization.gpu CSV")
    gwatch.add_argument("--out", required=True)
    gwatch.add_argument("--interval", type=float, default=1.0)

    fk = sub.add_parser("fake-server", help="serve a FAKE vLLM+Triton-compatible API for GPU-less smoke tests")
    fk.add_argument("--host", default="127.0.0.1")
    fk.add_argument("--port", type=int, default=18000)
    fk.add_argument("--ttft-ms", type=float, default=20.0)
    fk.add_argument("--itl-ms", type=float, default=2.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("generate", "health", "info", "bench", "gpu-watch"):
        fn = {"generate": _generate, "health": _health, "info": _info, "bench": _bench,
              "gpu-watch": _gpu_watch}[args.command]
        try:
            return asyncio.run(fn(args))
        except KeyboardInterrupt:
            return 130
    return {"report": _report, "validate-config": _validate, "gateway": _gateway,
            "fake-server": _fake_server}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
