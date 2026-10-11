"""Turn raw benchmark results into summary.csv / summary.json / summary.md and PNG plots.

If no measured records exist, a report explicitly marked PENDING is written instead.
Nothing here ever fabricates numbers: every value is computed from requests.jsonl.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from llmbench.metrics import output_agreement, relative_change, summarize

# Categorical slots 1-2 of the validated default palette (scripts/validate_palette.js: all pass).
SERIES_STYLE = {
    "vllm": {"color": "#2a78d6", "marker": "o", "label": "vLLM"},
    "triton": {"color": "#eb6834", "marker": "s", "label": "Triton + TRT-LLM"},
}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

CSV_COLUMNS = [
    "backend", "concurrency", "requests", "successes", "failures", "failure_rate", "window_s",
    "request_throughput_rps", "output_throughput_tok_s", "ttft_p50_ms", "ttft_p95_ms",
    "e2e_p50_ms", "e2e_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "output_length_mismatches",
    "output_tokens_uncounted", "gpu_memory_used_peak_mib", "gpu_memory_total_mib",
    "gpu_utilization_mean_pct", "gpu_samples",
]

# (summary key, title, unit, higher_is_better)
PLOTS = [
    ("ttft_p50_ms", "Time to first token, p50", "ms", False),
    ("e2e_p95_ms", "End-to-end latency, p95", "ms", False),
    ("output_throughput_tok_s", "Output token throughput", "tokens/s", True),
    ("request_throughput_rps", "Request throughput", "requests/s", True),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_results(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    """Find every <dir>/<backend>/requests.jsonl under the given paths. Later paths win if a
    backend appears twice (e.g. a re-run)."""
    found: dict[str, dict[str, Any]] = {}
    for root in map(Path, paths):
        for req_file in sorted(root.rglob("requests.jsonl")):
            d = req_file.parent
            meta_file = d / "meta.json"
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            records = _read_jsonl(req_file)
            backend = meta.get("backend") or (records[0]["backend"] if records else d.name)
            found[backend] = {"dir": str(d), "meta": meta, "records": records,
                              "gpu": _read_jsonl(d / "gpu.jsonl")}
    return found


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    return str(v)


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:+.1f}%"


def write_pending(out_dir: Path, reason: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "summary.md"
    md.write_text(
        "# Benchmark summary — PENDING\n\n"
        f"No measured results were found ({reason}).\n\n"
        "No numbers are reported until a real GPU run exists. See README → *Benchmark* for the "
        "commands that produce `results/<run>/<backend>/requests.jsonl`.\n")
    (out_dir / "summary.json").write_text(json.dumps({"status": "pending", "reason": reason}, indent=2))
    return md


def build_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in results.values():
        rows.extend(summarize(data["records"], data["gpu"]))
    return rows


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    by = {(r["backend"], r["concurrency"]): r for r in rows}
    levels = sorted({r["concurrency"] for r in rows})
    metrics = [("ttft_p50_ms", "TTFT p50 (ms)"), ("ttft_p95_ms", "TTFT p95 (ms)"),
               ("e2e_p50_ms", "E2E p50 (ms)"), ("e2e_p95_ms", "E2E p95 (ms)"),
               ("tpot_p50_ms", "TPOT p50 (ms)"),
               ("output_throughput_tok_s", "Output tok/s"), ("request_throughput_rps", "Req/s"),
               ("failures", "Failures"), ("gpu_memory_used_peak_mib", "GPU mem peak (MiB)")]
    lines = ["| Concurrency | Metric | vLLM | Triton + TRT-LLM | Triton vs vLLM |",
             "|---:|---|---:|---:|---:|"]
    for c in levels:
        v, t = by.get(("vllm", c), {}), by.get(("triton", c), {})
        for key, label in metrics:
            lines.append(f"| {c} | {label} | {_fmt(v.get(key))} | {_fmt(t.get(key))} | "
                         f"{_pct(relative_change(t.get(key), v.get(key)))} |")
    return "\n".join(lines)


def _per_backend_table(rows: list[dict[str, Any]]) -> str:
    cols = ["concurrency", "requests", "failures", "ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms",
            "e2e_p95_ms", "output_throughput_tok_s", "request_throughput_rps", "gpu_memory_used_peak_mib"]
    out = []
    for backend in sorted({r["backend"] for r in rows}):
        out.append(f"\n**{SERIES_STYLE.get(backend, {}).get('label', backend)}**\n")
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "---:|" * len(cols))
        for r in (r for r in rows if r["backend"] == backend):
            out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def _run_info(results: dict[str, dict[str, Any]]) -> str:
    lines = ["| | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]

    def row(label: str, fn) -> None:
        vals = []
        for d in results.values():
            try:
                vals.append(str(fn(d["meta"])).replace("|", "\\|").replace("\n", " "))
            except Exception:
                vals.append("–")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    row("Status", lambda m: m.get("status"))
    row("Started (UTC)", lambda m: m.get("started_utc"))
    row("Server version", lambda m: (m["server_info"].get("version") or m["server_info"].get("server")))
    row("GPU (nvidia-smi)", lambda m: m.get("gpu_info"))
    row("Model", lambda m: f'{m["config"]["model_id"]}@{m["config"]["model_revision"][:12]}')
    row("Sampling", lambda m: json.dumps(m["config"]["sampling"]))
    row("Prompts sha256", lambda m: m["prompts_sha256"][:16])
    row("Images (pod)", lambda m: m["extra"].get("pod_images", "–"))
    return "\n".join(lines)


def _caveats(rows: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[str]:
    notes = []
    for r in rows:
        if r["failures"]:
            notes.append(f"{r['backend']} @ c={r['concurrency']}: {r['failures']} failed requests "
                         f"(e.g. {r['error_examples'][:1]}).")
        if r["output_length_mismatches"]:
            notes.append(f"{r['backend']} @ c={r['concurrency']}: {r['output_length_mismatches']} requests "
                         "did not produce exactly max_tokens — throughput is not like-for-like there.")
        if r["output_tokens_uncounted"]:
            notes.append(f"{r['backend']} @ c={r['concurrency']}: {r['output_tokens_uncounted']} successful "
                         "requests had no token count and are excluded from tok/s.")
    shas = {d["meta"].get("prompts_sha256") for d in results.values()}
    if len(shas) > 1:
        notes.append("Backends were run with DIFFERENT prompt files — results are not comparable.")
    samplings = {json.dumps(d["meta"].get("config", {}).get("sampling"), sort_keys=True) for d in results.values()}
    if len(samplings) > 1:
        notes.append("Backends were run with DIFFERENT sampling settings — results are not comparable.")
    return notes


def make_plots(rows: list[dict[str, Any]], out_dir: Path, synthetic: bool = False) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    backends = [b for b in SERIES_STYLE if any(r["backend"] == b for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for ax, (key, title, unit, higher_better) in zip(axes.flat, PLOTS):
        ax.set_facecolor(SURFACE)
        ends: list[tuple[float, float, str]] = []
        for b in backends:
            pts = sorted((r["concurrency"], r[key]) for r in rows if r["backend"] == b and r.get(key) is not None)
            if not pts:
                continue
            xs, ys = zip(*pts)
            st = SERIES_STYLE[b]
            ax.plot(xs, ys, color=st["color"], linewidth=2, marker=st["marker"], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=st["label"], solid_capstyle="round")
            ends.append((xs[-1], ys[-1], st["label"]))
        # Direct end labels; nudge apart when the two series finish close together.
        span = max((y for _, y, _ in ends), default=0) or 1.0
        ends.sort(key=lambda e: e[1])
        for i, (x, y, label) in enumerate(ends):
            dy = 0
            if len(ends) == 2 and abs(ends[1][1] - ends[0][1]) < 0.08 * span:
                dy = -7 if i == 0 else 7
            ax.annotate(label, (x, y), xytext=(8, dy), textcoords="offset points",
                        va="center", fontsize=8, color=INK_MUTED, annotation_clip=False)
        ax.set_xscale("log", base=2)
        levels = sorted({r["concurrency"] for r in rows})
        ax.set_xticks(levels, [str(c) for c in levels])
        ax.minorticks_off()
        ax.set_title(f"{title} ({'higher' if higher_better else 'lower'} is better)", loc="left",
                     fontsize=10, color=INK)
        ax.set_ylabel(unit, color=INK_MUTED, fontsize=9)
        ax.set_xlabel("concurrent requests", color=INK_MUTED, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.96), frameon=False,
                   ncol=len(handles), fontsize=9)
    title = "vLLM vs. Triton + TensorRT-LLM — same GPU, model, prompts and sampling"
    if synthetic:
        title = "SYNTHETIC — FAKE SERVER, NOT A BENCHMARK — " + title
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.92))
    path = out_dir / "comparison.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def generate_report(result_paths: list[str | Path], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    results = load_results(result_paths)
    rows = build_rows(results)
    if not rows:
        return write_pending(out, "no requests.jsonl with phase=measure under "
                             + ", ".join(map(str, result_paths)))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    agreement = None
    if "vllm" in results and "triton" in results:
        agreement = output_agreement(results["vllm"]["records"], results["triton"]["records"])
    (out / "summary.json").write_text(json.dumps(
        {"status": "measured", "sources": {b: d["dir"] for b, d in results.items()},
         "rows": rows, "output_agreement": agreement}, indent=2))
    synthetic = any("FAKE" in json.dumps(d["meta"].get("server_info", {})) for d in results.values())
    plot = make_plots(rows, out, synthetic=synthetic)

    parts = ["# Benchmark summary\n"]
    if synthetic:
        parts.append("> **SYNTHETIC RESULTS** — at least one run targeted the FAKE test server. "
                     "These numbers say nothing about vLLM or TensorRT-LLM performance.\n")
    parts += [
             "Generated by `llmbench report` from raw `requests.jsonl` files: "
             + ", ".join(f"`{d['dir']}`" for d in results.values()) + ".\n",
             "## Runs\n", _run_info(results), ""]
    if plot:
        parts += ["## Plots\n", f"![comparison]({plot.name})\n"]
    if len(results) > 1:
        parts += ["## Side by side\n", "Relative change is (Triton − vLLM) / vLLM. For latency, negative is "
                  "better; for throughput, positive is better.\n", _comparison_table(rows), ""]
    parts += ["## Per backend\n", _per_backend_table(rows), ""]
    if agreement:
        parts += ["## Greedy output agreement (concurrency 1)\n",
                  "Exact equality is not expected across engines (different kernels and reduction order), "
                  "but very low prefix agreement would suggest a sampling or config mismatch.\n",
                  f"- prompts compared: {agreement.get('prompts_compared')}",
                  f"- exact matches: {agreement.get('exact_match', '–')} "
                  f"({_pct(agreement.get('exact_match_rate')) if agreement.get('exact_match_rate') is not None else '–'})".replace("+", ""),
                  f"- mean common-prefix fraction: {_fmt(agreement.get('mean_common_prefix_frac'), 3)}", ""]
    caveats = _caveats(rows, results)
    parts += ["## Automatic caveats\n", "\n".join(f"- {c}" for c in caveats) if caveats else "- none detected", ""]
    md = out / "summary.md"
    md.write_text("\n".join(parts))
    return md
