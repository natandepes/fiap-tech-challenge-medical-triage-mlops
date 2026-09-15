from __future__ import annotations

import argparse
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from triage_api.config import MODEL_PATH, ONNX_MODEL_PATH
from triage_api.model import TriageModel
from triage_api.onnx_model import OnnxTriageModel
from triage_api.samples import sample_reports

SAMPLE_REPORTS = sample_reports()

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
METRICS = ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[min(rank, len(ordered)) - 1]


def _summarize(name: str, latencies_ms: list[float]) -> dict:
    return {
        "name": name,
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "max_ms": max(latencies_ms),
    }


def measure(name: str, model, requests: int, warmup: int) -> dict:
    latencies_ms: list[float] = []
    for i in range(warmup + requests):
        text = SAMPLE_REPORTS[i % len(SAMPLE_REPORTS)]
        start = time.perf_counter_ns()
        model.predict(text)
        elapsed_ns = time.perf_counter_ns() - start
        if i >= warmup:
            latencies_ms.append(elapsed_ns / 1_000_000)
    return _summarize(name, latencies_ms)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return f"| {' | '.join(padded)} |"

    separator = f"| {' | '.join('-' * width for width in widths)} |"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def to_markdown(results: list[dict], requests: int, warmup: int) -> str:
    table = _render_table(
        ["Model", "mean (ms)", "p50", "p95", "p99", "max"],
        [[f"`{r['name']}`", *(f"{r[metric]:.4f}" for metric in METRICS)] for r in results],
    )
    sklearn_mean, onnx_mean = results[0]["mean_ms"], results[1]["mean_ms"]
    speedup = sklearn_mean / onnx_mean
    lines = [
        "# ONNX latency comparison (Stage 4)",
        "",
        f"- Measured at: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- Samples per model: {requests} (after {warmup} warmup)",
        "- Both models loaded in-process, no web layer — same methodology as the Stage 1 "
        "`model (in-process)` baseline in `benchmarks/latency_baseline.md`",
        "- Timer: `time.perf_counter_ns`, serialized requests, no concurrency",
        "- Inputs: real abstracts from `data/sample_triage.csv`",
        "",
        table,
        "",
        f"ONNX Runtime is **{speedup:.2f}x** faster than the scikit-learn pipeline on mean "
        "latency for the same inputs and the same model weights.",
        "",
        "Reproduce: `make latency-onnx`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare in-process latency: scikit-learn pipeline vs. ONNX Runtime"
    )
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--out", type=Path, default=BENCHMARKS_DIR / "onnx_latency_comparison.md")
    args = parser.parse_args()

    sklearn_model = TriageModel.load(MODEL_PATH)
    onnx_model = OnnxTriageModel.load(ONNX_MODEL_PATH)

    results = [
        measure("scikit-learn pipeline", sklearn_model, args.requests, args.warmup),
        measure("ONNX Runtime", onnx_model, args.requests, args.warmup),
    ]

    markdown = to_markdown(results, args.requests, args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
