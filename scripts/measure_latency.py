from __future__ import annotations

import argparse
import contextlib
import math
import socket
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from triage_api.config import MODEL_PATH
from triage_api.model import TriageModel

SAMPLE_REPORTS = [
    "CT scan of the chest: no acute abnormality identified. Routine review.",
    "The laboratory panel demonstrates mildly elevated inflammatory markers.",
    "Impression: acute intracranial hemorrhage on CT scan of the cranium. "
    "Immediate clinical attention required.",
    "MRI of the lumbar spine: mild degenerative changes consistent with age.",
    "Reported signs of septic shock with rising lactate. Rapid deterioration noted.",
]

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
DOCKER_IMAGE = "triage-api:latest"
CONTAINER_PORT = 8000
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


def measure_model(requests: int, warmup: int) -> dict:
    model = TriageModel.load(MODEL_PATH)
    latencies_ms: list[float] = []
    for i in range(warmup + requests):
        text = SAMPLE_REPORTS[i % len(SAMPLE_REPORTS)]
        start = time.perf_counter_ns()
        model.predict(text)
        elapsed_ns = time.perf_counter_ns() - start
        if i >= warmup:
            latencies_ms.append(elapsed_ns / 1_000_000)
    return _summarize("model (in-process)", latencies_ms)


def measure_api(base_url: str, requests: int, warmup: int) -> dict:
    latencies_ms: list[float] = []
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        for i in range(warmup + requests):
            payload = {"text": SAMPLE_REPORTS[i % len(SAMPLE_REPORTS)]}
            start = time.perf_counter_ns()
            response = client.post("/predict", json=payload)
            elapsed_ns = time.perf_counter_ns() - start
            response.raise_for_status()
            if i >= warmup:
                latencies_ms.append(elapsed_ns / 1_000_000)
    return _summarize("API (Docker container)", latencies_ms)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_until_healthy(base_url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(httpx.RequestError):
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                return
        time.sleep(0.2)
    raise RuntimeError(f"API did not become healthy within {timeout:.0f}s")


def _start_container(image: str) -> tuple[str, str]:
    port = _free_port()
    started = subprocess.run(
        ["docker", "run", "--rm", "--detach", "--publish", f"{port}:{CONTAINER_PORT}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    return started.stdout.strip(), f"http://127.0.0.1:{port}"


def _stop_container(container_id: str) -> None:
    subprocess.run(["docker", "stop", container_id], check=False, capture_output=True)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return f"| {' | '.join(padded)} |"

    separator = f"| {' | '.join('-' * width for width in widths)} |"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def to_markdown(results: list[dict], requests: int, warmup: int) -> str:
    table = _render_table(
        ["Stage", "mean (ms)", "p50", "p95", "p99", "max"],
        [[f"`{r['name']}`", *(f"{r[metric]:.3f}" for metric in METRICS)] for r in results],
    )
    lines = [
        "# Latency baseline (Stage 1)",
        "",
        f"- Measured at: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- Samples per stage: {requests} (after {warmup} warmup)",
        "- Model: TF-IDF + Logistic Regression (scikit-learn), no optimization",
        "- Timer: `time.perf_counter_ns`, serialized requests, no concurrency",
        f"- API served from the `{DOCKER_IMAGE}` container over loopback",
        "",
        table,
        "",
        "`model (in-process)` is the classifier call with no web layer; it is the number Stage 4 "
        "compares against the ONNX build. `API (Docker container)` is the full request path, "
        "dominated by framework and serialization overhead.",
        "",
        "Reproduce: `make latency`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure /predict latency, model-only and end-to-end"
    )
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--image", default=DOCKER_IMAGE, help="inference image to run")
    parser.add_argument(
        "--base-url",
        default=None,
        help="measure against an already-running API instead of starting a container",
    )
    parser.add_argument("--out", type=Path, default=BENCHMARKS_DIR / "latency_baseline.md")
    args = parser.parse_args()

    results = [measure_model(args.requests, args.warmup)]

    if args.base_url:
        results.append(measure_api(args.base_url, args.requests, args.warmup))
    else:
        container_id, base_url = _start_container(args.image)
        try:
            _wait_until_healthy(base_url)
            results.append(measure_api(base_url, args.requests, args.warmup))
        finally:
            _stop_container(container_id)

    markdown = to_markdown(results, args.requests, args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
