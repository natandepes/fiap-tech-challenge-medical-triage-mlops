from __future__ import annotations

import contextlib
import math
import socket
import statistics
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx

from triage_api.metrics import HEALTH_PATH
from triage_api.samples import sample_reports

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
DOCKER_IMAGE = "triage-api:latest"
CONTAINER_PORT = 8000
METRICS = ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
SAMPLE_REPORTS = sample_reports()


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(name: str, latencies_ms: list[float]) -> dict:
    return {
        "name": name,
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "max_ms": max(latencies_ms),
    }


def measure_model(name: str, model, requests: int, warmup: int) -> dict:
    latencies_ms: list[float] = []
    for i in range(warmup + requests):
        text = SAMPLE_REPORTS[i % len(SAMPLE_REPORTS)]
        start = time.perf_counter_ns()
        model.predict(text)
        elapsed_ns = time.perf_counter_ns() - start
        if i >= warmup:
            latencies_ms.append(elapsed_ns / 1_000_000)
    return summarize(name, latencies_ms)


def measure_api(name: str, base_url: str, requests: int, warmup: int) -> dict:
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
    return summarize(name, latencies_ms)


def served_backend(base_url: str) -> str:
    response = httpx.get(f"{base_url}{HEALTH_PATH}", timeout=5.0)
    response.raise_for_status()
    return response.json()["model_backend"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_until_healthy(base_url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(httpx.RequestError):
            if httpx.get(f"{base_url}{HEALTH_PATH}", timeout=1.0).status_code == 200:
                return
        time.sleep(0.2)
    raise RuntimeError(f"API did not become healthy within {timeout:.0f}s")


def _require_backend(base_url: str, expected: str) -> None:
    actual = served_backend(base_url)
    if actual != expected:
        raise RuntimeError(f"container serves the {actual!r} backend, expected {expected!r}")


@contextmanager
def api_container(backend: str, image: str = DOCKER_IMAGE) -> Iterator[str]:
    port = _free_port()
    started = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--detach",
            "--env",
            f"TRIAGE_MODEL_BACKEND={backend}",
            "--publish",
            f"{port}:{CONTAINER_PORT}",
            image,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    container_id = started.stdout.strip()
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_healthy(base_url)
        _require_backend(base_url, backend)
        yield base_url
    finally:
        subprocess.run(["docker", "stop", container_id], check=False, capture_output=True)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return f"| {' | '.join(padded)} |"

    separator = f"| {' | '.join('-' * width for width in widths)} |"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def metrics_table(first_column: str, results: list[dict], decimals: int = 3) -> str:
    return render_table(
        [first_column, "mean (ms)", "p50", "p95", "p99", "max"],
        [[f"`{r['name']}`", *(f"{r[metric]:.{decimals}f}" for metric in METRICS)] for r in results],
    )


def measured_at() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_report(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown)
    print(markdown)
