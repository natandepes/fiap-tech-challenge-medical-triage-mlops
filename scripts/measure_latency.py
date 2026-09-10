from __future__ import annotations

import argparse
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

SAMPLE_REPORTS = [
    "CT scan of the chest: no acute abnormality identified. Routine review.",
    "The laboratory panel demonstrates mildly elevated inflammatory markers.",
    "Impression: acute intracranial hemorrhage on CT scan of the cranium. "
    "Immediate clinical attention required.",
    "MRI of the lumbar spine: mild degenerative changes consistent with age.",
    "Reported signs of septic shock with rising lactate. Rapid deterioration noted.",
]


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def run(base_url: str, requests: int, warmup: int) -> dict:
    latencies_ms: list[float] = []
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        for i in range(warmup + requests):
            payload = {"text": SAMPLE_REPORTS[i % len(SAMPLE_REPORTS)]}
            start = time.perf_counter()
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            if i >= warmup:
                latencies_ms.append((time.perf_counter() - start) * 1000)

    return {
        "base_url": base_url,
        "requests": requests,
        "warmup": warmup,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "mean_ms": round(statistics.fmean(latencies_ms), 2),
        "p50_ms": round(_percentile(latencies_ms, 50), 2),
        "p95_ms": round(_percentile(latencies_ms, 95), 2),
        "p99_ms": round(_percentile(latencies_ms, 99), 2),
        "max_ms": round(max(latencies_ms), 2),
    }


def to_markdown(result: dict) -> str:
    return (
        "# Latency baseline (Stage 1)\n\n"
        f"- Endpoint: `POST {result['base_url']}/predict`\n"
        f"- Requests measured: {result['requests']} (after {result['warmup']} warmup)\n"
        f"- Measured at: {result['timestamp']}\n"
        "- Model: TF-IDF + Logistic Regression (scikit-learn), no optimization\n"
        "- Host: single uvicorn worker over loopback; serialized requests, no concurrency\n\n"
        "| Metric | Value (ms) |\n"
        "| --- | --- |\n"
        f"| mean | {result['mean_ms']} |\n"
        f"| p50 | {result['p50_ms']} |\n"
        f"| p95 | {result['p95_ms']} |\n"
        f"| p99 | {result['p99_ms']} |\n"
        f"| max | {result['max_ms']} |\n\n"
        "Stage 4 re-runs this script against the ONNX build and records the comparison.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure /predict latency against a running API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "benchmarks" / "latency_baseline.md",
    )
    args = parser.parse_args()

    result = run(args.base_url, args.requests, args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == "__main__":
    main()
