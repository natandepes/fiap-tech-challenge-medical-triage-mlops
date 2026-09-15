from __future__ import annotations

import argparse
from pathlib import Path

from latency_bench import (
    BENCHMARKS_DIR,
    DOCKER_IMAGE,
    api_container,
    measure_api,
    measure_model,
    measured_at,
    metrics_table,
    served_backend,
    write_report,
)
from triage_api.config import MODEL_PATH
from triage_api.model import TriageModel

BASELINE_BACKEND = "sklearn"


def to_markdown(results: list[dict], requests: int, warmup: int, api_source: str) -> str:
    lines = [
        "# Latency baseline (Stage 1)",
        "",
        f"- Measured at: {measured_at()}",
        f"- Samples per stage: {requests} (after {warmup} warmup)",
        "- Model: TF-IDF + Logistic Regression (scikit-learn), no optimization",
        "- Inputs: real abstracts from `data/sample_triage.csv`",
        "- Timer: `time.perf_counter_ns`, serialized requests, no concurrency",
        f"- {api_source}",
        "",
        metrics_table("Stage", results),
        "",
        "`model (in-process)` is the classifier call with no web layer. `API (Docker container)` "
        "is the full request path: the same prediction plus HTTP, FastAPI validation and JSON "
        "serialization.",
        "",
        "Stage 4 re-runs *both* of these measurements against the ONNX build, so the optimized "
        "numbers are comparable to this baseline stage for stage — see "
        "[`onnx_latency_comparison.md`](onnx_latency_comparison.md).",
        "",
        "Reproduce: `make latency`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the pre-optimization latency baseline, model-only and end-to-end"
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

    results = [
        measure_model(
            "model (in-process)", TriageModel.load(MODEL_PATH), args.requests, args.warmup
        )
    ]

    if args.base_url:
        api_source = (
            f"API measured at {args.base_url}, serving the "
            f"`{served_backend(args.base_url)}` backend"
        )
        results.append(
            measure_api("API (Docker container)", args.base_url, args.requests, args.warmup)
        )
    else:
        api_source = (
            f"API served from the `{args.image}` container over loopback, pinned to "
            f"`TRIAGE_MODEL_BACKEND={BASELINE_BACKEND}` so this baseline stays pre-optimization"
        )
        with api_container(BASELINE_BACKEND, args.image) as base_url:
            results.append(
                measure_api("API (Docker container)", base_url, args.requests, args.warmup)
            )

    write_report(args.out, to_markdown(results, args.requests, args.warmup, api_source))


if __name__ == "__main__":
    main()
