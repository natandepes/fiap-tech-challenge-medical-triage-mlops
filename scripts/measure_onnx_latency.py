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
    write_report,
)
from triage_api.config import MODEL_PATH, ONNX_MODEL_PATH
from triage_api.model import TriageModel
from triage_api.onnx_model import OnnxTriageModel


def measure_models(requests: int, warmup: int) -> list[dict]:
    return [
        measure_model("scikit-learn pipeline", TriageModel.load(MODEL_PATH), requests, warmup),
        measure_model("ONNX Runtime", OnnxTriageModel.load(ONNX_MODEL_PATH), requests, warmup),
    ]


def measure_apis(image: str, requests: int, warmup: int) -> list[dict]:
    results = []
    for name, backend in (("API + scikit-learn", "sklearn"), ("API + ONNX Runtime", "onnx")):
        with api_container(backend, image) as base_url:
            results.append(measure_api(name, base_url, requests, warmup))
    return results


def to_markdown(
    model_results: list[dict], api_results: list[dict], requests: int, warmup: int, image: str
) -> str:
    model_speedup = model_results[0]["mean_ms"] / model_results[1]["mean_ms"]
    api_speedup = api_results[0]["mean_ms"] / api_results[1]["mean_ms"]
    overhead_ms = api_results[1]["mean_ms"] - model_results[1]["mean_ms"]
    lines = [
        "# ONNX latency comparison (Stage 4)",
        "",
        f"- Measured at: {measured_at()}",
        f"- Samples per row: {requests} (after {warmup} warmup)",
        "- Inputs: real abstracts from `data/sample_triage.csv`",
        "- Timer: `time.perf_counter_ns`, serialized requests, no concurrency",
        "- Same model weights on both sides; only the inference backend changes",
        "- Every row below was measured in a single run on one machine, so the before/after "
        "numbers are directly comparable",
        "",
        "## Model only (in-process)",
        "",
        "The classifier call with no web layer; this isolates the optimization itself.",
        "",
        metrics_table("Model", model_results, decimals=4),
        "",
        f"ONNX Runtime is **{model_speedup:.2f}x** faster than the scikit-learn pipeline on mean "
        "latency.",
        "",
        "## End to end (API in the Docker container)",
        "",
        f"The full request path over loopback, from the `{image}` image, run twice with only "
        "`TRIAGE_MODEL_BACKEND` changed. This is the same measurement as the Stage 1 baseline in "
        "[`latency_baseline.md`](latency_baseline.md), so `API + scikit-learn` is the "
        "pre-optimization number and `API + ONNX Runtime` is what the shipped container serves.",
        "",
        metrics_table("Service", api_results, decimals=3),
        "",
        f"End to end the optimization is worth **{api_speedup:.2f}x** on mean latency. The gain is "
        f"smaller than the model-only figure because HTTP, FastAPI validation and JSON "
        f"serialization add a fixed ~{overhead_ms:.2f} ms per request that no model optimization "
        "can remove.",
        "",
        "Reproduce: `make latency-onnx`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scikit-learn against ONNX Runtime, in-process and through the API"
    )
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--image", default=DOCKER_IMAGE, help="inference image to run")
    parser.add_argument("--out", type=Path, default=BENCHMARKS_DIR / "onnx_latency_comparison.md")
    args = parser.parse_args()

    model_results = measure_models(args.requests, args.warmup)
    api_results = measure_apis(args.image, args.requests, args.warmup)

    write_report(
        args.out,
        to_markdown(model_results, api_results, args.requests, args.warmup, args.image),
    )


if __name__ == "__main__":
    main()
