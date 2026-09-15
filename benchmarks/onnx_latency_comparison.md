# ONNX latency comparison (Stage 4)

- Measured at: 2026-09-15T01:12:28+00:00
- Samples per row: 500 (after 25 warmup)
- Inputs: real abstracts from `data/sample_triage.csv`
- Timer: `time.perf_counter_ns`, serialized requests, no concurrency
- Same model weights on both sides; only the inference backend changes
- Every row below was measured in a single run on one machine, so the before/after numbers are directly comparable

## Model only (in-process)

The classifier call with no web layer — this isolates the optimization itself.

| Model                   | mean (ms) | p50    | p95    | p99    | max    |
| ----------------------- | --------- | ------ | ------ | ------ | ------ |
| `scikit-learn pipeline` | 0.3213    | 0.3053 | 0.4011 | 0.4933 | 1.1007 |
| `ONNX Runtime`          | 0.1398    | 0.1392 | 0.1799 | 0.1980 | 0.2789 |

ONNX Runtime is **2.30x** faster than the scikit-learn pipeline on mean latency.

## End to end (API in the Docker container)

The full request path over loopback, from the `triage-api:latest` image, run twice with only `TRIAGE_MODEL_BACKEND` changed. This is the same measurement as the Stage 1 baseline in [`latency_baseline.md`](latency_baseline.md), so `API + scikit-learn` is the pre-optimization number and `API + ONNX Runtime` is what the shipped container serves.

| Service              | mean (ms) | p50   | p95   | p99   | max   |
| -------------------- | --------- | ----- | ----- | ----- | ----- |
| `API + scikit-learn` | 1.749     | 1.735 | 1.980 | 2.094 | 2.292 |
| `API + ONNX Runtime` | 1.462     | 1.450 | 1.627 | 1.700 | 1.937 |

End to end the optimization is worth **1.20x** on mean latency. The gain is smaller than the model-only figure because HTTP, FastAPI validation and JSON serialization add a fixed ~1.32 ms per request that no model optimization can remove.

Reproduce: `make latency-onnx`.
