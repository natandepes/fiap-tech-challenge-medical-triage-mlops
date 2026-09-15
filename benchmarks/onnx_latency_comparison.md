# ONNX latency comparison (Stage 4)

- Measured at: 2026-09-15T01:36:47+00:00
- Samples per row: 500 (after 25 warmup)
- Inputs: real abstracts from `data/sample_triage.csv`
- Timer: `time.perf_counter_ns`, serialized requests, no concurrency
- Same model weights on both sides; only the inference backend changes
- Every row below was measured in a single run on one machine, so the before/after numbers are directly comparable

## Model only (in-process)

The classifier call with no web layer — this isolates the optimization itself.

| Model                   | mean (ms) | p50    | p95    | p99    | max    |
| ----------------------- | --------- | ------ | ------ | ------ | ------ |
| `scikit-learn pipeline` | 0.3255    | 0.3072 | 0.4381 | 0.5449 | 0.8470 |
| `ONNX Runtime`          | 0.1302    | 0.1150 | 0.1613 | 0.1731 | 4.1714 |

ONNX Runtime is **2.50x** faster than the scikit-learn pipeline on mean latency.

## End to end (API in the Docker container)

The full request path over loopback, from the `triage-api:latest` image, run twice with only `TRIAGE_MODEL_BACKEND` changed. This is the same measurement as the Stage 1 baseline in [`latency_baseline.md`](latency_baseline.md), so `API + scikit-learn` is the pre-optimization number and `API + ONNX Runtime` is what the shipped container serves.

| Service              | mean (ms) | p50   | p95   | p99   | max   |
| -------------------- | --------- | ----- | ----- | ----- | ----- |
| `API + scikit-learn` | 1.693     | 1.680 | 1.865 | 1.945 | 2.397 |
| `API + ONNX Runtime` | 1.424     | 1.405 | 1.618 | 1.735 | 2.248 |

End to end the optimization is worth **1.19x** on mean latency. The gain is smaller than the model-only figure because HTTP, FastAPI validation and JSON serialization add a fixed ~1.29 ms per request that no model optimization can remove.

Reproduce: `make latency-onnx`.
