# Latency baseline (Stage 1)

- Measured at: 2026-09-15T01:11:04+00:00
- Samples per stage: 500 (after 25 warmup)
- Model: TF-IDF + Logistic Regression (scikit-learn), no optimization
- Inputs: real abstracts from `data/sample_triage.csv`
- Timer: `time.perf_counter_ns`, serialized requests, no concurrency
- API served from the `triage-api:latest` container over loopback, pinned to `TRIAGE_MODEL_BACKEND=sklearn` so this baseline stays pre-optimization

| Stage                    | mean (ms) | p50   | p95   | p99   | max    |
| ------------------------ | --------- | ----- | ----- | ----- | ------ |
| `model (in-process)`     | 0.321     | 0.303 | 0.413 | 0.486 | 0.651  |
| `API (Docker container)` | 1.804     | 1.714 | 1.913 | 2.001 | 41.626 |

`model (in-process)` is the classifier call with no web layer. `API (Docker container)` is the full request path: the same prediction plus HTTP, FastAPI validation and JSON serialization.

Stage 4 re-runs *both* of these measurements against the ONNX build, so the optimized numbers are comparable to this baseline stage for stage — see [`onnx_latency_comparison.md`](onnx_latency_comparison.md).

Reproduce: `make latency`.
