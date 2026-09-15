# Latency baseline (Stage 1)

- Measured at: 2026-09-14T23:12:33+00:00
- Samples per stage: 500 (after 25 warmup)
- Model: TF-IDF + Logistic Regression (scikit-learn), no optimization
- Inputs: real abstracts from `data/sample_triage.csv`
- Timer: `time.perf_counter_ns`, serialized requests, no concurrency
- API served from the `triage-api:latest` container over loopback

| Stage                    | mean (ms) | p50   | p95   | p99   | max   |
| ------------------------ | --------- | ----- | ----- | ----- | ----- |
| `model (in-process)`     | 0.311     | 0.297 | 0.387 | 0.459 | 0.779 |
| `API (Docker container)` | 1.392     | 1.372 | 1.581 | 1.655 | 2.385 |

`model (in-process)` is the classifier call with no web layer; it is the number Stage 4 compares against the ONNX build. `API (Docker container)` is the full request path, dominated by framework and serialization overhead.

Reproduce: `make latency`.
