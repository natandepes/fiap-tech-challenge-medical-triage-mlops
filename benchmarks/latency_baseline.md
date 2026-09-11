# Latency baseline (Stage 1)

- Measured at: 2026-09-11T00:35:31+00:00
- Samples per stage: 500 (after 25 warmup)
- Model: TF-IDF + Logistic Regression (scikit-learn), no optimization
- Timer: `time.perf_counter_ns`, serialized requests, no concurrency
- API served from the `triage-api:latest` container over loopback

| Stage                    | mean (ms) | p50   | p95   | p99   | max   |
| ------------------------ | --------- | ----- | ----- | ----- | ----- |
| `model (in-process)`     | 0.280     | 0.265 | 0.346 | 0.414 | 0.577 |
| `API (Docker container)` | 1.462     | 1.449 | 1.648 | 1.823 | 2.492 |

`model (in-process)` is the classifier call with no web layer; it is the number Stage 4 compares against the ONNX build. `API (Docker container)` is the full request path, dominated by framework and serialization overhead.

Reproduce: `make latency`.
