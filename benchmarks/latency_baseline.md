# Latency baseline (Stage 1)

- Endpoint: `POST http://localhost:8000/predict`
- Requests measured: 500 (after 50 warmup)
- Measured at: 2026-09-10T01:58:30+00:00
- Model: TF-IDF + Logistic Regression (scikit-learn), no optimization
- Host: single uvicorn worker over loopback; serialized requests, no concurrency

| Metric | Value (ms) |
| --- | --- |
| mean | 1.13 |
| p50 | 1.12 |
| p95 | 1.26 |
| p99 | 1.35 |
| max | 1.68 |

Stage 4 re-runs this script against the ONNX build and records the comparison.
