# ONNX latency comparison (Stage 4)

- Measured at: 2026-09-14T23:11:12+00:00
- Samples per model: 500 (after 25 warmup)
- Both models loaded in-process, no web layer — same methodology as the Stage 1 `model (in-process)` baseline in `benchmarks/latency_baseline.md`
- Timer: `time.perf_counter_ns`, serialized requests, no concurrency
- Inputs: real abstracts from `data/sample_triage.csv`

| Model                   | mean (ms) | p50    | p95    | p99    | max    |
| ----------------------- | --------- | ------ | ------ | ------ | ------ |
| `scikit-learn pipeline` | 0.3118    | 0.2981 | 0.3822 | 0.4317 | 0.5410 |
| `ONNX Runtime`          | 0.1289    | 0.1175 | 0.1660 | 0.1783 | 0.1859 |

ONNX Runtime is **2.42x** faster than the scikit-learn pipeline on mean latency for the same inputs and the same model weights.

Reproduce: `make latency-onnx`.
