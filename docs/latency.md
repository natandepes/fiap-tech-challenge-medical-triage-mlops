# Latency measurement methodology

Both benchmarks share one harness (`scripts/latency_bench.py`): 500 serialized requests after 25
warmup, timed with `time.perf_counter_ns`, no concurrency, inputs drawn from real abstracts in
`data/sample_triage.csv`. Results land in [`benchmarks/`](../benchmarks/) as Markdown.

Two levels are measured, because they answer different questions:

- **model (in-process)** — the classifier call with no web layer. Isolates the optimization itself.
- **API (Docker container)** — the full request path over loopback: the same prediction plus HTTP,
  FastAPI validation and JSON serialization. This is what a caller actually feels.

## Stage 1 baseline — `make latency`

One command: trains the model, builds the inference image, runs it as a container, waits for
`/health`, fires the requests, tears the container down, and writes
[`benchmarks/latency_baseline.md`](../benchmarks/latency_baseline.md).

This is the **pre-optimization** baseline, so the container is started with
`TRIAGE_MODEL_BACKEND=sklearn` and the script asserts against `/health` that the container really is
serving that backend before it times anything. Without that pin the measurement would silently
drift: the image defaults to the ONNX backend, so an unpinned re-run would benchmark the optimized
model and still label itself "no optimization".

Pass `--base-url` to measure an already-running instance instead of starting a container; the report
then records whichever backend that instance reports.

## Stage 4 comparison — `make latency-onnx`

Trains, exports, validates, builds the image, then measures all four rows in a **single run on one
machine** so the before/after numbers are directly comparable — same model weights on both sides,
only the backend changes. Writes
[`benchmarks/onnx_latency_comparison.md`](../benchmarks/onnx_latency_comparison.md).

## Reading the numbers

The model-only speedup is much larger than the end-to-end one because HTTP, FastAPI validation and
JSON serialization add a fixed ~1.29 ms per request that no model optimization can remove. Both are
reported in the README rather than just the flattering one.

Absolute values are machine-dependent; what is reproducible is the ratio between the two backends
measured in the same run. Outliers in the `max` column reflect ordinary scheduling noise on a
developer laptop — the mean, p50, p95 and p99 columns are the stable signal.
