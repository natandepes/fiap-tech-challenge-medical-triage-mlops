# Medical Triage MLOps — Tech Challenge Phase 3

Automatic triage of free-text medical reports. A lightweight NLP classifier assigns an urgency
level — `normal`, `attention`, or `urgent` — and is served as a REST API in a Docker container.
The project's focus is the model lifecycle end to end: CI/CD, orchestrated retraining, monitoring,
and latency optimization.

Status: **Stage 1 complete** (architecture decision + API in Docker + latency baseline).

## Architecture decision (Stage 1)

### Batch vs. real-time — real-time

The value of triage is routing an urgent report to a clinician *before* the patient waits. A batch
job that scores reports on a schedule would leave an urgent case undetected for the length of the
batch window, which defeats the purpose. Each report is therefore scored **synchronously**: one
HTTP request in, one urgency label out, low latency, so the result can gate the patient queue in
real time. Bulk re-scoring of historical reports (e.g. after a model update) is the only batch-like
workload, and it can reuse the same endpoint offline.

### Serverless vs. always-on — always-on containers

A hospital produces a steady stream of reports with predictable daytime peaks; the load never
drops to zero and is never truly spiky. Scale-to-zero platforms (Lambda, Cloud Run min-instances 0)
would introduce cold starts precisely on the requests we are trying to keep fast, and this phase is
explicitly about latency. The service runs as an **always-on container, minimum 2 replicas** behind
a load balancer, autoscaling **up** on CPU/latency during peaks but never scaling below the floor.
The model is small, so keeping replicas warm is cheap.

### Cloud provider — AWS

| Concern | Choice |
| --- | --- |
| Compute | **ECS on Fargate** — containers without managing nodes; task min count 2, target-tracking autoscaling |
| Ingress | **Application Load Balancer** with health checks on `/health` |
| Image registry | **Amazon ECR** |
| Logs & infra metrics | **CloudWatch** (application metrics are handled by the self-hosted Prometheus/Grafana stack in Stage 3) |
| CI/CD | **GitHub Actions** builds, tests, and pushes the image to ECR (Stage 2) |

Serverless alternative: for a low-traffic pilot, **AWS App Runner** or **Cloud Run** would cut cost
and operational overhead, at the price of cold-start latency and less control over p99 — acceptable
for a proof of concept, not for production triage.

## Model

TF-IDF (1–2 grams) + Logistic Regression, scikit-learn. Lightweight, trains in seconds, and
converts cleanly to ONNX Runtime for the Stage 4 latency optimization. A heavy transformer would
contradict the low-latency premise.

On the held-out split the baseline scores ~0.99 macro-F1. The score is high because the corpus is
synthetic and the urgency vocabulary is fairly separable by design; the metric that matters for
this phase is the pipeline, not the ceiling of a toy dataset. Metrics are written to
`models/metrics.json` at training time.

### Dataset

No public medical-text dataset is labelled by urgency, so the labels would be synthetic in any
case. `triage_api.dataset` generates a reproducible corpus (~3000 rows, fixed seed) of report-style
sentences composed from per-urgency finding banks. `data/sample_triage.csv` holds a committed
sample; the full CSV is regenerated on demand and stays out of git. In Stage 2 this generator
becomes the ingestion task of the Airflow DAG.

## Run it

### Local (uv)

```bash
make install        # uv sync --extra dev
make train          # generate dataset + train -> models/model.joblib
make serve          # uvicorn on http://localhost:8000
make latency        # record the latency baseline -> benchmarks/
```

### Docker

```bash
make docker-build   # trains the model inside the image
make docker-run     # serves on http://localhost:8000
```

On WSL2, Docker needs *Docker Desktop → Settings → Resources → WSL Integration* enabled for the
current distro.

### Call the API

```bash
curl -s localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text": "CT of the cranium: acute intracranial hemorrhage. Immediate clinical attention required."}'
```

```json
{"urgency": "urgent", "confidence": 0.98, "probabilities": {"urgent": 0.98, "attention": 0.01, "normal": 0.01}}
```

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/predict` | `{"text": "..."}` → urgency, confidence, per-class probabilities |
| `GET` | `/health` | liveness + whether the model artifact is loaded |
| `GET` | `/docs` | OpenAPI UI |

## Latency baseline

```bash
make latency
```

One command: it trains the model, builds the inference image, runs it as a container, waits for
`/health`, fires 500 serialized requests, tears the container down, and writes
`benchmarks/latency_baseline.md`:

- `model (in-process)` — the classifier call with no web layer, the figure Stage 4 compares against
  the ONNX build.
- `API (Docker container)` — the full request path.

Pass `--base-url` to measure an already-running instance instead of starting a container. See
[`benchmarks/latency_baseline.md`](benchmarks/latency_baseline.md).

Recorded baseline (serialized requests, `time.perf_counter_ns`): model **0.28 ms** mean / **0.35 ms**
p95; API **1.46 ms** mean / **1.65 ms** p95. The request path is dominated by framework and
serialization overhead, so the Stage 4 comparison focuses on the in-process number.

## Repository layout

```
src/triage_api/      FastAPI app, model wrapper, dataset generator, training
scripts/             latency measurement
tests/               API + dataset smoke tests
benchmarks/          recorded latency numbers
Dockerfile           self-contained inference image (trains at build time)
```

## Roadmap

- **Stage 2** — GitHub Actions (lint + test on push) and an Airflow DAG (ingest → train → save).
- **Stage 3** — `prometheus_client` instrumentation and a `docker-compose.yml` bringing up API +
  Prometheus + Grafana with a dashboard.
- **Stage 4** — ONNX Runtime export and the documented latency comparison.
