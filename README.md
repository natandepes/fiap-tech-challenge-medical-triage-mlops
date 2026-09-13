# Medical Triage MLOps — Tech Challenge Phase 3

![CI](https://github.com/natandepes/fiap-tech-challenge-medical-triage-mlops/actions/workflows/ci.yml/badge.svg)

Automatic triage of free-text medical reports. A lightweight NLP classifier assigns an urgency
level — `normal`, `attention`, or `urgent` — and is served as a REST API in a Docker container.
The project's focus is the model lifecycle end to end: CI/CD, orchestrated retraining, monitoring,
and latency optimization.

Status: **Stage 1, 2 & 3 complete** (architecture decision + API in Docker + latency baseline;
CI/CD pipeline + orchestrated retraining DAG; Prometheus + Grafana monitoring stack).

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

## Stage 2 — CI/CD and retraining pipeline

### CI/CD (GitHub Actions)

The `CI` workflow (`.github/workflows/ci.yml`) runs on every `push` and `pull_request`, with four
jobs in parallel:

| Job | What it guards |
| --- | --- |
| `lint` | `ruff check .` and `ruff format --check .` — style and formatting |
| `test` | `pytest -q` — API and dataset smoke tests (Airflow absent, so the DAG test skips) |
| `dag` | installs the `airflow` extra, runs `airflow db migrate` then `airflow dags test triage_retraining 2026-01-01`, and finally `pytest tests/test_dag.py` so the DagBag assertions run with Airflow present |
| `docker-build` | builds the inference image and curls `/health` on the running container |

All jobs use `astral-sh/setup-uv` with the lockfile, so CI resolves the exact same dependency set
as local development. The badge under the title reflects the latest run on the default branch.

### Retraining pipeline (Airflow)

`dags/triage_retraining_dag.py` defines the `triage_retraining` DAG — a TaskFlow pipeline:

```
ingest ──> train ──> evaluate ──> publish
```

- **ingest** — regenerates the synthetic corpus (`triage_api.dataset.build_dataset`).
- **train** — fits the TF-IDF + Logistic Regression pipeline, writes `models/model.joblib` and
  `models/metrics.json`.
- **evaluate** — a macro-F1 gate: the run fails if `macro_f1` is below `TRIAGE_MIN_MACRO_F1`
  (default `0.80`).
- **publish** — promotes the artifact to `models/registry/model-<utc-timestamp>.joblib` and updates
  `models/registry/latest.txt`.

Schedule: `@weekly`, `catchup=False`. `models/` is gitignored, so DAG runs leave no commit noise.

### Run the DAG locally

```bash
uv sync --extra airflow
make dag-test        # airflow dags test triage_retraining 2026-01-01
```

`airflow dags test` executes the whole DAG against a local SQLite metastore — no scheduler or
webserver needed.

## Stage 3 — Monitoring and observability

### Instrumentation

`src/triage_api/metrics.py` uses `prometheus_client` to expose:

| Metric | Type | Labels | What it shows |
| --- | --- | --- | --- |
| `http_requests_total` | Counter | `method`, `path`, `status` | request count, sliceable by route and status code |
| `http_request_duration_seconds` | Histogram | `method`, `path` | request latency, for rate/percentile queries |
| `triage_predictions_total` | Counter | `urgency` | predictions served, by urgency label |

An ASGI middleware records the first two on every request except `/metrics` itself (so scraping the
endpoint doesn't inflate its own counters); `GET /metrics` renders the Prometheus text exposition
format directly — no sub-app mounting, so there is no trailing-slash redirect to trip up `curl` or a
scraper.

### Bring up the stack

```bash
make monitoring-up     # docker compose up --build -d  (api, prometheus, grafana)
make load              # scripts/generate_load.py — fires demo traffic at the API
make monitoring-down   # docker compose down
```

| Service | URL | Notes |
| --- | --- | --- |
| API | http://localhost:8000 | same image as Stage 1 |
| Prometheus | http://localhost:9090 | scrapes `api:8000/metrics` every 5s (`monitoring/prometheus.yml`) |
| Grafana | http://localhost:3000 | `admin` / `admin`; anonymous viewer access enabled |

### Dashboard

Grafana auto-provisions the **Medical Triage API** dashboard from
`monitoring/grafana/dashboards/triage-api.json` (committed, reproducible — no manual click-through
needed). Panels:

1. **Request rate by status** — `sum by (status) (rate(http_requests_total[1m]))`
2. **p95 request latency** — `histogram_quantile(0.95, ...http_request_duration_seconds_bucket...)`
3. **Error rate (5xx)** — 5xx request rate over total request rate
4. **Total requests** and **predictions by urgency** — traffic volume and the model's label mix

`make load` is the easy-to-forget step that gives the dashboard something to show — it fires a mix
of valid triage reports and a few invalid payloads (for the error-rate panel) at a running API.

### Test coverage

`tests/test_metrics.py` asserts `/metrics` exposes the three metric families and that the request
and prediction counters increment after real calls.

## Repository layout

```
src/triage_api/      FastAPI app, model wrapper, dataset generator, training, metrics
dags/                Airflow retraining DAG
scripts/             latency measurement, demo load generator
monitoring/          Prometheus scrape config + Grafana provisioning and dashboard JSON
tests/               API + dataset + DAG + metrics smoke tests
benchmarks/          recorded latency numbers
.github/workflows/   CI pipeline (lint, test, dag, docker-build)
Dockerfile           self-contained inference image (trains at build time)
docker-compose.yml   API + Prometheus + Grafana monitoring stack
```

## Roadmap

- **Stage 4** — ONNX Runtime export and the documented latency comparison.
