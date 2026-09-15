# Medical Triage MLOps: Tech Challenge Phase 3

![CI](https://github.com/natandepes/fiap-tech-challenge-medical-triage-mlops/actions/workflows/ci.yml/badge.svg)

Automatic triage of free-text medical reports. A lightweight NLP classifier assigns an urgency
level (`normal`, `attention`, or `urgent`) and is served as a REST API in a Docker container.
The focus is the model lifecycle end to end: CI/CD, orchestrated retraining, monitoring, and
latency optimization.

| | |
| --- | --- |
| **Model** | TF-IDF + Logistic Regression, **0.635 macro-F1** on 14,438 real medical abstracts (0.33 random baseline) |
| **Optimization** | ONNX Runtime: **2.50x faster** inference, **1.19x** end to end through the API |
| **Serving** | FastAPI in Docker, p95 **1.62 ms** per request |
| **CI/CD** | 4 GitHub Actions jobs: lint, tests, a real Airflow DAG run, Docker build + health check |
| **Monitoring** | Prometheus + Grafana via Compose, 6 provisioned panels |

**Contents**: [Quickstart](#quickstart) · [Architecture decision](#architecture-decision-stage-1) ·
[Model](#model) · [CI/CD](#cicd-stage-2) · [Retraining](#retraining-stage-2) ·
[Monitoring](#monitoring-stage-3) · [Latency](#latency-optimization-stage-4) ·
[Layout](#repository-layout)

## Quickstart

```bash
make install        # uv sync --extra dev
make train          # download corpus + train -> models/model.joblib
make serve          # uvicorn on http://localhost:8000
```

Or straight to the container:

```bash
make docker-build   # downloads the corpus and trains inside the image
make docker-run     # serves on http://localhost:8000
```

```bash
curl -s localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text": "Suppression of carbamazepine-induced rash with prednisone. We report our experience with 20 patients who developed a rash shortly after the introduction of carbamazepine and were treated with prednisone and an antihistamine."}'
```

```json
{"urgency": "attention", "confidence": 0.776, "probabilities": {"attention": 0.776, "normal": 0.12, "urgent": 0.104}}
```

That input is a real abstract from the corpus, and `attention` is the label the mapping assigns it.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/predict` | `{"text": "..."}` → urgency, confidence, per-class probabilities |
| `GET` | `/health` | liveness, whether the model is loaded, and which backend serves it |
| `GET` | `/metrics` | Prometheus exposition format |
| `GET` | `/docs` | OpenAPI UI |

All other entry points are `make` targets: `make test`, `make lint`, `make dag-test`,
`make monitoring-up`, `make load`, `make latency`, `make latency-onnx`.

On WSL2, Docker needs *Docker Desktop → Settings → Resources → WSL Integration* enabled for the
current distro.

## Architecture decision (Stage 1)

### Batch vs. real-time: real-time

The value of triage is routing an urgent report to a clinician *before* the patient waits. A batch
job that scores reports on a schedule would leave an urgent case undetected for the length of the
batch window, which defeats the purpose. Each report is therefore scored **synchronously**: one
HTTP request in, one urgency label out, low latency, so the result can gate the patient queue in
real time. Bulk re-scoring of historical reports (e.g. after a model update) is the only batch-like
workload, and it can reuse the same endpoint offline.

### Serverless vs. always-on: always-on containers

A hospital produces a steady stream of reports with predictable daytime peaks; the load never drops
to zero and is never truly spiky. Scale-to-zero platforms (Lambda, Cloud Run min-instances 0) would
introduce cold starts precisely on the requests we are trying to keep fast, and this phase is
explicitly about latency. The service runs as an **always-on container, minimum 2 replicas** behind
a load balancer, autoscaling **up** on CPU/latency during peaks but never below the floor. The
model is small, so keeping replicas warm is cheap.

### Cloud provider: AWS

| Concern | Choice |
| --- | --- |
| Compute | **ECS on Fargate**: containers without managing nodes; task min count 2, target-tracking autoscaling |
| Ingress | **Application Load Balancer** with health checks on `/health` |
| Image registry | **Amazon ECR** |
| Logs & infra metrics | **CloudWatch** (application metrics are handled by the self-hosted Prometheus/Grafana stack) |
| CI/CD | **GitHub Actions** builds, tests, and pushes the image to ECR |

Serverless alternative: for a low-traffic pilot, **AWS App Runner** or **Cloud Run** would cut cost
and operational overhead, at the price of cold-start latency and less control over p99; acceptable
for a proof of concept, not for production triage.

## Model

TF-IDF (unigrams) + Logistic Regression, scikit-learn. Lightweight, trains in ~15 s, and is
exported to ONNX Runtime for the Stage 4 optimization. The API serves the ONNX build by default.
A heavy transformer would contradict the low-latency premise.

On the held-out split (20%, stratified) it scores **0.635 macro-F1** across the three urgency
levels, against a 0.33 random baseline. That is a realistic number for real clinical abstracts
carrying an urgency label derived from their condition category: the label is a proxy, so part of
the gap is irreducible. Metrics are written to `models/metrics.json` at training time.

The vectorizer is deliberately unigram-only and uses the explicit token pattern `\w\w+`. Both
choices are forced by ONNX parity rather than by accuracy alone; see
[ONNX parity notes](docs/onnx-parity.md).

Trained on the [**Medical Abstracts TC Corpus**](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
14,438 real abstracts, CC BY-SA 3.0, downloaded from a pinned commit so ingestion is
reproducible. The corpus labels condition category, not urgency, so the triage target comes from a
documented mapping. Full corpus details, the mapping table and its clinical caveats:
[dataset notes](docs/dataset.md).

## CI/CD (Stage 2)

The `CI` workflow (`.github/workflows/ci.yml`) runs on every `push` and `pull_request`, four jobs
in parallel:

| Job | What it guards |
| --- | --- |
| `lint` | `ruff check .` and `ruff format --check .` |
| `test` | `pytest -q`: API, dataset mapping, metrics and ONNX parity |
| `dag` | installs Airflow, runs the **real** DAG via `airflow dags test`, then the DagBag assertions |
| `docker-build` | builds the inference image and curls `/health` on the running container |

All jobs use `astral-sh/setup-uv` with the lockfile, so CI resolves the exact same dependency set
as local development.

## Retraining (Stage 2)

`dags/triage_retraining_dag.py` defines the `triage_retraining` DAG, scheduled `@weekly`:

```
ingest ──> train ──> evaluate ──> export_onnx ──> validate_onnx ──> publish
```

`evaluate` is a macro-F1 quality gate and `validate_onnx` an accuracy-drift gate, so a weak model is
never converted and a lossy conversion is never published. `publish` promotes both artifacts to a
timestamped `models/registry/` entry.

```bash
uv sync --extra airflow
make dag-test        # runs the whole DAG against a local SQLite metastore
```

Task-by-task detail, and the **known limitation that `publish` does not yet feed the serving
path**: [retraining notes](docs/retraining.md).

## Monitoring (Stage 3)

```bash
make monitoring-up     # docker compose up --build -d  (api, prometheus, grafana)
make load              # fires demo traffic so the dashboard has something to show
make monitoring-down
```

| Service | URL | Notes |
| --- | --- | --- |
| API | http://localhost:8000 | same image as Stage 1 |
| Prometheus | http://localhost:9090 | scrapes `api:8000/metrics` every 5s |
| Grafana | http://localhost:3000 | `admin` / `admin`; anonymous viewer access enabled |

Grafana auto-provisions the **Medical Triage API** dashboard from
`monitoring/grafana/dashboards/triage-api.json`: request rate by status, p95 latency, 5xx and 4xx
error rates, total requests, and predictions by urgency.

`make load` is the easy-to-forget step: without traffic the panels are live but empty.

Metric definitions, panel queries and instrumentation detail: [monitoring notes](docs/monitoring.md).

## Latency optimization (Stage 4)

The trained pipeline is exported to **ONNX Runtime** (`triage_api.onnx_export`, via `skl2onnx`).
`OnnxTriageModel` implements the same `predict(text)` interface as the scikit-learn model, so the
backend is one environment variable: `TRIAGE_MODEL_BACKEND=onnx` (default) or `sklearn` to roll
back with no code change. `GET /health` reports which is loaded.

Measured in a single run on one machine, 500 requests after 25 warmup, same weights on both sides;
only the backend changes ([methodology](docs/latency.md)).

**Model only, in-process**, which isolates the optimization:

| Model | mean (ms) | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| `scikit-learn pipeline` | 0.3255 | 0.3072 | 0.4381 | 0.5449 |
| `ONNX Runtime` | 0.1302 | 0.1150 | 0.1613 | 0.1731 |

**ONNX Runtime is 2.50x faster** on mean latency, and the artifact is 954 KB against 1.0 MB.

**End to end, API in Docker**, the same image run twice with only the backend changed:

| Service | mean (ms) | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| `API + scikit-learn` | 1.693 | 1.680 | 1.865 | 1.945 |
| `API + ONNX Runtime` | 1.424 | 1.405 | 1.618 | 1.735 |

End to end the optimization is worth **1.19x**. The gain is smaller because HTTP, FastAPI
validation and JSON serialization add a fixed ~1.29 ms that no model optimization can remove. The
honest framing is "the model got 2.50x faster; the service it is wrapped in got 1.19x faster", and
the second number is the one a caller actually feels.

Reproduce with `make latency` (baseline) and `make latency-onnx` (comparison); full output in
[`benchmarks/`](benchmarks/).

## Repository layout

```
src/triage_api/      FastAPI app, model wrappers (sklearn + ONNX), corpus ingestion, training, metrics
dags/                Airflow retraining DAG
scripts/             latency measurement (sklearn baseline + ONNX comparison), demo load generator
monitoring/          Prometheus scrape config + Grafana provisioning and dashboard JSON
tests/               API + dataset mapping + DAG + metrics + ONNX parity tests
benchmarks/          recorded latency numbers (Stage 1 baseline, Stage 4 ONNX comparison)
docs/                deep dives: dataset, retraining, monitoring, latency, ONNX parity
.github/workflows/   CI pipeline (lint, test, dag, docker-build)
Dockerfile           self-contained inference image (trains + exports ONNX at build time)
docker-compose.yml   API + Prometheus + Grafana monitoring stack
```
