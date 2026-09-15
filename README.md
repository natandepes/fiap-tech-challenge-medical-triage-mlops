# Medical Triage MLOps — Tech Challenge Phase 3

![CI](https://github.com/natandepes/fiap-tech-challenge-medical-triage-mlops/actions/workflows/ci.yml/badge.svg)

Automatic triage of free-text medical reports. A lightweight NLP classifier assigns an urgency
level — `normal`, `attention`, or `urgent` — and is served as a REST API in a Docker container.
The project's focus is the model lifecycle end to end: CI/CD, orchestrated retraining, monitoring,
and latency optimization.

Trained on 14,438 real medical abstracts (Medical Abstracts TC Corpus).

Status: **Stage 1, 2, 3 & 4 complete** (architecture decision + API in Docker + latency baseline;
CI/CD pipeline + orchestrated retraining DAG; Prometheus + Grafana monitoring stack; ONNX Runtime
optimization shipped as the default serving backend).

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

TF-IDF (unigrams) + Logistic Regression, scikit-learn. Lightweight, trains in ~15 s, and is
exported to ONNX Runtime for the Stage 4 latency optimization (see below) — the API serves the
ONNX build by default. A heavy transformer would contradict the low-latency premise.

On the held-out split (20%, stratified) it scores **0.635 macro-F1** across the three urgency
levels, against a 0.33 random baseline. That is a realistic number for real clinical abstracts
carrying an urgency label derived from their condition category (see below) — the label is a proxy,
so part of the gap is irreducible. Metrics are written to `models/metrics.json` at training time.

The vectorizer is deliberately unigram-only and uses the explicit token pattern `\w\w+`. Both
choices are forced by ONNX parity rather than by accuracy alone — see
[Stage 4](#a-note-on-onnx-parity).

### Dataset

[**Medical Abstracts TC Corpus**](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
(Schopf, Braun & Matthes, NLPIR '22) — 14,438 real medical abstracts published by the
[sebis chair at TU München](https://wwwmatthes.in.tum.de) under CC BY-SA 3.0, downloadable as plain
CSV with no account required. `triage_api.dataset` downloads both official splits into `data/raw/`
(cached, so a second run is offline), maps them, and writes `data/triage.csv`.

The download URL is **pinned to commit `70a2d91`**, not `main`. A `raw.githubusercontent.com` URL at
a commit SHA is content-addressed, so the pin makes ingestion reproducible and tamper-evident in one
move — the recorded accuracy and latency numbers cannot silently drift because an upstream push
changed the data. Override with `TRIAGE_CORPUS_BASE_URL` to test against a newer revision.

Both official splits are concatenated and re-split 80/20 (stratified, seed 42) rather than using the
corpus's own train/test division: the published split is calibrated for the 5-class condition task,
and ours is a 3-class urgency task, so published baselines would not be comparable either way.

The corpus labels **condition category**, not urgency, so the triage target is derived from it with
an explicit, documented mapping:

| Corpus class | Urgency | Rationale |
| --- | --- | --- |
| Cardiovascular diseases | `urgent` | Acute coronary syndromes, infarction, aortic events — treatment windows measured in minutes |
| Neoplasms | `attention` | Require prompt oncologic workup and staging, but not same-hour intervention |
| Nervous system diseases | `attention` | Prompt specialist referral; the corpus mixes acute and chronic presentations |
| Digestive system diseases | `normal` | Predominantly chronic or electively managed in this corpus |
| General pathological conditions | `normal` | Catch-all class with no organ-specific acute pathway |

The mapping lives in `URGENCY_BY_CONDITION` in `src/triage_api/dataset.py`. It is a clinical
simplification and the honest limitation of this project: the *text* is real, the *urgency label*
is a proxy we defined. A production system would need clinician-adjudicated urgency labels. The
resulting distribution is `normal` 6,299 / `attention` 5,088 / `urgent` 3,051, and the imbalance is
handled with `class_weight="balanced"`.

Cite the corpus as Schopf, Braun & Matthes, *Evaluating Unsupervised Text Classification:
Zero-shot and Similarity-based Approaches*, NLPIR '22 ([doi](https://doi.org/10.1145/3582768.3582795)).

Raw corpus files and the built CSV stay out of git; `data/sample_triage.csv` holds a committed,
stratified 150-row sample used by the tests, the latency scripts and the load generator, so a clean
checkout can run everything except training without network access.

## Run it

### Local (uv)

```bash
make install        # uv sync --extra dev
make train          # download corpus + train -> models/model.joblib
make serve          # uvicorn on http://localhost:8000
make latency        # record the latency baseline -> benchmarks/
```

### Docker

```bash
make docker-build   # downloads the corpus and trains inside the image
make docker-run     # serves on http://localhost:8000
```

On WSL2, Docker needs *Docker Desktop → Settings → Resources → WSL Integration* enabled for the
current distro.

### Call the API

```bash
curl -s localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text": "Suppression of carbamazepine-induced rash with prednisone. We report our experience with 20 patients who developed a rash shortly after the introduction of carbamazepine and were treated with prednisone and an antihistamine."}'
```

```json
{"urgency": "attention", "confidence": 0.797, "probabilities": {"attention": 0.797, "normal": 0.114, "urgent": 0.089}}
```

That input is a real abstract from the corpus, and `attention` is the label the mapping assigns it.
Confidences sit in a realistic range rather than pinned near 1.0.

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

Recorded baseline (serialized requests, `time.perf_counter_ns`): model **0.311 ms** mean / **0.387 ms**
p95; API **1.392 ms** mean / **1.581 ms** p95. The request path is dominated by framework and
serialization overhead, so the Stage 4 comparison focuses on the in-process number.

## Stage 2 — CI/CD and retraining pipeline

### CI/CD (GitHub Actions)

The `CI` workflow (`.github/workflows/ci.yml`) runs on every `push` and `pull_request`, with four
jobs in parallel:

| Job | What it guards |
| --- | --- |
| `lint` | `ruff check .` and `ruff format --check .` — style and formatting |
| `test` | `pytest -q` — API, dataset-mapping and ONNX parity tests (Airflow absent, so the DAG test skips) |
| `dag` | installs the `airflow` extra, runs `airflow db migrate` then `airflow dags test triage_retraining 2026-01-01`, and finally `pytest tests/test_dag.py` so the DagBag assertions run with Airflow present |
| `docker-build` | builds the inference image and curls `/health` on the running container |

All jobs use `astral-sh/setup-uv` with the lockfile, so CI resolves the exact same dependency set
as local development. The badge under the title reflects the latest run on the default branch.

### Retraining pipeline (Airflow)

`dags/triage_retraining_dag.py` defines the `triage_retraining` DAG — a TaskFlow pipeline:

```
ingest ──> train ──> evaluate ──> export_onnx ──> validate_onnx ──> publish
```

- **ingest** — downloads the Medical Abstracts TC Corpus (cached in `data/raw/`) and writes the
  mapped `data/triage.csv` (`triage_api.dataset.build_dataset`).
- **train** — fits the TF-IDF + Logistic Regression pipeline, writes `models/model.joblib` and
  `models/metrics.json`.
- **evaluate** — a macro-F1 gate: the run fails if `macro_f1` is below `TRIAGE_MIN_MACRO_F1`
  (default `0.55`, against a measured 0.635 and a 0.33 random baseline).
- **export_onnx** — converts the freshly trained pipeline to `models/model.onnx` (Stage 4), only
  once the quality gate has passed.
- **validate_onnx** — re-evaluates the exported ONNX model on the same held-out test split and
  fails the run if its macro-F1 drifts from the sklearn model's by more than
  `TRIAGE_MAX_ONNX_F1_DRIFT` (default `0.02`). Catches conversion regressions before a bad ONNX
  artifact ever reaches the registry — `tests/test_onnx.py` checks per-prediction parity on fixed
  samples, this checks aggregate accuracy on a full held-out split.
- **publish** — promotes both artifacts to `models/registry/model-<utc-timestamp>.{joblib,onnx}`
  and updates `models/registry/latest.txt` / `latest_onnx.txt`.

Schedule: `@weekly`, `catchup=False`. `models/` is gitignored, so DAG runs leave no commit noise.

### Run the DAG locally

```bash
uv sync --extra airflow
make dag-test        # airflow dags test triage_retraining 2026-01-01
```

`airflow dags test` executes the whole DAG against a local SQLite metastore — no scheduler or
webserver needed.

### Known limitation: `publish` does not feed serving

`publish` promotes artifacts into `models/registry/` and updates `latest.txt` / `latest_onnx.txt`,
but nothing reads those pointers back. The API always loads the fixed paths `models/model.joblib` /
`models/model.onnx` (`MODEL_PATH` / `ONNX_MODEL_PATH` in `triage_api.config`), and the `Dockerfile`
trains and exports its own model at build time (`RUN python -m triage_api.train` /
`RUN python -m triage_api.onnx_export`), independently of any DAG run. So a completed retraining
run archives a versioned, validated pair of artifacts, but does not change what a running API
serves — the registry is a version history, not yet a deployment mechanism.

The same disconnect shows up in CI: the `dag` job in `.github/workflows/ci.yml` runs the real DAG
(`airflow dags test`, not a mock) on every push and PR, including a genuine `publish` step, but that
run happens on a throwaway GitHub Actions runner whose filesystem is discarded when the job ends —
nothing uploads or persists `models/registry/`. The `docker-build` job in the same workflow has no
dependency on `dag` and trains its own model independently inside the Docker build. CI therefore
proves the DAG is functionally correct end to end (the Stage 2 requirement), not that a retrain
updates a deployed model.

Closing this gap would mean having the API (or an entrypoint script) read `latest_onnx.txt` /
`latest.txt` at startup instead of a fixed path. Out of scope for this project as submitted; noted
here so a demo of `make dag-test` isn't mistaken for a live model update.

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
4. **Error rate (4xx)** — 4xx request rate over total request rate
5. **Total requests** and **predictions by urgency** — traffic volume and the model's label mix

`make load` is the easy-to-forget step that gives the dashboard something to show — it fires a mix
of valid triage reports and a few invalid payloads (for the error-rate panel) at a running API.

### Test coverage

`tests/test_metrics.py` asserts `/metrics` exposes the three metric families and that the request
and prediction counters increment after real calls.

## Stage 4 — Latency optimization and delivery

### Optimization

The trained scikit-learn pipeline (`models/model.joblib`) is exported to **ONNX Runtime**
(`triage_api.onnx_export`, using `skl2onnx`) as `models/model.onnx`, alongside a small
`model.classes.json` sidecar that records the class label order — with the converter's `zipmap`
output disabled for speed, onnxruntime's public API doesn't otherwise expose which output column
maps to which label.

`OnnxTriageModel` (`src/triage_api/onnx_model.py`) implements the same `predict(text)` interface as
the scikit-learn `TriageModel`, so the API can switch between them with one environment variable:

```bash
export TRIAGE_MODEL_BACKEND=onnx      # default — the optimized model
export TRIAGE_MODEL_BACKEND=sklearn   # rollback, no code change needed
```

`GET /health` reports which backend is currently loaded (`model_backend`).

### Latency comparison

```bash
make latency-onnx   # trains + exports + validates + runs scripts/measure_onnx_latency.py
```

In-process, same methodology as the Stage 1 `model (in-process)` baseline (500 requests, 25
warmup, `time.perf_counter_ns`, no concurrency):

| Model | mean (ms) | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| `scikit-learn pipeline` | 0.3118 | 0.2981 | 0.3822 | 0.4317 | 0.5410 |
| `ONNX Runtime` | 0.1289 | 0.1175 | 0.1660 | 0.1783 | 0.1859 |

**ONNX Runtime is 2.42x faster** than the scikit-learn pipeline on mean latency, for the same
inputs and the same model weights, and the exported artifact is 954 KB against 1.0 MB for the
joblib pipeline. Full output:
[`benchmarks/onnx_latency_comparison.md`](benchmarks/onnx_latency_comparison.md).

### A note on ONNX parity

Converting a TF-IDF pipeline to ONNX is not lossless, and on real clinical text the gap is big
enough to matter. Two problems showed up only once the synthetic corpus was replaced with real
abstracts:

1. **`skl2onnx` drops bigrams that span a discarded token.** scikit-learn's default token pattern
   `(?u)\b\w\w+\b` throws away one-character tokens *before* forming n-grams, so
   `"crosslinked D-dimer"` yields the bigram `crosslinked dimer` and `"p less than 0.001"` yields
   `than 001`. ONNX Runtime's RE2-based tokenizer does not reproduce those, so the exported model
   silently saw a different feature vector. Also, RE2 rejects the `(?u)` inline flag outright.
   Passing the explicit pattern `\w\w+` cut label disagreement from 6/600 to 1/600 with an
   identical vocabulary and identical accuracy.
2. **Bigrams made ONNX *slower* than scikit-learn.** With `ngram_range=(1, 2)` the vocabulary is
   233k features, the ONNX artifact is 11.2 MB and inference measured **0.6x** — a slowdown.
   Unigrams cut the artifact to 954 KB, run 2.42x faster, *and* score better (0.635 vs. 0.612
   macro-F1), since bigrams over long abstracts mostly add sparse noise.

What remains is float32-vs-float64 rounding: the ONNX and scikit-learn labels still disagree on
~0.6% of inputs, and in every observed case the top two classes were within 0.018 of each other —
genuine coin-flips rather than substantive disagreement. `tests/test_onnx.py` encodes exactly that
invariant instead of a blanket equality assertion: probabilities must track within `5e-2`, and a
label may differ *only* when the top-two gap is under `0.05`.

The takeaway for the report: the optimization is real, but it constrained the model. The
vectorizer's configuration is driven by what converts faithfully, not only by what scores best —
and here, fortunately, the two agreed.

### Delivery

- `Dockerfile` exports the ONNX artifact at build time and the runtime image defaults to
  `TRIAGE_MODEL_BACKEND=onnx` — the container built from `main` serves the optimized model, not
  just a benchmark script.
- `dags/triage_retraining_dag.py` adds `export_onnx` and `validate_onnx` tasks after the macro-F1
  quality gate; a failing model is never converted, and a conversion that drifts from the sklearn
  model's accuracy is never published. `publish` now promotes both `model-<timestamp>.joblib` and
  `model-<timestamp>.onnx` to `models/registry/`, so the served ONNX artifact stays in sync with
  weekly retraining instead of silently going stale.
- `tests/test_onnx.py` guards the conversion against the parity limits described above, on 60 real
  sample abstracts.

## Repository layout

```
src/triage_api/      FastAPI app, model wrappers (sklearn + ONNX), corpus ingestion, training, metrics
dags/                Airflow retraining DAG
scripts/             latency measurement (sklearn baseline + ONNX comparison), demo load generator
monitoring/          Prometheus scrape config + Grafana provisioning and dashboard JSON
tests/               API + dataset mapping + DAG + metrics + ONNX parity tests
benchmarks/          recorded latency numbers (Stage 1 baseline, Stage 4 ONNX comparison)
.github/workflows/   CI pipeline (lint, test, dag, docker-build)
Dockerfile           self-contained inference image (trains + exports ONNX at build time)
docker-compose.yml   API + Prometheus + Grafana monitoring stack
```
