# Retraining pipeline (Airflow)

`dags/triage_retraining_dag.py` defines the `triage_retraining` DAG, a TaskFlow pipeline:

```
ingest ──> train ──> evaluate ──> export_onnx ──> validate_onnx ──> publish
```

| Task | What it does |
| --- | --- |
| `ingest` | Downloads the Medical Abstracts TC Corpus (cached in `data/raw/`) and writes the mapped `data/triage.csv` (`triage_api.dataset.build_dataset`) |
| `train` | Fits the TF-IDF + Logistic Regression pipeline, writes `models/model.joblib` and `models/metrics.json` |
| `evaluate` | Quality gate: fails the run if `macro_f1` is below `TRIAGE_MIN_MACRO_F1` (default `0.55`, against a measured 0.635 and a 0.33 random baseline) |
| `export_onnx` | Converts the freshly trained pipeline to `models/model.onnx`, only once the quality gate has passed |
| `validate_onnx` | Re-evaluates the exported ONNX model on the same held-out split and fails if its macro-F1 drifts from the sklearn model's by more than `TRIAGE_MAX_ONNX_F1_DRIFT` (default `0.02`) |
| `publish` | Promotes both artifacts to `models/registry/model-<utc-timestamp>.{joblib,onnx}` and updates `latest.txt` / `latest_onnx.txt` |

`validate_onnx` and `tests/test_onnx.py` guard different things: the test checks per-prediction
parity on fixed samples, this task checks aggregate accuracy on a full held-out split, so a bad
ONNX artifact never reaches the registry.

Schedule: `@weekly`, `catchup=False`. `models/` is gitignored, so DAG runs leave no commit noise.

## Run it locally

```bash
uv sync --extra airflow
make dag-test        # airflow dags test triage_retraining 2026-01-01
```

`airflow dags test` executes the whole DAG against a local SQLite metastore, with no scheduler or
webserver needed.

## Known limitation: `publish` does not feed serving

`publish` promotes artifacts into `models/registry/` and updates `latest.txt` / `latest_onnx.txt`,
but nothing reads those pointers back. The API always loads the fixed paths `models/model.joblib` /
`models/model.onnx` (`MODEL_PATH` / `ONNX_MODEL_PATH` in `triage_api.config`), and the `Dockerfile`
trains and exports its own model at build time (`RUN python -m triage_api.train` /
`RUN python -m triage_api.onnx_export`), independently of any DAG run. So a completed retraining
run archives a versioned, validated pair of artifacts, but does not change what a running API
serves: the registry is a version history, not yet a deployment mechanism.

The same disconnect shows up in CI: the `dag` job in `.github/workflows/ci.yml` runs the real DAG
(`airflow dags test`, not a mock) on every push and PR, including a genuine `publish` step, but that
run happens on a throwaway GitHub Actions runner whose filesystem is discarded when the job ends;
nothing uploads or persists `models/registry/`. The `docker-build` job in the same workflow has no
dependency on `dag` and trains its own model independently inside the Docker build. CI therefore
proves the DAG is functionally correct end to end (the Stage 2 requirement), not that a retrain
updates a deployed model.

Closing this gap would mean having the API (or an entrypoint script) read `latest_onnx.txt` /
`latest.txt` at startup instead of a fixed path. Out of scope for this project as submitted; noted
here so a demo of `make dag-test` isn't mistaken for a live model update.
