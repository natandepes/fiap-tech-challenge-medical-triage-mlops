.PHONY: install data train test lint format serve docker-build docker-run latency dag-test \
	monitoring-up monitoring-down load export-onnx latency-onnx

install:
	uv sync --extra dev

data:
	uv run python -m triage_api.dataset

train:
	uv run python -m triage_api.train

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

dag-test:
	AIRFLOW__CORE__LOAD_EXAMPLES=False AIRFLOW__CORE__DAGS_FOLDER=$(PWD)/dags uv run airflow dags test triage_retraining 2026-01-01

serve:
	uv run uvicorn triage_api.main:app --app-dir src --reload

docker-build:
	docker build -t triage-api:latest .

docker-run:
	docker run --rm -p 8000:8000 triage-api:latest

latency: train docker-build
	uv run python scripts/measure_latency.py

export-onnx: train
	uv run python -m triage_api.onnx_export
	uv run python -m triage_api.onnx_validate

latency-onnx: export-onnx
	uv run python scripts/measure_onnx_latency.py

monitoring-up:
	docker compose up --build -d
	@echo "API http://localhost:8000  Prometheus http://localhost:9090  Grafana http://localhost:3000 (admin/admin)"

monitoring-down:
	docker compose down

load:
	uv run python scripts/generate_load.py
