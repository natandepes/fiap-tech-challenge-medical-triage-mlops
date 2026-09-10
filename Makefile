.PHONY: install data train test lint serve docker-build docker-run latency

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

serve:
	uv run uvicorn triage_api.main:app --app-dir src --reload

docker-build:
	docker build -t triage-api:latest .

docker-run:
	docker run --rm -p 8000:8000 triage-api:latest

latency:
	uv run python scripts/measure_latency.py
