from __future__ import annotations

import time

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

METRICS_PATH = "/metrics"
HEALTH_PATH = "/health"

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTION_COUNT = Counter(
    "triage_predictions_total",
    "Total triage predictions served, by predicted urgency",
    ["urgency"],
)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return route.path if route is not None else request.url.path


async def metrics_middleware(request: Request, call_next) -> Response:
    if request.url.path in (METRICS_PATH, HEALTH_PATH):
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    path = _route_template(request)
    REQUEST_COUNT.labels(method=request.method, path=path, status=response.status_code).inc()
    REQUEST_DURATION.labels(method=request.method, path=path).observe(duration)
    return response


def render_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def install_metrics(app: FastAPI) -> None:
    app.middleware("http")(metrics_middleware)
    app.add_api_route(METRICS_PATH, render_metrics, methods=["GET"])
