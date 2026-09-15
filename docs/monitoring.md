# Monitoring internals

## Instrumentation

`src/triage_api/metrics.py` uses `prometheus_client` to expose:

| Metric | Type | Labels | What it shows |
| --- | --- | --- | --- |
| `http_requests_total` | Counter | `method`, `path`, `status` | request count, sliceable by route and status code |
| `http_request_duration_seconds` | Histogram | `method`, `path` | request latency, for rate/percentile queries |
| `triage_predictions_total` | Counter | `urgency` | predictions served, by urgency label |

An ASGI middleware records the first two on every request except `/metrics` itself, so scraping the
endpoint doesn't inflate its own counters. `GET /metrics` renders the Prometheus text exposition
format directly, with no sub-app mounting, so there is no trailing-slash redirect to trip up `curl` or a
scraper.

## Dashboard queries

Grafana auto-provisions the **Medical Triage API** dashboard from
`monitoring/grafana/dashboards/triage-api.json` (committed, so it needs no manual click-through).

| Panel | Query |
| --- | --- |
| Request rate by status | `sum by (status) (rate(http_requests_total[1m]))` |
| p95 request latency | `histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[1m])))` |
| Error rate (5xx) | 5xx request rate over total request rate |
| Error rate (4xx) | 4xx request rate over total request rate |
| Total requests | `sum(http_requests_total)` |
| Predictions by urgency | `sum by (urgency) (triage_predictions_total)` |

Prometheus scrapes `api:8000/metrics` every 5s (`monitoring/prometheus.yml`).

## Generating traffic

`make load` runs `scripts/generate_load.py`, which fires a mix of valid triage reports and a few
invalid payloads at a running API; the invalid ones are what give the 4xx error-rate panel
something to plot. Without it the dashboard is live but empty.

## Test coverage

`tests/test_metrics.py` asserts `/metrics` exposes the three metric families and that the request
and prediction counters increment after real calls.
