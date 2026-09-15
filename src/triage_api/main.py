from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from triage_api import __version__
from triage_api.config import MODEL_BACKEND
from triage_api.metrics import HEALTH_PATH, PREDICTION_COUNT, install_metrics
from triage_api.model import TriageModel
from triage_api.model_loader import load_model
from triage_api.onnx_model import OnnxTriageModel
from triage_api.schemas import HealthResponse, TriageRequest, TriageResponse

_state: dict[str, TriageModel | OnnxTriageModel] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    _state["model"] = load_model()
    yield
    _state.clear()


app = FastAPI(title="Medical Triage API", version=__version__, lifespan=lifespan)
install_metrics(app)


@app.get(HEALTH_PATH, response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded = "model" in _state
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_backend=MODEL_BACKEND if model_loaded else None,
    )


@app.post("/predict", response_model=TriageResponse)
def predict(request: TriageRequest) -> TriageResponse:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    result = model.predict(request.text)
    PREDICTION_COUNT.labels(urgency=result.urgency).inc()
    return TriageResponse(
        urgency=result.urgency,
        confidence=result.confidence,
        probabilities=result.probabilities,
    )
