from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from triage_api.enums import Urgency

ReportText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=10000),
]


class TriageRequest(BaseModel):
    text: ReportText = Field(description="Free-text medical report")


class TriageResponse(BaseModel):
    urgency: Urgency
    confidence: float
    probabilities: dict[Urgency, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str | None = None
