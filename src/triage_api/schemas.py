from pydantic import BaseModel, Field

from triage_api.enums import Urgency


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000, description="Free-text medical report")


class TriageResponse(BaseModel):
    urgency: Urgency
    confidence: float
    probabilities: dict[Urgency, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
