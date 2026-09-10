from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000, description="Free-text medical report")


class TriageResponse(BaseModel):
    urgency: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
