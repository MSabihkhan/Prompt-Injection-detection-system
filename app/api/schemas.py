"""Request/response models for the classification API."""
from pydantic import BaseModel, Field

MAX_CHARS = 8000  # generous char cap; token truncation happens in Detector/tokenizer


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


class ClassifyBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class ClassifyResponse(BaseModel):
    is_injection: bool
    label: int
    score: float
    threshold: float
    model_version: str | int | None = None


class ClassifyBatchResponse(BaseModel):
    results: list[ClassifyResponse]


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    predicted_label: int = Field(..., ge=0, le=1)
    correct_label: int = Field(..., ge=0, le=1)
    model_version: str | int | None = None


class FeedbackResponse(BaseModel):
    status: str = "recorded"


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str | int | None = None
    threshold: float | None = None
