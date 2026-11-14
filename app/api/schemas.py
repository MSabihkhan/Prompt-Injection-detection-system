"""Request/response models for the classification API."""
from datetime import datetime

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


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str | int | None = None
    threshold: float | None = None


class DriftRun(BaseModel):
    """One row of drift_runs, written by the daily drift DAG (ops/monitoring/drift.py)."""
    created_at: datetime
    status: str
    drift_detected: bool
    n_reference: int | None = None
    n_current: int | None = None
    length_ks_p: float | None = None
    length_psi: float | None = None
    score_ks_p: float | None = None
    score_psi: float | None = None
    details: dict | None = None


class DriftResponse(BaseModel):
    runs: list[DriftRun]  # newest first
