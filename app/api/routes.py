from fastapi import APIRouter, Depends, HTTPException, Request, status

from .auth import require_api_key
from .feedback_store import make_feedback_row
from .schemas import (ClassifyBatchRequest, ClassifyBatchResponse, ClassifyRequest,
                      ClassifyResponse, FeedbackRequest, FeedbackResponse, HealthResponse)

router = APIRouter()


def _to_response(result: dict, det) -> ClassifyResponse:
    return ClassifyResponse(threshold=det.threshold, model_version=det.meta.get("version"), **result)


def _require_detector(request: Request):
    det = request.app.state.detector
    if det is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded; check /health")
    return det


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    det = request.app.state.detector
    if det is None:
        return HealthResponse(status="model_unavailable", model_name=request.app.state.model_name)
    return HealthResponse(status="ok", model_name=request.app.state.model_name,
                           model_version=det.meta.get("version"), threshold=det.threshold)


@router.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(require_api_key)])
def classify(body: ClassifyRequest, request: Request):
    det = _require_detector(request)
    result = det.predict(body.text)[0]
    return _to_response(result, det)


@router.post("/classify/batch", response_model=ClassifyBatchResponse, dependencies=[Depends(require_api_key)])
def classify_batch(body: ClassifyBatchRequest, request: Request):
    det = _require_detector(request)
    results = det.predict(body.texts)
    return ClassifyBatchResponse(results=[_to_response(r, det) for r in results])


@router.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_api_key)])
def feedback(body: FeedbackRequest, request: Request):
    row = make_feedback_row(body.text, body.predicted_label, body.correct_label, body.model_version)
    request.app.state.feedback_store.record(row)
    return FeedbackResponse()
