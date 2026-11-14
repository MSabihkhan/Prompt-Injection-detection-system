import logging
import time

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)

from .auth import require_api_key
from .schemas import (
    ClassifyBatchRequest,
    ClassifyBatchResponse,
    ClassifyRequest,
    ClassifyResponse,
    DriftResponse,
    HealthResponse,
)
from .store import make_prediction_row

logger = logging.getLogger("promptguard.api")
router = APIRouter()


def _to_response(result: dict, det) -> ClassifyResponse:
    return ClassifyResponse(threshold=det.threshold, model_version=det.meta.get("version"), **result)


def _require_detector(request: Request):
    det = request.app.state.detector
    if det is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded; check /health")
    return det


def _classify(det, texts: list[str], endpoint: str, request: Request, tasks: BackgroundTasks) -> list[dict]:
    t0 = time.perf_counter()
    results = det.predict(texts)
    per_text_ms = (time.perf_counter() - t0) * 1000 / len(texts)
    rows = [make_prediction_row(endpoint, t, r, det.threshold, det.meta.get("version"), per_text_ms)
            for t, r in zip(texts, results)]
    tasks.add_task(_log_predictions, request.app.state.store, rows)
    return results


def _log_predictions(store, rows: list[dict]) -> None:
    # Runs after the response is sent; monitoring must never break or slow classification.
    try:
        store.insert("predictions", rows)
    except Exception:
        logger.exception("Failed to log %d prediction(s)", len(rows))


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    det = request.app.state.detector
    if det is None:
        return HealthResponse(status="model_unavailable", model_name=request.app.state.model_name)
    return HealthResponse(status="ok", model_name=request.app.state.model_name,
                           model_version=det.meta.get("version"), threshold=det.threshold)


@router.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(require_api_key)])
def classify(body: ClassifyRequest, request: Request, tasks: BackgroundTasks):
    det = _require_detector(request)
    result = _classify(det, [body.text], "classify", request, tasks)[0]
    return _to_response(result, det)


@router.post("/classify/batch", response_model=ClassifyBatchResponse, dependencies=[Depends(require_api_key)])
def classify_batch(body: ClassifyBatchRequest, request: Request, tasks: BackgroundTasks):
    det = _require_detector(request)
    results = _classify(det, body.texts, "classify_batch", request, tasks)
    return ClassifyBatchResponse(results=[_to_response(r, det) for r in results])


@router.get("/drift", response_model=DriftResponse, dependencies=[Depends(require_api_key)])
def drift(request: Request, limit: int = Query(10, ge=1, le=100)):
    return DriftResponse(runs=request.app.state.store.recent("drift_runs", limit))
