"""FastAPI app: loads the champion detector once at startup, serves classify/feedback/health
plus a minimal demo page.

Run (from repo root):
    uvicorn app.api.main:app --reload
Env (read from .env in the repo root if present, via python-dotenv):
    PROMPTGUARD_MODEL   registry ref or local dir (default: champion)
    PROMPTGUARD_API_KEY required in X-API-Key header if set; unset disables auth (dev only)
    SUPABASE_URL / SUPABASE_KEY  feedback storage backend; falls back to a local file if unset
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ; must run before reading env vars below or in
                # feedback_store/auth (both read at import time, not request time)

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from .feedback_store import get_feedback_store
from .routes import router

logger = logging.getLogger("promptguard.api")

MODEL_NAME = "prompt-injection-detector"
MODEL_REF = os.environ.get("PROMPTGUARD_MODEL") or "champion"  # blank .env value counts as unset
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_name = MODEL_NAME
    app.state.feedback_store = get_feedback_store()
    if os.environ.get("PROMPTGUARD_SKIP_MODEL_LOAD") == "1":
        # Tests set this and inject their own stub detector; avoids a real registry round-trip
        # (DagsHub is the only registry — no local fallback to load instead).
        app.state.detector = None
    else:
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # app/ for `inference`
            from inference import Detector

            app.state.detector = Detector.load(MODEL_REF)
            logger.info("Loaded detector %s (version=%s)", MODEL_REF, app.state.detector.meta.get("version"))
        except Exception:
            # Service still starts so /health reports the problem instead of crashing the container.
            app.state.detector = None
            logger.exception("Failed to load detector '%s'; /classify will be unavailable", MODEL_REF)
    yield


app = FastAPI(title="PromptGuard", lifespan=lifespan)
app.include_router(router)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/")
def demo(request: Request):
    return templates.TemplateResponse(request, "index.html", {"model_name": MODEL_NAME})
