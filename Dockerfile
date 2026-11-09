# PromptGuard API image. Expects a pre-exported model at ./model (see README/CLAUDE.md:
#   python app/inference.py export --model champion --out model/
# before building, since CI has no remote MLflow to resolve "champion" from yet.

FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

FROM base AS deps
COPY requirements.txt .
RUN pip install --no-compile -r requirements.txt

FROM deps AS runtime
COPY app/ app/
COPY model/ model/
ENV PROMPTGUARD_MODEL=/app/model

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
