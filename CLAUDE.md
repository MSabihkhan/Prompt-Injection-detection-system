# PromptGuard

Prompt injection detector: text classifier (benign/injection) + score, served by FastAPI, wrapped in an MLOps pipeline (DVC, MLflow, Docker, CI/CD, Airflow retraining, drift monitoring). Effort goes into deployment, not model quality. Full spec: `Project Proposal.pdf`.

## Stack
Python 3.12 (proposal says 3.11; actual venv/Docker base is 3.12 — numpy==2.5.4 requires it) · scikit-learn (TF-IDF baseline), Hugging Face Transformers/PyTorch (pretrained model, planned) · DVC (remote: DagsHub) · MLflow registry · FastAPI/Uvicorn/Jinja2 · Docker + Compose · GitHub Actions · AWS (ECS Fargate, ECR, S3, Secrets Manager) · Airflow · Evidently + Grafana (monitoring/dashboards) · Supabase (DB) · pytest.

## Layout
- `app/` — pipeline scripts: `datadownload.py` → `preprocess.py` → `split.py` → `train.py` → `eval.py`; `inference.py` loads/exports the model.
- `app/api/` — FastAPI service: `main.py` (app + lifespan model load), `routes.py`, `schemas.py`, `auth.py`, `feedback_store.py` (Supabase or local-file fallback), `templates/index.html` (demo page).
- `tests/` — pytest, API tests use a stub `Detector` (no model/MLflow needed).
- `data/` — DVC-tracked (`data.dvc`), gitignored. Never commit data, `mlruns/`, `mlflow.db`, `model/`, `reports/`, `.env`.

## Commands (run from repo root, venv `.venv`)
- Install: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
- Pull data: `dvc pull`
- Train: `python app/train.py`
- Evaluate and gate: `python app/eval.py [--model N] [--promote]` (exit 1 = gate failed; used by CI)
- Predict: `python app/inference.py predict "text" --model champion`
- Export for Docker: `python app/inference.py export --model champion --out model/`
- Run API: `uvicorn app.api.main:app --reload` (env: `PROMPTGUARD_MODEL`, `PROMPTGUARD_API_KEY`, `SUPABASE_URL`/`SUPABASE_KEY`)
- Test: `pytest`

## Invariants (do not break)
- Use `preprocess.clean_texts` identically at train and inference.
- Test split is scored once. Threshold is picked on out-of-fold train+val scores, never on test.
- Splits use `SPLIT_SEED=42`; keep `split.py` and `train.py` in sync.
- A model is promoted (alias `champion`) only if it passes gates and beats the current champion.
- Each model version must link to its DVC data md5 and MLflow run.
- No prompt text stored by default. API key auth on endpoints. Secrets only via env/GitHub Secrets/Secrets Manager.
- Scope: English, direct prompts, binary labels, CPU serving. No multi-class, GPU, or output scanning.

## API (built)
`POST /classify`, `POST /classify/batch`, `POST /feedback`, `GET /health`, Jinja2 demo page. API key via `X-API-Key` (off if `PROMPTGUARD_API_KEY` unset). Feedback → Supabase (`supabase/schema.sql`) or local-file fallback. Metrics/dashboards → Grafana (not yet built).

## Docker / CI-CD
- `Dockerfile`: python:3.12-slim, expects a pre-exported `model/` dir (run `python app/inference.py export --model champion --out model/` before `docker build`).
- `docker-compose.yml`: API only for now; add MLflow/Airflow services once those are containerized.
- `.github/workflows/ci.yml`: ruff (scoped to `app/api`, `tests`) + pytest, on every push/PR. Runs today, no secrets needed.
- `.github/workflows/cd.yml`: build/push to ECR, deploy to ECS, on push to `main`. **Not active** — needs `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION`/`ECR_REPOSITORY`/`ECS_CLUSTER`/`ECS_SERVICE` secrets and a remote `MLFLOW_TRACKING_URI` secret (MLflow is local-only right now), none of which exist yet.

## Conventions
- Paths anchored with `Path(__file__)`, not cwd.
- Keep changes minimal; extend existing modules before adding new ones. Add pytest tests for new API/pipeline code. Run `pytest` before commit.
- Commit only when asked.

## Team ownership
Sabih: data + preprocessing, DVC push to DagsHub · Munam: API, DB (Supabase), tests, CI/CD · Fatima: Airflow/config, Grafana dashboard. Check before editing another owner's area.

## Notes
- Monitoring is Grafana, not CloudWatch (overrides proposal).
- DB is Supabase (not in original proposal).
