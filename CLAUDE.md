# PromptGuard

Prompt injection detector: text classifier (benign/injection) + score, served by FastAPI, wrapped in an MLOps pipeline (DVC, MLflow, Docker, CI/CD, Airflow retraining, drift monitoring). Effort goes into deployment, not model quality. Full spec: `Project Proposal.pdf`.

## Stack
Python 3.12 (proposal says 3.11; actual venv/Docker base is 3.12 — numpy==2.5.4 requires it) · scikit-learn (TF-IDF baseline), Hugging Face Transformers/PyTorch (pretrained model, planned) · DVC (remote: DagsHub) · MLflow registry (shared, hosted on DagsHub — see Commands) · FastAPI/Uvicorn/Jinja2 · Docker + Compose · GitHub Actions · AWS (ECS Fargate, ECR, S3, Secrets Manager) · Airflow · Evidently + Grafana (monitoring/dashboards) · Supabase (DB) · pytest.

## Layout
- `app/` — pipeline scripts: `datadownload.py` → `preprocess.py` → `split.py` → `train.py` → `eval.py`; `inference.py` loads/exports the model.
- `app/api/` — FastAPI service: `main.py` (app + lifespan model load), `routes.py`, `schemas.py`, `auth.py`, `feedback_store.py` (Supabase or local-file fallback), `templates/index.html` (demo page).
- `tests/` — pytest, API tests use a stub `Detector` (no model/MLflow needed).
- `data/` — DVC-tracked (`data.dvc`), gitignored. Never commit data, `mlruns/`, `mlflow.db`, `model/`, `reports/`, `.env`.

## Commands (run from repo root, venv `.venv`)
- Install: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
- MLflow registry is shared on DagsHub, no local fallback: `train.py`/`eval.py`/`inference.py` auto-load `.env` (python-dotenv) for `MLFLOW_TRACKING_URI`/`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` (DagsHub token) and fail fast with a clear error if `MLFLOW_TRACKING_URI` is unset. See `.env.example`. (`inference.py`'s `from_dir` path — used by Docker — needs no registry at all.)
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
- `.github/workflows/ci.yml`: `test` job (ruff scoped to `app/api`/`tests`, + pytest) runs on every push/PR, no secrets needed. `docker` job (export champion from DagsHub MLflow, build image, run it, smoke-test `/health` + `/classify`) always runs but skips its own steps cleanly if secrets `MLFLOW_TRACKING_URI`/`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` aren't set (all three as repo **secrets**, Settings > Secrets and variables > Actions > Secrets tab).
- `.github/workflows/cd.yml`: build/push to ECR, deploy to ECS, on push to `main`. **Disabled** (`if: false` on the job) until `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION`/`ECR_REPOSITORY`/`ECS_CLUSTER`/`ECS_SERVICE` secrets and a remote `MLFLOW_TRACKING_URI` secret exist (MLflow is local-only right now). Flip `if: false` to enable once that infra is ready.

## Conventions
- Paths anchored with `Path(__file__)`, not cwd.
- Keep changes minimal; extend existing modules before adding new ones. Add pytest tests for new API/pipeline code. Run `pytest` before commit.
- Commit only when asked.
- No AI attribution/co-author lines in commit messages or PR descriptions (no `Co-Authored-By: Claude...`, no "Generated with Claude Code" footer). Human authorship only.

## Team ownership
Sabih: data + preprocessing, DVC push to DagsHub · Munam: API, DB (Supabase), tests, CI/CD · Fatima: Airflow/config, Grafana dashboard. Check before editing another owner's area.

## Notes
- Monitoring is Grafana, not CloudWatch (overrides proposal).
- DB is Supabase (not in original proposal).
