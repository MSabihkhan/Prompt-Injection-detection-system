# PromptGuard

Prompt injection detector: text classifier (benign/injection) + score, served by FastAPI, wrapped in an MLOps pipeline (DVC, MLflow, Docker, CI/CD, Airflow retraining, drift monitoring). Effort goes into deployment, not model quality. Full spec: `Project Proposal.pdf`.

## Stack
Python 3.12 (proposal says 3.11; actual venv/Docker base is 3.12 — numpy==2.5.4 requires it) · scikit-learn (TF-IDF baseline), Hugging Face Transformers/PyTorch (pretrained model, planned) · DVC (remote: DagsHub) · MLflow registry (shared, hosted on DagsHub — see Commands) · FastAPI/Uvicorn/Jinja2 · Docker + Compose · GitHub Actions · Azure (Container Apps, Container Registry, Key Vault) · Airflow · Evidently + Grafana (monitoring/dashboards) · Supabase (DB) · pytest.

## Layout
- `app/` — pipeline scripts: `datadownload.py` → `preprocess.py` → `split.py` → `validate.py` → `train.py` → `eval.py` (stages `download` → `preprocess` → `split` → `validate` → `train` → `eval` in `dvc.yaml`); `inference.py` loads/exports the model.
- `app/api/` — FastAPI service: `main.py` (app + lifespan model load), `routes.py`, `schemas.py`, `auth.py`, `store.py` (Supabase or local-file fallback, for `predictions`/`drift_runs`), `templates/index.html` (demo page).
- `tests/` — pytest, API tests use a stub `Detector` (no model/MLflow needed).
- `data/` — DVC pipeline outputs (`dvc.yaml` + `dvc.lock`, no `data.dvc`), gitignored. Dataset download is pinned to a Hugging Face commit in `datadownload.py`. Never commit data, `mlruns/`, `mlflow.db`, `model/`, `reports/`, `.env`.

## Commands (run from repo root, venv `.venv`)
- Install: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
- MLflow registry is shared on DagsHub, no local fallback: `train.py`/`eval.py`/`inference.py` auto-load `.env` (python-dotenv) for `MLFLOW_TRACKING_URI`/`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` (DagsHub token) and fail fast with a clear error if `MLFLOW_TRACKING_URI` is unset. See `.env.example`. (`inference.py`'s `from_dir` path — used by Docker — needs no registry at all.)
- Pull data: `dvc pull`
- Full pipeline: `dvc repro` (download → … → eval; skips unchanged stages, `dvc repro --force` reruns all). Eval runs without `--promote`. `validate` exits 1 on bad data and stops the pipeline before training. After a data change: `dvc push`, commit `dvc.lock`.
- Data version for a model = `split` stage md5 in `dvc.lock` (`train.dvc_md5`), tagged on MLflow runs as `data_dvc_md5`.
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
- No prompt text stored by default. API key auth on endpoints. Secrets only via env/GitHub Secrets/Azure Key Vault (or Container App secrets).
- Scope: English, direct prompts, binary labels, CPU serving. No multi-class, GPU, or output scanning.

## API (built)
`POST /classify`, `POST /classify/batch`, `GET /drift?limit=N`, `GET /health`, Jinja2 demo page (classify only). API key via `X-API-Key` (off if `PROMPTGUARD_API_KEY` unset). Tables in `ops/sql/001_schema.sql`: classify endpoints log one `predictions` row per text (length/score/latency, never the text) in a background task that can't fail the request; `/drift` reads `drift_runs` (written by the Airflow drift DAG). Backend: Supabase, or JSONL files in `reports/` if unset. Supabase needs explicit `grant select, insert` to `service_role` on each table + its `_id_seq`. Dashboards → Grafana (`ops/grafana/`).

## Docker / CI-CD
- `Dockerfile`: python:3.12-slim, expects a pre-exported `model/` dir (run `python app/inference.py export --model champion --out model/` before `docker build`).
- `docker-compose.yml`: API only for now; add MLflow/Airflow services once those are containerized.
- `.github/workflows/ci.yml`: `test` job (ruff scoped to `app/api`/`tests`, + pytest) runs on every push/PR, no secrets needed. `docker` job (export champion from DagsHub MLflow, build image, run it, smoke-test `/health` + `/classify`) always runs but skips its own steps cleanly if secrets `MLFLOW_TRACKING_URI`/`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` aren't set (all three as repo **secrets**, Settings > Secrets and variables > Actions > Secrets tab).
- `.github/workflows/cd.yml`: Azure. Runs after CI succeeds on `main` (`workflow_run`): export champion, build, push to ACR, `az containerapp update`, smoke-test `/health`. Login via OIDC (federated credential, no client secret). **Disabled** (`if: false` on the job) until the Azure resources and secrets `AZURE_CLIENT_ID`/`AZURE_TENANT_ID`/`AZURE_SUBSCRIPTION_ID`/`ACR_NAME`/`AZURE_RESOURCE_GROUP`/`AZURE_CONTAINERAPP_NAME` exist; enable steps are in the file header. App runtime secrets (Supabase, API key) live on the Container App, not in the workflow.

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
