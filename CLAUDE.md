# PromptGuard

Prompt injection detector: text classifier (benign/injection) + score, served by FastAPI, wrapped in an MLOps pipeline (DVC, MLflow, Docker, CI/CD, Airflow retraining, drift monitoring). Effort goes into deployment, not model quality. Full spec: `Project Proposal.pdf`.

## Stack
Python 3.11 · scikit-learn (TF-IDF baseline), Hugging Face Transformers/PyTorch (pretrained model, planned) · DVC (remote: DagsHub) · MLflow registry · FastAPI/Uvicorn/Jinja2 · Docker + Compose · GitHub Actions · AWS (ECS Fargate, ECR, S3, Secrets Manager) · Airflow · Evidently + Grafana (monitoring/dashboards) · Supabase (DB) · pytest.

## Layout
- `app/` — pipeline scripts: `datadownload.py` → `preprocess.py` → `split.py` → `train.py` → `eval.py`; `inference.py` loads/exports the model.
- `data/` — DVC-tracked (`data.dvc`), gitignored. Never commit data, `mlruns/`, `model/`, `reports/`, `.env`.

## Commands (run from repo root, venv `.venv`)
- Install: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
- Pull data: `dvc pull`
- Train: `python app/train.py`
- Evaluate and gate: `python app/eval.py [--model N] [--promote]` (exit 1 = gate failed; used by CI)
- Predict: `python app/inference.py predict "text" --model champion`
- Export for Docker: `python app/inference.py export --model champion --out model/`

## Invariants (do not break)
- Use `preprocess.clean_texts` identically at train and inference.
- Test split is scored once. Threshold is picked on out-of-fold train+val scores, never on test.
- Splits use `SPLIT_SEED=42`; keep `split.py` and `train.py` in sync.
- A model is promoted (alias `champion`) only if it passes gates and beats the current champion.
- Each model version must link to its DVC data md5 and MLflow run.
- No prompt text stored by default. API key auth on endpoints. Secrets only via env/GitHub Secrets/Secrets Manager.
- Scope: English, direct prompts, binary labels, CPU serving. No multi-class, GPU, or output scanning.

## API (to build)
`POST /classify`, `POST /classify/batch`, `POST /feedback`, `GET /health`, Jinja2 demo page. Configurable block threshold. Inputs truncated/split to 512 tokens. Feedback/audit data → Supabase. Metrics/dashboards → Grafana.

## Conventions
- Paths anchored with `Path(__file__)`, not cwd.
- Keep changes minimal; extend existing modules before adding new ones. Add pytest tests for new API/pipeline code. Run `pytest` before commit.
- Commit only when asked.

## Team ownership
Sabih: data + preprocessing, DVC push to DagsHub · Munam: API, DB (Supabase), tests, CI/CD · Fatima: Airflow/config, Grafana dashboard. Check before editing another owner's area.

## Notes
- DVC data not yet pushed by Sabih — `dvc pull` will fail until then.
- Monitoring is Grafana, not CloudWatch (overrides proposal).
- DB is Supabase (not in original proposal).
