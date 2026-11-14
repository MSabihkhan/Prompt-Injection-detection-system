# Ops guide: Airflow, config, Grafana dashboard, Supabase (Fatima's part)

## 1. What this part is

| Piece | What it is | Files |
|---|---|---|
| Config | One place for thresholds and schedules | `ops/config.yaml`, `ops/config.py`, `ops/.env.ops.example` |
| Database side | Tables Grafana and Airflow read | `ops/sql/001_schema.sql`, `ops/db.py` |
| Dashboard | A **Grafana** monitoring dashboard (not a web frontend) | `ops/grafana/` |
| Drift check | KS + PSI on prompt length and score | `ops/monitoring/drift.py` |
| Airflow | Weekly retrain DAG, daily drift DAG | `ops/airflow/` |
| Helpers | Fake data, feedback export/merge, reference builder | `ops/scripts/` |

Flow: API writes `predictions` + `feedback` -> Grafana charts them -> daily `drift_daily` DAG compares recent
`predictions` with the training reference -> on drift it triggers `retrain_weekly` -> that exports feedback,
merges it into the TRAIN split, runs `app/train.py`, then `app/eval.py --promote` (the existing gates decide).

## 2. Setup (Windows, from the repo root)

```bash
git checkout -b fatima/airflow-grafana
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r ops/requirements-ops.txt
copy .env.example .env
copy ops\.env.ops.example ops\.env.ops
```
Install Docker Desktop. Ask Sabih to add you to the DagsHub repo (DVC + MLflow). Put the MLflow values in `.env`.

## 3. Work in this order

### Step 1: Grafana with fake data (no one else needed)
```bash
docker compose -f ops/docker-compose.ops.yml --env-file ops/.env.ops --profile local-db up -d
```
Add to `.env`: `DATABASE_URL=postgresql://pg:pg@localhost:5433/promptguard`, then:
```bash
.venv\Scripts\python.exe -m ops.scripts.seed_fake_data --days 14 --drift-last-days 1
```
Open http://localhost:3000, dashboard **PromptGuard overview**. Edit panels in the UI, then
Dashboard settings -> JSON model -> copy into `ops/grafana/dashboards/promptguard.json` and commit.

### Step 2: Drift check offline
```bash
.venv\Scripts\python.exe -m ops.monitoring.drift --report reports/drift_report.html
```
Seed with `--drift-last-days 1` and it should say `drift_detected: true`; seed without it and it says false.
Tune `ks_alpha`, `psi_threshold`, `min_current_rows` in `ops/config.yaml`.

### Step 3: Airflow
```bash
docker compose -f ops/docker-compose.ops.yml --env-file ops/.env.ops --profile local-db up -d --build
docker logs promptguard-airflow   # find the admin password
```
Open http://localhost:8080. Trigger `drift_daily` by hand first (it needs the reference file from step 2),
then `retrain_weekly` with config `{"force": true}`.

### Step 4: Real Supabase
1. Supabase -> SQL editor -> paste `ops/sql/001_schema.sql`. Then grant the API's role access and turn on Row
   Level Security (the schema file does neither; without the grants the API gets "permission denied"):
   ```sql
   grant select, insert on public.predictions, public.drift_runs, public.feedback to service_role;
   grant usage, select on sequence predictions_id_seq, drift_runs_id_seq, feedback_id_seq to service_role;
   alter table public.predictions enable row level security;
   alter table public.drift_runs enable row level security;
   alter table public.feedback enable row level security;
   ```
2. Supabase -> Connect -> **Session pooler** string (works over IPv4). Put the parts into `ops/.env.ops` (PG_*)
   and the full URL into `.env` as `DATABASE_URL`.
3. Restart without `--profile local-db`.

### Step 5: Real reference + real model
```bash
.venv\Scripts\python.exe -m ops.scripts.build_reference --model champion
```

### Evidently HTML report (optional)
Evidently needs pandas < 3, the project uses pandas 3.x, so Evidently lives in its own venv:
```bash
py -3.12 -m venv .venv-evidently
.venv-evidently\Scripts\python.exe -m pip install "pandas<3" evidently
```
Then add to `.env`: `EVIDENTLY_PY=C:\path\to\project\.venv-evidently\Scripts\python.exe`.
Inside Docker this is already set up (`/home/airflow/ev-venv`). Without it the drift check still works;
`report_written` is just false.

## 4. Team decisions

1. **Who writes `predictions`? Done.** The API (`app/api/routes.py`) inserts one row per classification from
   `/classify` and `/classify/batch`: label, score, threshold, `text_length`, `latency_ms`, model version, no
   prompt text. It runs as a background task, so a DB outage never fails a request. `GET /drift` returns recent
   `drift_runs` rows.
2. **Feedback table. Settled.** The shape in `001_schema.sql` matches what the API used, so `export_feedback.py`
   needs no change. Users no longer submit feedback: the `/feedback` route and the demo page's Correct/Wrong
   buttons were removed. The table stays because retraining reads it. A planned LLM judge will write rows where
   it disagrees with the model.
3. **DVC versioning of retrained data. Still open.** The data now comes from a DVC pipeline (`dvc.yaml`, stages
   download → preprocess → split → validate → train → eval; `data.dvc` is gone). `dvc pull --force` in the DAG
   works the same way once `dvc push` has been run. `train.py` now records the md5 of the `split` stage output
   from `dvc.lock`. `merge_feedback.py` still changes `train.csv` inside the container after the pull, so a
   retrained model's recorded data version won't match its actual data. Options: make feedback merging a DVC
   stage before `train`, or have retraining `dvc push` a new data version. Agree with Sabih.
4. **Redeploy after promotion.** The DAG exports the new champion to `model/`. Shipping it is the CD workflow
   (`.github/workflows/cd.yml`), which now targets Azure Container Apps and is disabled until the Azure
   resources exist.
5. **Score range.** If the champion uses a decision-function score (not 0..1), the dashboard histogram and
   fake data still work, but PSI/KS operate on whatever scale the model produces. That is fine. The current
   champion family (`linearsvc_char`) does produce margins, not probabilities.

## 5. Not tested yet

Docker, Grafana and Airflow were not run when this was written (no Docker in the authoring environment).
The drift maths, feedback merge and CLI were tested; the SQL, compose file, dashboard JSON and DAGs have only
been syntax-checked. Expect to fix small things on first run, and read `docker compose logs <service>`.
