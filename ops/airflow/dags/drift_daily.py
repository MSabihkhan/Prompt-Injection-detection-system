"""Daily drift check. If prompts have drifted, trigger retrain_weekly early.

run_drift_check (writes a row to drift_runs for Grafana) -> branch -> trigger_retrain | no_drift
"""
import json
import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

REPO = os.environ.get("PG_REPO", "/opt/airflow/repo")
PY = os.environ.get("PG_PY", "/home/airflow/pg-venv/bin/python")
sys.path.insert(0, REPO)
from ops.config import load_config

CFG = load_config()


def _branch(ti, **_) -> str:
    result = json.loads(ti.xcom_pull(task_ids="run_drift_check"))
    print(f"status={result['status']} drift={result['drift_detected']}")
    return "trigger_retrain" if result["drift_detected"] else "no_drift"


with DAG(
    dag_id="drift_daily",
    description="Compare recent prompts with the training data; trigger an early retrain on drift.",
    schedule=CFG["airflow"]["drift_schedule"],
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["promptguard", "drift"],
) as dag:
    run_drift_check = BashOperator(
        task_id="run_drift_check",
        bash_command=f"cd {REPO} && {PY} -m ops.monitoring.drift --write-db "
                     f"--report reports/drift_report.html",
    )
    branch = BranchPythonOperator(task_id="branch", python_callable=_branch)
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="retrain_weekly",
        conf={"force": True, "reason": "drift"},
        wait_for_completion=False,
    )
    no_drift = EmptyOperator(task_id="no_drift")

    run_drift_check >> branch >> [trigger_retrain, no_drift]
