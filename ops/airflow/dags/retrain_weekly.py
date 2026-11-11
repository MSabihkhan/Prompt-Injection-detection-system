"""Weekly retraining. Also triggered early by drift_daily (conf={"force": true}).

export_feedback -> enough_feedback? -> dvc_pull -> merge_feedback -> train -> evaluate_promote
                                                                        -> export_model (only if promoted)

The project's own gates decide promotion (app/eval.py): a new version becomes 'champion' only if it
passes the recall/precision gates AND beats the current champion.
"""
import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

REPO = os.environ.get("PG_REPO", "/opt/airflow/repo")
PY = os.environ.get("PG_PY", "/home/airflow/pg-venv/bin/python")
VBIN = os.path.dirname(PY)
sys.path.insert(0, REPO)
from ops.config import load_config

CFG = load_config()
CD = f"cd {REPO} && "
PROMOTE_FLAG = "--promote" if CFG["retrain"].get("promote") else ""


def _enough_feedback(ti, dag_run, **_) -> bool:
    n = int(ti.xcom_pull(task_ids="export_feedback") or 0)
    force = bool((dag_run.conf or {}).get("force"))
    need = CFG["retrain"]["min_feedback_rows"]
    print(f"feedback rows={n}, required={need}, forced={force}")
    return force or n >= need


with DAG(
    dag_id="retrain_weekly",
    description="Retrain on user feedback; promote only if the new model beats the champion.",
    schedule=CFG["retrain"]["schedule"],
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["promptguard", "retrain"],
) as dag:
    export_feedback = BashOperator(
        task_id="export_feedback",
        bash_command=CD + f"{PY} -m ops.scripts.export_feedback --out reports/feedback.csv",
    )
    enough_feedback = ShortCircuitOperator(task_id="enough_feedback", python_callable=_enough_feedback)

    # DagsHub credentials for DVC are written to .dvc/config.local (gitignored), never to git.
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=CD
        + f"{VBIN}/dvc remote modify --local origin auth basic && "
        + f"{VBIN}/dvc remote modify --local origin user \"$DAGSHUB_USER\" && "
        + f"{VBIN}/dvc remote modify --local origin password \"$DAGSHUB_TOKEN\" && "
        + f"{VBIN}/dvc pull --force",  # --force: data/ must match the committed version; the merge step re-applies feedback each run
    )
    merge_feedback = BashOperator(
        task_id="merge_feedback",
        bash_command=CD + f"{PY} -m ops.scripts.merge_feedback --feedback reports/feedback.csv",
    )
    train = BashOperator(task_id="train", bash_command=CD + f"{PY} app/train.py")

    # eval.py exits 1 when the gates fail. That is a normal outcome ("keep the old model"), not a
    # broken DAG, so exit code 1 is turned into a message; any other failure still fails the task.
    evaluate_promote = BashOperator(
        task_id="evaluate_promote",
        bash_command=CD + f"""
        out=$({PY} app/eval.py --model latest {PROMOTE_FLAG} 2>&1); rc=$?
        echo "$out"
        if [ $rc -ne 0 ] && [ $rc -ne 1 ]; then exit $rc; fi
        if echo "$out" | grep -q "^Promoted"; then echo PROMOTED; else echo NOT_PROMOTED; fi
        """,
    )
    promoted = ShortCircuitOperator(
        task_id="promoted",
        python_callable=lambda ti, **_: ti.xcom_pull(task_ids="evaluate_promote") == "PROMOTED",
    )
    export_model = BashOperator(
        task_id="export_model",
        bash_command=CD + f"{PY} app/inference.py export --model champion --out model/ --force "
        f"&& {PY} -m ops.scripts.build_reference --model champion",
    )

    (export_feedback >> enough_feedback >> dvc_pull >> merge_feedback >> train
     >> evaluate_promote >> promoted >> export_model)
