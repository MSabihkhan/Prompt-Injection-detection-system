"""Daily drift check: compare recent prompts with the training reference.

Features: prompt length (characters) and model score. A feature counts as drifted only when
BOTH the KS test is significant (p < ks_alpha) AND PSI >= psi_threshold. KS alone flags tiny,
harmless shifts once there is a lot of traffic; PSI alone is noisy on small samples.

Usage (repo root):
    python -m ops.monitoring.drift                       # read predictions from DATABASE_URL
    python -m ops.monitoring.drift --current-csv x.csv   # offline test, columns text_length,score
    python -m ops.monitoring.drift --write-db --report reports/drift.html

The LAST line printed is a JSON object, so Airflow can read it as the task result.
Exit code is always 0 unless something crashes; drift is reported in the JSON, not the exit code.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ops.config import ROOT, load_config


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index with quantile bins taken from the reference."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # reference is (almost) constant
        return 0.0 if np.allclose(cur, ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-4
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(reference, current, ks_alpha: float, psi_threshold: float) -> dict:
    ks = ks_2samp(reference, current)
    p = psi(reference, current)
    return {
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "psi": p,
        "ref_mean": float(np.mean(reference)),
        "cur_mean": float(np.mean(current)),
        "drifted": bool(ks.pvalue < ks_alpha and p >= psi_threshold),
    }


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame, cfg: dict) -> dict:
    d = cfg["drift"]
    result = {"n_reference": len(ref), "n_current": len(cur), "features": {}}
    if len(cur) < d["min_current_rows"]:
        result.update(status="insufficient_data", drift_detected=False)
        return result
    for feat in d["features"]:
        result["features"][feat] = feature_drift(
            ref[feat].dropna(), cur[feat].dropna(), d["ks_alpha"], d["psi_threshold"])
    result["status"] = "ok"
    result["drift_detected"] = any(f["drifted"] for f in result["features"].values())
    return result


def _evidently_in_subprocess(ref: pd.DataFrame, cur: pd.DataFrame, out: Path, features) -> bool:
    """Evidently 0.7 cannot be imported under the project's pandas 3, so if EVIDENTLY_PY points to
    a Python from a separate venv (pandas 2.x + evidently), run the report there."""
    import os
    import subprocess
    import tempfile

    py = os.environ.get("EVIDENTLY_PY")
    if not py:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ref_p, cur_p = Path(tmp) / "ref.csv", Path(tmp) / "cur.csv"
        ref.to_csv(ref_p, index=False)
        cur.to_csv(cur_p, index=False)
        proc = subprocess.run(
            [py, str(Path(__file__).with_name("evidently_report.py")), "--reference", str(ref_p),
             "--current", str(cur_p), "--out", str(out), "--columns", *features],
            capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Evidently report skipped: {proc.stderr.strip()[-300:]}", file=sys.stderr)
    return proc.returncode == 0


def write_evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, out: Path, features) -> bool:
    """Optional HTML report. Never fails the drift check. Tries Evidently in this interpreter,
    then in the separate venv named by EVIDENTLY_PY."""
    try:
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.report import Report
    except ImportError:
        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
        except ImportError:
            return _evidently_in_subprocess(ref, cur, out, features)
    try:
        report = Report(metrics=[DataDriftPreset(columns=list(features))])
        report.run(reference_data=ref[list(features)], current_data=cur[list(features)])
        out.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out))
        return True
    except Exception as e:
        print(f"Evidently report skipped: {e}", file=sys.stderr)
        return False


def load_current_from_db(days: int) -> pd.DataFrame:
    from ops.db import get_conn

    with get_conn() as conn:
        return pd.read_sql_query(
            "select text_length, score from predictions "
            "where created_at >= now() - make_interval(days => %(d)s)",
            conn, params={"d": days})


def write_run_to_db(result: dict) -> None:
    from psycopg2.extras import Json

    from ops.db import get_conn

    f = result.get("features", {})
    length, score = f.get("text_length", {}), f.get("score", {})
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "insert into drift_runs (status, drift_detected, n_reference, n_current, "
            "length_ks_p, length_psi, score_ks_p, score_psi, details) "
            "values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (result["status"], result["drift_detected"], result["n_reference"], result["n_current"],
             length.get("ks_p"), length.get("psi"), score.get("ks_p"), score.get("psi"), Json(result)))


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=str(ROOT / cfg["drift"]["reference_path"]))
    ap.add_argument("--current-csv", help="use this CSV instead of reading the predictions table")
    ap.add_argument("--days", type=int, default=cfg["drift"]["window_days"])
    ap.add_argument("--write-db", action="store_true", help="insert the result into drift_runs")
    ap.add_argument("--report", help="write an Evidently HTML report to this path (optional)")
    args = ap.parse_args(argv)

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current_csv) if args.current_csv else load_current_from_db(args.days)

    result = compute_drift(ref, cur, cfg)
    if args.report and result["status"] == "ok":
        result["report_written"] = write_evidently_report(
            ref, cur, Path(args.report), cfg["drift"]["features"])
    if args.write_db:
        write_run_to_db(result)

    print(json.dumps(result))  # must stay the last line
    return 0


if __name__ == "__main__":
    sys.exit(main())
