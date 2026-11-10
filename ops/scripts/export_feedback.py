"""Export recent user feedback to a CSV (columns: text, label, model_version).

label = the CORRECT label the user gave. The LAST line printed is the row count, which Airflow
uses to decide whether a retrain is worth running.

    python -m ops.scripts.export_feedback --out data/processed/feedback.csv
"""
import argparse

import pandas as pd

from ops.config import ROOT, load_config


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/processed/feedback.csv")
    ap.add_argument("--days", type=int, default=cfg["retrain"]["feedback_since_days"])
    args = ap.parse_args(argv)

    from ops.db import get_conn

    with get_conn() as conn:
        df = pd.read_sql_query(
            "select text, correct_label as label, model_version from feedback "
            "where created_at >= now() - make_interval(days => %(d)s) order by created_at",
            conn, params={"d": args.days})
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(len(df))  # must stay the last line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
