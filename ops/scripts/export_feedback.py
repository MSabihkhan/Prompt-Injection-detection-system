"""Export recent user feedback to a CSV (columns: text, label).

label = the CORRECT label the user gave. The LAST line printed is the row count, which Airflow
uses to decide whether a retrain is worth running.

Only `text` and `correct_label` are required in the feedback table. The time column is detected
(created_at, timestamp, ...) so a small naming difference does not break the export.

    python -m ops.scripts.export_feedback --out reports/feedback.csv
"""
import argparse

import pandas as pd

from ops.config import ROOT, load_config

TIME_COLUMNS = ("created_at", "timestamp", "inserted_at", "ts", "time")


def pick_time_column(columns):
    """First known time column present in the table, or None (then no date filter is applied)."""
    lower = {c.lower(): c for c in columns}
    for name in TIME_COLUMNS:
        if name in lower:
            return lower[name]
    return None


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/feedback.csv")
    ap.add_argument("--days", type=int, default=cfg["retrain"]["feedback_since_days"])
    args = ap.parse_args(argv)

    from psycopg2 import sql

    from ops.db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select column_name from information_schema.columns "
                        "where table_schema = 'public' and table_name = 'feedback'")
            columns = [r[0] for r in cur.fetchall()]
        missing = {"text", "correct_label"} - set(columns)
        if missing:
            raise SystemExit(f"feedback table is missing column(s): {sorted(missing)}; found {columns}")

        time_col = pick_time_column(columns)
        query = sql.SQL("select text, correct_label as label from feedback")
        params = {}
        if time_col:
            query += sql.SQL(" where {t} >= now() - make_interval(days => %(d)s) order by {t}").format(
                t=sql.Identifier(time_col))
            params = {"d": args.days}
        df = pd.read_sql_query(query.as_string(conn), conn, params=params)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(len(df))  # must stay the last line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
