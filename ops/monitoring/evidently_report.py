"""Standalone Evidently drift report. Runs in a SEPARATE virtualenv.

Why separate: the project pins pandas 3.x, but Evidently 0.7.x only works with pandas 2.x
(importing it under pandas 3 fails). drift.py calls this script with the other interpreter
(env var EVIDENTLY_PY) so both can live side by side.

    <evidently-venv>/python ops/monitoring/evidently_report.py \
        --reference ref.csv --current cur.csv --out reports/drift_report.html
"""
import argparse
import sys

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--columns", nargs="+", default=["text_length", "score"])
    args = ap.parse_args(argv)

    try:
        from evidently.legacy.metric_preset import DataDriftPreset  # evidently >= 0.7
        from evidently.legacy.report import Report
    except ImportError:
        from evidently.metric_preset import DataDriftPreset  # evidently < 0.7
        from evidently.report import Report

    ref = pd.read_csv(args.reference)[args.columns]
    cur = pd.read_csv(args.current)[args.columns]
    report = Report(metrics=[DataDriftPreset(columns=args.columns)])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
