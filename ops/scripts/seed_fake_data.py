"""Fill the database (and a reference CSV) with FAKE data so you can build the Grafana
dashboard and test the drift check before the real API logs anything.

    python -m ops.scripts.seed_fake_data --days 14                 # normal traffic
    python -m ops.scripts.seed_fake_data --days 14 --drift-last-days 1   # last day looks different
    python -m ops.scripts.seed_fake_data --reference-only          # just write the reference CSV
"""
import argparse
from datetime import UTC, datetime, timedelta

import numpy as np

from ops.config import ROOT, load_config


def make_rows(n, rng, start, end, drifted=False, version="1"):
    """Normal: mostly short benign prompts with low scores, ~25% injections with high scores.
    Drifted: longer prompts and more mid-range scores."""
    ts = [start + (end - start) * float(x) for x in np.sort(rng.random(n))]
    is_inj = rng.random(n) < (0.45 if drifted else 0.25)
    length = rng.lognormal(mean=(5.6 if drifted else 4.4), sigma=0.6, size=n).astype(int) + 5
    score = np.where(is_inj, rng.beta(6, 2, n), rng.beta(1.5, 8 if not drifted else 4, n))
    latency = rng.gamma(3.0, 12.0, n) + 15
    return [
        dict(created_at=t, model_version=version, label=int(s >= 0.5), score=float(s),
             threshold=0.5, text_length=int(L), latency_ms=float(lat))
        for t, s, L, lat in zip(ts, score, length, latency)
    ]


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--per-day", type=int, default=300)
    ap.add_argument("--drift-last-days", type=float, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference-only", action="store_true")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # reference CSV (what the model saw at training time)
    import pandas as pd
    ref = pd.DataFrame(make_rows(3000, rng, datetime(2026, 1, 1), datetime(2026, 1, 2)))[["text_length", "score"]]
    ref_path = ROOT / cfg["drift"]["reference_path"]
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(ref_path, index=False)
    print(f"Wrote fake reference: {ref_path}")
    if args.reference_only:
        return 0

    from psycopg2.extras import execute_values

    from ops.db import get_conn

    now = datetime.now(UTC)
    split = now - timedelta(days=args.drift_last_days)
    start = now - timedelta(days=args.days)
    half = start + (split - start) / 2
    rows = (make_rows(int(args.per_day * (half - start).total_seconds() / 86400), rng, start, half, version="1")
            + make_rows(int(args.per_day * (split - half).total_seconds() / 86400), rng, half, split, version="2"))
    if args.drift_last_days > 0:
        rows += make_rows(int(args.per_day * args.drift_last_days), rng, split, now, drifted=True, version="2")

    cols = ["created_at", "model_version", "label", "score", "threshold", "text_length", "latency_ms"]
    fb = []
    for r in rows[:: max(1, len(rows) // 120)]:  # ~120 feedback rows
        correct = r["label"] if rng.random() < 0.9 else 1 - r["label"]
        fb.append((r["created_at"], f"fake prompt {len(fb)}", r["label"], correct, r["model_version"]))

    with get_conn() as conn, conn.cursor() as cur:
        execute_values(cur, f"insert into predictions ({','.join(cols)}) values %s",
                       [tuple(r[c] for c in cols) for r in rows])
        execute_values(cur, "insert into feedback (created_at, text, predicted_label, correct_label, "
                            "model_version) values %s", fb)
    print(f"Inserted {len(rows)} fake predictions and {len(fb)} fake feedback rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
