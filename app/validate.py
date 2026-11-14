"""Data checks that must pass before training (DVC stage `validate`).

Fails (exit 1) on: missing columns, null/empty text, labels outside {0, 1}, a split missing a
class, too few rows, or the same prompt appearing in more than one split (leakage).
Writes data/processed/validation.json either way.
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
SPLITS = ("train", "val", "test")
MIN_ROWS = {"train": 1000, "val": 200, "test": 200}


def validate(splits: dict[str, pd.DataFrame]) -> dict:
    checks = {}
    for name, df in splits.items():
        has_cols = {"text", "label"} <= set(df.columns)
        checks[f"{name}_has_columns"] = has_cols
        if not has_cols:
            continue
        text = df["text"]
        checks[f"{name}_no_empty_text"] = bool(text.notna().all() and (text.astype(str).str.strip() != "").all())
        checks[f"{name}_labels_binary"] = bool(df["label"].isin([0, 1]).all())
        checks[f"{name}_both_classes"] = set(df["label"]) == {0, 1}
        checks[f"{name}_min_rows"] = len(df) >= MIN_ROWS[name]

    keys = {n: set(df["text"].astype(str).str.lower()) for n, df in splits.items() if "text" in df}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in keys and b in keys:
            checks[f"no_overlap_{a}_{b}"] = not (keys[a] & keys[b])

    return {"passed": all(checks.values()), "rows": {n: len(df) for n, df in splits.items()},
            "checks": checks}


def main() -> int:
    splits = {s: pd.read_csv(PROCESSED / "splits" / f"{s}.csv") for s in SPLITS}
    result = validate(splits)
    (PROCESSED / "validation.json").write_text(json.dumps(result, indent=2))
    for name, ok in result["checks"].items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print("Validation", "PASSED" if result["passed"] else "FAILED")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
