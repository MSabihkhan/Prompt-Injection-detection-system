"""Add user feedback to the TRAIN split only (val/test are never touched).

Rules (they protect the project's invariants):
  * texts are cleaned with preprocess.clean_text, the same function used at train/inference time
  * a text that already appears in val or test is dropped (no leakage into evaluation)
  * a text given both labels in feedback is dropped (conflicting votes)
  * for a text already in train, the feedback label wins
  * the original train.csv is backed up once as train.base.csv

    python -m ops.scripts.merge_feedback --feedback data/processed/feedback.csv
"""
import argparse
import sys

import pandas as pd

from ops.config import ROOT

sys.path.insert(0, str(ROOT / "app"))
from preprocess import clean_text

SPLITS = ROOT / "data" / "processed" / "splits"


def merge(train: pd.DataFrame, holdout_texts: set, feedback: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fb = feedback[["text", "label"]].copy()
    fb["text"] = fb["text"].astype(str).map(clean_text)
    fb = fb[fb["text"] != ""]
    n_in = len(fb)

    conflict = fb.groupby("text")["label"].transform("nunique") > 1
    n_conflict = int(conflict.sum())
    fb = fb[~conflict].drop_duplicates("text", keep="last")
    n_leak = int(fb["text"].isin(holdout_texts).sum())
    fb = fb[~fb["text"].isin(holdout_texts)]

    base = train[~train["text"].isin(set(fb["text"]))]  # feedback label overrides existing rows
    extra_cols = [c for c in train.columns if c not in ("text", "label")]
    fb = fb.assign(**{c: None for c in extra_cols})[train.columns]
    merged = pd.concat([base, fb], ignore_index=True)
    stats = {"feedback_in": n_in, "dropped_conflict": n_conflict, "dropped_leak": n_leak,
             "added_or_updated": len(fb), "train_before": len(train), "train_after": len(merged)}
    return merged, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feedback", default="data/processed/feedback.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    fb_path = ROOT / args.feedback
    feedback = pd.read_csv(fb_path) if fb_path.exists() else pd.DataFrame(columns=["text", "label"])
    if feedback.empty:
        print("No feedback rows; train split unchanged.")
        return 0

    train = pd.read_csv(SPLITS / "train.csv")
    holdout = set()
    for name in ("val", "test"):
        holdout |= set(pd.read_csv(SPLITS / f"{name}.csv")["text"].astype(str))

    merged, stats = merge(train, holdout, feedback)
    print(stats)
    if args.dry_run:
        return 0
    backup = SPLITS / "train.base.csv"
    if not backup.exists():
        train.to_csv(backup, index=False)
    merged.to_csv(SPLITS / "train.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
