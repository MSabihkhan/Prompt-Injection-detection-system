"""Pre-training data validation (app/validate.py). No files, no DVC."""
import pandas as pd
from validate import MIN_ROWS, validate


def make(n, start=0):
    return pd.DataFrame({"text": [f"prompt {i}" for i in range(start, start + n)],
                         "label": [i % 2 for i in range(n)]})


def good_splits():
    return {"train": make(MIN_ROWS["train"]),
            "val": make(MIN_ROWS["val"], 10_000),
            "test": make(MIN_ROWS["test"], 20_000)}


def test_clean_splits_pass():
    assert validate(good_splits())["passed"]


def test_leakage_between_splits_fails():
    s = good_splits()
    s["test"].loc[0, "text"] = s["train"].loc[0, "text"].upper()  # same prompt, different case
    r = validate(s)
    assert not r["passed"] and not r["checks"]["no_overlap_train_test"]


def test_missing_class_fails():
    s = good_splits()
    s["val"]["label"] = 0
    assert not validate(s)["checks"]["val_both_classes"]


def test_bad_label_and_empty_text_fail():
    s = good_splits()
    s["train"].loc[0, "label"] = 2
    s["train"].loc[1, "text"] = "  "
    c = validate(s)["checks"]
    assert not c["train_labels_binary"] and not c["train_no_empty_text"]


def test_too_few_rows_fails():
    s = good_splits()
    s["train"] = make(10)
    assert not validate(s)["checks"]["train_min_rows"]


def test_missing_column_fails():
    s = good_splits()
    s["test"] = s["test"].rename(columns={"label": "y"})
    assert not validate(s)["passed"]
