# preprocess.py
import re
import unicodedata
from pathlib import Path
import json
import pandas as pd


def clean_text(s: str) -> str:
    """NFKC-normalize and collapse whitespace. Reuse this exact function at inference."""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def _drop_conflicts_and_dups(df: pd.DataFrame, key: str):
    """Drop every row whose key appears with both labels, then drop duplicate keys."""
    conflict = df.groupby(key)["label"].transform("nunique") > 1
    n_conflict = int(conflict.sum())
    df = df[~conflict]

    before = len(df)
    df = df.drop_duplicates(subset=key)
    return df, n_conflict, before - len(df)


def preprocess(
    path: str | Path = "../data/Raw/safeguard_all.csv",
    out_path: str | Path | None = "../data/processed/clean.csv",
):
    df = pd.read_csv(path)
    report = {"rows_raw": len(df)}

    # 1. Validity checks
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    report["after_validity"] = len(df)

    # 2 + 3. Exact duplicates and conflicting labels (raw text)
    df, n_conf, n_dup = _drop_conflicts_and_dups(df, "text")
    report["exact_conflict_rows"] = n_conf
    report["exact_dups"] = n_dup

    # 4. Light normalization (NFKC + whitespace)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"] != ""].copy()  # normalization can occasionally empty a row

    # 5. Near-duplicates and conflicts on a lowercased key (text itself keeps its casing)
    df["key"] = df["text"].str.lower()
    df, n_conf, n_dup = _drop_conflicts_and_dups(df, "key")
    report["near_conflict_rows"] = n_conf
    report["near_dups"] = n_dup
    df = df.drop(columns="key").reset_index(drop=True)

    # 6. Class balance check
    share = df["label"].value_counts(normalize=True)
    report["label_share"] = share.round(3).to_dict()
    report["class_weight"] = "balanced" if share.min() < 0.4 else None
    report["rows_final"] = len(df)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df, report


if __name__ == "__main__":
    _, report = preprocess()
    Path("../data/processed").mkdir(parents=True, exist_ok=True)
    Path("../data/processed/preprocess_report.json").write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        print(f"{k}: {v}")