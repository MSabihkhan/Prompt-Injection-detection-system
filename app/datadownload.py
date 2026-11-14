"""Download the raw dataset from Hugging Face (DVC stage `download`)."""
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import pandas as pd
from datasets import load_dataset

DATASET = "xTRam1/safe-guard-prompt-injection"
REVISION = "a3a877d608f37b7d20d9945671902df895ecdb46"  # pinned commit, so re-runs get identical data
OUT = Path(__file__).resolve().parent.parent / "data" / "Raw" / "safeguard_all.csv"


def main():
    ds = load_dataset(DATASET, revision=REVISION, token=False)
    df = pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.shape)
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
