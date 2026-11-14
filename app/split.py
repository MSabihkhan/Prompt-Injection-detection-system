from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def split_and_save(
    path: str | Path = PROCESSED / "clean.csv",
    out_dir: str | Path = PROCESSED / "splits",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    drop_duplicates: bool = True,
):
    df = pd.read_csv(path)

    if drop_duplicates:
        df = df.drop_duplicates(subset="text").reset_index(drop=True)

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val["label"],
        random_state=seed,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        part.reset_index(drop=True).to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows -> {out_dir / f'{name}.csv'}")

    return splits


if __name__ == "__main__":
    split_and_save()