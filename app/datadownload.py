import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import pandas as pd
from datasets import load_dataset

ds = load_dataset("xTRam1/safe-guard-prompt-injection", token=False)

df = pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)
df.to_csv("./data/Raw/safeguard_all.csv", index=False)

print(df.shape)
print(df["label"].value_counts())