"""Build the drift reference (text_length, score) from the TRAIN split, scored by a model.

Run after each promotion so the reference matches the model in production:
    python -m ops.scripts.build_reference --model champion
Needs MLFLOW_TRACKING_URI etc. in .env (it loads the model). Use --no-model for lengths only.
"""
import argparse
import sys

import pandas as pd

from ops.config import ROOT, load_config

sys.path.insert(0, str(ROOT / "app"))


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="champion")
    ap.add_argument("--out", default=cfg["drift"]["reference_path"])
    ap.add_argument("--no-model", action="store_true", help="lengths only (score column left empty)")
    args = ap.parse_args(argv)

    train = pd.read_csv(ROOT / "data" / "processed" / "splits" / "train.csv")
    ref = pd.DataFrame({"text_length": train["text"].astype(str).str.len()})
    if args.no_model:
        ref["score"] = float("nan")
    else:
        from inference import Detector

        ref["score"] = Detector.load(args.model).score(train["text"].astype(str).tolist())
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(out, index=False)
    print(f"Wrote {len(ref)} reference rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
