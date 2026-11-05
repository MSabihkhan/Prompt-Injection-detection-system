"""Load the prompt-injection detector and classify text.

Usage (from repo root):
    python app/inference.py predict "Ignore previous instructions..." --model champion
    python app/inference.py predict "hello" --model model/            # local exported dir
    python app/inference.py export --model champion --out model/      # for Docker

`--model` is a registry reference (alias like "champion", a version number, or "latest")
or a path to a directory produced by `export`.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from preprocess import clean_texts

MODEL_NAME = "prompt-injection-detector"
META_FILE = "detector.json"  # written next to MLmodel by `export`


# ---------- Scoring helpers (shared with train.py / eval.py) ----------
def scores(pipe, X) -> np.ndarray:
    """Positive-class score: probability if available, else SVM/ridge margin."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def score_type(pipe) -> str:
    return "probability" if hasattr(pipe, "predict_proba") else "decision_function"


def default_threshold(pipe) -> float:
    """Threshold equivalent to pipe.predict(): 0.5 for probabilities, 0 for margins."""
    return 0.5 if score_type(pipe) == "probability" else 0.0


# ---------- Registry resolution ----------
def resolve_version(ref: str, client: MlflowClient | None = None):
    """Turn 'latest' / '3' / 'champion' into a ModelVersion."""
    client = client or MlflowClient()
    if ref == "latest":
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise LookupError(f"No versions registered for {MODEL_NAME}")
        return max(versions, key=lambda v: int(v.version))
    if ref.isdigit():
        return client.get_model_version(MODEL_NAME, ref)
    try:
        return client.get_model_version_by_alias(MODEL_NAME, ref)
    except MlflowException:
        # Fallback for servers without alias support: eval.py also sets tag alias.<ref>=true
        tagged = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
                  if v.tags.get(f"alias.{ref}") == "true"]
        if not tagged:
            raise LookupError(
                f"No version of {MODEL_NAME} has alias '{ref}'. "
                f"Run `python app/eval.py --promote` first, or pass a version number."
            ) from None
        return max(tagged, key=lambda v: int(v.version))


# ---------- Detector ----------
class Detector:
    def __init__(self, pipe, threshold: float | None = None, meta: dict | None = None):
        self.pipe = pipe
        # Models trained before the `clean` step was added need cleaning applied here.
        self.has_clean_step = "clean" in getattr(pipe, "named_steps", {})
        self.score_type = score_type(pipe)
        self.threshold = float(threshold) if threshold is not None else default_threshold(pipe)
        self.meta = meta or {}

    @classmethod
    def from_registry(cls, ref: str = "champion"):
        v = resolve_version(ref)
        pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{v.version}")
        meta = {"model_name": MODEL_NAME, "version": v.version, "run_id": v.run_id, **v.tags}
        return cls(pipe, v.tags.get("threshold"), meta)

    @classmethod
    def from_dir(cls, path: str | Path):
        path = Path(path)
        pipe = mlflow.sklearn.load_model(str(path))
        meta_path = path / META_FILE
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(pipe, meta.get("threshold"), meta)

    @classmethod
    def load(cls, model: str):
        return cls.from_dir(model) if Path(model).is_dir() else cls.from_registry(model)

    def _prep(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return list(texts) if self.has_clean_step else clean_texts(texts)

    def score(self, texts) -> np.ndarray:
        return scores(self.pipe, self._prep(texts))

    def predict(self, texts) -> list[dict]:
        return [
            {"is_injection": bool(s >= self.threshold), "label": int(s >= self.threshold), "score": float(s)}
            for s in self.score(texts)
        ]

    def info(self) -> dict:
        return {"threshold": self.threshold, "score_type": self.score_type,
                "has_clean_step": self.has_clean_step, **self.meta}


# ---------- Export (registry -> local dir, used when building the Docker image) ----------
def export(ref: str, out: str | Path, force: bool = False) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty; pass --force to overwrite")
        shutil.rmtree(out)
    v = resolve_version(ref)
    mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{v.version}", dst_path=str(out))
    pipe = mlflow.sklearn.load_model(str(out))  # sanity check: it loads
    meta = {
        "model_name": MODEL_NAME, "version": v.version, "run_id": v.run_id,
        "threshold": float(v.tags.get("threshold", default_threshold(pipe))),
        "score_type": score_type(pipe),
        "threshold_source": "eval" if "threshold" in v.tags else "default",
        **{k: val for k, val in v.tags.items() if k != "threshold"},
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2))
    print(f"Exported {MODEL_NAME} v{v.version} -> {out} (threshold={meta['threshold']:.4f})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="classify one or more texts")
    p.add_argument("texts", nargs="+")
    p.add_argument("--model", default="champion")

    e = sub.add_parser("export", help="download a registered version to a local dir")
    e.add_argument("--model", default="champion")
    e.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "model"))
    e.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "export":
        export(args.model, args.out, args.force)
        return 0

    det = Detector.load(args.model)
    print(json.dumps({"model": det.info(), "predictions": [
        {"text": t, **r} for t, r in zip(args.texts, det.predict(args.texts))
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
