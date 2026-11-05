"""Train prompt-injection classifiers: sweep on val, refit best on train+val, score test once."""
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ---------- Config (paths anchored to the repo, so cwd doesn't matter) ----------
ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "processed" / "splits"
REPORT = ROOT / "data" / "processed" / "preprocess_report.json"
DVC_FILE = ROOT / "data.dvc"

EXPERIMENT = "prompt-injection-tfidf"
MODEL_NAME = "prompt-injection-detector"
SPLIT_SEED = 42  # must match split.py
SEED = 42

WORD = dict(analyzer="word", ngram_range=(1, 2), max_features=50_000)
CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), max_features=100_000)
TREE_WORD = dict(WORD, max_features=20_000)


def load_class_weight() -> tuple[dict, str | None]:
    if not REPORT.exists():
        print(f"WARNING: {REPORT} not found; training without class weights.")
        return {}, None
    report = json.loads(REPORT.read_text())
    return report, report.get("class_weight")  # "balanced" or None


REPORT_DATA, CW = load_class_weight()


def cw(model_cls, **kw):
    """Add class_weight only for models that support it."""
    return model_cls(class_weight=CW, **kw) if CW else model_cls(**kw)


# (run name, tfidf config, model factory)
EXPERIMENTS = [
    ("logreg_word",       WORD,      lambda: cw(LogisticRegression, max_iter=1000)),
    ("logreg_char",       CHAR,      lambda: cw(LogisticRegression, max_iter=1000)),
    ("linearsvc_word",    WORD,      lambda: cw(LinearSVC, C=1.0)),
    ("linearsvc_char",    CHAR,      lambda: cw(LinearSVC, C=1.0)),
    ("sgd_modhuber_word", WORD,      lambda: cw(SGDClassifier, loss="modified_huber", random_state=SEED)),
    ("ridge_word",        WORD,      lambda: cw(RidgeClassifier)),
    ("multinb_word",      WORD,      lambda: MultinomialNB()),
    ("complementnb_word", WORD,      lambda: ComplementNB()),
    ("randomforest_word", TREE_WORD, lambda: cw(RandomForestClassifier, n_estimators=300, n_jobs=-1, random_state=SEED)),
    ("extratrees_word",   TREE_WORD, lambda: cw(ExtraTreesClassifier, n_estimators=300, n_jobs=-1, random_state=SEED)),
]


def dvc_md5() -> str:
    m = re.search(r"md5:\s*([0-9a-f]+)", DVC_FILE.read_text()) if DVC_FILE.exists() else None
    return m.group(1) if m else "unknown"


def build_pipe(feat, model) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, lowercase=True, **feat)),
        ("clf", model),
    ])


def get_scores(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def evaluate(pipe, df, prefix):
    preds = pipe.predict(df["text"])
    y = df["label"]
    return preds, {
        f"{prefix}_accuracy": accuracy_score(y, preds),
        f"{prefix}_f1": f1_score(y, preds),
        f"{prefix}_precision": precision_score(y, preds),
        f"{prefix}_recall": recall_score(y, preds),
        f"{prefix}_roc_auc": roc_auc_score(y, get_scores(pipe, df["text"])),
    }


def log_cm(df, preds, name):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(df["label"], preds, ax=ax)
    mlflow.log_figure(fig, name)
    plt.close(fig)


def log_params(model, feat, **extra):
    mlflow.log_params({
        "model": type(model).__name__,
        **{f"tfidf__{k}": str(v) for k, v in feat.items()},
        **{f"clf__{k}": str(v) for k, v in model.get_params().items()},
        **extra,
    })


def sweep(train, val):
    """One MLflow run per experiment, fit on train, scored on val. Returns a leaderboard."""
    results = []
    for name, feat, make_model in EXPERIMENTS:
        model = make_model()
        pipe = build_pipe(feat, model)

        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"data_dvc_md5": dvc_md5(), "split_seed": str(SPLIT_SEED), "stage": "sweep"})
            mlflow.log_dict(REPORT_DATA, "preprocess_report.json")
            log_params(model, feat, n_train=len(train), n_val=len(val))

            t0 = time.perf_counter()
            pipe.fit(train["text"], train["label"])
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            pipe.predict(val["text"])
            predict_ms = (time.perf_counter() - t0) / len(val) * 1000

            preds, metrics = evaluate(pipe, val, "val")
            metrics.update(fit_seconds=fit_s, predict_ms_per_sample=predict_ms)
            mlflow.log_metrics(metrics)
            log_cm(val, preds, "confusion_matrix_val.png")

            results.append({"run": name, "run_id": run.info.run_id, **metrics})
            print(f"{name:20s} val_f1={metrics['val_f1']:.4f}  fit={fit_s:.1f}s")

    return pd.DataFrame(results).sort_values("val_f1", ascending=False).reset_index(drop=True)


def train_final(best, leaderboard, train, val, test):
    """Refit the winner on train+val, score the test set once, register the model."""
    feat, make_model = next((f, m) for n, f, m in EXPERIMENTS if n == best["run"])
    model = make_model()
    final = build_pipe(feat, model)
    train_val = pd.concat([train, val], ignore_index=True)
    final.fit(train_val["text"], train_val["label"])

    with mlflow.start_run(run_name=f"FINAL_{best['run']}"):
        mlflow.set_tags({"data_dvc_md5": dvc_md5(), "split_seed": str(SPLIT_SEED),
                         "stage": "final", "selected_from": best["run_id"]})
        mlflow.log_dict(REPORT_DATA, "preprocess_report.json")
        log_params(model, feat, n_train=len(train_val), n_test=len(test))

        preds, metrics = evaluate(final, test, "test")
        mlflow.log_metrics(metrics)
        mlflow.log_dict(leaderboard.to_dict("records"), "leaderboard.json")
        log_cm(test, preds, "confusion_matrix_test.png")

        sample = train_val["text"].head(5)
        mlflow.sklearn.log_model(
            final, name="model",  # use artifact_path="model" on MLflow < 3
            signature=infer_signature(sample, final.predict(sample)),
            input_example=sample.tolist(),
            registered_model_name=MODEL_NAME,
        )
        print(f"\nFINAL {best['run']}: test_f1={metrics['test_f1']:.4f}")


def main():
    mlflow.set_experiment(EXPERIMENT)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    train, val, test = (pd.read_csv(SPLITS / f"{s}.csv") for s in ("train", "val", "test"))

    leaderboard = sweep(train, val)
    cols = ["run", "val_f1", "val_recall", "val_precision", "val_roc_auc"]
    print("\n", leaderboard[cols].to_string(index=False))

    train_final(leaderboard.iloc[0], leaderboard, train, val, test)


if __name__ == "__main__":
    main()
