"""Evaluate a registered detector version, pick its decision threshold, gate it, optionally promote it.

Usage (from repo root):
    python app/eval.py                          # evaluate latest version, log to MLflow
    python app/eval.py --model 2 --promote      # evaluate v2; set alias 'champion' if it passes and beats current
    python app/eval.py --no-log                 # local dry run: no MLflow writes

Threshold is chosen on out-of-fold scores from train+val (never on test), as the highest
threshold reaching --target-recall, and never above the model's default threshold, so it
can only make the model *more* sensitive. Test is then scored once at that threshold.

Exit code 0 = gates passed, 1 = gates failed (for CI).
"""
import argparse
import json
import sys
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from inference import MODEL_NAME, Detector, resolve_version
from train import EXPERIMENT, ROOT, SEED, SPLIT_SEED, SPLITS, dvc_md5

REPORTS = ROOT / "reports"
CHAMPION = "champion"

FULLWIDTH = {i: i + 0xFEE0 for i in range(0x21, 0x7F)}
LEET = str.maketrans("aeiost", "431057")
PERTURBATIONS = {
    "uppercase": str.upper,
    "lowercase": str.lower,
    "extra_whitespace": lambda t: t.replace(" ", "  \n\t "),
    "fullwidth_unicode": lambda t: t.translate(FULLWIDTH),  # NFKC in clean step should undo this
    "leetspeak": lambda t: t.translate(LEET),
    "benign_prefix": lambda t: "Can you help me write a polite email to my landlord? " + t,
    "benign_suffix": lambda t: t + " Thanks so much, have a great day!",
}
LENGTH_BINS = [0, 10, 50, 200, np.inf]
LENGTH_LABELS = ["1-10 words", "11-50 words", "51-200 words", "200+ words"]


# ---------- Threshold ----------
def oof_scores(det: Detector, df: pd.DataFrame, folds: int) -> np.ndarray:
    """Out-of-fold scores on train+val by refitting a clone of the pipeline per fold."""
    method = "predict_proba" if det.score_type == "probability" else "decision_function"
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    s = cross_val_predict(clone(det.pipe), det._prep(df["text"]), df["label"].to_numpy(),
                          cv=cv, method=method)
    return s[:, 1] if s.ndim == 2 else s


def choose_threshold(y, s, target_recall: float, default: float) -> tuple[float, float]:
    """Highest threshold with recall >= target, capped at the default. Returns (chosen, uncapped)."""
    _, rec, thr = precision_recall_curve(y, s)
    ok = thr[rec[:-1] >= target_recall]
    uncapped = float(ok.max()) if len(ok) else float(thr.min())
    return min(uncapped, default), uncapped


# ---------- Metrics ----------
def metrics_at(y, s, t, prefix: str) -> dict:
    pred = (s >= t).astype(int)
    return {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_f1": f1_score(y, pred, zero_division=0),
        f"{prefix}_precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, pred, zero_division=0),
        f"{prefix}_false_negatives": int(((pred == 0) & (y == 1)).sum()),
        f"{prefix}_false_positives": int(((pred == 1) & (y == 0)).sum()),
    }


def slice_report(df: pd.DataFrame, s, t) -> pd.DataFrame:
    words = df["text"].str.split().str.len()
    buckets = pd.cut(words, LENGTH_BINS, labels=LENGTH_LABELS)
    pred = (s >= t).astype(int)
    rows = []
    for b in LENGTH_LABELS:
        m = (buckets == b).to_numpy()
        y, p = df["label"].to_numpy()[m], pred[m]
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        rows.append({
            "slice": b, "n": int(m.sum()), "n_injection": n_pos,
            "recall": ((p == 1) & (y == 1)).sum() / n_pos if n_pos else np.nan,
            "false_positive_rate": ((p == 1) & (y == 0)).sum() / n_neg if n_neg else np.nan,
        })
    return pd.DataFrame(rows)


def robustness_report(det: Detector, df: pd.DataFrame, t) -> pd.DataFrame:
    """Detection rate on perturbed injections, false-positive rate on perturbed safe prompts."""
    inj = df.loc[df["label"] == 1, "text"].tolist()
    safe = df.loc[df["label"] == 0, "text"].tolist()
    rows = [{"perturbation": "none",
             "injection_recall": float((det.score(inj) >= t).mean()),
             "safe_false_positive_rate": float((det.score(safe) >= t).mean())}]
    for name, fn in PERTURBATIONS.items():
        rows.append({
            "perturbation": name,
            "injection_recall": float((det.score([fn(x) for x in inj]) >= t).mean()),
            "safe_false_positive_rate": float((det.score([fn(x) for x in safe]) >= t).mean()),
        })
    return pd.DataFrame(rows)


def latency_ms(det: Detector, texts: list[str], n: int = 200) -> dict:
    sample = pd.Series(texts).sample(min(n, len(texts)), random_state=SEED).tolist()
    det.predict(sample[:1])  # warm-up
    times = []
    for t in sample:
        t0 = time.perf_counter()
        det.predict([t])
        times.append((time.perf_counter() - t0) * 1000)
    return {"latency_p50_ms": float(np.percentile(times, 50)),
            "latency_p95_ms": float(np.percentile(times, 95))}


# ---------- Figures ----------
def save_figures(y, s, t, out_dir):
    pred = (s >= t).astype(int)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, display_labels=["safe", "injection"])
    ax.set_title(f"Test @ threshold {t:.3f}")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix_test.png"); plt.close(fig)

    prec, rec, thr = precision_recall_curve(y, s)
    i = np.searchsorted(thr, t).clip(0, len(thr) - 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec)
    ax.scatter([rec[i]], [prec[i]], color="red", zorder=3, label=f"threshold {t:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "pr_curve_test.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s[y == 0], bins=60, alpha=0.6, label="safe")
    ax.hist(s[y == 1], bins=60, alpha=0.6, label="injection")
    ax.axvline(t, color="red", linestyle="--", label="threshold")
    ax.set(xlabel="Score", ylabel="Count", title="Score distribution (test)", yscale="log")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "score_hist_test.png"); plt.close(fig)


# ---------- Registry ----------
def current_champion(client):
    try:
        return resolve_version(CHAMPION, client)
    except LookupError:
        return None


def promote(client, version: str):
    try:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    except MlflowException:  # server without alias support: tag-based fallback
        for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if v.tags.get(f"alias.{CHAMPION}") == "true" and v.version != version:
                client.delete_model_version_tag(MODEL_NAME, v.version, f"alias.{CHAMPION}")
        client.set_model_version_tag(MODEL_NAME, version, f"alias.{CHAMPION}", "true")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="latest", help="version number, alias, or 'latest'")
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--min-recall", type=float, default=0.97, help="gate on test recall")
    ap.add_argument("--min-precision", type=float, default=0.95, help="gate on test precision")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--promote", action="store_true", help=f"set alias '{CHAMPION}' if gates pass")
    ap.add_argument("--no-log", action="store_true", help="skip all MLflow writes")
    args = ap.parse_args(argv)

    client = MlflowClient()
    v = resolve_version(args.model, client)
    det = Detector.from_registry(str(v.version))
    default_t = Detector(det.pipe).threshold  # ignore any previously stored threshold
    print(f"Evaluating {MODEL_NAME} v{v.version} ({det.score_type}, clean step: {det.has_clean_step})")

    trained_md5 = client.get_run(v.run_id).data.tags.get("data_dvc_md5")
    if trained_md5 and trained_md5 != dvc_md5():
        print(f"WARNING: model trained on data {trained_md5}, current data is {dvc_md5()}")

    train, val, test = (pd.read_csv(SPLITS / f"{s}.csv") for s in ("train", "val", "test"))
    train_val = pd.concat([train, val], ignore_index=True)

    # 1. Threshold from out-of-fold scores (test untouched)
    oof = oof_scores(det, train_val, args.folds)
    t, t_uncapped = choose_threshold(train_val["label"].to_numpy(), oof, args.target_recall, default_t)
    det.threshold = t
    print(f"Threshold: {t:.4f} (default {default_t:.4f}, uncapped {t_uncapped:.4f})")

    # 2. Test metrics at default and chosen threshold
    y, s = test["label"].to_numpy(), det.score(test["text"])
    metrics = {
        "test_roc_auc": roc_auc_score(y, s),
        "test_pr_auc": average_precision_score(y, s),
        **metrics_at(y, s, default_t, "test_default"),
        **metrics_at(y, s, t, "test_at_threshold"),
        **latency_ms(det, test["text"].tolist()),
    }
    slices = slice_report(test, s, t)
    robust = robustness_report(det, test, t)
    for _, r in robust.iterrows():
        metrics[f"robust_{r['perturbation']}_recall"] = r["injection_recall"]
        metrics[f"robust_{r['perturbation']}_fpr"] = r["safe_false_positive_rate"]

    # 3. Gates
    gates = {
        "recall": metrics["test_at_threshold_recall"] >= args.min_recall,
        "precision": metrics["test_at_threshold_precision"] >= args.min_precision,
    }
    passed = all(gates.values())

    # 4. Local report
    out_dir = REPORTS / f"eval_v{v.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = test.assign(score=s, pred=(s >= t).astype(int))
    errors = errors[errors["pred"] != errors["label"]].assign(
        error=lambda d: np.where(d["label"] == 1, "false_negative", "false_positive")
    ).sort_values(["error", "score"])
    errors.to_csv(out_dir / "errors.csv", index=False)
    slices.to_csv(out_dir / "slices.csv", index=False)
    robust.to_csv(out_dir / "robustness.csv", index=False)
    save_figures(y, s, t, out_dir)
    summary = {"model_version": v.version, "threshold": t, "default_threshold": default_t,
               "score_type": det.score_type, "gates": gates, "passed": passed, "metrics": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    # 5. Print
    print(f"\nTest @ default   : recall={metrics['test_default_recall']:.4f} "
          f"precision={metrics['test_default_precision']:.4f} f1={metrics['test_default_f1']:.4f}")
    print(f"Test @ threshold : recall={metrics['test_at_threshold_recall']:.4f} "
          f"precision={metrics['test_at_threshold_precision']:.4f} f1={metrics['test_at_threshold_f1']:.4f} "
          f"(FN={metrics['test_at_threshold_false_negatives']}, FP={metrics['test_at_threshold_false_positives']})")
    print(f"ROC-AUC={metrics['test_roc_auc']:.4f}  PR-AUC={metrics['test_pr_auc']:.4f}  "
          f"latency p50={metrics['latency_p50_ms']:.2f}ms p95={metrics['latency_p95_ms']:.2f}ms")
    print("\nBy length:\n" + slices.to_string(index=False, float_format="%.3f"))
    print("\nRobustness:\n" + robust.to_string(index=False, float_format="%.3f"))
    print(f"\nGates {gates} -> {'PASSED' if passed else 'FAILED'}   report: {out_dir}")

    if args.no_log:
        return 0 if passed else 1

    # 6. MLflow: eval run + version tags (threshold travels with the model via tags)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"EVAL_v{v.version}") as run:
        mlflow.set_tags({"stage": "eval", "model_version": v.version, "model_run_id": v.run_id,
                         "data_dvc_md5": dvc_md5(), "split_seed": str(SPLIT_SEED),
                         "gate_passed": str(passed)})
        mlflow.log_params({"threshold": t, "default_threshold": default_t,
                           "target_recall": args.target_recall, "min_recall": args.min_recall,
                           "min_precision": args.min_precision, "folds": args.folds})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(out_dir))

    for k, val in {"threshold": t, "score_type": det.score_type, "threshold_target_recall": args.target_recall,
                   "test_f1_at_threshold": metrics["test_at_threshold_f1"],
                   "test_recall_at_threshold": metrics["test_at_threshold_recall"],
                   "test_precision_at_threshold": metrics["test_at_threshold_precision"],
                   "gate_passed": passed, "eval_run_id": run.info.run_id}.items():
        client.set_model_version_tag(MODEL_NAME, v.version, k, str(val))

    # 7. Promotion
    if args.promote:
        champ = current_champion(client)
        champ_f1 = float(champ.tags.get("test_f1_at_threshold", 0)) if champ else -1
        if not passed:
            print(f"Not promoting v{v.version}: gates failed.")
        elif champ and champ.version != v.version and metrics["test_at_threshold_f1"] < champ_f1:
            print(f"Not promoting v{v.version}: f1 {metrics['test_at_threshold_f1']:.4f} "
                  f"< champion v{champ.version} {champ_f1:.4f}")
        else:
            promote(client, v.version)
            print(f"Promoted v{v.version} to '{CHAMPION}'.")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
