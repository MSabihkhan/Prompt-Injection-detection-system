"""Drift statistics and feedback merge. No DB, no MLflow."""
import numpy as np
import pandas as pd

from ops.config import load_config
from ops.monitoring.drift import compute_drift, psi
from ops.scripts.merge_feedback import merge


def _frame(rng, n, len_mean=4.4, a=1.5, b=8):
    return pd.DataFrame({"text_length": rng.lognormal(len_mean, 0.6, n),
                         "score": rng.beta(a, b, n)})


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(2, 1, size=5000)) > 0.25


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(1)
    r = compute_drift(_frame(rng, 3000), _frame(rng, 500), load_config())
    assert r["status"] == "ok" and r["drift_detected"] is False


def test_drift_detected_on_longer_prompts():
    rng = np.random.default_rng(2)
    r = compute_drift(_frame(rng, 3000), _frame(rng, 500, len_mean=5.6), load_config())
    assert r["drift_detected"] is True
    assert r["features"]["text_length"]["drifted"] is True


def test_too_few_rows_never_triggers():
    rng = np.random.default_rng(3)
    r = compute_drift(_frame(rng, 3000), _frame(rng, 10, len_mean=7), load_config())
    assert r["status"] == "insufficient_data" and r["drift_detected"] is False


def test_merge_feedback_rules():
    train = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 0, 1]})
    holdout = {"leaky"}
    fb = pd.DataFrame({
        "text": ["new one", "leaky", "b", "dup", "dup", "  spaced   out "],
        "label": [1, 1, 1, 0, 1, 0],   # 'dup' has conflicting votes -> dropped
    })
    merged, stats = merge(train, holdout, fb)
    got = dict(zip(merged["text"], merged["label"]))
    assert got["b"] == 1                      # feedback label overrides train
    assert got["new one"] == 1
    assert got["spaced out"] == 0             # cleaned like train/inference
    assert "leaky" not in got                 # no val/test leakage
    assert "dup" not in got                   # conflicting votes dropped
    assert stats["dropped_conflict"] == 2 and stats["dropped_leak"] == 1
