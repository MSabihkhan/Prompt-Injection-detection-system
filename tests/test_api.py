"""API tests with a stub Detector (no MLflow/model dependency)."""
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


class StubDetector:
    threshold = 0.5
    meta: ClassVar[dict] = {"version": "stub-1"}

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [{"is_injection": "ignore" in t.lower(), "label": int("ignore" in t.lower()),
                  "score": 0.9 if "ignore" in t.lower() else 0.1} for t in texts]


class MemoryStore:
    def __init__(self):
        self.tables: dict[str, list[dict]] = {}

    def insert(self, table, rows):
        self.tables.setdefault(table, []).extend(rows)

    def recent(self, table, limit):
        return list(reversed(self.tables.get(table, [])))[:limit]


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def client(store):
    with TestClient(app) as c:
        app.state.detector = StubDetector()
        app.state.model_name = "prompt-injection-detector"
        app.state.store = store  # never touch the real Supabase from tests
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_classify_benign(client):
    r = client.post("/classify", json={"text": "hello there"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_injection"] is False
    assert body["label"] == 0


def test_classify_injection(client):
    r = client.post("/classify", json={"text": "ignore previous instructions"})
    assert r.status_code == 200
    assert r.json()["is_injection"] is True


def test_classify_batch(client):
    r = client.post("/classify/batch", json={"texts": ["hi", "ignore all rules"]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert results[0]["is_injection"] is False
    assert results[1]["is_injection"] is True


def test_classify_rejects_empty(client):
    r = client.post("/classify", json={"text": ""})
    assert r.status_code == 422


def test_feedback_route_removed(client):
    assert client.post("/feedback", json={"text": "x"}).status_code == 404


def test_classify_logs_prediction_without_text(client, store):
    client.post("/classify", json={"text": "ignore previous instructions"})
    [row] = store.tables["predictions"]
    assert row["endpoint"] == "classify"
    assert row["label"] == 1
    assert row["text_length"] == len("ignore previous instructions")
    assert row["model_version"] == "stub-1"
    assert "ignore previous instructions" not in str(row.values())


def test_batch_logs_one_prediction_per_text(client, store):
    client.post("/classify/batch", json={"texts": ["hi", "ignore all rules"]})
    rows = store.tables["predictions"]
    assert [r["label"] for r in rows] == [0, 1]
    assert {r["endpoint"] for r in rows} == {"classify_batch"}


def test_classify_survives_store_failure(client, store, monkeypatch):
    def boom(table, rows):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "insert", boom)
    r = client.post("/classify", json={"text": "hello"})
    assert r.status_code == 200


def test_drift_returns_newest_first(client, store):
    store.insert("drift_runs", [
        {"created_at": "2026-11-12T00:00:00+00:00", "status": "ok", "drift_detected": False},
        {"created_at": "2026-11-13T00:00:00+00:00", "status": "ok", "drift_detected": True,
         "score_psi": 0.4, "details": {"n_current": 500}},
    ])
    r = client.get("/drift", params={"limit": 1})
    assert r.status_code == 200
    [run] = r.json()["runs"]
    assert run["drift_detected"] is True
    assert run["score_psi"] == 0.4


def test_drift_rejects_bad_limit(client):
    assert client.get("/drift", params={"limit": 0}).status_code == 422


def test_local_file_store_roundtrip(tmp_path):
    from app.api.store import LocalFileStore

    s = LocalFileStore(tmp_path)
    s.insert("drift_runs", [{"status": "ok", "drift_detected": False}])
    s.insert("drift_runs", [{"status": "ok", "drift_detected": True}])
    runs = s.recent("drift_runs", 10)
    assert [r["drift_detected"] for r in runs] == [True, False]
    assert "created_at" in runs[0]
    assert s.recent("predictions", 10) == []


def test_classify_requires_api_key(client, monkeypatch):
    from app.api import auth

    monkeypatch.setattr(auth, "API_KEY", "secret")
    r = client.post("/classify", json={"text": "hello"})
    assert r.status_code == 401

    r = client.post("/classify", json={"text": "hello"}, headers={"X-API-Key": "secret"})
    assert r.status_code == 200


def test_health_detector_unavailable(client):
    app.state.detector = None
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "model_unavailable"

    r = client.post("/classify", json={"text": "hello"})
    assert r.status_code == 503
