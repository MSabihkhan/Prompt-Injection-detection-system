"""DB access for the API: predictions, drift_runs (schema: ops/sql/001_schema.sql).
Uses Supabase if SUPABASE_URL/SUPABASE_KEY are set, else one JSONL file per table under reports/
(dev fallback, gitignored). Swap the backend here only.
"""
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_DIR = ROOT / "reports"


class Store(Protocol):
    def insert(self, table: str, rows: list[dict]) -> None: ...
    def recent(self, table: str, limit: int) -> list[dict]: ...


class LocalFileStore:
    def __init__(self, directory: Path = LOCAL_DIR):
        self.directory = directory

    def _path(self, table: str) -> Path:
        return self.directory / f"{table}.jsonl"

    def insert(self, table: str, rows: list[dict]) -> None:
        path = self._path(table)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC).isoformat()
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"created_at": now, **row}) + "\n")

    def recent(self, table: str, limit: int) -> list[dict]:
        path = self._path(table)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in reversed(lines[-limit:])]


class SupabaseStore:
    def __init__(self, url: str, key: str):
        from supabase import create_client  # imported lazily: optional dependency

        self.client = create_client(url, key)

    def insert(self, table: str, rows: list[dict]) -> None:
        self.client.table(table).insert(rows).execute()  # created_at defaults in the DB

    def recent(self, table: str, limit: int) -> list[dict]:
        res = self.client.table(table).select("*").order("created_at", desc=True).limit(limit).execute()
        return res.data


def get_store() -> Store:
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if url and key:
        return SupabaseStore(url, key)
    return LocalFileStore()


def make_prediction_row(endpoint: str, text: str, result: dict, threshold: float,
                        model_version: str | int | None, latency_ms: float) -> dict:
    # Length only, never the prompt text (project rule: no prompt text stored by default).
    return {
        "endpoint": endpoint,
        "model_version": str(model_version) if model_version is not None else None,
        "label": result["label"],
        "score": result["score"],
        "threshold": threshold,
        "text_length": len(text),
        "latency_ms": latency_ms,
    }
