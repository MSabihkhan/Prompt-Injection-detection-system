"""Feedback storage. Uses Supabase if SUPABASE_URL/SUPABASE_KEY are set, else appends to a local
JSONL file under reports/ (dev fallback, gitignored). Swap the backend here only.
"""
import json
import os
import time
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_FEEDBACK_FILE = ROOT / "reports" / "feedback.jsonl"


class FeedbackStore(Protocol):
    def record(self, row: dict) -> None: ...


class LocalFileFeedbackStore:
    def __init__(self, path: Path = LOCAL_FEEDBACK_FILE):
        self.path = path

    def record(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


class SupabaseFeedbackStore:
    def __init__(self, url: str, key: str, table: str = "feedback"):
        from supabase import create_client  # imported lazily: optional dependency

        self.client = create_client(url, key)
        self.table = table

    def record(self, row: dict) -> None:
        self.client.table(self.table).insert(row).execute()


def get_feedback_store() -> FeedbackStore:
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if url and key:
        return SupabaseFeedbackStore(url, key)
    return LocalFileFeedbackStore()


def make_feedback_row(text: str, predicted_label: int, correct_label: int, model_version: str | None) -> dict:
    return {
        "text": text,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "model_version": model_version,
        "created_at": time.time(),
    }
