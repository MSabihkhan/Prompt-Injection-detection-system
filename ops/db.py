"""Postgres connection helper. Works for local Postgres (docker) and Supabase's Postgres.

Set DATABASE_URL in .env, e.g.
  local:    postgresql://pg:pg@localhost:5433/promptguard
  supabase: postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
"""
import os

from dotenv import load_dotenv

load_dotenv()


def get_conn():
    import psycopg2  # imported lazily so unit tests that don't touch the DB need no driver

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. See ops/.env.ops.example.")
    return psycopg2.connect(url)
