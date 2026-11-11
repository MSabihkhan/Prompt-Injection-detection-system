"""Postgres connection helper. Works for local Postgres (docker) and Supabase's Postgres.

Preferred: set the separate PG_* variables (PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD,
PG_SSLMODE). The password is then passed as-is, so special characters need no URL-encoding.
Fallback: DATABASE_URL (then special characters in the password must be URL-encoded).
"""
import os

from dotenv import load_dotenv

load_dotenv()


def get_conn():
    import psycopg2  # imported lazily so unit tests that don't touch the DB need no driver

    host = os.environ.get("PG_HOST")
    if host:
        return psycopg2.connect(
            host=host,
            port=os.environ.get("PG_PORT", "5432"),
            dbname=os.environ.get("PG_DB", "postgres"),
            user=os.environ.get("PG_USER"),
            password=os.environ.get("PG_PASSWORD"),
            sslmode=os.environ.get("PG_SSLMODE", "prefer"),
        )
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set PG_HOST/PG_USER/PG_PASSWORD (preferred) or DATABASE_URL. See ops/.env.ops.example.")
    return psycopg2.connect(url)
