"""API key auth. Set PROMPTGUARD_API_KEY; if unset, auth is disabled (local dev only)."""
import os

from fastapi import Header, HTTPException, status

API_KEY = os.environ.get("PROMPTGUARD_API_KEY")


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY is None:
        return  # no key configured -> auth disabled (dev convenience, not for prod)
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
