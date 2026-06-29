"""
Velozen AI — API key authentication.

Reads the expected key from the VELOZEN_API_KEY environment variable.
Clients send it in the X-API-Key header.

Swap this dependency for OAuth2 / JWT when moving to multi-tenant production.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

load_dotenv()

_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> str:
    expected = os.getenv("VELOZEN_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is not configured with an API key.",
        )
    if key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return key
