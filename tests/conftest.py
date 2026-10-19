"""Keep pytest collection and destructive integration tests off the business DB."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from tests.db_safety import verify_isolated_test_database


# app.py constructs an app during import. Hide DATABASE_URL before collecting
# tests, including modules that import app at the top level. Keep the original
# process and .env values only for comparison; never print either secret.
_PROCESS_BUSINESS_URL = os.environ.get("DATABASE_URL")
_DOTENV_BUSINESS_URL = dotenv_values(Path(__file__).resolve().parents[1] / ".env").get("DATABASE_URL")
_ISOLATED_DB_URL = os.environ.get("TEST_DATABASE_URL")
os.environ["DATABASE_URL"] = ""


@pytest.fixture(scope="module")
def isolated_test_database(request) -> str:
    if not _ISOLATED_DB_URL:
        pytest.skip("requires an explicit PostgreSQL TEST_DATABASE_URL")
    verify_isolated_test_database(
        _ISOLATED_DB_URL,
        business_urls=(_PROCESS_BUSINESS_URL, _DOTENV_BUSINESS_URL),
    )
    os.environ["DATABASE_URL"] = _ISOLATED_DB_URL
    request.module.DATABASE_URL = _ISOLATED_DB_URL
    try:
        yield _ISOLATED_DB_URL
    finally:
        os.environ["DATABASE_URL"] = ""
