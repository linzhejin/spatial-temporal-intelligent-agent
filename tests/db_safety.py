"""Guard destructive PostgreSQL integration tests against business databases."""

from __future__ import annotations

import os
import re

from psycopg.conninfo import conninfo_to_dict

from storage import database

_TEST_DATABASE_NAME = re.compile(r"(?:^|_)test(?:_|$)", re.IGNORECASE)


def verify_isolated_test_database(url: str, *, business_urls: tuple[str | None, ...] = ()) -> None:
    """Validate the explicit test target before migrations or fixture cleanup."""
    if not url:
        raise RuntimeError("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    target_name = conninfo_to_dict(url).get("dbname")
    if not target_name or not _TEST_DATABASE_NAME.search(target_name):
        raise RuntimeError("TEST_DATABASE_URL must name a dedicated test database")

    for business_url in (os.environ.get("DATABASE_URL"), *business_urls):
        if business_url:
            business_name = conninfo_to_dict(business_url).get("dbname")
            if business_name and business_name.casefold() == target_name.casefold():
                raise RuntimeError("TEST_DATABASE_URL points to the business database")

    with database.connect(url) as conn:
        actual_name = conn.execute("SELECT current_database() AS name").fetchone()["name"]
    if actual_name != target_name:
        raise RuntimeError("TEST_DATABASE_URL resolved to an unexpected database")
