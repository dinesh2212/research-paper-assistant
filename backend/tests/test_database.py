"""Database lifecycle tests."""

from pathlib import Path

import pytest
from research_paper_assistant.db import Database
from sqlalchemy import text


@pytest.mark.asyncio
async def test_database_initializes_file_and_responds(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "application.db"
    database = Database(f"sqlite+aiosqlite:///{database_path}")

    try:
        await database.initialize()
        await database.ping()
        async with database.session() as session:
            table_names = set(
                await session.scalars(
                    text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                    )
                )
            )
    finally:
        await database.close()

    assert database_path.exists()
    assert table_names == {"papers", "ingestion_jobs"}
