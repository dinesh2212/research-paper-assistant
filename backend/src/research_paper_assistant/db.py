"""SQLite engine lifecycle and connectivity checks."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from research_paper_assistant.models import Base


class Database:
    """Own the application's asynchronous database engine."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

        if make_url(database_url).drivername.startswith("sqlite"):
            event.listen(self._engine.sync_engine, "connect", self._enable_foreign_keys)

    @staticmethod
    def _enable_foreign_keys(dbapi_connection: Any, _: Any) -> None:
        """Enable SQLite foreign-key enforcement for every pooled connection."""

        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def prepare_storage(self) -> None:
        """Create the parent directory for a file-backed SQLite database."""

        url = make_url(self._database_url)
        if not url.drivername.startswith("sqlite") or not url.database:
            return
        if url.database == ":memory:":
            return
        Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Prepare storage, create tables, and verify connectivity."""

        self.prepare_storage()
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        await self.ping()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a short-lived asynchronous ORM session."""

        async with self._session_factory() as session:
            yield session

    async def ping(self) -> None:
        """Execute a minimal connectivity query."""

        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def close(self) -> None:
        """Release pooled database resources."""

        await self._engine.dispose()
