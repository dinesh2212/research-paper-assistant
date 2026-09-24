"""Shared test fixtures."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from research_paper_assistant.core.config import Settings
from research_paper_assistant.db import Database
from research_paper_assistant.main import create_app
from research_paper_assistant.schemas.health import ComponentHealth, ReadinessResponse
from research_paper_assistant.services.papers import PaperService


class StubHealthChecker:
    """Configurable health checker for API tests."""

    def __init__(self, status: str = "ok") -> None:
        component_status = "ok" if status == "ok" else "error"
        self.response = ReadinessResponse(
            status="ok" if status == "ok" else "degraded",
            services={
                "database": ComponentHealth(
                    status=component_status,
                    latency_ms=1.0,
                    detail="test",
                )
            },
        )

    async def check_readiness(self) -> ReadinessResponse:
        return self.response


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        paper_storage_path=tmp_path / "papers",
        max_pdf_size_bytes=1024 * 1024,
    )


@pytest.fixture
async def database(settings: Settings) -> AsyncIterator[Database]:
    database = Database(settings.database_url)
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
async def paper_service(settings: Settings, database: Database) -> PaperService:
    service = PaperService(
        database=database,
        storage_root=settings.paper_storage_path,
        max_pdf_size_bytes=settings.max_pdf_size_bytes,
    )
    await service.initialize()
    return service


@pytest.fixture
async def client(
    settings: Settings,
    paper_service: PaperService,
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        settings=settings,
        health_checker=StubHealthChecker(),
        paper_service=paper_service,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
