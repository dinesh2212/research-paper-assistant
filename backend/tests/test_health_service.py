"""Dependency health service tests."""

from pathlib import Path

import httpx
import pytest
from research_paper_assistant.core.config import Settings
from research_paper_assistant.db import Database
from research_paper_assistant.services.health import HealthService


def healthy_dependency_responses(request: httpx.Request) -> httpx.Response:
    """Return representative responses from local dependencies."""

    if request.url.path == "/healthz":
        return httpx.Response(200, text="healthz check passed")
    if request.url.path == "/":
        return httpx.Response(200, json={"version": "1.18.2"})
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.32.5"})
    if request.url.path == "/api/tags":
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "gemma4:12b"},
                    {"name": "qwen3-embedding:0.6b"},
                ]
            },
        )
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_all_dependencies_are_healthy(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'health.db'}",
    )
    database = Database(settings.database_url)
    await database.initialize()
    transport = httpx.MockTransport(healthy_dependency_responses)

    try:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await HealthService(settings, database, client).check_readiness()
    finally:
        await database.close()

    assert result.status == "ok"
    assert set(result.services) == {"database", "qdrant", "ollama"}
    assert all(component.status == "ok" for component in result.services.values())


@pytest.mark.asyncio
async def test_missing_ollama_model_degrades_readiness(tmp_path: Path) -> None:
    def missing_model_response(request: httpx.Request) -> httpx.Response:
        response = healthy_dependency_responses(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma4:12b"}]})
        return response

    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'health.db'}",
    )
    database = Database(settings.database_url)
    await database.initialize()
    transport = httpx.MockTransport(missing_model_response)

    try:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await HealthService(settings, database, client).check_readiness()
    finally:
        await database.close()

    assert result.status == "degraded"
    assert result.services["ollama"].status == "error"
    assert "qwen3-embedding:0.6b" in result.services["ollama"].detail
