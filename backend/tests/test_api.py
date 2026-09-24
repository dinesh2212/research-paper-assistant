"""API contract tests."""

import httpx
from research_paper_assistant.core.config import Settings
from research_paper_assistant.main import create_app
from research_paper_assistant.schemas.health import ComponentHealth, ReadinessResponse


class DegradedHealthChecker:
    """Return a deterministic failed dependency state."""

    async def check_readiness(self) -> ReadinessResponse:
        return ReadinessResponse(
            status="degraded",
            services={
                "qdrant": ComponentHealth(
                    status="error",
                    latency_ms=2.0,
                    detail="connection failed",
                )
            },
        )


async def test_application_metadata(client: httpx.AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "Research Paper Assistant",
        "version": "0.1.0",
        "environment": "test",
        "docs": "/docs",
    }


async def test_liveness(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_when_dependencies_are_healthy(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readiness_returns_503_when_dependency_fails(settings: Settings) -> None:
    app = create_app(settings=settings, health_checker=DegradedHealthChecker())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["services"]["qdrant"]["status"] == "error"


async def test_empty_paper_library(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/papers")

    assert response.status_code == 200
    assert response.json() == []
