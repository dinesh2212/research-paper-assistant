"""Dependency health checks."""

import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Protocol

import httpx

from research_paper_assistant.core.config import Settings
from research_paper_assistant.db import Database
from research_paper_assistant.schemas.health import ComponentHealth, ReadinessResponse


class HealthChecker(Protocol):
    """Interface consumed by the health API."""

    async def check_readiness(self) -> ReadinessResponse:
        """Return aggregate dependency readiness."""


class HealthService:
    """Check all dependencies required to serve application requests."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._settings = settings
        self._database = database
        self._http_client = http_client

    async def check_readiness(self) -> ReadinessResponse:
        """Check dependencies concurrently and aggregate their results."""

        database, qdrant, ollama = await asyncio.gather(
            self._timed_check(self._check_database),
            self._timed_check(self._check_qdrant),
            self._timed_check(self._check_ollama),
        )
        services = {
            "database": database,
            "qdrant": qdrant,
            "ollama": ollama,
        }
        status = "ok" if all(item.status == "ok" for item in services.values()) else "degraded"
        return ReadinessResponse(status=status, services=services)

    async def _timed_check(
        self,
        check: Callable[[], Awaitable[str]],
    ) -> ComponentHealth:
        started_at = perf_counter()
        try:
            detail = await check()
        except Exception as exc:
            latency_ms = (perf_counter() - started_at) * 1_000
            return ComponentHealth(
                status="error",
                latency_ms=round(latency_ms, 2),
                detail=f"{type(exc).__name__}: {str(exc)[:200]}",
            )

        latency_ms = (perf_counter() - started_at) * 1_000
        return ComponentHealth(
            status="ok",
            latency_ms=round(latency_ms, 2),
            detail=detail,
        )

    async def _check_database(self) -> str:
        await self._database.ping()
        return "SQLite reachable"

    async def _check_qdrant(self) -> str:
        health_response = await self._http_client.get(
            f"{self._settings.qdrant_url.rstrip('/')}/healthz"
        )
        health_response.raise_for_status()

        version_response = await self._http_client.get(self._settings.qdrant_url)
        version_response.raise_for_status()
        version = version_response.json().get("version", "unknown")
        return f"Qdrant {version} reachable"

    async def _check_ollama(self) -> str:
        base_url = self._settings.ollama_url.rstrip("/")
        version_response, tags_response = await asyncio.gather(
            self._http_client.get(f"{base_url}/api/version"),
            self._http_client.get(f"{base_url}/api/tags"),
        )
        version_response.raise_for_status()
        tags_response.raise_for_status()

        version = version_response.json().get("version", "unknown")
        model_names = {
            item.get("name")
            for item in tags_response.json().get("models", [])
            if isinstance(item, dict)
        }
        required_models = {
            self._settings.ollama_generation_model,
            self._settings.ollama_embedding_model,
        }
        missing_models = sorted(required_models - model_names)
        if missing_models:
            missing = ", ".join(missing_models)
            raise RuntimeError(f"Missing required Ollama models: {missing}")

        return f"Ollama {version} reachable; required models available"
