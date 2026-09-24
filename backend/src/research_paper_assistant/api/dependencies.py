"""FastAPI request dependencies."""

from typing import cast

from fastapi import HTTPException, Request

from research_paper_assistant.services.health import HealthChecker
from research_paper_assistant.services.papers import PaperService
from research_paper_assistant.services.retrieval import RetrievalService


async def get_health_checker(request: Request) -> HealthChecker:
    """Return the application health checker."""

    return cast(HealthChecker, request.app.state.health_checker)


async def get_paper_service(request: Request) -> PaperService:
    """Return the application paper service."""

    return cast(PaperService, request.app.state.paper_service)


async def get_retrieval_service(request: Request) -> RetrievalService:
    service = getattr(request.app.state, "retrieval_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Retrieval service is not initialized")
    return cast(RetrievalService, service)
