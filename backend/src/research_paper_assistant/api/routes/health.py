"""Liveness and dependency readiness routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from research_paper_assistant.api.dependencies import get_health_checker
from research_paper_assistant.schemas.health import LivenessResponse, ReadinessResponse
from research_paper_assistant.services.health import HealthChecker

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Report whether the API process is running."""

    return LivenessResponse()


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(
    response: Response,
    health_checker: Annotated[HealthChecker, Depends(get_health_checker)],
) -> ReadinessResponse:
    """Report whether all required local dependencies are available."""

    result = await health_checker.check_readiness()
    if result.status == "degraded":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
