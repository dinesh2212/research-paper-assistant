"""Health endpoint response models."""

from typing import Literal

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    """Health state for one required component."""

    status: Literal["ok", "error"]
    latency_ms: float
    detail: str


class LivenessResponse(BaseModel):
    """Minimal process liveness response."""

    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    """Aggregate dependency readiness response."""

    status: Literal["ok", "degraded"]
    services: dict[str, ComponentHealth]
