"""Paper library API models."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

PaperStatus = Literal["pending", "processing", "ready", "failed"]


class PaperSummary(BaseModel):
    """Public summary for one uploaded paper."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    original_filename: str
    size_bytes: int
    page_count: int
    status: PaperStatus
    created_at: datetime


class PaperDetail(PaperSummary):
    """Detailed metadata and ingestion state for one paper."""

    error_message: str | None
    updated_at: datetime
