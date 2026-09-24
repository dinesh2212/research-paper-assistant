"""Contracts for inspecting retrieval before connecting answer generation."""

from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", allow_inf_nan=False)

    question: str = Field(min_length=1, max_length=2000)
    paper_ids: list[UUID] = Field(min_length=1, max_length=20)
    limit: int = Field(default=8, ge=1, le=8)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)

    @model_validator(mode="after")
    def unique_selection(self) -> Self:
        if len(set(self.paper_ids)) != len(self.paper_ids):
            raise ValueError("paper_ids must not contain duplicates")
        return self


class RetrievedSource(BaseModel):
    """One source in selection order; source_id is scoped to this response."""

    source_id: int
    paper_id: UUID
    paper_title: str
    page_number: int
    chunk_id: UUID
    excerpt: str
    score: float


class RetrievalResponse(BaseModel):
    question: str
    paper_ids: list[UUID]
    candidate_count: int
    sources: list[RetrievedSource]


class ChunkPayload(BaseModel):
    paper_id: UUID
    chunk_id: UUID
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    embedding_model: str


class ScoredChunk(BaseModel):
    """Validated internal Qdrant result; raw vectors are never returned publicly."""

    model_config = ConfigDict(allow_inf_nan=False)

    id: UUID
    score: float
    payload: ChunkPayload
    vector: list[float] = Field(min_length=1)


class QueryResult(BaseModel):
    points: list[ScoredChunk]
