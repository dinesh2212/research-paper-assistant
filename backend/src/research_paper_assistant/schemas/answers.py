"""Answer contracts and the structured output requested from the local model."""

from pydantic import BaseModel, ConfigDict, Field

from research_paper_assistant.schemas.retrieval import RetrievalResponse


class CitedStatement(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    text: str = Field(min_length=1, max_length=4000)
    source_ids: list[int] = Field(min_length=1, max_length=8)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    insufficient_evidence: bool
    statements: list[CitedStatement] = Field(max_length=12)


class AnswerResponse(RetrievalResponse):
    answer: str
    insufficient_evidence: bool
