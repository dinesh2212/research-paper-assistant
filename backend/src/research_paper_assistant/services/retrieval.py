"""Paper-filtered dense retrieval with deterministic diversity selection."""

import asyncio
import math
from contextlib import AsyncExitStack
from uuid import UUID, uuid5

from research_paper_assistant.models import Paper
from research_paper_assistant.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedSource,
    ScoredChunk,
)
from research_paper_assistant.services.indexing import IndexingError, OllamaEmbedder, QdrantStore
from research_paper_assistant.services.papers import PaperService, PaperStateError

QUERY_INSTRUCTION = (
    "Given a research question, retrieve passages from scientific papers that answer it."
)


def select_diverse(
    candidates: list[ScoredChunk], limit: int, weight: float = 0.7
) -> list[ScoredChunk]:
    """Keep the top hit, then use maximal marginal relevance to reduce repetition.

    Exact duplicate text is suppressed within a paper; copies in distinct
    papers remain independent sources. MMR ordering is not descending score.
    """

    remaining: list[ScoredChunk] = []
    seen: set[tuple[UUID, str]] = set()
    for candidate in sorted(candidates, key=lambda item: (-item.score, str(item.id))):
        key = (candidate.payload.paper_id, " ".join(candidate.payload.text.split()))
        if key not in seen:
            remaining.append(candidate)
            seen.add(key)
    if not remaining:
        return []
    normalized: dict[UUID, list[float]] = {}
    for item in remaining:
        norm = math.hypot(*item.vector)
        normalized[item.id] = [value / norm for value in item.vector]
    selected = [remaining.pop(0)]
    while remaining and len(selected) < limit:

        def marginal_relevance(item: ScoredChunk) -> float:
            similarity = max(
                sum(a * b for a, b in zip(normalized[item.id], normalized[other.id], strict=True))
                for other in selected
            )
            return weight * item.score - (1 - weight) * similarity

        best = max(remaining, key=marginal_relevance)
        selected.append(best)
        remaining.remove(best)
    return selected


class RetrievalService:
    def __init__(
        self,
        papers: PaperService,
        embedder: OllamaEmbedder,
        vectors: QdrantStore,
        timeout_seconds: float = 30,
    ) -> None:
        self.papers = papers
        self.embedder = embedder
        self.vectors = vectors
        self.timeout_seconds = timeout_seconds

    async def _ready_papers(self, ids: list[UUID]) -> dict[UUID, Paper]:
        papers: dict[UUID, Paper] = {}
        for paper_id in ids:
            paper = await self.papers.get_paper(paper_id)
            if paper.status != "ready":
                raise PaperStateError(
                    f"Paper {paper_id} is {paper.status}; select only ready papers"
                )
            papers[paper_id] = paper
        return papers

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        # Reject pending/processing selections promptly, before waiting on locks.
        await self._ready_papers(request.paper_ids)
        async with asyncio.timeout(self.timeout_seconds), AsyncExitStack() as stack:
            # Globally ordered acquisition prevents deadlock for overlapping selections.
            for paper_id in sorted(request.paper_ids):
                await stack.enter_async_context(self.papers.paper_lock(paper_id))
            papers = await self._ready_papers(request.paper_ids)
            query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {request.question}"
            try:
                embeddings = await self.embedder.embed([query])
                candidates = await self.vectors.query(
                    embeddings[0],
                    request.paper_ids,
                    self.embedder.model,
                    limit=20,
                    score_threshold=request.score_threshold,
                )
                seen_ids: set[UUID] = set()
                for candidate in candidates:
                    payload = candidate.payload
                    paper = papers.get(payload.paper_id)
                    if (
                        paper is None
                        or payload.page_number > paper.page_count
                        or candidate.id != payload.chunk_id
                        or candidate.id
                        != uuid5(payload.paper_id, f"{payload.page_number}:{payload.chunk_index}")
                        or payload.embedding_model != self.embedder.model
                        or not payload.text.strip()
                        or len(candidate.vector) != self.embedder.dimensions
                        or not any(candidate.vector)
                        or not math.isfinite(math.hypot(*candidate.vector))
                        or candidate.id in seen_ids
                    ):
                        raise IndexingError("The selected paper index contains invalid source data")
                    seen_ids.add(candidate.id)
                eligible = [
                    item
                    for item in candidates
                    if request.score_threshold is None or item.score >= request.score_threshold
                ]
                selected = select_diverse(eligible, request.limit)
            except IndexingError as exc:
                raise IndexingError(
                    "Retrieval unavailable. Check Ollama, Qdrant, and the selected paper indexes."
                ) from exc
            return RetrievalResponse(
                question=request.question,
                paper_ids=request.paper_ids,
                candidate_count=len(eligible),
                sources=[
                    RetrievedSource(
                        source_id=index + 1,
                        paper_id=item.payload.paper_id,
                        paper_title=papers[item.payload.paper_id].title,
                        page_number=item.payload.page_number,
                        chunk_id=item.id,
                        excerpt=item.payload.text,
                        score=item.score,
                    )
                    for index, item in enumerate(selected)
                ],
            )
