"""One lifespan-managed worker draining the durable SQLite ingestion queue."""

import asyncio
import logging
from contextlib import suppress
from uuid import UUID

from anyio import to_thread
from sqlalchemy import select, update

from research_paper_assistant.core.config import Settings
from research_paper_assistant.db import Database
from research_paper_assistant.models import IngestionJob, Paper
from research_paper_assistant.services.extraction import ExtractionError, chunk_pages, extract_pages
from research_paper_assistant.services.indexing import IndexingError, OllamaEmbedder, QdrantStore
from research_paper_assistant.services.papers import PaperNotFoundError, PaperService

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        database: Database,
        papers: PaperService,
        embedder: OllamaEmbedder,
        vectors: QdrantStore,
        settings: Settings,
    ) -> None:
        self.database = database
        self.papers = papers
        self.embedder = embedder
        self.vectors = vectors
        self.settings = settings
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Requeue work interrupted by a previous process before starting the worker."""

        async with self.database.session() as session:
            await session.execute(
                update(Paper)
                .where(Paper.status == "processing")
                .values(
                    status="pending",
                    error_message=None,
                )
            )
            pending = select(Paper.id).where(Paper.status == "pending")
            await session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.paper_id.in_(pending),
                )
                .values(status="pending", error_message=None)
            )
            await session.commit()
        self._task = asyncio.create_task(self._run(), name="paper-ingestion")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                async with self.database.session() as session:
                    paper_id = await session.scalar(
                        select(IngestionJob.paper_id)
                        .join(Paper)
                        .where(
                            IngestionJob.status == "pending",
                            Paper.status == "pending",
                        )
                        .order_by(IngestionJob.created_at)
                        .limit(1)
                    )
                if paper_id is not None:
                    await self.process(UUID(paper_id))
                    continue
            except Exception:
                # A transient DB failure must not permanently kill the queue consumer.
                logger.exception("Ingestion queue operation failed")
            await asyncio.sleep(self.settings.ingestion_poll_seconds)

    async def _set_status(self, paper_id: UUID, status: str, error: str | None = None) -> None:
        async with self.database.session() as session:
            await session.execute(
                update(Paper)
                .where(Paper.id == str(paper_id))
                .values(
                    status=status,
                    error_message=error,
                )
            )
            await session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.paper_id == str(paper_id),
                )
                .values(status=status, error_message=error)
            )
            await session.commit()

    async def process(self, paper_id: UUID) -> None:
        """Index one paper, keeping the paper and job states in the same transaction."""

        async with self.papers.paper_lock(paper_id):
            try:
                paper = await self.papers.get_paper(paper_id)
            except PaperNotFoundError:
                return  # Deleted after the queue scan.
            if paper.status != "pending":
                return
            await self._set_status(paper_id, "processing")
            try:
                # Always clean stale partial points before a new/recovered attempt.
                await self.vectors.delete_paper(str(paper_id))
                pages = await to_thread.run_sync(extract_pages, self.papers.stored_path(paper))
                chunks = await to_thread.run_sync(
                    chunk_pages,
                    paper_id,
                    pages,
                    self.settings.chunk_tokens,
                    self.settings.chunk_overlap_tokens,
                )
                await self.vectors.ensure_collection()
                size = self.settings.embedding_batch_size
                for start in range(0, len(chunks), size):
                    batch = chunks[start : start + size]
                    embeddings = await self.embedder.embed([chunk.text for chunk in batch])
                    await self.vectors.upsert(
                        str(paper_id),
                        paper.title,
                        batch,
                        embeddings,
                        self.embedder.model,
                    )
                await self._set_status(paper_id, "ready")
            except Exception as exc:
                logger.exception("Ingestion failed for paper %s", paper_id)
                error = (
                    str(exc)
                    if isinstance(exc, (ExtractionError, IndexingError))
                    else ("Ingestion failed unexpectedly. Check the backend logs and retry.")
                )
                try:
                    await self.vectors.delete_paper(str(paper_id))
                except IndexingError:
                    error += " Partial vector cleanup unconfirmed; retry after Qdrant recovers."
                await self._set_status(paper_id, "failed", error)
            # Cancellation deliberately leaves processing persisted. Startup requeues it,
            # and the next attempt removes any partially acknowledged vectors first.
