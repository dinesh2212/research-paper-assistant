"""Paper metadata and local PDF storage operations."""

import asyncio
import hashlib
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4
from weakref import WeakValueDictionary

import pymupdf
from anyio import to_thread
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from research_paper_assistant.db import Database
from research_paper_assistant.models import IngestionJob, Paper
from research_paper_assistant.services.indexing import IndexingError, QdrantStore


class PaperError(Exception):
    """Base class for expected paper-library failures."""


class InvalidPdfError(PaperError):
    """The upload is not an accepted, readable PDF."""


class PdfTooLargeError(PaperError):
    """The upload exceeds the configured byte limit."""


class DuplicatePaperError(PaperError):
    """The same PDF content is already stored."""


class PaperNotFoundError(PaperError):
    """No paper exists for the requested identifier."""


class PaperCleanupError(PaperError):
    """A stored PDF could not be removed safely."""


class PaperStateError(PaperError):
    """The requested operation is not valid in the paper's current state."""


class PaperService:
    """Coordinate paper records with generated-name filesystem storage."""

    _chunk_size = 1024 * 1024
    _accepted_content_types = {"application/pdf", "application/x-pdf"}

    def __init__(
        self,
        database: Database,
        storage_root: Path,
        max_pdf_size_bytes: int,
        vector_store: QdrantStore | None = None,
    ) -> None:
        self._database = database
        self._storage_root = storage_root.expanduser().resolve()
        self._temporary_root = self._storage_root / ".tmp"
        self._max_pdf_size_bytes = max_pdf_size_bytes
        self._vector_store = vector_store
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def paper_lock(self, paper_id: UUID) -> asyncio.Lock:
        """Serialize indexing, retries, and deletion within this single worker process."""

        key = str(paper_id)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def initialize(self) -> None:
        """Create private storage directories if they do not exist."""

        await to_thread.run_sync(lambda: self._temporary_root.mkdir(parents=True, exist_ok=True))

    async def list_papers(self) -> list[Paper]:
        """Return all papers, newest first."""

        async with self._database.session() as session:
            result = await session.scalars(select(Paper).order_by(Paper.created_at.desc()))
            return list(result)

    async def get_paper(self, paper_id: UUID) -> Paper:
        """Return a paper or raise a domain-level not-found error."""

        async with self._database.session() as session:
            paper = await session.get(Paper, str(paper_id))
            if paper is None:
                raise PaperNotFoundError("Paper not found")
            return paper

    async def create_paper(self, upload: UploadFile) -> Paper:
        """Validate, store, and persist one PDF upload."""

        original_filename, title = self._validate_upload_metadata(upload)
        paper_id = uuid4()
        temporary_path = self._temporary_root / f"{paper_id}.upload"
        stored_filename = f"{paper_id}.pdf"
        destination = self._storage_root / stored_filename

        try:
            digest, size_bytes = await to_thread.run_sync(
                self._copy_and_hash,
                upload.file,
                temporary_path,
            )
            await self._ensure_not_duplicate(digest)
            page_count = await to_thread.run_sync(self._validate_pdf_document, temporary_path)
            await to_thread.run_sync(temporary_path.replace, destination)

            paper = Paper(
                id=str(paper_id),
                title=title,
                original_filename=original_filename,
                stored_filename=stored_filename,
                sha256=digest,
                size_bytes=size_bytes,
                page_count=page_count,
                status="pending",
            )
            ingestion_job = IngestionJob(
                id=str(uuid4()),
                paper_id=str(paper_id),
                status="pending",
            )

            try:
                async with self._database.session() as session:
                    session.add(paper)
                    await session.flush()
                    session.add(ingestion_job)
                    await session.commit()
                    await session.refresh(paper)
            except IntegrityError as exc:
                await self._unlink_if_present(destination)
                if "papers.sha256" in str(exc.orig).lower():
                    raise DuplicatePaperError("This PDF is already in the library") from exc
                raise
            except SQLAlchemyError:
                await self._unlink_if_present(destination)
                raise

            return paper
        finally:
            await self._unlink_if_present(temporary_path)

    async def delete_paper(self, paper_id: UUID) -> None:
        """Remove vectors and PDF before metadata; preserve records on cleanup failure."""

        async with self.paper_lock(paper_id), self._database.session() as session:
            paper = await session.get(Paper, str(paper_id))
            if paper is None:
                raise PaperNotFoundError("Paper not found")

            target = self._safe_stored_path(paper.stored_filename)
            if self._vector_store is not None:
                # Persist a non-ready state BEFORE external changes. If the process
                # crashes or later file/DB cleanup fails, missing vectors cannot
                # leave behind a paper incorrectly advertised as searchable.
                paper.status = "failed"
                paper.error_message = "Deletion incomplete; retry deletion or ingestion."
                jobs = await session.scalars(
                    select(IngestionJob).where(IngestionJob.paper_id == str(paper_id))
                )
                for job in jobs:
                    job.status = "failed"
                    job.error_message = paper.error_message
                await session.commit()
                try:
                    await self._vector_store.delete_paper(str(paper_id))
                except IndexingError as exc:
                    raise PaperCleanupError(
                        "Unable to delete indexed vectors; paper retained. "
                        "Restore Qdrant and retry deletion."
                    ) from exc
            try:
                await to_thread.run_sync(target.unlink, True)
            except OSError as exc:
                raise PaperCleanupError("Unable to delete the stored PDF") from exc

            await session.delete(paper)
            await session.commit()

    async def retry_ingestion(self, paper_id: UUID) -> Paper:
        """Requeue a failed paper and its persisted ingestion job."""

        async with self.paper_lock(paper_id), self._database.session() as session:
            paper = await session.get(Paper, str(paper_id))
            if paper is None:
                raise PaperNotFoundError("Paper not found")
            if paper.status != "failed":
                raise PaperStateError("Only failed papers can be retried")
            paper.status = "pending"
            paper.error_message = None
            jobs = list(
                await session.scalars(
                    select(IngestionJob).where(IngestionJob.paper_id == str(paper_id))
                )
            )
            if not jobs:
                jobs = [IngestionJob(id=str(uuid4()), paper_id=str(paper_id))]
                session.add_all(jobs)
            for job in jobs:
                job.status = "pending"
                job.error_message = None
            await session.commit()
            await session.refresh(paper)
            return paper

    def stored_path(self, paper: Paper) -> Path:
        """Return the validated storage path for a paper."""

        return self._safe_stored_path(paper.stored_filename)

    def _validate_upload_metadata(self, upload: UploadFile) -> tuple[str, str]:
        filename = upload.filename
        if not filename:
            raise InvalidPdfError("A PDF filename is required")

        try:
            original_filename = Path(filename).name
        except ValueError as exc:
            raise InvalidPdfError("The PDF filename is invalid") from exc

        if len(original_filename) > 255:
            raise InvalidPdfError("The PDF filename is too long")
        if Path(original_filename).suffix.lower() != ".pdf":
            raise InvalidPdfError("The uploaded filename must end in .pdf")

        content_type = (upload.content_type or "").split(";", maxsplit=1)[0].lower()
        if content_type not in self._accepted_content_types:
            raise InvalidPdfError("The upload must use a PDF content type")

        title = Path(original_filename).stem.strip()
        if not title:
            raise InvalidPdfError("The PDF filename must contain a title")
        return original_filename, title[:500]

    def _copy_and_hash(self, source: BinaryIO, destination: Path) -> tuple[str, int]:
        source.seek(0)
        digest = hashlib.sha256()
        total = 0
        signature = b""

        try:
            with destination.open("xb") as output:
                while chunk := source.read(self._chunk_size):
                    total += len(chunk)
                    if total > self._max_pdf_size_bytes:
                        raise PdfTooLargeError(
                            f"PDF exceeds the {self._max_pdf_size_bytes}-byte upload limit"
                        )
                    if len(signature) < 5:
                        signature += chunk[: 5 - len(signature)]
                    digest.update(chunk)
                    output.write(chunk)
        except FileExistsError as exc:
            raise PaperCleanupError("A temporary upload path already exists") from exc

        if total == 0:
            raise InvalidPdfError("The uploaded PDF is empty")
        if signature != b"%PDF-":
            raise InvalidPdfError("The upload does not have a valid PDF signature")
        return digest.hexdigest(), total

    @staticmethod
    def _validate_pdf_document(path: Path) -> int:
        try:
            with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
                if document.needs_pass:
                    raise InvalidPdfError("Password-protected PDFs are not supported")
                page_count = int(document.page_count)
                if page_count < 1:
                    raise InvalidPdfError("The PDF contains no pages")
                document.load_page(0)
                return page_count
        except InvalidPdfError:
            raise
        except (pymupdf.EmptyFileError, pymupdf.FileDataError) as exc:
            raise InvalidPdfError("The PDF is corrupt or unreadable") from exc

    async def _ensure_not_duplicate(self, digest: str) -> None:
        async with self._database.session() as session:
            existing_id = await session.scalar(select(Paper.id).where(Paper.sha256 == digest))
            if existing_id is not None:
                raise DuplicatePaperError("This PDF is already in the library")

    def _safe_stored_path(self, stored_filename: str) -> Path:
        if Path(stored_filename).name != stored_filename:
            raise PaperCleanupError("Stored PDF path is invalid")
        target = self._storage_root / stored_filename
        if target.is_symlink() or target.parent != self._storage_root:
            raise PaperCleanupError("Stored PDF path is unsafe")
        return target

    @staticmethod
    async def _unlink_if_present(path: Path) -> None:
        try:
            await to_thread.run_sync(path.unlink, True)
        except OSError:
            # Cleanup is best-effort while preserving the original domain error.
            return
