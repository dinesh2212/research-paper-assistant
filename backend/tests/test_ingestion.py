"""Ingestion lifecycle with real SQLite/PDF processing and simulated HTTP services."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pymupdf
import pytest
from research_paper_assistant.core.config import Settings
from research_paper_assistant.db import Database
from research_paper_assistant.main import create_app
from research_paper_assistant.models import IngestionJob, Paper
from research_paper_assistant.services.indexing import IndexingError, OllamaEmbedder, QdrantStore
from research_paper_assistant.services.ingestion import IngestionWorker
from research_paper_assistant.services.papers import PaperService
from sqlalchemy import select

from .conftest import StubHealthChecker
from .test_papers_api import make_pdf


class LocalServices:
    """Model the remote contracts, including acknowledged writes and failures."""

    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.collection: dict[str, Any] | None = None
        self.requests: list[httpx.Request] = []
        self.embed_calls = 0
        self.fail_embedding_call: int | None = None
        self.fail_cleanup_on_embedding_error = False
        self.fail_delete = False
        self.fail_upsert = False
        self.upsert_status = "completed"
        self.embedding_started = asyncio.Event()
        self.embedding_gate = asyncio.Event()
        self.embedding_gate.set()

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/api/embed":
            self.embed_calls += 1
            self.embedding_started.set()
            await self.embedding_gate.wait()
            if self.embed_calls == self.fail_embedding_call:
                self.fail_delete = self.fail_cleanup_on_embedding_error
                return httpx.Response(503, json={"error": "Model unavailable"})
            assert body["truncate"] is False
            return httpx.Response(200, json={"embeddings": [[0.6, 0.8] for _ in body["input"]]})
        if request.url.path.endswith("/points/delete"):
            if self.fail_delete:
                return httpx.Response(503)
            if self.collection is None:
                return httpx.Response(404)
            paper_id = body["filter"]["must"][0]["match"]["value"]
            self.points = {
                key: point
                for key, point in self.points.items()
                if point["payload"]["paper_id"] != paper_id
            }
        elif request.url.path.endswith("/points"):
            if self.fail_upsert:
                return httpx.Response(503)
            for point in body["points"]:
                self.points[point["id"]] = point
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "status": self.upsert_status,
                    },
                },
            )
        elif request.url.path.endswith("/index"):
            assert body["field_name"] == "paper_id"
        elif request.method == "GET":
            if self.collection is None:
                return httpx.Response(404)
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "config": {"params": self.collection},
                    },
                },
            )
        elif request.method == "PUT":
            self.collection = body
            return httpx.Response(200, json={"status": "ok", "result": True})
        else:
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")
        return httpx.Response(200, json={"status": "ok", "result": {"status": "completed"}})


@dataclass
class Runtime:
    api: httpx.AsyncClient
    database: Database
    papers: PaperService
    worker: IngestionWorker
    remote: LocalServices

    async def upload(self, content: bytes | None = None) -> UUID:
        response = await self.api.post(
            "/api/v1/papers",
            files={
                "file": (
                    "evidence.pdf",
                    make_pdf() if content is None else content,
                    "application/pdf",
                ),
            },
        )
        assert response.status_code == 201
        assert response.json()["status"] == "pending"
        return UUID(response.json()["id"])

    async def assert_state(self, paper_id: UUID, state: str) -> dict[str, Any]:
        response = await self.api.get(f"/api/v1/papers/{paper_id}")
        data: dict[str, Any] = response.json()
        assert data["status"] == state
        async with self.database.session() as session:
            jobs = list(
                await session.scalars(
                    select(IngestionJob).where(
                        IngestionJob.paper_id == str(paper_id),
                    )
                )
            )
        assert jobs
        assert all(
            job.status == state and job.error_message == data["error_message"] for job in jobs
        )
        return data


@pytest.fixture
async def runtime(settings: Settings, database: Database) -> AsyncIterator[Runtime]:
    settings.embedding_dimensions = 2
    settings.embedding_batch_size = 1
    settings.ingestion_poll_seconds = 0.01
    remote = LocalServices()
    async with httpx.AsyncClient(transport=httpx.MockTransport(remote.handle)) as client:
        store = QdrantStore(client, settings.qdrant_url, settings.qdrant_collection, 2)
        papers = PaperService(database, settings.paper_storage_path, 1024 * 1024, store)
        await papers.initialize()
        worker = IngestionWorker(
            database,
            papers,
            OllamaEmbedder(client, settings.ollama_url, "test-embedding", 2),
            store,
            settings,
        )
        app = create_app(settings, StubHealthChecker(), papers)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            try:
                yield Runtime(api, database, papers, worker, remote)
            finally:
                await worker.stop()


def two_page_pdf() -> bytes:
    with pymupdf.open() as doc:  # type: ignore[no-untyped-call]
        doc.new_page().insert_text((72, 72), "The first experiment supports conclusion alpha.")
        doc.new_page().insert_text((72, 72), "The second experiment supports conclusion beta.")
        return bytes(doc.tobytes())


async def test_ingestion_stores_citations_and_deletion_is_scoped(runtime: Runtime) -> None:
    paper_id = await runtime.upload(two_page_pdf())
    other_id = await runtime.upload(make_pdf("Unrelated paper"))
    await runtime.worker.process(paper_id)
    await runtime.worker.process(other_id)
    await runtime.assert_state(paper_id, "ready")
    points = [
        point
        for point in runtime.remote.points.values()
        if point["payload"]["paper_id"] == str(paper_id)
    ]
    assert {point["payload"]["page_number"] for point in points} == {1, 2}
    assert all(point["payload"]["chunk_id"] == point["id"] for point in points)
    assert all(point["payload"]["paper_title"] == "evidence" for point in points)
    assert all(point["payload"]["embedding_model"] == "test-embedding" for point in points)
    assert "alpha" in points[0]["payload"]["text"]
    assert "beta" in points[1]["payload"]["text"]
    assert runtime.remote.embed_calls == 3  # Batch size one; page text never concatenated.
    response = await runtime.api.delete(f"/api/v1/papers/{paper_id}")
    assert response.status_code == 204
    assert len(runtime.remote.points) == 1
    assert next(iter(runtime.remote.points.values()))["payload"]["paper_id"] == str(other_id)
    async with runtime.database.session() as session:
        assert await session.get(Paper, str(paper_id)) is None
        assert (
            await session.scalar(
                select(IngestionJob.id).where(
                    IngestionJob.paper_id == str(paper_id),
                )
            )
            is None
        )


async def test_blank_pdf_fails_without_embedding(runtime: Runtime) -> None:
    paper_id = await runtime.upload(make_pdf(""))
    await runtime.worker.process(paper_id)
    data = await runtime.assert_state(paper_id, "failed")
    assert "OCR" in data["error_message"]
    assert runtime.remote.embed_calls == 0


async def test_partial_failure_cleanup_and_retry(runtime: Runtime) -> None:
    runtime.remote.fail_embedding_call = 2
    paper_id = await runtime.upload(two_page_pdf())
    await runtime.worker.process(paper_id)
    data = await runtime.assert_state(paper_id, "failed")
    assert "Ollama" in data["error_message"]
    assert not runtime.remote.points
    response = await runtime.api.post(f"/api/v1/papers/{paper_id}/retry")
    assert response.status_code == 202
    await runtime.assert_state(paper_id, "pending")
    await runtime.worker.process(paper_id)
    assert (await runtime.assert_state(paper_id, "ready"))["error_message"] is None
    assert len(runtime.remote.points) == 2
    assert (await runtime.api.post(f"/api/v1/papers/{paper_id}/retry")).status_code == 409
    assert (await runtime.api.post(f"/api/v1/papers/{uuid4()}/retry")).status_code == 404


async def test_qdrant_cleanup_failure_preserves_paper_and_file(runtime: Runtime) -> None:
    paper_id = await runtime.upload()
    await runtime.worker.process(paper_id)
    runtime.remote.fail_delete = True
    response = await runtime.api.delete(f"/api/v1/papers/{paper_id}")
    assert response.status_code == 500
    assert "paper retained" in response.json()["detail"]
    paper = await runtime.papers.get_paper(paper_id)
    assert paper.status == "failed"
    assert runtime.papers.stored_path(paper).is_file()
    assert runtime.remote.points
    runtime.remote.fail_delete = False
    assert (await runtime.api.delete(f"/api/v1/papers/{paper_id}")).status_code == 204


async def test_failed_partial_cleanup_is_reported_and_retry_removes_stale_points(
    runtime: Runtime,
) -> None:
    runtime.remote.fail_embedding_call = 2
    runtime.remote.fail_cleanup_on_embedding_error = True
    paper_id = await runtime.upload(two_page_pdf())
    await runtime.worker.process(paper_id)
    data = await runtime.assert_state(paper_id, "failed")
    assert "cleanup unconfirmed" in data["error_message"]
    assert len(runtime.remote.points) == 1
    runtime.remote.fail_delete = False
    assert (await runtime.api.post(f"/api/v1/papers/{paper_id}/retry")).status_code == 202
    await runtime.worker.process(paper_id)
    await runtime.assert_state(paper_id, "ready")
    assert len(runtime.remote.points) == 2


async def test_failed_paper_does_not_block_following_queue_job(runtime: Runtime) -> None:
    blank_id = await runtime.upload(make_pdf(""))
    text_id = await runtime.upload(make_pdf("A usable paper after a failed paper"))
    await runtime.worker.start()
    async with asyncio.timeout(3):
        while (await runtime.papers.get_paper(text_id)).status != "ready":  # noqa: ASYNC110
            await asyncio.sleep(0.01)
    await runtime.assert_state(blank_id, "failed")
    await runtime.assert_state(text_id, "ready")


async def test_file_cleanup_failure_marks_paper_failed_after_vectors_removed(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = await runtime.upload()
    await runtime.worker.process(paper_id)
    original = Path.unlink

    def fail(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".pdf":
            raise PermissionError("locked")
        original(path, missing_ok)

    monkeypatch.setattr(Path, "unlink", fail)
    assert (await runtime.api.delete(f"/api/v1/papers/{paper_id}")).status_code == 500
    assert not runtime.remote.points
    assert (
        "Deletion incomplete" in (await runtime.assert_state(paper_id, "failed"))["error_message"]
    )


async def test_restart_recovers_interrupted_jobs_without_duplicate_points(runtime: Runtime) -> None:
    paper_id = await runtime.upload(two_page_pdf())
    # Interrupt after the first batch has been stored and the second has started.
    original_handle = runtime.remote.handle
    second_started = asyncio.Event()

    async def gate_second(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed" and runtime.remote.embed_calls == 1:
            runtime.remote.embedding_gate.clear()
            runtime.remote.embedding_started.clear()
            second_started.set()
        return await original_handle(request)

    runtime.worker.embedder.client = httpx.AsyncClient(transport=httpx.MockTransport(gate_second))
    try:
        await runtime.worker.start()
        await asyncio.wait_for(second_started.wait(), timeout=3)
        assert len(runtime.remote.points) == 1
        await runtime.worker.stop()
        await runtime.assert_state(paper_id, "processing")
        runtime.remote.embedding_gate.set()
        await runtime.worker.start()
        async with asyncio.timeout(3):
            while (await runtime.papers.get_paper(paper_id)).status != "ready":  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        await runtime.assert_state(paper_id, "ready")
        assert len(runtime.remote.points) == 2
    finally:
        await runtime.worker.stop()
        await runtime.worker.embedder.client.aclose()


async def test_delete_waits_for_indexing_and_leaves_no_orphan_points(runtime: Runtime) -> None:
    paper_id = await runtime.upload()
    runtime.remote.embedding_gate.clear()
    process = asyncio.create_task(runtime.worker.process(paper_id))
    deletion: asyncio.Task[httpx.Response] | None = None
    try:
        await asyncio.wait_for(runtime.remote.embedding_started.wait(), timeout=3)
        await runtime.assert_state(paper_id, "processing")
        deletion = asyncio.create_task(runtime.api.delete(f"/api/v1/papers/{paper_id}"))
        await asyncio.sleep(0.01)
        assert not deletion.done()
        runtime.remote.embedding_gate.set()
        await asyncio.wait_for(process, timeout=3)
        assert (await asyncio.wait_for(deletion, timeout=3)).status_code == 204
        assert not runtime.remote.points
        # A previously queued reference cannot resurrect a deleted paper.
        await runtime.worker.process(paper_id)
        assert not runtime.remote.points
    finally:
        runtime.remote.embedding_gate.set()
        await process
        if deletion is not None:
            await deletion


@pytest.mark.parametrize("mode", ["outage", "unconfirmed", "incompatible"])
async def test_bad_qdrant_results_never_mark_ready(runtime: Runtime, mode: str) -> None:
    if mode == "outage":
        runtime.remote.fail_upsert = True
    elif mode == "unconfirmed":
        runtime.remote.upsert_status = "acknowledged"
    else:
        runtime.remote.collection = {"vectors": {"size": 3, "distance": "Cosine"}}
    paper_id = await runtime.upload()
    await runtime.worker.process(paper_id)
    await runtime.assert_state(paper_id, "failed")
    assert not runtime.remote.points


@pytest.mark.parametrize(
    "vectors",
    [[], [[1.0]], [[0.0, 0.0]], [[1.0, 2.0], [1.0, 2.0]], [["NaN", 1.0]], [["Infinity", 1.0]]],
)
async def test_invalid_embeddings_rejected(vectors: list[list[Any]]) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": vectors}),
        )
    ) as client:
        embedder = OllamaEmbedder(client, "http://ollama", "test", 2)
        with pytest.raises(IndexingError):
            await embedder.embed(["source evidence"])


async def test_real_app_lifespan_automatically_ingests_upload(settings: Settings) -> None:
    settings.embedding_dimensions = 2
    settings.ingestion_poll_seconds = 0.01
    remote = LocalServices()
    app = create_app(
        settings, StubHealthChecker(), indexing_transport=httpx.MockTransport(remote.handle)
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api,
    ):
        response = await api.post(
            "/api/v1/papers",
            files={
                "file": ("lifecycle.pdf", make_pdf(), "application/pdf"),
            },
        )
        assert response.status_code == 201
        paper_id = response.json()["id"]
        async with asyncio.timeout(3):
            while True:
                if (await api.get(f"/api/v1/papers/{paper_id}")).json()["status"] == "ready":
                    break
                await asyncio.sleep(0.01)
        assert len(remote.points) == 1
    assert not any(task.get_name() == "paper-ingestion" for task in asyncio.all_tasks())
