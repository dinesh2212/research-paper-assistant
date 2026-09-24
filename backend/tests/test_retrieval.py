"""Retrieval contract, isolation, citation validation, and selection tests."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4, uuid5

import httpx
import pytest
from research_paper_assistant.core.config import Settings
from research_paper_assistant.db import Database
from research_paper_assistant.main import create_app
from research_paper_assistant.models import Paper
from research_paper_assistant.schemas.retrieval import ScoredChunk
from research_paper_assistant.services.indexing import OllamaEmbedder, QdrantStore
from research_paper_assistant.services.papers import PaperService
from research_paper_assistant.services.retrieval import RetrievalService, select_diverse
from sqlalchemy import update

from .conftest import StubHealthChecker
from .test_papers_api import make_pdf

ENDPOINT = "/api/v1/debug/retrieval"


def point(
    paper_id: str,
    index: int = 0,
    score: float = 0.9,
    text: str = "Source evidence",
    vector: list[float] | None = None,
) -> dict[str, Any]:
    chunk_id = str(uuid5(UUID(paper_id), f"1:{index}"))
    return {
        "id": chunk_id,
        "score": score,
        "vector": [1.0, 0.0] if vector is None else vector,
        "payload": {
            "paper_id": paper_id,
            "paper_title": "Untrusted/stale vector-store title",
            "page_number": 1,
            "chunk_id": chunk_id,
            "chunk_index": index,
            "text": text,
            "embedding_model": "test-embedding",
        },
    }


class SearchServices:
    def __init__(self) -> None:
        self.points: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []
        self.failure = ""
        self.ignore_filter = False
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.gate.set()

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            self.started.set()
            await self.gate.wait()
            if self.failure == "ollama":
                return httpx.Response(503)
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        if request.url.path.endswith("/points/delete"):
            paper_id = body["filter"]["must"][0]["match"]["value"]
            self.points = [item for item in self.points if item["payload"]["paper_id"] != paper_id]
            return httpx.Response(200, json={"status": "ok", "result": {"status": "completed"}})
        assert request.url.path.endswith("/points/query")
        if self.failure in {"qdrant", "missing_collection"}:
            return httpx.Response(404 if self.failure == "missing_collection" else 503)
        if self.failure == "malformed":
            return httpx.Response(200, json={"status": "ok", "result": {}})
        ids = body["filter"]["must"][0]["match"]["any"]
        matches = [
            item for item in self.points if self.ignore_filter or item["payload"]["paper_id"] in ids
        ]
        return httpx.Response(200, json={"status": "ok", "result": {"points": matches}})


@dataclass
class SearchRuntime:
    api: httpx.AsyncClient
    database: Database
    remote: SearchServices
    papers: PaperService
    service: RetrievalService

    async def add_paper(self, status: str = "ready") -> str:
        response = await self.api.post(
            "/api/v1/papers",
            files={
                "file": ("Original title.pdf", make_pdf(str(uuid4())), "application/pdf"),
            },
        )
        assert response.status_code == 201
        paper_id: str = response.json()["id"]
        async with self.database.session() as session:
            await session.execute(update(Paper).where(Paper.id == paper_id).values(status=status))
            await session.commit()
        return paper_id

    async def search(self, ids: list[str], **options: Any) -> httpx.Response:
        return await self.api.post(
            ENDPOINT,
            json={
                "question": "What evidence supports this conclusion?",
                "paper_ids": ids,
                **options,
            },
        )


@pytest.fixture
async def search_runtime(settings: Settings, database: Database) -> AsyncIterator[SearchRuntime]:
    remote = SearchServices()
    async with httpx.AsyncClient(transport=httpx.MockTransport(remote.handle)) as http:
        store = QdrantStore(http, settings.qdrant_url, settings.qdrant_collection, 2)
        papers = PaperService(database, settings.paper_storage_path, 1024 * 1024, store)
        await papers.initialize()
        service = RetrievalService(
            papers, OllamaEmbedder(http, settings.ollama_url, "test-embedding", 2), store
        )
        app = create_app(settings, StubHealthChecker(), papers, retrieval_service=service)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            yield SearchRuntime(api, database, remote, papers, service)


async def test_timeout_releases_paper_locks(search_runtime: SearchRuntime) -> None:
    paper_id = await search_runtime.add_paper()
    search_runtime.remote.gate.clear()
    search_runtime.service.timeout_seconds = 0.02
    response = await asyncio.wait_for(search_runtime.search([paper_id]), timeout=1)
    assert response.status_code == 503
    assert "timed out" in response.json()["detail"]
    assert not search_runtime.papers.paper_lock(UUID(paper_id)).locked()
    search_runtime.remote.gate.set()
    search_runtime.service.timeout_seconds = 30
    assert (await search_runtime.search([paper_id])).status_code == 200


async def test_selected_papers_only_and_citation_contract(search_runtime: SearchRuntime) -> None:
    first, second, excluded = [await search_runtime.add_paper() for _ in range(3)]
    search_runtime.remote.points = [
        point(first, text="First evidence"),
        point(second, text="Second evidence"),
        point(excluded),
    ]
    response = await search_runtime.search([first, second])
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["candidate_count"] == 2
    assert {source["paper_id"] for source in result["sources"]} == {first, second}
    assert [source["source_id"] for source in result["sources"]] == [1, 2]
    for source in result["sources"]:
        assert source["paper_title"] == "Original title"
        assert source["page_number"] == 1
        assert source["chunk_id"] == str(uuid5(UUID(source["paper_id"]), "1:0"))
        assert "vector" not in source
        assert source["score"] == 0.9
    embed, query = search_runtime.remote.requests
    prompt = json.loads(embed.content)
    assert prompt["input"][0].startswith("Instruct: ")
    assert "\nQuery: What evidence" in prompt["input"][0]
    assert prompt["truncate"] is False
    body = json.loads(query.content)
    assert body["limit"] == 20
    assert body["filter"]["must"] == [
        {"key": "paper_id", "match": {"any": [first, second]}},
        {"key": "embedding_model", "match": {"value": "test-embedding"}},
    ]
    assert all(request.url.path != "/api/generate" for request in search_runtime.remote.requests)


@pytest.mark.parametrize("status", ["pending", "processing", "failed"])
async def test_unready_selection_rejected_before_network(
    search_runtime: SearchRuntime, status: str
) -> None:
    ready = await search_runtime.add_paper()
    unready = await search_runtime.add_paper(status)
    response = await search_runtime.search([ready, unready])
    assert response.status_code == 409
    assert status in response.json()["detail"]
    assert not search_runtime.remote.requests


async def test_missing_and_deleted_papers_rejected(search_runtime: SearchRuntime) -> None:
    assert (await search_runtime.search([str(uuid4())])).status_code == 404
    paper_id = await search_runtime.add_paper()
    assert (await search_runtime.api.delete(f"/api/v1/papers/{paper_id}")).status_code == 204
    assert (await search_runtime.search([paper_id])).status_code == 404


@pytest.mark.parametrize(
    "options",
    [
        {"question": "  \n\t"},
        {"question": "x" * 2001},
        {"paper_ids": []},
        {"paper_ids": ["invalid"]},
        {"paper_ids": [str(uuid4()) for _ in range(21)]},
        {"limit": 0},
        {"limit": 9},
        {"score_threshold": 1.1},
        {"unknown": "value"},
    ],
)
async def test_invalid_requests_rejected(
    search_runtime: SearchRuntime, options: dict[str, Any]
) -> None:
    response = await search_runtime.search([str(uuid4())], **options)
    assert response.status_code == 422
    assert not search_runtime.remote.requests


async def test_duplicate_ids_rejected(search_runtime: SearchRuntime) -> None:
    paper_id = str(uuid4())
    assert (await search_runtime.search([paper_id, paper_id])).status_code == 422


async def test_empty_and_threshold_filtered_results(search_runtime: SearchRuntime) -> None:
    paper_id = await search_runtime.add_paper()
    result = (await search_runtime.search([paper_id])).json()
    assert result["sources"] == [] and result["candidate_count"] == 0
    search_runtime.remote.points = [point(paper_id, score=0.4)]
    result = (await search_runtime.search([paper_id], score_threshold=0.6)).json()
    assert result["sources"] == [] and result["candidate_count"] == 0
    assert json.loads(search_runtime.remote.requests[-1].content)["score_threshold"] == 0.6


@pytest.mark.parametrize("failure", ["ollama", "qdrant", "missing_collection", "malformed"])
async def test_dependency_failures_return_503(search_runtime: SearchRuntime, failure: str) -> None:
    paper_id = await search_runtime.add_paper()
    search_runtime.remote.failure = failure
    response = await search_runtime.search([paper_id])
    assert response.status_code == 503
    assert "Retrieval unavailable" in response.json()["detail"]


@pytest.mark.parametrize(
    "corruption",
    [
        "other_paper",
        "page",
        "chunk_id",
        "index",
        "model",
        "text",
        "dimension",
        "zero",
        "nonfinite",
        "score",
        "duplicates",
    ],
)
async def test_invalid_or_out_of_scope_sources_never_escape(
    search_runtime: SearchRuntime,
    corruption: str,
) -> None:
    paper_id = await search_runtime.add_paper()
    item = point(paper_id)
    payload = item["payload"]
    if corruption == "other_paper":
        payload["paper_id"] = str(uuid4())
    elif corruption == "page":
        payload["page_number"] = 2
    elif corruption == "chunk_id":
        payload["chunk_id"] = str(uuid4())
    elif corruption == "index":
        payload["chunk_index"] = 2
    elif corruption == "model":
        payload["embedding_model"] = "wrong-model"
    elif corruption == "text":
        payload["text"] = "  "
    elif corruption == "dimension":
        item["vector"] = [1.0]
    elif corruption == "zero":
        item["vector"] = [0.0, 0.0]
    elif corruption == "nonfinite":
        item["vector"] = ["NaN", 1.0]
    elif corruption == "score":
        item["score"] = "Infinity"
    search_runtime.remote.points = [item, item] if corruption == "duplicates" else [item]
    search_runtime.remote.ignore_filter = True
    assert (await search_runtime.search([paper_id])).status_code == 503


def test_mmr_keeps_relevance_and_prefers_complementary_passages() -> None:
    paper_id = str(uuid4())
    candidates = [
        ScoredChunk.model_validate(item)
        for item in [
            point(paper_id, 0, 0.95, "First finding", [1.0, 0.0]),
            point(paper_id, 1, 0.94, "Nearly identical finding", [0.999, 0.01]),
            point(paper_id, 2, 0.85, "Complementary evidence", [0.6, 0.8]),
            point(paper_id, 3, 0.9, "First finding", [1.0, 0.0]),
        ]
    ]
    selected = select_diverse(candidates, 2)
    assert [item.payload.chunk_index for item in selected] == [0, 2]
    assert len(select_diverse(candidates, 8)) == 3
    assert select_diverse(list(reversed(candidates)), 2) == selected


async def test_default_and_requested_result_limits(search_runtime: SearchRuntime) -> None:
    paper_id = await search_runtime.add_paper()
    search_runtime.remote.points = [
        point(paper_id, index, text=f"Evidence {index}") for index in range(20)
    ]
    result = (await search_runtime.search([paper_id])).json()
    assert result["candidate_count"] == 20
    assert len(result["sources"]) == 8
    assert len((await search_runtime.search([paper_id], limit=6)).json()["sources"]) == 6


@pytest.mark.parametrize(
    ("environment", "enabled", "present"),
    [
        ("production", False, False),
        ("production", True, True),
        ("development", False, True),
    ],
)
async def test_debug_route_visibility(
    settings: Settings, environment: Any, enabled: bool, present: bool
) -> None:
    settings.app_env = environment
    settings.retrieval_debug_enabled = enabled
    app = create_app(settings, StubHealthChecker())
    assert (ENDPOINT in app.openapi()["paths"]) == present
    if not present:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            assert (await api.post(ENDPOINT, json={})).status_code == 404


async def test_deletion_cannot_invalidate_sources_during_retrieval(
    search_runtime: SearchRuntime,
) -> None:
    paper_id = await search_runtime.add_paper()
    search_runtime.remote.points = [point(paper_id)]
    search_runtime.remote.gate.clear()
    searching = asyncio.create_task(search_runtime.search([paper_id]))
    deletion: asyncio.Task[httpx.Response] | None = None
    try:
        await asyncio.wait_for(search_runtime.remote.started.wait(), timeout=3)
        deletion = asyncio.create_task(search_runtime.api.delete(f"/api/v1/papers/{paper_id}"))
        await asyncio.sleep(0.01)
        assert not deletion.done()
        search_runtime.remote.gate.set()
        result = await asyncio.wait_for(searching, timeout=3)
        assert result.status_code == 200
        assert len(result.json()["sources"]) == 1
        assert (await asyncio.wait_for(deletion, timeout=3)).status_code == 204
        assert (await search_runtime.search([paper_id])).status_code == 404
    finally:
        search_runtime.remote.gate.set()
        await searching
        if deletion is not None:
            await deletion
