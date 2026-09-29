"""Answer grounding contracts, Ollama integration, and production API behavior."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from research_paper_assistant.core.config import Settings
from research_paper_assistant.main import create_app
from research_paper_assistant.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedSource,
)
from research_paper_assistant.services.answers import (
    INSUFFICIENT_EVIDENCE,
    AnswerService,
    GenerationError,
    build_answer_chain,
)
from research_paper_assistant.services.indexing import IndexingError
from research_paper_assistant.services.papers import PaperNotFoundError, PaperStateError
from research_paper_assistant.services.retrieval import RetrievalService


@pytest.fixture
def evidence() -> RetrievalResponse:
    paper_id = uuid4()
    return RetrievalResponse(
        question="What changed?",
        paper_ids=[paper_id],
        candidate_count=1,
        sources=[
            RetrievedSource(
                source_id=1,
                paper_id=paper_id,
                paper_title="Battery study",
                page_number=3,
                chunk_id=uuid4(),
                excerpt="Capacity decreased by ten percent.",
                score=0.8,
            )
        ],
    )


def request_for(evidence: RetrievalResponse) -> RetrievalRequest:
    return RetrievalRequest(question=evidence.question, paper_ids=evidence.paper_ids)


def service_for(evidence: RetrievalResponse, output: Any) -> AnswerService:
    retrieval = AsyncMock(spec=RetrievalService)
    retrieval.retrieve.return_value = evidence

    async def generate(_: Any) -> Any:
        return output

    return AnswerService(retrieval, RunnableLambda(generate))


async def test_actual_langchain_ollama_chain_and_citation_rendering(
    evidence: RetrievalResponse,
) -> None:
    calls: list[dict[str, Any]] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "insufficient_evidence": False,
                            "statements": [{"text": "Capacity fell by 10%.", "source_ids": [1]}],
                        }
                    ),
                },
                "done": True,
                "done_reason": "stop",
            },
        )

    retrieval = AsyncMock(spec=RetrievalService)
    retrieval.retrieve.return_value = evidence
    settings = Settings(_env_file=None, ollama_generation_model="test-model")
    service = AnswerService(retrieval, build_answer_chain(settings, httpx.MockTransport(ollama)))
    result = await service.answer(request_for(evidence))
    assert result.answer == "Capacity fell by 10%. [1]"
    assert not result.insufficient_evidence
    assert result.sources == evidence.sources
    retrieval.retrieve.assert_awaited_once_with(request_for(evidence))
    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["format"]["type"] == "object"
    assert "maxLength" not in calls[0]["format"]["$defs"]["CitedStatement"]["properties"]["text"]
    assert calls[0]["think"] is False
    assert calls[0]["options"]["num_ctx"] == 16384
    prompt = calls[0]["messages"][-1]["content"]
    assert evidence.question in prompt
    assert evidence.sources[0].excerpt in prompt
    assert '"source_id": 1' in prompt


async def test_no_sources_skips_model(evidence: RetrievalResponse) -> None:
    evidence.sources = []
    service = service_for(evidence, None)
    service.chain = AsyncMock()
    result = await service.answer(request_for(evidence))
    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.insufficient_evidence
    service.chain.ainvoke.assert_not_awaited()


async def test_model_can_abstain_with_retrieved_sources(evidence: RetrievalResponse) -> None:
    service = service_for(evidence, {"insufficient_evidence": True, "statements": []})
    result = await service.answer(request_for(evidence))
    assert result.insufficient_evidence
    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.sources == evidence.sources


@pytest.mark.parametrize(
    "output",
    [
        None,
        "invalid JSON",
        {"insufficient_evidence": False, "statements": []},
        {"insufficient_evidence": False, "statements": [{"text": "Claim", "source_ids": []}]},
        {"insufficient_evidence": False, "statements": [{"text": "Claim", "source_ids": [99]}]},
        {"insufficient_evidence": False, "statements": [{"text": "Claim", "source_ids": [True]}]},
        {"insufficient_evidence": False, "statements": [{"text": "x" * 4001, "source_ids": [1]}]},
        {"insufficient_evidence": False, "statements": [{"text": "Claim", "source_ids": [1, 1]}]},
        {"insufficient_evidence": False, "statements": [{"text": "Claim [99]", "source_ids": [1]}]},
        {"insufficient_evidence": True, "statements": [{"text": "Claim", "source_ids": [1]}]},
    ],
)
async def test_invalid_or_uncited_answers_are_rejected(
    evidence: RetrievalResponse, output: Any
) -> None:
    with pytest.raises(GenerationError, match="valid cited answer"):
        await service_for(evidence, output).answer(request_for(evidence))


async def test_generation_timeout_releases_slot(evidence: RetrievalResponse) -> None:
    service = service_for(evidence, {})

    async def slow(_: Any) -> Any:
        await asyncio.sleep(10)

    service.chain = RunnableLambda(slow)
    service.timeout_seconds = 0.01
    with pytest.raises(GenerationError, match="timed out"):
        await service.answer(request_for(evidence))
    assert not service._generation_lock.locked()


async def test_model_connection_failure_is_reported(evidence: RetrievalResponse) -> None:
    service = service_for(evidence, {})

    async def unavailable(_: Any) -> Any:
        raise ConnectionError("Ollama is offline")

    service.chain = RunnableLambda(unavailable)
    with pytest.raises(GenerationError, match="Check Ollama"):
        await service.answer(request_for(evidence))
    assert not service._generation_lock.locked()


async def test_context_budget_matches_returned_excerpts(evidence: RetrievalResponse) -> None:
    evidence.sources[0].excerpt = "x" * 30000
    service = service_for(evidence, {})

    async def inspect_prompt(value: Any) -> Any:
        sources = json.loads(value["sources"])
        assert len(sources[0]["excerpt"]) == 24000
        return {"insufficient_evidence": True, "statements": []}

    service.chain = RunnableLambda(inspect_prompt)
    result = await service.answer(request_for(evidence))
    assert result.sources[0].excerpt == "x" * 24000
    assert len(evidence.sources[0].excerpt) == 30000


async def test_answer_route_available_in_production_and_validates_selection(
    evidence: RetrievalResponse,
) -> None:
    service = service_for(evidence, {"insufficient_evidence": True, "statements": []})
    app = create_app(
        Settings(_env_file=None, app_env="production", retrieval_debug_enabled=False),
        answer_service=service,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/answers", json=request_for(evidence).model_dump(mode="json")
        )
        assert response.status_code == 200
        assert response.json()["insufficient_evidence"] is True
        assert (await client.post("/api/v1/debug/retrieval", json={})).status_code == 404
        response = await client.post("/api/v1/answers", json={"question": "Q", "paper_ids": []})
        assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (PaperNotFoundError("Missing paper"), 404),
        (PaperStateError("Paper processing"), 409),
        (IndexingError("Retrieval unavailable"), 503),
        (GenerationError("Generation unavailable"), 503),
        (TimeoutError(), 503),
    ],
)
async def test_answer_api_errors(
    evidence: RetrievalResponse, error: Exception, status: int
) -> None:
    service = AsyncMock(spec=AnswerService)
    service.answer.side_effect = error
    app = create_app(Settings(_env_file=None), answer_service=service)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/answers", json=request_for(evidence).model_dump(mode="json")
        )
    assert response.status_code == status
