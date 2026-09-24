"""Opt-in development endpoint for evaluating retrieval without an LLM."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from research_paper_assistant.api.dependencies import get_retrieval_service
from research_paper_assistant.schemas.retrieval import RetrievalRequest, RetrievalResponse
from research_paper_assistant.services.indexing import IndexingError
from research_paper_assistant.services.papers import PaperNotFoundError, PaperStateError
from research_paper_assistant.services.retrieval import RetrievalService

router = APIRouter(prefix="/debug/retrieval", tags=["retrieval development"])
logger = logging.getLogger(__name__)


@router.post("", response_model=RetrievalResponse)
async def retrieve(
    request: RetrievalRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrievalResponse:
    """Find paper-filtered source excerpts; scores do not establish answerability."""

    try:
        return await service.retrieve(request)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PaperStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except IndexingError as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="Retrieval timed out; try again.") from exc
