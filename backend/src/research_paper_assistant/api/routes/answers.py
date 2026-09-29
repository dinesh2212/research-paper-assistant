"""Generate an answer from explicitly selected ready papers."""

import logging
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request

from research_paper_assistant.schemas.answers import AnswerResponse
from research_paper_assistant.schemas.retrieval import RetrievalRequest
from research_paper_assistant.services.answers import AnswerService, GenerationError
from research_paper_assistant.services.indexing import IndexingError
from research_paper_assistant.services.papers import PaperNotFoundError, PaperStateError

router = APIRouter(prefix="/answers", tags=["answers"])
logger = logging.getLogger(__name__)


async def get_answer_service(request: Request) -> AnswerService:
    service = getattr(request.app.state, "answer_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Answer service is not initialized")
    return cast(AnswerService, service)


@router.post("", response_model=AnswerResponse)
async def answer(
    request: RetrievalRequest,
    service: Annotated[AnswerService, Depends(get_answer_service)],
) -> AnswerResponse:
    try:
        return await service.answer(request)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PaperStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (IndexingError, GenerationError) as exc:
        logger.exception("Answer request failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="Retrieval timed out; try again.") from exc
