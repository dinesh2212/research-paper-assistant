"""Paper library routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from research_paper_assistant.api.dependencies import get_paper_service
from research_paper_assistant.schemas.papers import PaperDetail, PaperSummary
from research_paper_assistant.services.papers import (
    DuplicatePaperError,
    InvalidPdfError,
    PaperCleanupError,
    PaperNotFoundError,
    PaperService,
    PaperStateError,
    PdfTooLargeError,
)

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("", response_model=list[PaperSummary])
async def list_papers(
    paper_service: Annotated[PaperService, Depends(get_paper_service)],
) -> list[PaperSummary]:
    """Return the paper library, newest first."""

    papers = await paper_service.list_papers()
    return [PaperSummary.model_validate(paper) for paper in papers]


@router.post("", response_model=PaperDetail, status_code=status.HTTP_201_CREATED)
async def upload_paper(
    file: Annotated[UploadFile, File(description="Text-based research paper PDF")],
    paper_service: Annotated[PaperService, Depends(get_paper_service)],
) -> PaperDetail:
    """Validate and store one research-paper PDF."""

    try:
        paper = await paper_service.create_paper(file)
    except PdfTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except DuplicatePaperError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except InvalidPdfError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return PaperDetail.model_validate(paper)


@router.post("/{paper_id}/retry", response_model=PaperDetail, status_code=status.HTTP_202_ACCEPTED)
async def retry_ingestion(
    paper_id: UUID,
    paper_service: Annotated[PaperService, Depends(get_paper_service)],
) -> PaperDetail:
    """Requeue a failed ingestion after its underlying problem is resolved."""

    try:
        paper = await paper_service.retry_ingestion(paper_id)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PaperStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return PaperDetail.model_validate(paper)


@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper(
    paper_id: UUID,
    paper_service: Annotated[PaperService, Depends(get_paper_service)],
) -> PaperDetail:
    """Return detailed metadata for one paper."""

    try:
        paper = await paper_service.get_paper(paper_id)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return PaperDetail.model_validate(paper)


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(
    paper_id: UUID,
    paper_service: Annotated[PaperService, Depends(get_paper_service)],
) -> Response:
    """Delete one paper and its locally stored PDF."""

    try:
        await paper_service.delete_paper(paper_id)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PaperCleanupError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
