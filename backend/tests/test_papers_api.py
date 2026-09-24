"""Paper upload, metadata, and deletion API tests."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pymupdf
import pytest
from research_paper_assistant.core.config import Settings
from research_paper_assistant.db import Database
from research_paper_assistant.models import IngestionJob, Paper
from sqlalchemy import func, select


def make_pdf(text: str = "Grounded research evidence") -> bytes:
    """Create a small, structurally valid one-page PDF."""

    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return bytes(document.tobytes())  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]


async def upload_pdf(
    client: httpx.AsyncClient,
    content: bytes | None = None,
    filename: str = "paper.pdf",
    content_type: str = "application/pdf",
) -> httpx.Response:
    """Upload a test PDF through the public API."""

    return await client.post(
        "/api/v1/papers",
        files={"file": (filename, make_pdf() if content is None else content, content_type)},
    )


async def test_upload_list_detail_and_delete_paper(
    client: httpx.AsyncClient,
    settings: Settings,
    database: Database,
) -> None:
    pdf = make_pdf("A reproducible result")

    upload_response = await upload_pdf(client, pdf, "experiment.pdf")

    assert upload_response.status_code == 201
    uploaded = upload_response.json()
    assert uploaded["title"] == "experiment"
    assert uploaded["original_filename"] == "experiment.pdf"
    assert uploaded["status"] == "pending"
    assert uploaded["page_count"] == 1
    assert uploaded["size_bytes"] == len(pdf)
    paper_id = uploaded["id"]

    stored_files = list(settings.paper_storage_path.glob("*.pdf"))
    assert [path.name for path in stored_files] == [f"{paper_id}.pdf"]

    list_response = await client.get("/api/v1/papers")
    assert list_response.status_code == 200
    assert [paper["id"] for paper in list_response.json()] == [paper_id]

    detail_response = await client.get(f"/api/v1/papers/{paper_id}")
    assert detail_response.status_code == 200
    assert detail_response.json() == uploaded

    async with database.session() as session:
        job_count = await session.scalar(
            select(func.count()).select_from(IngestionJob).where(IngestionJob.paper_id == paper_id)
        )
    assert job_count == 1

    delete_response = await client.delete(f"/api/v1/papers/{paper_id}")
    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert not stored_files[0].exists()

    assert (await client.get(f"/api/v1/papers/{paper_id}")).status_code == 404
    async with database.session() as session:
        remaining_jobs = await session.scalar(select(func.count()).select_from(IngestionJob))
    assert remaining_jobs == 0


async def test_duplicate_pdf_content_is_rejected(
    client: httpx.AsyncClient,
    settings: Settings,
) -> None:
    pdf = make_pdf()
    first = await upload_pdf(client, pdf, "first.pdf")
    duplicate = await upload_pdf(client, pdf, "renamed.pdf")

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "This PDF is already in the library"
    assert len(list(settings.paper_storage_path.glob("*.pdf"))) == 1


@pytest.mark.parametrize(
    ("content", "filename", "content_type", "expected_detail"),
    [
        (b"not a pdf", "paper.pdf", "application/pdf", "valid PDF signature"),
        (b"%PDF-not-a-document", "paper.pdf", "application/pdf", "corrupt or unreadable"),
        (make_pdf(), "paper.txt", "application/pdf", "must end in .pdf"),
        (make_pdf(), "paper.pdf", "text/plain", "PDF content type"),
    ],
)
async def test_invalid_pdf_uploads_are_rejected(
    client: httpx.AsyncClient,
    content: bytes,
    filename: str,
    content_type: str,
    expected_detail: str,
) -> None:
    response = await upload_pdf(client, content, filename, content_type)

    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]


async def test_oversized_pdf_is_rejected(client: httpx.AsyncClient) -> None:
    oversized = b"%PDF-" + (b"x" * (1024 * 1024))

    response = await upload_pdf(client, oversized)

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]


async def test_user_filename_is_not_used_as_storage_path(
    client: httpx.AsyncClient,
    settings: Settings,
) -> None:
    response = await upload_pdf(client, filename="../../outside.pdf")

    assert response.status_code == 201
    result = response.json()
    assert result["original_filename"] == "outside.pdf"
    assert (settings.paper_storage_path / f"{result['id']}.pdf").is_file()
    assert not (settings.paper_storage_path.parent / "outside.pdf").exists()


async def test_delete_cleanup_failure_preserves_database_record(
    client: httpx.AsyncClient,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = await upload_pdf(client)
    paper_id = response.json()["id"]
    original_unlink = Path.unlink

    def fail_pdf_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.suffix == ".pdf":
            raise PermissionError("locked")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pdf_unlink)

    delete_response = await client.delete(f"/api/v1/papers/{paper_id}")

    assert delete_response.status_code == 500
    assert delete_response.json()["detail"] == "Unable to delete the stored PDF"
    async with database.session() as session:
        assert await session.get(Paper, paper_id) is not None


async def test_missing_paper_returns_404(client: httpx.AsyncClient) -> None:
    missing_id = uuid4()

    assert (await client.get(f"/api/v1/papers/{missing_id}")).status_code == 404
    assert (await client.delete(f"/api/v1/papers/{missing_id}")).status_code == 404


async def test_empty_upload_is_rejected(client: httpx.AsyncClient) -> None:
    response = await upload_pdf(client, content=b"")

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded PDF is empty"


async def test_invalid_paper_identifier_returns_422(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/papers/not-a-uuid")

    assert response.status_code == 422


async def test_library_is_sorted_newest_first(client: httpx.AsyncClient) -> None:
    first = await upload_pdf(client, make_pdf("first"), "first.pdf")
    second = await upload_pdf(client, make_pdf("second"), "second.pdf")

    response = await client.get("/api/v1/papers")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [
        second.json()["id"],
        first.json()["id"],
    ]
