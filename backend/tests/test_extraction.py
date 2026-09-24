"""Extraction, page citations, and chunk coverage tests."""

from pathlib import Path
from uuid import UUID, uuid4

import pymupdf
import pytest
from research_paper_assistant.core.config import Settings
from research_paper_assistant.services.extraction import (
    ExtractionError,
    PageText,
    chunk_pages,
    extract_pages,
    normalize_text,
)


def test_normalization_preserves_paragraphs_and_scientific_content() -> None:
    assert normalize_text("  ﬁndings\x00\t  x² = 4\n continue\n\n Next\u00a0part\u00ad ") == (
        "findings x² = 4 continue\n\nNext part"
    )


def test_extraction_keeps_physical_page_numbers_and_skips_blanks(tmp_path: Path) -> None:
    path = tmp_path / "pages.pdf"
    with pymupdf.open() as doc:  # type: ignore[no-untyped-call]
        doc.new_page().insert_text((72, 72), "First page evidence")
        doc.new_page()
        doc.new_page().insert_text((72, 72), "Third page evidence")
        doc.save(path)
    pages = extract_pages(path)
    assert [(page.page_number, page.text) for page in pages] == [
        (1, "First page evidence"),
        (3, "Third page evidence"),
    ]


@pytest.mark.parametrize("image_only", [False, True])
def test_empty_and_scanned_pdf_fail_with_ocr_explanation(tmp_path: Path, image_only: bool) -> None:
    path = tmp_path / "unreadable.pdf"
    with pymupdf.open() as doc:  # type: ignore[no-untyped-call]
        page = doc.new_page()
        if image_only:
            # A synthetic raster page: no extractable text layer.
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 16, 16), False)  # type: ignore[no-untyped-call]
            pixmap.clear_with(255)  # type: ignore[no-untyped-call]
            page.insert_image((0, 0, 200, 200), pixmap=pixmap)
        doc.save(path)
    with pytest.raises(ExtractionError, match="Scanned or empty PDFs require OCR"):
        extract_pages(path)


def test_corrupt_or_missing_stored_pdf_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="Unable to extract"):
        extract_pages(tmp_path / "missing.pdf")


def test_chunk_coverage_overlap_and_stable_citation_ids() -> None:
    paper_id = uuid4()
    text = " ".join(f"evidence{index:04d}" for index in range(1000))
    pages = [PageText(2, text), PageText(5, "A short final page.")]
    chunks = chunk_pages(paper_id, pages)
    assert chunks == chunk_pages(paper_id, pages)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert UUID(chunk.id).version == 5
        assert len(chunk.text) <= 2800
    first_page = [chunk for chunk in chunks if chunk.page_number == 2]
    starts = [text.index(chunk.text) for chunk in first_page]
    assert starts[0] == 0
    assert starts[-1] + len(first_page[-1].text) == len(text)
    for previous, current, chunk in zip(starts, starts[1:], first_page, strict=False):
        assert 399 <= previous + len(chunk.text) - current <= 400
    assert chunks[-1].page_number == 5
    assert chunks[-1].text == pages[-1].text
    assert chunks[-1].chunk_index == 0
    assert chunk_pages(uuid4(), pages)[0].id != chunks[0].id


def test_unbroken_text_chunks_progress_without_losing_characters() -> None:
    chunks = chunk_pages(uuid4(), [PageText(1, "x" * 7200)])
    assert [len(chunk.text) for chunk in chunks] == [2800, 2800, 2400]


def test_invalid_chunk_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(uuid4(), [], 100, 100)
    with pytest.raises(ValueError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(chunk_tokens=100, chunk_overlap_tokens=100)
