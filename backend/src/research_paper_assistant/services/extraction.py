"""Page-preserving PDF extraction and deterministic, bounded text chunks."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid5

import pymupdf


class ExtractionError(Exception):
    """The PDF cannot supply usable text without unsupported OCR."""


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    id: str
    page_number: int
    chunk_index: int
    text: str


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph breaks."""

    # Compatibility normalization (NFKC) would turn x² into x2 and H₂O into
    # H2O. Preserve scientific symbols; expand only typographic ligatures.
    text = unicodedata.normalize("NFC", text).translate(
        {
            ord("ﬀ"): "ff",
            ord("ﬁ"): "fi",
            ord("ﬂ"): "fl",
            ord("ﬃ"): "ffi",
            ord("ﬄ"): "ffl",
            ord("ﬅ"): "st",
            ord("ﬆ"): "st",
        }
    )
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(" ".join(part.split()) for part in paragraphs if part.strip())


def extract_pages(path: Path) -> list[PageText]:
    """Read physical PDF pages (1-based), skipping blank/image-only pages.

    Mixed PDFs retain extractable text only. A wholly image-only or empty PDF
    fails clearly; no OCR, header removal, or page renumbering is performed.
    """

    pages: list[PageText] = []
    try:
        with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
            if document.needs_pass:
                raise ExtractionError("Password-protected PDFs are not supported")
            for index in range(document.page_count):
                page = document.load_page(index)
                text = normalize_text(str(page.get_text("text", sort=True)))
                if any(character.isalnum() for character in text):
                    pages.append(PageText(page_number=index + 1, text=text))
    except ExtractionError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExtractionError("Unable to extract text from the stored PDF") from exc
    if not pages:
        raise ExtractionError(
            "No extractable text found. Scanned or empty PDFs require OCR, "
            "which is not supported yet."
        )
    return pages


def chunk_pages(
    paper_id: UUID,
    pages: list[PageText],
    chunk_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[TextChunk]:
    """Split within pages using ~4 characters/token for English research prose.

    These are token estimates, not Qwen tokenizer counts. End cuts prefer
    whitespace; overlap is approximately the configured character budget.
    Exact normalized source substrings and deterministic UUIDs retain citations.
    """

    if chunk_tokens < 1 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size")
    size, overlap = chunk_tokens * 4, overlap_tokens * 4
    chunks: list[TextChunk] = []
    for page in pages:
        start = 0
        page_index = 0
        while start < len(page.text):
            end = min(start + size, len(page.text))
            if end < len(page.text):
                # Only prefer a boundary near the end; long words still advance.
                boundary = page.text.rfind(" ", start + max(size // 2, overlap + 1), end)
                if boundary > start:
                    end = boundary
            text = page.text[start:end].strip()
            if text:
                chunks.append(
                    TextChunk(
                        id=str(uuid5(paper_id, f"{page.page_number}:{page_index}")),
                        page_number=page.page_number,
                        chunk_index=page_index,
                        text=text,
                    )
                )
                page_index += 1
            if end == len(page.text):
                break
            start = max(start + 1, end - overlap)
    return chunks
