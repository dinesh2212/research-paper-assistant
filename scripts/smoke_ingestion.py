"""Exercise a running local stack; remove only PDFs created by this smoke test.

Run from the repository root with `uv run python scripts/smoke_ingestion.py`.
Uses the configured Qdrant collection and localhost frontend API proxy.
"""

import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pymupdf
from research_paper_assistant.core.config import Settings
from sqlalchemy.engine import make_url


def make_document(blank: bool = False) -> bytes:
    with pymupdf.open() as document:
        for number in range(1 if blank else 18):
            page = document.new_page()
            if not blank:
                page.insert_text(
                    (72, 72),
                    f"Ingestion smoke test page {number + 1}. "
                    "Reproducible evidence for page-level citations.",
                )
        if not blank:
            # An incompressible attachment makes this exceed Nginx's old 1 MiB
            # default without changing the extracted text or making 1,000 pages.
            document.embfile_add("smoke-padding.bin", random.Random(0).randbytes(1_100_000))
        return bytes(document.tobytes())


def wait_for_ingestion(client: httpx.Client, paper_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/papers/{paper_id}")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        if data["status"] in {"ready", "failed"}:
            return data
        time.sleep(0.25)
    raise TimeoutError(f"Ingestion did not finish for smoke paper {paper_id}")


def main() -> None:
    settings = Settings()
    created: list[str] = []
    report: dict[str, Any] = {}
    with (
        httpx.Client(base_url="http://127.0.0.1:5173", timeout=180) as api,
        httpx.Client(
            base_url=f"{settings.qdrant_url}/collections/{settings.qdrant_collection}",
            timeout=30,
        ) as qdrant,
    ):
        try:
            readiness = api.get("/api/v1/health/ready")
            readiness.raise_for_status()
            report["readiness"] = readiness.json()["status"]
            for blank in (False, True):
                document = make_document(blank)
                response = api.post(
                    "/api/v1/papers",
                    files={
                        "file": (f"ingestion-smoke-{uuid4()}.pdf", document, "application/pdf"),
                    },
                )
                response.raise_for_status()
                assert response.status_code == 201, response.text
                paper_id = response.json()["id"]
                created.append(paper_id)
                start = time.monotonic()
                paper = wait_for_ingestion(api, paper_id)
                if blank:
                    assert paper["status"] == "failed", paper
                    assert "OCR" in paper["error_message"], paper
                    retry = api.post(f"/api/v1/papers/{paper_id}/retry")
                    assert retry.status_code == 202, retry.text
                    assert wait_for_ingestion(api, paper_id)["status"] == "failed"
                    report["blank_pdf"] = "failed with OCR explanation; retry verified"
                    continue
                assert paper["status"] == "ready", paper
                points_response = qdrant.post(
                    "points/scroll",
                    json={
                        "filter": {"must": [{"key": "paper_id", "match": {"value": paper_id}}]},
                        "limit": 100,
                        "with_payload": True,
                        "with_vector": True,
                    },
                )
                points_response.raise_for_status()
                points = points_response.json()["result"]["points"]
                assert len(points) == 18, len(points)
                assert {point["payload"]["page_number"] for point in points} == set(range(1, 19))
                for point in points:
                    payload = point["payload"]
                    assert payload["chunk_id"] == point["id"]
                    assert f"page {payload['page_number']}." in payload["text"]
                    assert payload["embedding_model"] == settings.ollama_embedding_model
                    assert len(point["vector"]) == settings.embedding_dimensions
                report["indexed_pdf"] = {
                    "bytes": len(document),
                    "pages": 18,
                    "chunks": len(points),
                    "dimensions": settings.embedding_dimensions,
                    "seconds": round(time.monotonic() - start, 2),
                }
        finally:
            for paper_id in created:
                response = api.delete(f"/api/v1/papers/{paper_id}")
                response.raise_for_status()
                assert response.status_code == 204
                count = qdrant.post(
                    "points/count",
                    json={
                        "filter": {"must": [{"key": "paper_id", "match": {"value": paper_id}}]},
                        "exact": True,
                    },
                )
                count.raise_for_status()
                assert count.json()["result"]["count"] == 0
                assert api.get(f"/api/v1/papers/{paper_id}").status_code == 404
                assert not (settings.paper_storage_path / f"{paper_id}.pdf").exists()
            database_path = make_url(settings.database_url).database
            if created and database_path:
                with sqlite3.connect(
                    f"{Path(database_path).resolve().as_uri()}?mode=ro", uri=True
                ) as db:
                    for paper_id in created:
                        assert (
                            db.execute(
                                "SELECT count(*) FROM ingestion_jobs WHERE paper_id = ?",
                                (paper_id,),
                            ).fetchone()[0]
                            == 0
                        )
            report["cleanup"] = (
                f"Removed {len(created)} smoke PDFs, records, jobs, and all their vectors"
            )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
