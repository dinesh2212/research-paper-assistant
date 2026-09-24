"""Live synthetic retrieval baseline; clean up only uploads made by this run."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pymupdf
from research_paper_assistant.core.config import Settings
from research_paper_assistant.services.extraction import normalize_text
from smoke_ingestion import wait_for_ingestion

ENDPOINT = "/api/v1/debug/retrieval"


def make_pdf(pages: list[str]) -> bytes:
    with pymupdf.open() as document:
        for text in pages:
            page = document.new_page()
            assert page.insert_textbox((72, 72, 520, 750), text, fontsize=11) >= 0
        return bytes(document.tobytes())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = json.loads((root / "evals/retrieval_baseline.json").read_text())
    settings = Settings()
    ids: dict[str, str] = {}
    created: list[str] = []
    report: dict[str, Any] = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "corpus": "evals/retrieval_baseline.json",
        "embedding_model": settings.ollama_embedding_model,
        "scope": "Synthetic smoke baseline, not a representative research-paper benchmark",
    }
    with (
        httpx.Client(base_url="http://127.0.0.1:5173", timeout=60) as api,
        httpx.Client(
            base_url=f"{settings.qdrant_url}/collections/{settings.qdrant_collection}",
            timeout=30,
        ) as qdrant,
    ):
        try:
            api.get("/api/v1/health/ready").raise_for_status()
            # Check activation before creating any fixtures.
            probe = api.post(ENDPOINT, json={"question": "probe", "paper_ids": []})
            assert probe.status_code == 422, "Enable RETRIEVAL_DEBUG_ENABLED in the local backend"
            documents: dict[str, dict[str, Any]] = {}
            for paper in corpus["papers"]:
                response = api.post(
                    "/api/v1/papers",
                    files={
                        "file": (
                            f"{paper['title']}-{uuid4()}.pdf",
                            make_pdf(paper["pages"]),
                            "application/pdf",
                        ),
                    },
                )
                response.raise_for_status()
                paper_id = response.json()["id"]
                created.append(paper_id)
                ids[paper["key"]] = paper_id
                documents[paper_id] = paper
                state = wait_for_ingestion(api, paper_id)
                assert state["status"] == "ready", state

            def search(question: str, selected: list[str], **options: Any) -> dict[str, Any]:
                response = api.post(
                    ENDPOINT,
                    json={
                        "question": question,
                        "paper_ids": selected,
                        **options,
                    },
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                assert result["candidate_count"] <= 20
                assert len(result["sources"]) <= options.get("limit", 8)
                for source in result["sources"]:
                    assert source["paper_id"] in selected, "Paper filter violation"
                    doc = documents[source["paper_id"]]
                    assert 1 <= source["page_number"] <= len(doc["pages"])
                    original = normalize_text(doc["pages"][source["page_number"] - 1])
                    assert source["excerpt"] == original, "Citation does not match source page"
                    assert source["paper_title"].startswith(doc["title"])
                return result

            cases = []
            for case in corpus["questions"]:
                result = search(case["question"], list(ids.values()), limit=3)
                expected = (ids[case["paper"]], case["page"])
                ranked = [
                    (source["paper_id"], source["page_number"]) for source in result["sources"]
                ]
                cases.append(
                    {
                        "question": case["question"],
                        "expected_paper": case["paper"],
                        "expected_page": case["page"],
                        "top_1_correct": bool(ranked and ranked[0] == expected),
                        "hit_at_3": expected in ranked,
                        "top_score": result["sources"][0]["score"] if ranked else None,
                    }
                )
            report["cases"] = cases
            report["top_1_accuracy"] = sum(case["top_1_correct"] for case in cases) / len(cases)
            report["hit_at_3"] = sum(case["hit_at_3"] for case in cases) / len(cases)
            report["citation_checks"] = "All returned excerpts match their labeled source page"
            single = search(corpus["questions"][0]["question"], [ids["batteries"]])
            assert single["candidate_count"] == 3
            multi = search(
                corpus["questions"][3]["question"], [ids["batteries"], ids["irrigation"]]
            )
            assert multi["candidate_count"] == 6
            default = search(corpus["questions"][0]["question"], list(ids.values()))
            assert default["candidate_count"] == 9 and len(default["sources"]) == 8
            excluded = search(corpus["questions"][-1]["question"], [ids["batteries"]])
            assert all(source["paper_id"] == ids["batteries"] for source in excluded["sources"])
            empty = search(
                "What opera did Mozart compose?", list(ids.values()), score_threshold=1.0
            )
            assert empty["sources"] == []
            missing = api.post(
                ENDPOINT, json={"question": "Evidence?", "paper_ids": [str(uuid4())]}
            )
            assert missing.status_code == 404
            report["filter_and_contract_checks"] = (
                "Paper scope, exclusion, limits, empty results, and missing ID passed"
            )
        finally:
            failures = []
            for paper_id in created:
                try:
                    api.delete(f"/api/v1/papers/{paper_id}").raise_for_status()
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
                except Exception as exc:
                    failures.append(f"{paper_id}: {exc}")
            assert not failures, f"Smoke cleanup incomplete: {failures}"
            report["cleanup"] = f"Removed all {len(created)} test papers and their vectors"
        print(json.dumps(report, indent=2))
        assert report["top_1_accuracy"] >= 0.8 and report["hit_at_3"] == 1.0, report


if __name__ == "__main__":
    main()
