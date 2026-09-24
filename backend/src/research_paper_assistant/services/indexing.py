"""Small, validated HTTP adapters for local embeddings and vector storage."""

from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from research_paper_assistant.schemas.retrieval import QueryResult, ScoredChunk
from research_paper_assistant.services.extraction import TextChunk


class IndexingError(Exception):
    """An embedding or vector-storage operation failed."""


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    embeddings: list[list[float]] = Field(min_length=1)


class OllamaEmbedder:
    def __init__(self, client: httpx.AsyncClient, url: str, model: str, dimensions: int) -> None:
        self.client = client
        self.url = url.rstrip("/")
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.post(
                f"{self.url}/api/embed",
                json={"model": self.model, "input": texts, "truncate": False},
            )
            response.raise_for_status()
            vectors = EmbeddingResponse.model_validate(response.json()).embeddings
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            raise IndexingError(
                "Embedding generation failed. Check Ollama and the configured "
                "embedding model, then retry ingestion."
            ) from exc
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimensions or not any(vector) for vector in vectors
        ):
            raise IndexingError(
                "Ollama returned an invalid embedding count, dimension, or zero vector"
            )
        return vectors


class QdrantStore:
    def __init__(
        self, client: httpx.AsyncClient, url: str, collection: str, dimensions: int
    ) -> None:
        self.client = client
        self.collection_url = f"{url.rstrip('/')}/collections/{quote(collection, safe='')}"
        self.dimensions = dimensions

    async def _request(
        self,
        method: str,
        suffix: str = "",
        body: dict[str, Any] | None = None,
        *,
        missing_ok: bool = False,
    ) -> dict[str, Any] | None:
        try:
            response = await self.client.request(
                method, f"{self.collection_url}{suffix}", json=body
            )
            if missing_ok and response.status_code == 404:
                return None
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            if not isinstance(data, dict) or data.get("status") != "ok":
                raise ValueError("Unexpected Qdrant response")
            if (
                method != "GET"
                and (suffix.startswith("/points?") or suffix.startswith("/points/delete"))
                and data.get("result", {}).get("status") != "completed"
            ):
                raise ValueError("Qdrant operation did not complete")
            return data
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise IndexingError(
                "Vector storage failed. Check Qdrant, then retry the operation."
            ) from exc

    async def ensure_collection(self) -> None:
        data = await self._request("GET", missing_ok=True)
        if data is None:
            await self._request(
                "PUT",
                body={
                    "vectors": {"size": self.dimensions, "distance": "Cosine"},
                },
            )
        else:
            vectors = data.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
            if vectors.get("size") != self.dimensions or vectors.get("distance") != "Cosine":
                raise IndexingError(
                    "Qdrant collection has incompatible vectors. Configure a new "
                    "collection for this embedding model."
                )
        await self._request(
            "PUT",
            "/index?wait=true",
            {
                "field_name": "paper_id",
                "field_schema": "keyword",
            },
        )

    async def delete_paper(self, paper_id: str) -> None:
        await self._request(
            "POST",
            "/points/delete?wait=true",
            {
                "filter": {"must": [{"key": "paper_id", "match": {"value": paper_id}}]},
            },
            missing_ok=True,
        )

    async def query(
        self,
        vector: list[float],
        paper_ids: list[UUID],
        model: str,
        limit: int = 20,
        score_threshold: float | None = None,
    ) -> list[ScoredChunk]:
        if not paper_ids:
            raise IndexingError("Retrieval requires an explicit paper selection")
        body: dict[str, Any] = {
            "query": vector,
            "filter": {
                "must": [
                    {"key": "paper_id", "match": {"any": [str(value) for value in paper_ids]}},
                    {"key": "embedding_model", "match": {"value": model}},
                ]
            },
            "limit": limit,
            "with_payload": True,
            "with_vector": True,
        }
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        data = await self._request("POST", "/points/query", body)
        try:
            result = QueryResult.model_validate(data["result"] if data is not None else None)
            if len(result.points) > limit:
                raise ValueError("Qdrant exceeded the candidate limit")
            return result.points
        except (ValueError, KeyError) as exc:
            raise IndexingError("Qdrant returned invalid retrieval results") from exc

    async def upsert(
        self,
        paper_id: str,
        title: str,
        chunks: list[TextChunk],
        vectors: list[list[float]],
        model: str,
    ) -> None:
        points = [
            {
                "id": chunk.id,
                "vector": vector,
                "payload": {
                    "paper_id": paper_id,
                    "paper_title": title,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "embedding_model": model,
                },
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._request("PUT", "/points?wait=true", {"points": points})
