"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research_paper_assistant import __version__
from research_paper_assistant.api.router import api_router
from research_paper_assistant.api.routes.retrieval import router as retrieval_router
from research_paper_assistant.core.config import Settings, get_settings
from research_paper_assistant.db import Database
from research_paper_assistant.services.health import HealthChecker, HealthService
from research_paper_assistant.services.indexing import OllamaEmbedder, QdrantStore
from research_paper_assistant.services.ingestion import IngestionWorker
from research_paper_assistant.services.papers import PaperService
from research_paper_assistant.services.retrieval import RetrievalService


def create_app(
    settings: Settings | None = None,
    health_checker: HealthChecker | None = None,
    paper_service: PaperService | None = None,
    indexing_transport: httpx.AsyncBaseTransport | None = None,
    retrieval_service: RetrievalService | None = None,
) -> FastAPI:
    """Create a configured FastAPI application."""

    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database: Database | None = None
        http_client: httpx.AsyncClient | None = None
        indexing_client: httpx.AsyncClient | None = None
        ingestion_worker: IngestionWorker | None = None

        if paper_service is None or health_checker is None:
            database = Database(app_settings.database_url)
            await database.initialize()

        if paper_service is None:
            if database is None:
                raise RuntimeError("Database initialization failed")
            indexing_client = httpx.AsyncClient(
                timeout=app_settings.indexing_timeout_seconds,
                transport=indexing_transport,
            )
            vectors = QdrantStore(
                indexing_client,
                app_settings.qdrant_url,
                app_settings.qdrant_collection,
                app_settings.embedding_dimensions,
            )
            runtime_paper_service = PaperService(
                database=database,
                storage_root=app_settings.paper_storage_path,
                max_pdf_size_bytes=app_settings.max_pdf_size_bytes,
                vector_store=vectors,
            )
            await runtime_paper_service.initialize()
            application.state.paper_service = runtime_paper_service
            ingestion_worker = IngestionWorker(
                database,
                runtime_paper_service,
                OllamaEmbedder(
                    indexing_client,
                    app_settings.ollama_url,
                    app_settings.ollama_embedding_model,
                    app_settings.embedding_dimensions,
                ),
                vectors,
                app_settings,
            )
            application.state.retrieval_service = retrieval_service or RetrievalService(
                runtime_paper_service,
                ingestion_worker.embedder,
                vectors,
            )
        else:
            application.state.paper_service = paper_service

        if health_checker is None:
            if database is None:
                raise RuntimeError("Database initialization failed")
            http_client = httpx.AsyncClient(timeout=app_settings.health_timeout_seconds)
            application.state.health_checker = HealthService(
                settings=app_settings,
                database=database,
                http_client=http_client,
            )
        else:
            application.state.health_checker = health_checker

        try:
            if ingestion_worker is not None:
                await ingestion_worker.start()
            yield
        finally:
            if ingestion_worker is not None:
                await ingestion_worker.stop()
            if indexing_client is not None:
                await indexing_client.aclose()
            if http_client is not None:
                await http_client.aclose()
            if database is not None:
                await database.close()

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    if health_checker is not None:
        application.state.health_checker = health_checker
    if paper_service is not None:
        application.state.paper_service = paper_service
    if retrieval_service is not None:
        application.state.retrieval_service = retrieval_service
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix=app_settings.api_prefix)
    if app_settings.app_env != "production" or app_settings.retrieval_debug_enabled:
        application.include_router(retrieval_router, prefix=app_settings.api_prefix)

    @application.get("/", tags=["application"])
    async def application_metadata() -> dict[str, Any]:
        return {
            "name": app_settings.app_name,
            "version": __version__,
            "environment": app_settings.app_env,
            "docs": "/docs",
        }

    return application


app = create_app()
