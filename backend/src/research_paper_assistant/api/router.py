"""Top-level API router."""

from fastapi import APIRouter

from research_paper_assistant.api.routes import health, papers

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(papers.router)
