"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.schemas import HealthResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the HTTP application with explicit, testable dependencies."""

    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Later phases initialize shared provider, graph, and storage clients here.
        yield

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="A self-corrective RAG assistant for technical documentation.",
        lifespan=lifespan,
    )
    application.state.settings = app_settings

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        return HealthResponse(
            version=app_settings.app_version,
            provider_configured=app_settings.provider_configured,
        )

    return application


app = create_app()
