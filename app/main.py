"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import ApplicationServices, router
from app.bootstrap import build_document_store, build_runtime_services
from app.checkpoints import SQLiteCheckpointManager
from app.config import Settings, get_settings
from app.observability import RequestContextMiddleware, configure_logging
from app.repository import SQLiteRepository
from app.schemas import HealthResponse


def create_app(
    settings: Settings | None = None,
    services: ApplicationServices | None = None,
) -> FastAPI:
    """Create the HTTP application with explicit, testable dependencies."""

    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        checkpoint_manager: SQLiteCheckpointManager | None = None
        try:
            if services is not None:
                services.repository.initialize()
                application.state.services = services
            else:
                repository = SQLiteRepository(app_settings.application_db_path)
                repository.initialize()
                document_store = build_document_store(app_settings)
                if app_settings.provider_configured:
                    checkpoint_manager = SQLiteCheckpointManager(app_settings.checkpoint_db_path)
                    checkpoint_manager.initialize()
                    checkpoint_manager.cleanup(app_settings.checkpoint_retention_days)
                    runtime = build_runtime_services(
                        app_settings,
                        store=document_store,
                        checkpoint_manager=checkpoint_manager,
                    )
                    application.state.services = ApplicationServices(
                        query=runtime.query,
                        ingestion=runtime.ingestion,
                        documents=runtime.documents,
                        repository=repository,
                    )
                else:
                    application.state.services = ApplicationServices(
                        query=None,
                        ingestion=None,
                        documents=document_store,
                        repository=repository,
                    )
            yield
        finally:
            if checkpoint_manager is not None:
                checkpoint_manager.close()

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="A self-corrective RAG assistant for technical documentation.",
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.add_middleware(RequestContextMiddleware)
    application.include_router(router)

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        return HealthResponse(
            version=app_settings.app_version,
            provider_configured=app_settings.provider_configured,
        )

    return application


app = create_app()
