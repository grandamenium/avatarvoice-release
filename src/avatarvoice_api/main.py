"""FastAPI application entry point for AvatarVoice API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .dependencies import APIState
from .routes import (
    health_router,
    analyze_router,
    voices_router,
    generate_router,
    optimize_router,
    pipeline_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    APIState.initialize(settings)
    yield
    # Shutdown
    APIState.cleanup()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(analyze_router)
    app.include_router(voices_router)
    app.include_router(generate_router)
    app.include_router(optimize_router)
    app.include_router(pipeline_router)

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "avatarvoice_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
