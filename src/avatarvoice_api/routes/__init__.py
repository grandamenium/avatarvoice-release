"""API routes for AvatarVoice."""

from .health import router as health_router
from .analyze import router as analyze_router
from .voices import router as voices_router
from .generate import router as generate_router
from .optimize import router as optimize_router
from .pipeline import router as pipeline_router

__all__ = [
    "health_router",
    "analyze_router",
    "voices_router",
    "generate_router",
    "optimize_router",
    "pipeline_router",
]
