"""API route for text optimization."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from voicematch.prompt_optimizer import PromptOptimizer


router = APIRouter(prefix="/optimize", tags=["optimize"])


class OptimizeRequest(BaseModel):
    """Request model for text optimization."""

    text: str = Field(..., description="Text to optimize for TTS")
    emotion: str = Field(
        default="neutral",
        description="Emotion to optimize for (e.g., 'happy', 'sad', 'NEU')",
    )


class OptimizeResponse(BaseModel):
    """Response model for text optimization."""

    original: str = Field(..., description="Original input text")
    optimized: str = Field(..., description="Optimized text for TTS")
    emotion: str = Field(..., description="Emotion used for optimization")


# Global optimizer instance (lazy initialized)
_optimizer: PromptOptimizer | None = None


def get_optimizer() -> PromptOptimizer:
    """Get or create PromptOptimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptOptimizer()
    return _optimizer


@router.post("/prompt", response_model=OptimizeResponse)
async def optimize_prompt(request: OptimizeRequest) -> OptimizeResponse:
    """Optimize text for TTS based on emotion.

    Takes user-provided text and the detected/selected emotion, and returns
    an optimized version with improved punctuation and grammar for better
    TTS synthesis.

    Args:
        request: The optimization request containing text and emotion

    Returns:
        OptimizeResponse with original and optimized text
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty",
        )

    try:
        optimizer = get_optimizer()
        optimized = await optimizer.optimize(request.text, request.emotion)

        return OptimizeResponse(
            original=request.text,
            optimized=optimized,
            emotion=request.emotion,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization configuration error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}",
        )
