"""Pipeline Orchestrator for end-to-end AvatarVoice generation.

This module coordinates the full pipeline:
1. Analyze image/video via Gemini -> demographics + emotion
2. Query database for matching voice actors
3. Create preview audio from top match
4. Optimize user text for TTS (optional)
5. Send to VibeVoice for speech generation
6. Return generated audio
"""

import base64
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Any, Optional

from voicematch.api import VoiceMatchAPI
from vibevoice_client import VibeVoiceClient
from vibevoice_client.models import GenerationConfig, VoiceReference

# Import prompt_optimizer conditionally (may not exist yet)
try:
    from voicematch.prompt_optimizer import PromptOptimizer
    HAS_PROMPT_OPTIMIZER = True
except ImportError:
    HAS_PROMPT_OPTIMIZER = False
    PromptOptimizer = None


# Mapping from Gemini emotion to CREMA-D emotion code
EMOTION_MAPPING = {
    "anger": "ANG",
    "disgust": "DIS",
    "fear": "FEA",
    "happy": "HAP",
    "neutral": "NEU",
    "sad": "SAD",
    "ambiguous": "NEU",
}


def map_gemini_emotion_to_cremad(emotion: str) -> str:
    """Map Gemini-detected emotion to CREMA-D emotion code.

    Args:
        emotion: Emotion string from Gemini (e.g., 'happy', 'sad')

    Returns:
        CREMA-D emotion code (e.g., 'HAP', 'SAD')
    """
    emotion_lower = emotion.lower() if emotion else "neutral"
    return EMOTION_MAPPING.get(emotion_lower, "NEU")


@dataclass
class PipelineResult:
    """Result of the complete pipeline."""
    audio_bytes: bytes
    duration: float
    voice_actor: str
    emotion: str
    original_text: str
    optimized_text: Optional[str]
    demographics: Dict[str, Any]


@dataclass
class PipelineProgress:
    """Progress update during pipeline execution."""
    step: str
    message: str
    progress: int
    data: Optional[Dict[str, Any]] = None


class PipelineOrchestrator:
    """Orchestrates the full AvatarVoice pipeline.

    Coordinates image analysis, voice matching, prompt optimization,
    and TTS generation into a single streaming workflow.
    """

    def __init__(
        self,
        voicematch_api: VoiceMatchAPI,
        vibevoice_client: VibeVoiceClient,
        prompt_optimizer: Optional["PromptOptimizer"] = None,
    ):
        """Initialize the pipeline orchestrator.

        Args:
            voicematch_api: VoiceMatch API instance for image analysis and voice matching.
            vibevoice_client: VibeVoice client for TTS generation.
            prompt_optimizer: Optional PromptOptimizer for text optimization.
        """
        self.voicematch = voicematch_api
        self.vibevoice = vibevoice_client
        self.optimizer = prompt_optimizer

    async def generate_streaming(
        self,
        media_bytes: bytes,
        mime_type: str,
        text: str,
        cfg_scale: float = 2.0,
        optimize_prompt: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the full pipeline with streaming progress updates.

        Args:
            media_bytes: Raw image/video bytes.
            mime_type: MIME type of the media (e.g., 'image/jpeg').
            text: Text to synthesize.
            cfg_scale: Voice consistency parameter (1.0-2.0).
            optimize_prompt: Whether to optimize text for TTS.

        Yields:
            Progress updates and final result as dictionaries.
        """
        demographics = {}
        emotion_code = "NEU"

        try:
            # Step 1: Analyze image/video
            yield {
                "step": "analyzing",
                "message": "Analyzing avatar...",
                "progress": 0,
            }

            analysis = await self.voicematch.analyze_image_bytes(media_bytes, mime_type)

            # Extract demographics for the response
            demographics = {
                "age": analysis.estimated_age,
                "gender": analysis.gender if isinstance(analysis.gender, str) else analysis.gender.value,
                "race": analysis.race if isinstance(analysis.race, str) else analysis.race.value,
            }

            # Get emotion and map to CREMA-D code
            emotion_str = analysis.emotion if isinstance(analysis.emotion, str) else analysis.emotion.value
            emotion_code = map_gemini_emotion_to_cremad(emotion_str)

            yield {
                "step": "demographics",
                "message": f"Detected: {demographics['gender']}, {demographics['age']}yo, {demographics['race']}, {emotion_str}",
                "progress": 20,
                "data": demographics,
            }

        except Exception as e:
            yield {
                "step": "error",
                "message": f"Image analysis failed: {str(e)}",
                "progress": 0,
            }
            return

        try:
            # Step 2: Find matching voices
            yield {
                "step": "matching",
                "message": "Finding matching voices...",
                "progress": 40,
            }

            matches = self.voicematch.find_matches(analysis, limit=5)

            if not matches:
                yield {
                    "step": "error",
                    "message": "No matching voice actors found",
                    "progress": 40,
                }
                return

            best_match = matches[0]
            actor_id = best_match.actor.id

        except Exception as e:
            yield {
                "step": "error",
                "message": f"Voice matching failed: {str(e)}",
                "progress": 40,
            }
            return

        try:
            # Step 3: Get voice sample for the matched actor
            sample_path = self.voicematch.get_voice_sample(
                actor_id=actor_id,
                emotion=emotion_code,
                duration_seconds=10.0,
            )

            if sample_path is None:
                yield {
                    "step": "error",
                    "message": f"No voice samples found for actor {actor_id}",
                    "progress": 50,
                }
                return

        except Exception as e:
            yield {
                "step": "error",
                "message": f"Failed to get voice sample: {str(e)}",
                "progress": 50,
            }
            return

        # Step 4: Optimize prompt (optional)
        optimized_text = text
        if optimize_prompt and self.optimizer is not None:
            try:
                yield {
                    "step": "optimizing",
                    "message": "Optimizing text for TTS...",
                    "progress": 60,
                }
                optimized_text = await self.optimizer.optimize(text, emotion_code)
            except Exception as e:
                # Optimization failure is non-fatal, use original text
                optimized_text = text
                yield {
                    "step": "optimizing",
                    "message": f"Optimization skipped: {str(e)}",
                    "progress": 60,
                }
        elif optimize_prompt and self.optimizer is None:
            yield {
                "step": "optimizing",
                "message": "Prompt optimizer not available, using original text",
                "progress": 60,
            }

        try:
            # Step 5: Generate speech with VibeVoice
            yield {
                "step": "generating",
                "message": "Generating speech...",
                "progress": 80,
            }

            config = GenerationConfig(
                text=optimized_text,
                voice_reference=VoiceReference(audio_path=sample_path),
                cfg_scale=cfg_scale,
            )

            result = await self.vibevoice.generate(config)

            if not result.audio_bytes:
                yield {
                    "step": "error",
                    "message": "TTS generation produced no audio",
                    "progress": 80,
                }
                return

        except Exception as e:
            yield {
                "step": "error",
                "message": f"Speech generation failed: {str(e)}",
                "progress": 80,
            }
            return

        # Step 6: Return complete result
        yield {
            "step": "complete",
            "audio_base64": base64.b64encode(result.audio_bytes).decode(),
            "duration": result.duration_seconds,
            "voice_actor": actor_id,
            "emotion": emotion_code,
            "original_text": text,
            "optimized_text": optimized_text if optimize_prompt else None,
            "demographics": demographics,
        }

    async def generate(
        self,
        media_bytes: bytes,
        mime_type: str,
        text: str,
        cfg_scale: float = 2.0,
        optimize_prompt: bool = True,
    ) -> PipelineResult:
        """Run the full pipeline and return the final result.

        Non-streaming version that waits for completion.

        Args:
            media_bytes: Raw image/video bytes.
            mime_type: MIME type of the media.
            text: Text to synthesize.
            cfg_scale: Voice consistency parameter.
            optimize_prompt: Whether to optimize text for TTS.

        Returns:
            PipelineResult with audio data and metadata.

        Raises:
            ValueError: If pipeline fails at any step.
        """
        result = None
        error_message = None

        async for update in self.generate_streaming(
            media_bytes=media_bytes,
            mime_type=mime_type,
            text=text,
            cfg_scale=cfg_scale,
            optimize_prompt=optimize_prompt,
        ):
            if update.get("step") == "complete":
                result = update
            elif update.get("step") == "error":
                error_message = update.get("message", "Unknown error")

        if error_message:
            raise ValueError(error_message)

        if result is None:
            raise ValueError("Pipeline completed without result")

        return PipelineResult(
            audio_bytes=base64.b64decode(result["audio_base64"]),
            duration=result["duration"],
            voice_actor=result["voice_actor"],
            emotion=result["emotion"],
            original_text=result["original_text"],
            optimized_text=result.get("optimized_text"),
            demographics=result["demographics"],
        )
