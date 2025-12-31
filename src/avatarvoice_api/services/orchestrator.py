"""Orchestrator service for AvatarVoice pipeline."""

from pathlib import Path
from typing import Optional, List

from voicematch import VoiceMatchAPI, AvatarAnalysisResponse, MatchResultResponse
from vibevoice_client import VibeVoiceClient, GenerationConfig, VoiceReference, GenerationResult
from vibevoice_client.models import EmotionType


class AvatarVoiceOrchestrator:
    """Orchestrates the full avatar-to-voice pipeline.

    Coordinates between VoiceMatch (demographics → voice matching)
    and VibeVoice (TTS generation).
    """

    def __init__(
        self,
        voicematch_api: VoiceMatchAPI,
        vibevoice_client: VibeVoiceClient,
    ):
        """Initialize orchestrator.

        Args:
            voicematch_api: VoiceMatch API instance.
            vibevoice_client: VibeVoice client instance.
        """
        self.voicematch = voicematch_api
        self.vibevoice = vibevoice_client

    async def analyze_avatar(
        self,
        image_path: Optional[Path] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
    ) -> AvatarAnalysisResponse:
        """Analyze an avatar image for demographics.

        Args:
            image_path: Path to image file.
            image_bytes: Raw image bytes.
            mime_type: MIME type if using bytes.

        Returns:
            Demographics analysis.

        Raises:
            ValueError: If neither path nor bytes provided.
        """
        if image_path:
            return await self.voicematch.analyze_image(image_path)
        elif image_bytes:
            return await self.voicematch.analyze_image_bytes(image_bytes, mime_type)
        else:
            raise ValueError("Either image_path or image_bytes must be provided")

    def find_matching_voices(
        self,
        analysis: AvatarAnalysisResponse,
        limit: int = 5,
        emotion_filter: Optional[str] = None,
    ) -> List[MatchResultResponse]:
        """Find voice actors matching the demographics.

        Args:
            analysis: Demographics from analyze_avatar.
            limit: Maximum matches to return.
            emotion_filter: Optional CREMA-D emotion code.

        Returns:
            List of matching voices sorted by score.
        """
        return self.voicematch.find_matches(
            analysis,
            limit=limit,
            emotion_filter=emotion_filter,
        )

    async def generate_voice(
        self,
        text: str,
        actor_id: Optional[str] = None,
        voice_reference_path: Optional[Path] = None,
        emotion: str = "neutral",
        **generation_params,
    ) -> GenerationResult:
        """Generate speech using matched voice or reference.

        Args:
            text: Text to synthesize.
            actor_id: Actor ID to use as voice reference.
            voice_reference_path: Alternative reference audio path.
            emotion: Target emotion for generation.
            **generation_params: Additional TTS parameters.

        Returns:
            Generated audio result.
        """
        # Build voice reference
        voice_ref = None

        if actor_id:
            # Get voice sample from CREMA-D actor
            sample_path = self.voicematch.get_voice_sample(
                actor_id=actor_id,
                emotion=self._emotion_to_code(emotion),
            )
            if sample_path:
                voice_ref = VoiceReference(audio_path=sample_path)

        elif voice_reference_path:
            voice_ref = VoiceReference(audio_path=voice_reference_path)

        # Map emotion
        emotion_type = self._map_emotion(emotion)

        # Build config
        config = GenerationConfig(
            text=text,
            voice_reference=voice_ref,
            emotion=emotion_type,
            **generation_params,
        )

        return await self.vibevoice.generate(config)

    async def full_pipeline(
        self,
        image_path: Optional[Path] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        text: str = "",
        **generation_params,
    ) -> tuple[AvatarAnalysisResponse, List[MatchResultResponse], Optional[GenerationResult]]:
        """Run the complete avatar-to-voice pipeline.

        Args:
            image_path: Path to avatar image.
            image_bytes: Raw avatar image bytes.
            mime_type: Image MIME type.
            text: Text to synthesize.
            **generation_params: Additional TTS parameters.

        Returns:
            Tuple of (analysis, matches, generation_result).
        """
        # Step 1: Analyze avatar
        analysis = await self.analyze_avatar(
            image_path=image_path,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        # Step 2: Find matching voices
        matches = self.find_matching_voices(
            analysis,
            limit=5,
            emotion_filter=self._emotion_to_code(analysis.emotion),
        )

        # Step 3: Generate voice using best match
        generation = None
        if text and matches:
            best_match = matches[0]
            generation = await self.generate_voice(
                text=text,
                actor_id=best_match.actor.id,
                emotion=analysis.emotion,
                **generation_params,
            )

        return analysis, matches, generation

    def _emotion_to_code(self, emotion: str) -> str:
        """Convert emotion string to CREMA-D code."""
        mapping = {
            "anger": "ANG",
            "angry": "ANG",
            "disgust": "DIS",
            "fear": "FEA",
            "happy": "HAP",
            "neutral": "NEU",
            "sad": "SAD",
        }
        return mapping.get(emotion.lower(), "NEU")

    def _map_emotion(self, emotion: str) -> Optional[EmotionType]:
        """Map emotion string to EmotionType enum."""
        mapping = {
            "neutral": EmotionType.NEUTRAL,
            "happy": EmotionType.HAPPY,
            "sad": EmotionType.SAD,
            "angry": EmotionType.ANGRY,
            "anger": EmotionType.ANGRY,
            "fear": EmotionType.FEAR,
            "disgust": EmotionType.DISGUST,
            "surprise": EmotionType.SURPRISE,
        }
        return mapping.get(emotion.lower())
