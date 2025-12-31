"""Clean API for VoiceMatch functionality.

This module provides a UI-agnostic API for the VoiceMatch system.
It can be used by FastAPI, CLI, or any other interface.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

from .audio_processor import AudioProcessor
from .config import Config, ConfigError
from .database import VoiceDatabase, Actor, DatabaseError as DBError
from .exceptions import (
    ConfigurationError,
    DatabaseError,
    ImageAnalysisError,
    UnsupportedFormatError,
    VoiceNotFoundError,
)
from .gemini_analyzer import AvatarAnalysis, GeminiAnalyzer, GeminiError
from .models import (
    ActorResponse,
    AvatarAnalysisResponse,
    Emotion,
    Gender,
    MatchResultResponse,
    Race,
    EMOTION_CODES,
)
from .voice_matcher import MatchResult, VoiceMatcher


# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


class VoiceMatchAPI:
    """Clean API for VoiceMatch functionality.

    Can be used by FastAPI, CLI, or any other interface.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize with optional config override.

        Args:
            config: Optional Config instance. If not provided, loads from environment.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        try:
            self._config = config or Config.load()
        except ConfigError as e:
            raise ConfigurationError(str(e)) from e

        self._analyzer: Optional[GeminiAnalyzer] = None
        self._database: Optional[VoiceDatabase] = None
        self._matcher: Optional[VoiceMatcher] = None
        self._audio_processor = AudioProcessor()

    @property
    def analyzer(self) -> GeminiAnalyzer:
        """Lazy-load the Gemini analyzer."""
        if self._analyzer is None:
            self._analyzer = GeminiAnalyzer(
                api_key=self._config.gemini_api_key
            )
        return self._analyzer

    @property
    def database(self) -> VoiceDatabase:
        """Lazy-load and connect to the database."""
        if self._database is None:
            try:
                self._database = VoiceDatabase(self._config.database_path)
                self._database.connect()
            except DBError as e:
                raise DatabaseError(str(e)) from e
        return self._database

    @property
    def matcher(self) -> VoiceMatcher:
        """Lazy-load the voice matcher."""
        if self._matcher is None:
            self._matcher = VoiceMatcher(self.database)
        return self._matcher

    def _validate_image_format(self, image_path: Path) -> None:
        """Validate that the image format is supported.

        Args:
            image_path: Path to the image file.

        Raises:
            UnsupportedFormatError: If format is not supported.
        """
        ext = image_path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported image format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

    def _validate_mime_type(self, mime_type: str) -> None:
        """Validate that the MIME type is supported.

        Args:
            mime_type: MIME type string.

        Raises:
            UnsupportedFormatError: If MIME type is not supported.
        """
        if mime_type not in MIME_TO_EXT:
            raise UnsupportedFormatError(
                f"Unsupported MIME type: {mime_type}. Supported: {', '.join(MIME_TO_EXT.keys())}"
            )

    def _analysis_to_response(self, analysis: AvatarAnalysis) -> AvatarAnalysisResponse:
        """Convert internal AvatarAnalysis to API response model.

        Args:
            analysis: Internal analysis result.

        Returns:
            AvatarAnalysisResponse model.
        """
        return AvatarAnalysisResponse(
            estimated_age=analysis.estimated_age,
            age_range=analysis.age_range,
            gender=Gender(analysis.gender),
            race=Race(analysis.race),
            ethnicity=analysis.ethnicity,
            emotion=Emotion(analysis.emotion),
            confidence=analysis.confidence,
            raw_response=analysis.raw_response,
        )

    def _actor_to_response(self, actor: Actor, sample_count: int = 0) -> ActorResponse:
        """Convert internal Actor to API response model.

        Args:
            actor: Internal actor object.
            sample_count: Number of audio samples for this actor.

        Returns:
            ActorResponse model.
        """
        return ActorResponse(
            id=actor.id,
            age=actor.age,
            sex=actor.sex,
            race=actor.race,
            ethnicity=actor.ethnicity if actor.ethnicity != "Not Hispanic" else None,
            sample_count=sample_count,
        )

    def _match_to_response(self, match: MatchResult) -> MatchResultResponse:
        """Convert internal MatchResult to API response model.

        Args:
            match: Internal match result.

        Returns:
            MatchResultResponse model.
        """
        sample_count = self.database.get_clip_count_for_actor(match.actor.id)
        return MatchResultResponse(
            actor=self._actor_to_response(match.actor, sample_count),
            score=match.score,
            match_details=match.match_details,
        )

    async def analyze_image(self, image_path: Path) -> AvatarAnalysisResponse:
        """Analyze an image file and return demographic analysis.

        Args:
            image_path: Path to image file (jpg, png, gif, webp).

        Returns:
            AvatarAnalysisResponse with demographics.

        Raises:
            FileNotFoundError: If image doesn't exist.
            UnsupportedFormatError: If unsupported file type.
            ImageAnalysisError: If API call fails.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self._validate_image_format(image_path)

        try:
            # Run synchronous analysis in thread pool
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None, self.analyzer.analyze_image, image_path
            )
            return self._analysis_to_response(analysis)
        except GeminiError as e:
            raise ImageAnalysisError(str(e)) from e

    async def analyze_image_bytes(
        self, image_data: bytes, mime_type: str
    ) -> AvatarAnalysisResponse:
        """Analyze image from raw bytes (for API uploads).

        Args:
            image_data: Raw image bytes.
            mime_type: MIME type (image/jpeg, image/png, etc.).

        Returns:
            AvatarAnalysisResponse with demographics.

        Raises:
            UnsupportedFormatError: If invalid MIME type.
            ImageAnalysisError: If analysis fails.
        """
        self._validate_mime_type(mime_type)

        try:
            # Create a temporary file for the image
            import tempfile
            ext = MIME_TO_EXT[mime_type]

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(image_data)
                temp_path = Path(f.name)

            try:
                result = await self.analyze_image(temp_path)
                return result
            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)

        except (FileNotFoundError, UnsupportedFormatError):
            raise
        except Exception as e:
            raise ImageAnalysisError(f"Failed to analyze image: {e}") from e

    def find_matches(
        self,
        analysis: AvatarAnalysisResponse,
        limit: int = 5,
        emotion_filter: Optional[str] = None,
    ) -> List[MatchResultResponse]:
        """Find voice actors matching the analysis.

        Args:
            analysis: Demographics from analyze_image.
            limit: Max results to return.
            emotion_filter: Optional emotion code (ANG, DIS, FEA, HAP, NEU, SAD).

        Returns:
            List of MatchResultResponse sorted by score.
        """
        # Convert response model back to internal AvatarAnalysis
        internal_analysis = AvatarAnalysis(
            estimated_age=analysis.estimated_age,
            age_range=analysis.age_range,
            gender=analysis.gender if isinstance(analysis.gender, str) else analysis.gender.value,
            race=analysis.race if isinstance(analysis.race, str) else analysis.race.value,
            ethnicity=analysis.ethnicity,
            emotion=analysis.emotion if isinstance(analysis.emotion, str) else analysis.emotion.value,
            confidence=analysis.confidence,
            raw_response=analysis.raw_response or {},
        )

        # Get matches from the matcher
        matches = self.matcher.find_matches(internal_analysis, limit=limit)

        # Filter by emotion if specified
        if emotion_filter and emotion_filter in EMOTION_CODES:
            filtered_matches = []
            for match in matches:
                clips = self.database.get_clips_for_actor(
                    match.actor.id, emotion=emotion_filter
                )
                if clips:
                    filtered_matches.append(match)
            matches = filtered_matches[:limit]

        return [self._match_to_response(m) for m in matches]

    def get_voice_sample(
        self,
        actor_id: str,
        emotion: str = "NEU",
        duration_seconds: float = 10.0,
    ) -> Optional[Path]:
        """Get a voice sample for an actor.

        Args:
            actor_id: Actor identifier from CREMA-D.
            emotion: Emotion code for sample selection (ANG, DIS, FEA, HAP, NEU, SAD).
            duration_seconds: Target sample duration.

        Returns:
            Path to audio file, or None if not found.

        Raises:
            VoiceNotFoundError: If actor doesn't exist.
        """
        # Verify actor exists
        actor = self.database.get_actor_by_id(actor_id)
        if actor is None:
            raise VoiceNotFoundError(f"Actor not found: {actor_id}")

        # Get clips for the actor with the specified emotion
        clips = self.database.get_clips_for_actor(actor_id, emotion=emotion)

        # Fallback to neutral if no clips for requested emotion
        if not clips and emotion != "NEU":
            clips = self.database.get_clips_for_actor(actor_id, emotion="NEU")

        # Fallback to any clips
        if not clips:
            clips = self.database.get_clips_for_actor(actor_id)

        if not clips:
            return None

        # For single clip, just return it
        if len(clips) == 1:
            clip_path = clips[0].filepath
            if clip_path.exists():
                return clip_path
            # Try with data_dir prefix
            full_path = self._config.data_dir / clip_path
            return full_path if full_path.exists() else None

        # For multiple clips, concatenate to target duration
        clip_paths = []
        for clip in clips:
            path = clip.filepath
            if not path.exists():
                path = self._config.data_dir / clip.filepath
            if path.exists():
                clip_paths.append(path)

        if not clip_paths:
            return None

        # Create output path
        output_path = self._config.output_dir / f"{actor_id}_{emotion}_sample.wav"

        # Configure audio processor for target duration
        from .audio_processor import AudioConfig
        config = AudioConfig(target_duration_ms=int(duration_seconds * 1000))
        processor = AudioProcessor(config)

        return processor.concatenate_clips(clip_paths, output_path)

    def list_voices(
        self,
        gender: Optional[str] = None,
        race: Optional[str] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[ActorResponse], int]:
        """List available voice actors with filtering.

        Args:
            gender: Filter by gender (Male, Female).
            race: Filter by race.
            age_min: Minimum age.
            age_max: Maximum age.
            limit: Results per page.
            offset: Pagination offset.

        Returns:
            Tuple of (actors list, total count).
        """
        # Query all matching actors
        all_actors = self.database.query_actors(
            sex=gender,
            race=race,
            min_age=age_min,
            max_age=age_max,
        )

        total = len(all_actors)

        # Apply pagination
        paginated = all_actors[offset : offset + limit]

        # Convert to response models
        actors = []
        for actor in paginated:
            sample_count = self.database.get_clip_count_for_actor(actor.id)
            actors.append(self._actor_to_response(actor, sample_count))

        return actors, total

    def get_actor_details(self, actor_id: str) -> Optional[ActorResponse]:
        """Get full details for a specific actor.

        Args:
            actor_id: The actor's ID.

        Returns:
            ActorResponse if found, None otherwise.
        """
        actor = self.database.get_actor_by_id(actor_id)
        if actor is None:
            return None

        sample_count = self.database.get_clip_count_for_actor(actor_id)
        return self._actor_to_response(actor, sample_count)

    def close(self) -> None:
        """Clean up resources."""
        if self._database is not None:
            self._database.close()
            self._database = None

    def __enter__(self) -> "VoiceMatchAPI":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
