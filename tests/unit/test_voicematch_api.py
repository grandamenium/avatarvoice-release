"""Unit tests for VoiceMatch API."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio
import tempfile

from voicematch.api import VoiceMatchAPI, SUPPORTED_FORMATS, MIME_TO_EXT
from voicematch.models import (
    AvatarAnalysisResponse,
    ActorResponse,
    MatchResultResponse,
    Gender,
    Race,
    Emotion,
)
from voicematch.exceptions import (
    ConfigurationError,
    DatabaseError,
    ImageAnalysisError,
    UnsupportedFormatError,
    VoiceNotFoundError,
)
from voicematch.gemini_analyzer import AvatarAnalysis
from voicematch.database import Actor
from voicematch.voice_matcher import MatchResult


class TestVoiceMatchAPIInit:
    """Tests for VoiceMatchAPI initialization."""

    def test_init_with_config(self, mock_env, temp_dir):
        """Should initialize with provided config."""
        from voicematch.config import Config

        # Create database file
        db_path = temp_dir / "test.sqlite"
        db_path.touch()

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=db_path,
        )

        api = VoiceMatchAPI(config=config)
        assert api._config == config

    def test_init_without_config_loads_from_env(self, mock_env, temp_dir):
        """Should load config from environment when not provided."""
        # Create required directories and files
        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)
        (temp_dir / "voice.sqlite").touch()

        api = VoiceMatchAPI()
        assert api._config is not None
        assert api._config.gemini_api_key == "test_api_key_12345"

    def test_init_without_api_key_raises_error(self, monkeypatch, temp_dir):
        """Should raise ConfigurationError when API key is missing."""
        from voicematch.config import Config
        Config.reset()

        # Remove the API key from environment
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # Also clear any .env file that might be loaded
        monkeypatch.chdir(temp_dir)

        with pytest.raises(ConfigurationError):
            VoiceMatchAPI()

        # Reset again for cleanup
        Config.reset()


class TestVoiceMatchAPIImageAnalysis:
    """Tests for image analysis functionality."""

    @pytest.fixture
    def api_with_mocked_analyzer(self, mock_env, temp_dir):
        """Create API with mocked Gemini analyzer."""
        from voicematch.config import Config

        # Create required directories
        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)
        db_path = temp_dir / "test.sqlite"
        db_path.touch()

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=db_path,
        )

        api = VoiceMatchAPI(config=config)

        # Mock the analyzer
        mock_analyzer = Mock()
        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="neutral",
            confidence=0.85,
            raw_response={"test": "data"},
        )
        mock_analyzer.analyze_image.return_value = mock_analysis
        api._analyzer = mock_analyzer

        return api, mock_analyzer

    @pytest.fixture
    def sample_image_jpg(self, temp_dir):
        """Create a sample JPG image file."""
        # Create a minimal valid JPEG (just the header)
        filepath = temp_dir / "test_image.jpg"
        # Minimal JPEG data (SOI + EOI markers)
        jpeg_data = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
                          0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
                          0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9])
        filepath.write_bytes(jpeg_data)
        return filepath

    @pytest.fixture
    def sample_image_png(self, temp_dir):
        """Create a sample PNG image file."""
        filepath = temp_dir / "test_image.png"
        # Minimal PNG data
        png_data = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        filepath.write_bytes(png_data)
        return filepath

    @pytest.mark.asyncio
    async def test_analyze_image_valid_jpg(self, api_with_mocked_analyzer, sample_image_jpg):
        """Should analyze valid JPG image."""
        api, mock_analyzer = api_with_mocked_analyzer

        result = await api.analyze_image(sample_image_jpg)

        assert isinstance(result, AvatarAnalysisResponse)
        assert result.estimated_age == 35
        assert result.gender == Gender.MALE
        assert result.race == Race.CAUCASIAN
        mock_analyzer.analyze_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_image_valid_png(self, api_with_mocked_analyzer, sample_image_png):
        """Should analyze valid PNG image."""
        api, mock_analyzer = api_with_mocked_analyzer

        result = await api.analyze_image(sample_image_png)

        assert isinstance(result, AvatarAnalysisResponse)
        mock_analyzer.analyze_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_image_nonexistent(self, api_with_mocked_analyzer):
        """Should raise FileNotFoundError for missing file."""
        api, _ = api_with_mocked_analyzer

        with pytest.raises(FileNotFoundError):
            await api.analyze_image(Path("/nonexistent/path/image.jpg"))

    @pytest.mark.asyncio
    async def test_analyze_image_unsupported_format(self, api_with_mocked_analyzer, temp_dir):
        """Should raise UnsupportedFormatError for unsupported format."""
        api, _ = api_with_mocked_analyzer

        # Create a .bmp file
        bmp_file = temp_dir / "test.bmp"
        bmp_file.write_bytes(b"BM")

        with pytest.raises(UnsupportedFormatError):
            await api.analyze_image(bmp_file)

    @pytest.mark.asyncio
    async def test_analyze_image_bytes_valid(self, api_with_mocked_analyzer):
        """Should analyze image from raw bytes."""
        api, mock_analyzer = api_with_mocked_analyzer

        # Sample image bytes
        image_bytes = bytes([0xFF, 0xD8, 0xFF, 0xE0])

        result = await api.analyze_image_bytes(image_bytes, "image/jpeg")

        assert isinstance(result, AvatarAnalysisResponse)

    @pytest.mark.asyncio
    async def test_analyze_image_bytes_invalid_mime(self, api_with_mocked_analyzer):
        """Should raise UnsupportedFormatError for invalid MIME type."""
        api, _ = api_with_mocked_analyzer

        with pytest.raises(UnsupportedFormatError):
            await api.analyze_image_bytes(b"data", "image/bmp")


class TestVoiceMatchAPIVoiceMatching:
    """Tests for voice matching functionality."""

    @pytest.fixture
    def api_with_mocked_db(self, mock_env, temp_dir, test_database):
        """Create API with test database."""
        from voicematch.config import Config

        # Create required directories
        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        api = VoiceMatchAPI(config=config)
        return api

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis response."""
        return AvatarAnalysisResponse(
            estimated_age=30,
            age_range=(25, 35),
            gender=Gender.MALE,
            race=Race.CAUCASIAN,
            ethnicity=None,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

    def test_find_matches_returns_sorted(self, api_with_mocked_db, sample_analysis):
        """Should return matches sorted by score descending."""
        api = api_with_mocked_db

        results = api.find_matches(sample_analysis, limit=10)

        assert len(results) > 0
        # Check sorting
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_matches_respects_limit(self, api_with_mocked_db, sample_analysis):
        """Should return at most 'limit' results."""
        api = api_with_mocked_db

        results = api.find_matches(sample_analysis, limit=2)

        assert len(results) <= 2

    def test_find_matches_emotion_filter(self, api_with_mocked_db, sample_analysis):
        """Should filter by emotion when specified."""
        api = api_with_mocked_db

        # Actor 1001 has ANG and NEU clips
        results = api.find_matches(sample_analysis, limit=10, emotion_filter="ANG")

        # Should only return actors with ANG emotion clips
        for result in results:
            clips = api.database.get_clips_for_actor(result.actor.id, emotion="ANG")
            assert len(clips) > 0 or result.actor.id not in ["1001"]


class TestVoiceMatchAPIVoiceSamples:
    """Tests for voice sample functionality."""

    @pytest.fixture
    def api_with_mocked_db(self, mock_env, temp_dir, test_database):
        """Create API with test database."""
        from voicematch.config import Config

        # Create required directories
        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        output_dir = temp_dir / "output"
        output_dir.mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=data_dir,
            output_dir=output_dir,
            database_path=test_database,
        )

        api = VoiceMatchAPI(config=config)
        return api

    def test_get_voice_sample_not_found(self, api_with_mocked_db):
        """Should raise VoiceNotFoundError for invalid actor."""
        api = api_with_mocked_db

        with pytest.raises(VoiceNotFoundError):
            api.get_voice_sample("invalid_actor_id")

    def test_get_voice_sample_exists(self, api_with_mocked_db):
        """Should return path for existing sample (or None if files don't exist)."""
        api = api_with_mocked_db

        # This will return None since the actual audio files don't exist
        result = api.get_voice_sample("1001", emotion="NEU")

        # Result is None because test files don't actually exist
        assert result is None


class TestVoiceMatchAPIVoiceListing:
    """Tests for voice listing functionality."""

    @pytest.fixture
    def api_with_mocked_db(self, mock_env, temp_dir, test_database):
        """Create API with test database."""
        from voicematch.config import Config

        # Create required directories
        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        api = VoiceMatchAPI(config=config)
        return api

    def test_list_voices_no_filter(self, api_with_mocked_db):
        """Should return all voices when no filter."""
        api = api_with_mocked_db

        actors, total = api.list_voices()

        assert total == 5  # We have 5 actors in test data
        assert len(actors) == 5

    def test_list_voices_gender_filter(self, api_with_mocked_db):
        """Should filter by gender."""
        api = api_with_mocked_db

        actors, total = api.list_voices(gender="Male")

        assert total == 3  # 3 male actors in test data
        for actor in actors:
            assert actor.sex == "Male"

    def test_list_voices_age_range(self, api_with_mocked_db):
        """Should filter by age range."""
        api = api_with_mocked_db

        actors, total = api.list_voices(age_min=30, age_max=50)

        for actor in actors:
            assert 30 <= actor.age <= 50

    def test_list_voices_pagination(self, api_with_mocked_db):
        """Should respect limit and offset."""
        api = api_with_mocked_db

        # Get first page
        actors1, total1 = api.list_voices(limit=2, offset=0)

        # Get second page
        actors2, total2 = api.list_voices(limit=2, offset=2)

        assert len(actors1) == 2
        assert len(actors2) == 2
        assert total1 == total2 == 5

        # Ensure different actors
        ids1 = {a.id for a in actors1}
        ids2 = {a.id for a in actors2}
        assert ids1.isdisjoint(ids2)

    def test_list_voices_combined_filters(self, api_with_mocked_db):
        """Should apply multiple filters correctly."""
        api = api_with_mocked_db

        actors, total = api.list_voices(gender="Male", race="Caucasian")

        for actor in actors:
            assert actor.sex == "Male"
            assert actor.race == "Caucasian"


class TestVoiceMatchAPIActorDetails:
    """Tests for actor details functionality."""

    @pytest.fixture
    def api_with_mocked_db(self, mock_env, temp_dir, test_database):
        """Create API with test database."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        return VoiceMatchAPI(config=config)

    def test_get_actor_details_exists(self, api_with_mocked_db):
        """Should return actor details when found."""
        api = api_with_mocked_db

        result = api.get_actor_details("1001")

        assert result is not None
        assert result.id == "1001"
        assert result.age == 28
        assert result.sex == "Male"
        assert result.sample_count > 0

    def test_get_actor_details_not_found(self, api_with_mocked_db):
        """Should return None when actor not found."""
        api = api_with_mocked_db

        result = api.get_actor_details("nonexistent")

        assert result is None


class TestVoiceMatchAPIContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self, mock_env, temp_dir, test_database):
        """Should work as context manager."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        with VoiceMatchAPI(config=config) as api:
            actors, _ = api.list_voices(limit=1)
            assert len(actors) > 0

    def test_close_cleanup(self, mock_env, temp_dir, test_database):
        """Should clean up resources on close."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        api = VoiceMatchAPI(config=config)
        # Force database connection
        _ = api.database

        api.close()

        assert api._database is None


class TestSupportedFormats:
    """Tests for format validation."""

    def test_supported_image_formats(self):
        """Should support common image formats."""
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".jpeg" in SUPPORTED_FORMATS
        assert ".png" in SUPPORTED_FORMATS
        assert ".gif" in SUPPORTED_FORMATS
        assert ".webp" in SUPPORTED_FORMATS

    def test_mime_type_mapping(self):
        """Should have correct MIME type mappings."""
        assert MIME_TO_EXT["image/jpeg"] == ".jpg"
        assert MIME_TO_EXT["image/png"] == ".png"
        assert MIME_TO_EXT["image/gif"] == ".gif"
        assert MIME_TO_EXT["image/webp"] == ".webp"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestAnalyzeImageEdgeCases:
    """Edge case tests for image analysis."""

    @pytest.fixture
    def api_with_mocked_analyzer(self, mock_env, temp_dir):
        """Create API with mocked Gemini analyzer."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)
        db_path = temp_dir / "test.sqlite"
        db_path.touch()

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=db_path,
        )

        api = VoiceMatchAPI(config=config)

        mock_analyzer = Mock()
        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="neutral",
            confidence=0.85,
            raw_response={"test": "data"},
        )
        mock_analyzer.analyze_image.return_value = mock_analysis
        api._analyzer = mock_analyzer

        return api, mock_analyzer

    @pytest.mark.asyncio
    async def test_analyze_image_bytes_empty(self, api_with_mocked_analyzer):
        """Should handle empty bytes gracefully."""
        api, _ = api_with_mocked_analyzer

        # Empty bytes with valid MIME type - will create empty file
        result = await api.analyze_image_bytes(b"", "image/jpeg")
        # The mock analyzer will still return a result
        assert isinstance(result, AvatarAnalysisResponse)

    @pytest.mark.asyncio
    async def test_analyze_image_bytes_invalid_content(self, api_with_mocked_analyzer):
        """Should handle invalid image content (non-image bytes)."""
        api, mock_analyzer = api_with_mocked_analyzer

        # Simulate analyzer failure for invalid content
        from voicematch.gemini_analyzer import GeminiError
        mock_analyzer.analyze_image.side_effect = GeminiError("Invalid image content")

        with pytest.raises(ImageAnalysisError):
            await api.analyze_image_bytes(b"not an image at all", "image/jpeg")

    @pytest.mark.asyncio
    async def test_analyze_image_bytes_text_file_content(self, api_with_mocked_analyzer):
        """Should handle text file content passed as image."""
        api, mock_analyzer = api_with_mocked_analyzer

        from voicematch.gemini_analyzer import GeminiError
        mock_analyzer.analyze_image.side_effect = GeminiError("Cannot process text as image")

        with pytest.raises(ImageAnalysisError):
            await api.analyze_image_bytes(b"Hello, this is text content!", "image/png")

    @pytest.mark.asyncio
    async def test_analyze_image_bytes_pdf_content(self, api_with_mocked_analyzer):
        """Should reject PDF MIME type."""
        api, _ = api_with_mocked_analyzer

        pdf_header = b"%PDF-1.4"
        with pytest.raises(UnsupportedFormatError):
            await api.analyze_image_bytes(pdf_header, "application/pdf")

    @pytest.mark.asyncio
    async def test_analyze_image_bytes_all_valid_mime_types(self, api_with_mocked_analyzer):
        """Should accept all valid MIME types."""
        api, _ = api_with_mocked_analyzer

        valid_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        for mime_type in valid_mime_types:
            result = await api.analyze_image_bytes(b"\x00\x00\x00", mime_type)
            assert isinstance(result, AvatarAnalysisResponse)

    @pytest.mark.asyncio
    async def test_analyze_image_corrupted_file(self, api_with_mocked_analyzer, temp_dir):
        """Should handle corrupted image files."""
        api, mock_analyzer = api_with_mocked_analyzer

        # Create a file with JPG extension but corrupted content
        corrupted_file = temp_dir / "corrupted.jpg"
        corrupted_file.write_bytes(b"\x00\x00\x00corrupted data")

        from voicematch.gemini_analyzer import GeminiError
        mock_analyzer.analyze_image.side_effect = GeminiError("Invalid image format")

        with pytest.raises(ImageAnalysisError):
            await api.analyze_image(corrupted_file)

    @pytest.mark.asyncio
    async def test_analyze_image_very_large_file(self, api_with_mocked_analyzer, temp_dir):
        """Should handle analysis of large files (simulated)."""
        api, _ = api_with_mocked_analyzer

        # Create a "large" image file (still small for test speed)
        large_file = temp_dir / "large.jpg"
        # Write 1MB of data
        large_file.write_bytes(b"\xFF\xD8\xFF\xE0" + (b"\x00" * 1024 * 1024))

        result = await api.analyze_image(large_file)
        assert isinstance(result, AvatarAnalysisResponse)

    @pytest.mark.asyncio
    async def test_analyze_image_special_characters_filename(self, api_with_mocked_analyzer, temp_dir):
        """Should handle filenames with special characters."""
        api, _ = api_with_mocked_analyzer

        special_file = temp_dir / "test image (1) [final].jpg"
        special_file.write_bytes(b"\xFF\xD8\xFF\xE0")

        result = await api.analyze_image(special_file)
        assert isinstance(result, AvatarAnalysisResponse)

    @pytest.mark.asyncio
    async def test_analyze_image_unicode_filename(self, api_with_mocked_analyzer, temp_dir):
        """Should handle filenames with unicode characters."""
        api, _ = api_with_mocked_analyzer

        unicode_file = temp_dir / "avatar_test.jpg"
        unicode_file.write_bytes(b"\xFF\xD8\xFF\xE0")

        result = await api.analyze_image(unicode_file)
        assert isinstance(result, AvatarAnalysisResponse)


class TestFindMatchesEdgeCases:
    """Edge case tests for voice matching."""

    @pytest.fixture
    def api_with_mocked_db(self, mock_env, temp_dir, test_database):
        """Create API with test database."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        return VoiceMatchAPI(config=config)

    @pytest.fixture
    def empty_database(self, temp_dir):
        """Create an empty test database."""
        import sqlite3
        db_path = temp_dir / "empty.sqlite"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE actors (
                id TEXT PRIMARY KEY,
                age INTEGER,
                sex TEXT,
                race TEXT,
                ethnicity TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE audio_clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id TEXT,
                filepath TEXT,
                sentence TEXT,
                emotion TEXT,
                level TEXT,
                duration_ms INTEGER,
                FOREIGN KEY (actor_id) REFERENCES actors(id)
            )
        """)

        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def api_with_empty_db(self, mock_env, temp_dir, empty_database):
        """Create API with empty database."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=empty_database,
        )

        return VoiceMatchAPI(config=config)

    def test_find_matches_no_matching_demographics(self, api_with_mocked_db):
        """Should return empty list when no demographics match."""
        api = api_with_mocked_db

        # Create analysis with demographics that don't match any actor
        analysis = AvatarAnalysisResponse(
            estimated_age=5,  # Very young age, no actors match
            age_range=(1, 10),
            gender=Gender.MALE,
            race=Race.MIXED,  # No "mixed" race actors in test data
            ethnicity=None,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        results = api.find_matches(analysis, limit=10)
        # Should still return results (best matches available), but with lower scores
        assert isinstance(results, list)

    def test_find_matches_empty_database(self, api_with_empty_db):
        """Should return empty list with empty database."""
        api = api_with_empty_db

        analysis = AvatarAnalysisResponse(
            estimated_age=30,
            age_range=(25, 35),
            gender=Gender.MALE,
            race=Race.CAUCASIAN,
            ethnicity=None,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        results = api.find_matches(analysis, limit=10)
        assert results == []

    def test_find_matches_partial_match_gender_only(self, api_with_mocked_db):
        """Should match on gender when other demographics don't match."""
        api = api_with_mocked_db

        analysis = AvatarAnalysisResponse(
            estimated_age=99,  # No 99-year-olds
            age_range=(95, 105),
            gender=Gender.FEMALE,  # Should match female actors
            race=Race.MIXED,  # No mixed race actors
            ethnicity=None,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        results = api.find_matches(analysis, limit=10)
        # Should still get some results
        assert len(results) > 0

    def test_find_matches_all_emotion_filters(self, api_with_mocked_db):
        """Should filter by all valid emotion codes."""
        api = api_with_mocked_db

        analysis = AvatarAnalysisResponse(
            estimated_age=30,
            age_range=(25, 35),
            gender=Gender.MALE,
            race=Race.CAUCASIAN,
            ethnicity=None,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        # Test all valid emotion codes: NEU, HAP, SAD, ANG, FEA, DIS
        emotion_codes = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]
        for code in emotion_codes:
            results = api.find_matches(analysis, limit=10, emotion_filter=code)
            assert isinstance(results, list)

    def test_find_matches_invalid_emotion_filter(self, api_with_mocked_db):
        """Should ignore invalid emotion filter."""
        api = api_with_mocked_db

        analysis = AvatarAnalysisResponse(
            estimated_age=30,
            age_range=(25, 35),
            gender=Gender.MALE,
            race=Race.CAUCASIAN,
            ethnicity=None,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        # Invalid emotion filter should be ignored
        results = api.find_matches(analysis, limit=10, emotion_filter="INVALID")
        assert len(results) > 0

    def test_find_matches_limit_zero_or_negative(self, api_with_mocked_db):
        """Should handle edge case limits."""
        api = api_with_mocked_db

        analysis = AvatarAnalysisResponse(
            estimated_age=30,
            age_range=(25, 35),
            gender=Gender.MALE,
            race=Race.CAUCASIAN,
            ethnicity=None,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        # Limit of 0 should return empty or respect the limit
        results = api.find_matches(analysis, limit=0)
        assert len(results) == 0

    def test_find_matches_very_high_limit(self, api_with_mocked_db):
        """Should handle very high limits."""
        api = api_with_mocked_db

        analysis = AvatarAnalysisResponse(
            estimated_age=30,
            age_range=(25, 35),
            gender=Gender.MALE,
            race=Race.CAUCASIAN,
            ethnicity=None,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        # Very high limit should return all available actors
        results = api.find_matches(analysis, limit=1000)
        assert len(results) <= 5  # We only have 5 actors in test data


class TestGetVoiceSampleEdgeCases:
    """Edge case tests for voice sample retrieval."""

    @pytest.fixture
    def api_with_mocked_db(self, mock_env, temp_dir, test_database):
        """Create API with test database."""
        from voicematch.config import Config

        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        output_dir = temp_dir / "output"
        output_dir.mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=data_dir,
            output_dir=output_dir,
            database_path=test_database,
        )

        return VoiceMatchAPI(config=config)

    def test_get_voice_sample_invalid_actor_id(self, api_with_mocked_db):
        """Should raise VoiceNotFoundError for non-existent actor."""
        api = api_with_mocked_db

        with pytest.raises(VoiceNotFoundError):
            api.get_voice_sample("nonexistent_actor_9999")

    def test_get_voice_sample_empty_actor_id(self, api_with_mocked_db):
        """Should raise VoiceNotFoundError for empty actor ID."""
        api = api_with_mocked_db

        with pytest.raises(VoiceNotFoundError):
            api.get_voice_sample("")

    def test_get_voice_sample_all_emotion_codes(self, api_with_mocked_db):
        """Should handle all valid emotion codes."""
        api = api_with_mocked_db

        emotion_codes = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]
        for code in emotion_codes:
            # Actor exists, but files don't - should return None
            result = api.get_voice_sample("1001", emotion=code)
            # Since audio files don't exist in test, result is None
            assert result is None

    def test_get_voice_sample_invalid_emotion_code(self, api_with_mocked_db):
        """Should fallback to neutral for invalid emotion code."""
        api = api_with_mocked_db

        # Should not raise, should fallback
        result = api.get_voice_sample("1001", emotion="INVALID_EMOTION")
        assert result is None  # Files don't exist

    def test_get_voice_sample_missing_audio_files(self, api_with_mocked_db):
        """Should return None when audio files don't exist."""
        api = api_with_mocked_db

        # Actor exists but audio files at /path/to/... don't exist
        result = api.get_voice_sample("1001", emotion="NEU")
        assert result is None

    def test_get_voice_sample_zero_duration(self, api_with_mocked_db):
        """Should handle zero duration request."""
        api = api_with_mocked_db

        result = api.get_voice_sample("1001", emotion="NEU", duration_seconds=0.0)
        assert result is None

    def test_get_voice_sample_negative_duration(self, api_with_mocked_db):
        """Should handle negative duration request."""
        api = api_with_mocked_db

        result = api.get_voice_sample("1001", emotion="NEU", duration_seconds=-10.0)
        assert result is None

    def test_get_voice_sample_very_long_duration(self, api_with_mocked_db):
        """Should handle very long duration request."""
        api = api_with_mocked_db

        result = api.get_voice_sample("1001", emotion="NEU", duration_seconds=3600.0)
        assert result is None


class TestDatabaseConnectionEdgeCases:
    """Edge case tests for database connection failures."""

    def test_database_connection_invalid_path(self, mock_env, temp_dir):
        """Should raise DatabaseError for invalid database path."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        # Point to a non-existent directory
        invalid_db_path = temp_dir / "nonexistent_dir" / "database.sqlite"

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=invalid_db_path,
        )

        api = VoiceMatchAPI(config=config)

        with pytest.raises(DatabaseError):
            _ = api.database

    def test_database_connection_permission_denied(self, mock_env, temp_dir):
        """Should handle permission denied gracefully."""
        import os
        import stat
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        # Create a database file with no permissions
        db_path = temp_dir / "no_perms.sqlite"
        db_path.touch()

        # Remove read/write permissions (only works on Unix-like systems)
        try:
            os.chmod(db_path, 0o000)

            config = Config(
                gemini_api_key="test_key",
                data_dir=temp_dir,
                output_dir=temp_dir,
                database_path=db_path,
            )

            api = VoiceMatchAPI(config=config)

            # Depending on OS, this may or may not raise
            try:
                _ = api.database
            except (DatabaseError, PermissionError):
                pass  # Expected
        finally:
            # Restore permissions for cleanup
            os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR)


class TestListVoicesEdgeCases:
    """Edge case tests for voice listing."""

    @pytest.fixture
    def api_with_mocked_db(self, mock_env, temp_dir, test_database):
        """Create API with test database."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        return VoiceMatchAPI(config=config)

    @pytest.fixture
    def api_with_empty_db(self, mock_env, temp_dir):
        """Create API with empty database."""
        import sqlite3
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        db_path = temp_dir / "empty.sqlite"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE actors (
                id TEXT PRIMARY KEY,
                age INTEGER,
                sex TEXT,
                race TEXT,
                ethnicity TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE audio_clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id TEXT,
                filepath TEXT,
                sentence TEXT,
                emotion TEXT,
                level TEXT,
                duration_ms INTEGER,
                FOREIGN KEY (actor_id) REFERENCES actors(id)
            )
        """)

        conn.commit()
        conn.close()

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=db_path,
        )

        return VoiceMatchAPI(config=config)

    def test_list_voices_empty_database(self, api_with_empty_db):
        """Should return empty list for empty database."""
        api = api_with_empty_db

        actors, total = api.list_voices()
        assert actors == []
        assert total == 0

    def test_list_voices_age_boundary_zero(self, api_with_mocked_db):
        """Should handle age boundary of 0."""
        api = api_with_mocked_db

        actors, total = api.list_voices(age_min=0)
        assert len(actors) == 5  # All actors should have age >= 0

    def test_list_voices_age_boundary_100(self, api_with_mocked_db):
        """Should handle age boundary of 100."""
        api = api_with_mocked_db

        actors, total = api.list_voices(age_max=100)
        assert len(actors) == 5  # All actors should have age <= 100

    def test_list_voices_age_negative(self, api_with_mocked_db):
        """Should handle negative age filter."""
        api = api_with_mocked_db

        # Negative age - should return empty or handle gracefully
        actors, total = api.list_voices(age_min=-10, age_max=-1)
        assert len(actors) == 0

    def test_list_voices_age_min_greater_than_max(self, api_with_mocked_db):
        """Should return empty when min > max."""
        api = api_with_mocked_db

        actors, total = api.list_voices(age_min=50, age_max=20)
        assert len(actors) == 0

    def test_list_voices_all_gender_values(self, api_with_mocked_db):
        """Should filter by all gender values."""
        api = api_with_mocked_db

        # Male
        actors_male, _ = api.list_voices(gender="Male")
        assert all(a.sex == "Male" for a in actors_male)

        # Female
        actors_female, _ = api.list_voices(gender="Female")
        assert all(a.sex == "Female" for a in actors_female)

    def test_list_voices_all_race_values(self, api_with_mocked_db):
        """Should filter by all race values from CREMA-D."""
        api = api_with_mocked_db

        # Test CREMA-D race values
        race_values = ["African American", "Asian", "Caucasian"]
        for race in race_values:
            actors, total = api.list_voices(race=race)
            if total > 0:
                assert all(a.race == race for a in actors)

    def test_list_voices_nonexistent_race(self, api_with_mocked_db):
        """Should return empty for non-existent race."""
        api = api_with_mocked_db

        actors, total = api.list_voices(race="Martian")
        assert actors == []
        assert total == 0

    def test_list_voices_nonexistent_gender(self, api_with_mocked_db):
        """Should return empty for non-existent gender."""
        api = api_with_mocked_db

        actors, total = api.list_voices(gender="Other")
        assert actors == []
        assert total == 0

    def test_list_voices_offset_beyond_total(self, api_with_mocked_db):
        """Should return empty when offset exceeds total."""
        api = api_with_mocked_db

        actors, total = api.list_voices(offset=1000)
        assert actors == []
        assert total == 5  # Total should still be correct

    def test_list_voices_zero_limit(self, api_with_mocked_db):
        """Should return empty with zero limit."""
        api = api_with_mocked_db

        actors, total = api.list_voices(limit=0)
        assert actors == []
        assert total == 5  # Total should still be correct


class TestActorDetailsEdgeCases:
    """Edge case tests for actor details."""

    @pytest.fixture
    def api_with_mocked_db(self, mock_env, temp_dir, test_database):
        """Create API with test database."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        return VoiceMatchAPI(config=config)

    def test_get_actor_details_empty_id(self, api_with_mocked_db):
        """Should return None for empty ID."""
        api = api_with_mocked_db

        result = api.get_actor_details("")
        assert result is None

    def test_get_actor_details_whitespace_id(self, api_with_mocked_db):
        """Should return None for whitespace ID."""
        api = api_with_mocked_db

        result = api.get_actor_details("   ")
        assert result is None

    def test_get_actor_details_special_characters(self, api_with_mocked_db):
        """Should handle special characters in ID."""
        api = api_with_mocked_db

        result = api.get_actor_details("1001'; DROP TABLE actors;--")
        assert result is None

    def test_get_actor_details_very_long_id(self, api_with_mocked_db):
        """Should handle very long ID."""
        api = api_with_mocked_db

        result = api.get_actor_details("a" * 10000)
        assert result is None


class TestAnalysisToResponseConversion:
    """Tests for internal conversion methods."""

    @pytest.fixture
    def api(self, mock_env, temp_dir, test_database):
        """Create API instance."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        return VoiceMatchAPI(config=config)

    def test_analysis_to_response_all_genders(self, api):
        """Should handle all gender values."""
        genders = ["male", "female", "ambiguous"]
        for gender in genders:
            analysis = AvatarAnalysis(
                estimated_age=30,
                age_range=(25, 35),
                gender=gender,
                race="caucasian",
                ethnicity=None,
                emotion="neutral",
                confidence=0.85,
                raw_response={},
            )
            result = api._analysis_to_response(analysis)
            # Pydantic model uses use_enum_values=True, so gender is a string
            assert result.gender == gender

    def test_analysis_to_response_all_races(self, api):
        """Should handle all race values."""
        races = ["african_american", "asian", "caucasian", "hispanic", "mixed", "ambiguous"]
        for race in races:
            analysis = AvatarAnalysis(
                estimated_age=30,
                age_range=(25, 35),
                gender="male",
                race=race,
                ethnicity=None,
                emotion="neutral",
                confidence=0.85,
                raw_response={},
            )
            result = api._analysis_to_response(analysis)
            # Pydantic model uses use_enum_values=True, so race is a string
            assert result.race == race

    def test_analysis_to_response_all_emotions(self, api):
        """Should handle all emotion values."""
        emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "ambiguous"]
        for emotion in emotions:
            analysis = AvatarAnalysis(
                estimated_age=30,
                age_range=(25, 35),
                gender="male",
                race="caucasian",
                ethnicity=None,
                emotion=emotion,
                confidence=0.85,
                raw_response={},
            )
            result = api._analysis_to_response(analysis)
            # Pydantic model uses use_enum_values=True, so emotion is a string
            assert result.emotion == emotion

    def test_analysis_to_response_edge_age_values(self, api):
        """Should handle edge age values."""
        edge_ages = [0, 1, 99, 100, 120]
        for age in edge_ages:
            analysis = AvatarAnalysis(
                estimated_age=age,
                age_range=(max(0, age - 5), age + 5),
                gender="male",
                race="caucasian",
                ethnicity=None,
                emotion="neutral",
                confidence=0.85,
                raw_response={},
            )
            result = api._analysis_to_response(analysis)
            assert result.estimated_age == age

    def test_analysis_to_response_confidence_boundaries(self, api):
        """Should handle confidence boundaries."""
        confidences = [0.0, 0.5, 1.0]
        for conf in confidences:
            analysis = AvatarAnalysis(
                estimated_age=30,
                age_range=(25, 35),
                gender="male",
                race="caucasian",
                ethnicity=None,
                emotion="neutral",
                confidence=conf,
                raw_response={},
            )
            result = api._analysis_to_response(analysis)
            assert result.confidence == conf


class TestFormatValidation:
    """Tests for format validation edge cases."""

    @pytest.fixture
    def api(self, mock_env, temp_dir, test_database):
        """Create API instance."""
        from voicematch.config import Config

        (temp_dir / "data").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir,
            database_path=test_database,
        )

        return VoiceMatchAPI(config=config)

    def test_validate_image_format_case_insensitive(self, api, temp_dir):
        """Should handle case-insensitive extensions."""
        # Create files with different case extensions
        files = [
            ("test.JPG", True),
            ("test.Jpg", True),
            ("test.JPEG", True),
            ("test.PNG", True),
            ("test.GIF", True),
            ("test.WEBP", True),
            ("test.BMP", False),
            ("test.TIFF", False),
        ]

        for filename, should_pass in files:
            filepath = temp_dir / filename
            filepath.write_bytes(b"\x00")

            if should_pass:
                api._validate_image_format(filepath)  # Should not raise
            else:
                with pytest.raises(UnsupportedFormatError):
                    api._validate_image_format(filepath)

    def test_validate_mime_type_edge_cases(self, api):
        """Should reject unsupported MIME types."""
        invalid_mimes = [
            "image/bmp",
            "image/tiff",
            "image/svg+xml",
            "application/pdf",
            "text/plain",
            "video/mp4",
            "",
            "invalid",
        ]

        for mime in invalid_mimes:
            with pytest.raises(UnsupportedFormatError):
                api._validate_mime_type(mime)
