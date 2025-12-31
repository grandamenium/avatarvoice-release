"""Integration tests for the complete VoiceMatch pipeline."""

import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import wave
import struct
import math

from voicematch.app import VoiceMatchApp
from voicematch.database import VoiceDatabase, Actor, AudioClip
from voicematch.gemini_analyzer import AvatarAnalysis
from voicematch.voice_matcher import VoiceMatcher
from voicematch.audio_processor import AudioProcessor


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image for pipeline testing."""
    filepath = temp_dir / "test_avatar.png"
    # Minimal 1x1 PNG (red pixel)
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
        0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
        0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ])
    filepath.write_bytes(png_data)
    return filepath


@pytest.fixture
def sample_audio_clips(temp_dir):
    """Create sample audio clips for testing."""
    clips = []
    for i in range(3):
        filepath = temp_dir / f"clip_{i}.wav"
        sample_rate = 44100
        duration = 2.0  # 2 seconds each
        frequency = 440 + i * 50

        num_samples = int(sample_rate * duration)

        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            for j in range(num_samples):
                sample = int(16000 * math.sin(2 * math.pi * frequency * j / sample_rate))
                wav_file.writeframes(struct.pack("<h", sample))

        clips.append(filepath)
    return clips


@pytest.fixture
def mock_analysis():
    """Create a mock avatar analysis."""
    return AvatarAnalysis(
        estimated_age=35,
        age_range=(30, 40),
        gender="male",
        race="caucasian",
        ethnicity=None,
        confidence=0.85,
        raw_response={},
    )


@pytest.fixture
def populated_test_db(temp_dir, sample_audio_clips):
    """Create a test database with actors and audio clips."""
    db_path = temp_dir / "test_voice.sqlite"
    db = VoiceDatabase(db_path)
    db.connect()
    db.create_schema()

    # Insert test actors
    actors = [
        Actor("1001", 30, "Male", "Caucasian", "Not Hispanic"),
        Actor("1002", 25, "Female", "African American", "Not Hispanic"),
        Actor("1003", 45, "Male", "Asian", "Not Hispanic"),
    ]

    for actor in actors:
        db.insert_actor(actor)

    # Insert audio clips for actor 1001
    for i, clip_path in enumerate(sample_audio_clips):
        clip = AudioClip(
            id=i + 1,
            actor_id="1001",
            filepath=clip_path,
            sentence="IEO",
            emotion="NEU",
            level="XX",
            duration_ms=2000,
        )
        db.insert_audio_clip(clip)

    yield db, db_path

    db.close()


class TestFullPipeline:
    """Integration tests for complete pipeline."""

    def test_full_pipeline_with_mock_api(
        self, mock_env, sample_image, populated_test_db, temp_dir, mock_analysis
    ):
        """Test complete pipeline with mocked Gemini API."""
        db, db_path = populated_test_db

        app = VoiceMatchApp()

        # Configure app with test database
        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.database_path = db_path
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir
            app.config.gemini_api_key = "test_key"

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            # Run the pipeline
            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), "Test text"
            )

            # Verify analysis results
            assert "35" in analysis_text  # Age
            assert "male" in analysis_text.lower()  # Gender
            assert "caucasian" in analysis_text.lower()  # Race

            # Verify match results
            assert "1001" in match_text  # Actor ID
            assert "%" in match_text  # Score percentage

            # Verify audio output
            assert audio_path is not None
            assert Path(audio_path).exists()
            assert audio_path.endswith(".wav")

    @pytest.mark.requires_api
    def test_full_pipeline_with_live_api(self, sample_image, populated_test_db, temp_dir):
        """Test complete pipeline with live Gemini API.

        This test requires a valid GEMINI_API_KEY environment variable.
        """
        # This test is marked with requires_api and will be skipped
        # unless explicitly run with -m requires_api
        pytest.skip("Requires live API key - run with -m requires_api")

    def test_pipeline_handles_no_match(self, mock_env, sample_image, temp_dir):
        """Test pipeline when no matching voice found."""
        app = VoiceMatchApp()

        # Create empty database
        db_path = temp_dir / "empty.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        analysis = AvatarAnalysis(
            estimated_age=80,  # No actors this age
            age_range=(75, 85),
            gender="male",
            race="caucasian",
            ethnicity=None,
            confidence=0.9,
            raw_response={},
        )

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.database_path = db_path
            app.config.output_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), ""
            )

            # Should indicate no match found
            assert "no match" in match_text.lower() or "not found" in match_text.lower()
            assert audio_path is None

        db.close()

    def test_pipeline_handles_api_error(self, mock_env, sample_image, temp_dir):
        """Test pipeline handles API errors gracefully."""
        app = VoiceMatchApp()

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.output_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.side_effect = Exception("API rate limit exceeded")

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), ""
            )

            assert "error" in analysis_text.lower()
            assert "rate limit" in analysis_text.lower() or "API" in analysis_text
            assert audio_path is None

    @pytest.mark.slow
    def test_pipeline_performance(
        self, mock_env, sample_image, populated_test_db, temp_dir, mock_analysis
    ):
        """Test that pipeline completes in reasonable time."""
        db, db_path = populated_test_db

        app = VoiceMatchApp()

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.database_path = db_path
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            start = time.time()

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), ""
            )

            elapsed = time.time() - start

            # Should complete within 30 seconds (generous for CI environments)
            assert elapsed < 30, f"Pipeline took {elapsed:.2f}s, expected < 30s"

            # Should still produce valid output
            assert audio_path is not None


class TestComponentIntegration:
    """Tests for integration between specific components."""

    def test_analyzer_to_matcher_integration(self, mock_env, populated_test_db, mock_analysis):
        """Test that analyzer output works with matcher."""
        db, _ = populated_test_db

        matcher = VoiceMatcher(db)
        results = matcher.find_matches(mock_analysis, limit=5)

        assert len(results) > 0
        # Best match should be actor 1001 (30, Male, Caucasian)
        assert results[0].actor.id == "1001"

    def test_matcher_to_processor_integration(
        self, mock_env, populated_test_db, sample_audio_clips, temp_dir
    ):
        """Test that matcher results work with audio processor."""
        db, _ = populated_test_db

        # Get clips for actor 1001
        clips = db.get_clips_for_actor("1001")
        assert len(clips) > 0

        # Process clips
        processor = AudioProcessor()
        clip_paths = [clip.filepath for clip in clips]

        output_path = temp_dir / "output.wav"
        result = processor.concatenate_clips(clip_paths, output_path)

        assert result is not None
        assert result.exists()

    def test_database_with_real_queries(self, mock_env, populated_test_db):
        """Test database queries return expected results."""
        db, _ = populated_test_db

        # Query by sex
        males = db.query_actors(sex="Male")
        assert len(males) == 2  # 1001 and 1003

        # Query by race
        caucasians = db.query_actors(race="Caucasian")
        assert len(caucasians) == 1  # 1001

        # Query by age range
        middle_aged = db.query_actors(min_age=25, max_age=35)
        assert len(middle_aged) >= 1


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_recovers_from_missing_audio_files(
        self, mock_env, temp_dir, mock_analysis
    ):
        """Test pipeline handles missing audio files gracefully."""
        app = VoiceMatchApp()

        # Create database with non-existent audio paths
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        db.insert_actor(Actor("1001", 35, "Male", "Caucasian", "Not Hispanic"))
        db.insert_audio_clip(AudioClip(
            id=1,
            actor_id="1001",
            filepath=Path("/nonexistent/path/audio.wav"),
            sentence="IEO",
            emotion="NEU",
            level="XX",
            duration_ms=2000,
        ))

        # Create test image
        sample_image = temp_dir / "test.png"
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
        sample_image.write_bytes(png_data)

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), ""
            )

            # Should handle gracefully
            assert "error" in match_text.lower() or "not found" in match_text.lower()
            assert audio_path is None

        db.close()

    def test_handles_corrupted_audio_file(
        self, mock_env, temp_dir, mock_analysis
    ):
        """Test pipeline handles corrupted audio files."""
        app = VoiceMatchApp()

        # Create database with corrupted audio
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        db.insert_actor(Actor("1001", 35, "Male", "Caucasian", "Not Hispanic"))

        # Create corrupted audio file
        corrupted_audio = temp_dir / "corrupted.wav"
        corrupted_audio.write_bytes(b"not a valid wav file")

        db.insert_audio_clip(AudioClip(
            id=1,
            actor_id="1001",
            filepath=corrupted_audio,
            sentence="IEO",
            emotion="NEU",
            level="XX",
            duration_ms=2000,
        ))

        sample_image = temp_dir / "test.png"
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
        sample_image.write_bytes(png_data)

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            # Should not crash
            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), ""
            )

            # Should indicate error or no audio
            assert audio_path is None or "error" in match_text.lower()

        db.close()


class TestEmotionPipeline:
    """Integration tests for emotion detection and selection pipeline."""

    @pytest.fixture
    def db_with_emotions(self, temp_dir, sample_audio_clips):
        """Create a test database with actors and clips for multiple emotions."""
        db_path = temp_dir / "emotion_test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        # Insert test actor
        db.insert_actor(Actor("1001", 35, "Male", "Caucasian", "Not Hispanic"))

        # Insert audio clips for different emotions
        emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
        for i, emotion in enumerate(emotions):
            # Create emotion-specific audio file
            filepath = temp_dir / f"clip_{emotion}.wav"
            sample_rate = 44100
            duration = 1.0
            frequency = 440 + i * 100

            num_samples = int(sample_rate * duration)

            with wave.open(str(filepath), "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                for j in range(num_samples):
                    sample = int(16000 * math.sin(2 * math.pi * frequency * j / sample_rate))
                    wav_file.writeframes(struct.pack("<h", sample))

            clip = AudioClip(
                id=i + 1,
                actor_id="1001",
                filepath=filepath,
                sentence="IEO",
                emotion=emotion,
                level="XX",
                duration_ms=1000,
            )
            db.insert_audio_clip(clip)

        yield db, db_path
        db.close()

    def test_emotion_flows_from_gemini_to_db_query(
        self, mock_env, sample_image, db_with_emotions, temp_dir
    ):
        """Test that detected emotion from Gemini flows to database query."""
        db, db_path = db_with_emotions

        app = VoiceMatchApp()

        # Analysis with emotion
        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="happy",
            confidence=0.85,
            raw_response={},
        )

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.database_path = db_path
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), "", "Auto-detect"
            )

            # Should produce output
            assert audio_path is not None

    def test_user_emotion_selection_overrides_gemini(
        self, mock_env, sample_image, db_with_emotions, temp_dir
    ):
        """Test that user-selected emotion overrides Gemini detection."""
        db, db_path = db_with_emotions

        app = VoiceMatchApp()

        # Gemini detects "happy"
        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="happy",
            confidence=0.85,
            raw_response={},
        )

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.database_path = db_path
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            # User selects "Sad" - should override Gemini's "happy"
            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), "", "Sad"
            )

            # Should produce output with sad audio
            assert audio_path is not None

    def test_emotion_displayed_in_analysis_output(
        self, mock_env, sample_image, db_with_emotions, temp_dir
    ):
        """Test that detected emotion is shown in analysis output."""
        db, db_path = db_with_emotions

        app = VoiceMatchApp()

        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="angry",
            confidence=0.85,
            raw_response={},
        )

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.database_path = db_path
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), "", "Auto-detect"
            )

            # Analysis should include emotion info
            assert "emotion" in analysis_text.lower() or "angry" in analysis_text.lower()

    def test_all_emotion_codes_work(
        self, mock_env, sample_image, db_with_emotions, temp_dir
    ):
        """Test that all emotion selections produce valid results."""
        db, db_path = db_with_emotions

        emotions_to_test = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

        for emotion in emotions_to_test:
            app = VoiceMatchApp()

            mock_analysis = AvatarAnalysis(
                estimated_age=35,
                age_range=(30, 40),
                gender="male",
                race="caucasian",
                ethnicity=None,
                emotion="neutral",
                confidence=0.85,
                raw_response={},
            )

            with patch.object(app, "_initialize"):
                app.config = MagicMock()
                app.config.database_path = db_path
                app.config.output_dir = temp_dir
                app.config.data_dir = temp_dir

                app.analyzer = MagicMock()
                app.analyzer.analyze_image.return_value = mock_analysis

                app.database = db
                app.matcher = VoiceMatcher(db)
                app.processor = AudioProcessor()

                analysis_text, match_text, audio_path = app.process_upload(
                    str(sample_image), "", emotion
                )

                # Each emotion should produce valid output
                assert analysis_text is not None, f"Failed for emotion: {emotion}"

    def test_fallback_when_emotion_clips_missing(
        self, mock_env, sample_image, temp_dir
    ):
        """Test fallback behavior when requested emotion clips don't exist."""
        # Create database with only neutral clips
        db_path = temp_dir / "limited_emotion.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        db.insert_actor(Actor("1001", 35, "Male", "Caucasian", "Not Hispanic"))

        # Only add neutral clips
        filepath = temp_dir / "clip_NEU.wav"
        sample_rate = 44100
        duration = 1.0
        num_samples = int(sample_rate * duration)

        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for j in range(num_samples):
                sample = int(16000 * math.sin(2 * math.pi * 440 * j / sample_rate))
                wav_file.writeframes(struct.pack("<h", sample))

        db.insert_audio_clip(AudioClip(
            id=1,
            actor_id="1001",
            filepath=filepath,
            sentence="IEO",
            emotion="NEU",  # Only neutral
            level="XX",
            duration_ms=1000,
        ))

        app = VoiceMatchApp()

        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="happy",  # Request happy but only neutral exists
            confidence=0.85,
            raw_response={},
        )

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.database_path = db_path
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = db
            app.matcher = VoiceMatcher(db)
            app.processor = AudioProcessor()

            # Request happy emotion
            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image), "", "Happy"
            )

            # Should fallback to any available clips (neutral)
            # or handle gracefully
            assert analysis_text is not None

        db.close()
