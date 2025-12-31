"""Unit tests for SQLite database operations."""

import pytest
import sqlite3
from pathlib import Path

from voicematch.database import (
    VoiceDatabase,
    Actor,
    AudioClip,
    DatabaseError,
)


class TestDatabaseSchema:
    """Tests for database schema creation."""

    def test_database_schema_created_correctly(self, temp_dir):
        """Test that database tables and indexes are created correctly."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        # Verify tables exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check actors table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actors'")
        assert cursor.fetchone() is not None

        # Check audio_clips table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audio_clips'")
        assert cursor.fetchone() is not None

        # Check indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_actor_sex'")
        assert cursor.fetchone() is not None

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_actor_race'")
        assert cursor.fetchone() is not None

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_actor_age'")
        assert cursor.fetchone() is not None

        conn.close()
        db.close()

    def test_actors_table_populated(self, temp_dir):
        """Test that actors can be inserted and retrieved."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        actor = Actor(
            id="1001",
            age=51,
            sex="Male",
            race="Caucasian",
            ethnicity="Not Hispanic",
        )
        db.insert_actor(actor)

        # Retrieve the actor
        actors = db.query_actors()
        assert len(actors) == 1
        assert actors[0].id == "1001"
        assert actors[0].age == 51
        assert actors[0].sex == "Male"

        db.close()

    def test_audio_clips_linked_to_actors(self, temp_dir):
        """Test that audio clips are correctly linked to actors."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        # Insert actor first
        actor = Actor(id="1001", age=51, sex="Male", race="Caucasian", ethnicity="Not Hispanic")
        db.insert_actor(actor)

        # Insert audio clip
        clip = AudioClip(
            id=0,  # Will be auto-assigned
            actor_id="1001",
            filepath=Path("/path/to/1001_IEO_ANG_HI.wav"),
            sentence="IEO",
            emotion="ANG",
            level="HI",
            duration_ms=2500,
        )
        db.insert_audio_clip(clip)

        # Retrieve clips for actor
        clips = db.get_clips_for_actor("1001")
        assert len(clips) == 1
        assert clips[0].actor_id == "1001"
        assert clips[0].emotion == "ANG"

        db.close()


class TestDatabaseQueries:
    """Tests for database query operations."""

    def test_query_actors_by_sex(self, temp_dir):
        """Test filtering actors by sex."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        # Insert multiple actors
        db.insert_actor(Actor("1001", 51, "Male", "Caucasian", "Not Hispanic"))
        db.insert_actor(Actor("1002", 21, "Female", "Caucasian", "Not Hispanic"))
        db.insert_actor(Actor("1003", 35, "Male", "Asian", "Not Hispanic"))

        # Query by sex
        males = db.query_actors(sex="Male")
        assert len(males) == 2

        females = db.query_actors(sex="Female")
        assert len(females) == 1

        db.close()

    def test_query_actors_by_race(self, temp_dir):
        """Test filtering actors by race."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        db.insert_actor(Actor("1001", 51, "Male", "Caucasian", "Not Hispanic"))
        db.insert_actor(Actor("1002", 21, "Female", "Asian", "Not Hispanic"))
        db.insert_actor(Actor("1003", 35, "Male", "African American", "Not Hispanic"))

        asian = db.query_actors(race="Asian")
        assert len(asian) == 1
        assert asian[0].id == "1002"

        db.close()

    def test_query_actors_by_age_range(self, temp_dir):
        """Test filtering actors by age range."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        db.insert_actor(Actor("1001", 25, "Male", "Caucasian", "Not Hispanic"))
        db.insert_actor(Actor("1002", 35, "Female", "Caucasian", "Not Hispanic"))
        db.insert_actor(Actor("1003", 50, "Male", "Caucasian", "Not Hispanic"))

        # Query actors between 30 and 40
        actors = db.query_actors(min_age=30, max_age=40)
        assert len(actors) == 1
        assert actors[0].id == "1002"

        # Query actors 30 and older
        actors = db.query_actors(min_age=30)
        assert len(actors) == 2

        db.close()

    def test_get_clips_for_actor_filtered_by_emotion(self, temp_dir):
        """Test filtering clips by emotion."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        db.insert_actor(Actor("1001", 51, "Male", "Caucasian", "Not Hispanic"))

        db.insert_audio_clip(
            AudioClip(0, "1001", Path("/p/1001_IEO_ANG_HI.wav"), "IEO", "ANG", "HI", 2500)
        )
        db.insert_audio_clip(
            AudioClip(0, "1001", Path("/p/1001_IEO_NEU_XX.wav"), "IEO", "NEU", "XX", 2200)
        )
        db.insert_audio_clip(
            AudioClip(0, "1001", Path("/p/1001_TIE_HAP_MD.wav"), "TIE", "HAP", "MD", 2800)
        )

        # Filter by emotion
        neutral_clips = db.get_clips_for_actor("1001", emotion="NEU")
        assert len(neutral_clips) == 1

        all_clips = db.get_clips_for_actor("1001")
        assert len(all_clips) == 3

        db.close()


class TestDatabaseEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_missing_database(self, temp_dir):
        """Test that missing database is handled gracefully."""
        db_path = temp_dir / "nonexistent" / "test.sqlite"
        db = VoiceDatabase(db_path)

        # Should raise error when connecting to non-existent directory
        with pytest.raises(DatabaseError):
            db.connect()

    def test_handles_duplicate_actor_id(self, temp_dir):
        """Test handling of duplicate actor IDs."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        db.insert_actor(Actor("1001", 51, "Male", "Caucasian", "Not Hispanic"))

        # Attempting to insert duplicate should raise error or update
        with pytest.raises(DatabaseError):
            db.insert_actor(Actor("1001", 52, "Male", "Asian", "Not Hispanic"))

        db.close()

    def test_query_returns_empty_for_no_matches(self, temp_dir):
        """Test that queries return empty list when no matches found."""
        db_path = temp_dir / "test.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        db.insert_actor(Actor("1001", 51, "Male", "Caucasian", "Not Hispanic"))

        # Query for non-existent criteria
        actors = db.query_actors(race="Asian")
        assert actors == []

        clips = db.get_clips_for_actor("9999")
        assert clips == []

        db.close()

    def test_database_context_manager(self, temp_dir):
        """Test using database as context manager."""
        db_path = temp_dir / "test.sqlite"

        with VoiceDatabase(db_path) as db:
            db.create_schema()
            db.insert_actor(Actor("1001", 51, "Male", "Caucasian", "Not Hispanic"))
            actors = db.query_actors()
            assert len(actors) == 1

        # Connection should be closed after context
        # Reopening should work fine
        with VoiceDatabase(db_path) as db:
            actors = db.query_actors()
            assert len(actors) == 1
