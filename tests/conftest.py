"""Shared pytest fixtures for VoiceMatch tests."""

import pytest
from pathlib import Path
import tempfile
import wave
import struct
import sqlite3
import math


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "requires_api: requires live API access")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio_file(temp_dir):
    """Generate a sample WAV file with a sine wave.

    Creates a 1-second mono WAV file at 44100 Hz with a 440 Hz sine wave.
    """
    filepath = temp_dir / "sample.wav"
    sample_rate = 44100
    duration = 1.0  # seconds
    frequency = 440  # Hz (A4 note)
    amplitude = 32767  # Max for 16-bit audio

    num_samples = int(sample_rate * duration)

    with wave.open(str(filepath), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            sample = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav_file.writeframes(struct.pack("<h", sample))

    return filepath


@pytest.fixture
def mock_gemini_response():
    """Return a mock Gemini API response."""
    return {
        "estimated_age": 35,
        "age_range_min": 30,
        "age_range_max": 40,
        "gender": "male",
        "race": "caucasian",
        "ethnicity_notes": None,
        "confidence": 0.85,
    }


@pytest.fixture
def sample_demographics_csv(temp_dir):
    """Create a sample VideoDemographics.csv."""
    csv_content = """ActorID,Age,Sex,Race,Ethnicity
1001,28,Male,African American,Not Hispanic
1002,34,Female,Caucasian,Not Hispanic
1003,45,Male,Asian,Not Hispanic
1004,29,Female,African American,Hispanic
1005,52,Male,Caucasian,Not Hispanic
"""
    csv_path = temp_dir / "VideoDemographics.csv"
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def test_database(temp_dir, sample_demographics_csv):
    """Create a test SQLite database with sample data."""
    db_path = temp_dir / "test_voice.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables
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

    # Insert sample actors
    actors = [
        ("1001", 28, "Male", "African American", "Not Hispanic"),
        ("1002", 34, "Female", "Caucasian", "Not Hispanic"),
        ("1003", 45, "Male", "Asian", "Not Hispanic"),
        ("1004", 29, "Female", "African American", "Hispanic"),
        ("1005", 52, "Male", "Caucasian", "Not Hispanic"),
    ]

    cursor.executemany(
        "INSERT INTO actors (id, age, sex, race, ethnicity) VALUES (?, ?, ?, ?, ?)",
        actors,
    )

    # Insert sample audio clips
    clips = [
        ("1001", "/path/to/1001_IEO_ANG_HI.wav", "IEO", "ANG", "HI", 2500),
        ("1001", "/path/to/1001_IEO_NEU_XX.wav", "IEO", "NEU", "XX", 2200),
        ("1002", "/path/to/1002_TIE_HAP_MD.wav", "TIE", "HAP", "MD", 2800),
        ("1002", "/path/to/1002_TIE_NEU_XX.wav", "TIE", "NEU", "XX", 2400),
        ("1003", "/path/to/1003_IOM_SAD_LO.wav", "IOM", "SAD", "LO", 3000),
    ]

    cursor.executemany(
        """INSERT INTO audio_clips (actor_id, filepath, sentence, emotion, level, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?)""",
        clips,
    )

    # Create indexes
    cursor.execute("CREATE INDEX idx_actor_sex ON actors(sex)")
    cursor.execute("CREATE INDEX idx_actor_race ON actors(race)")
    cursor.execute("CREATE INDEX idx_actor_age ON actors(age)")
    cursor.execute("CREATE INDEX idx_clip_actor ON audio_clips(actor_id)")

    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def mock_env(temp_dir, monkeypatch):
    """Set up mock environment variables for testing."""
    # Reset Config singleton
    from voicematch.config import Config

    Config.reset()

    # Create a .env file
    env_file = temp_dir / ".env"
    env_file.write_text("GEMINI_API_KEY=test_api_key_12345\n")

    # Set environment variables
    monkeypatch.setenv("GEMINI_API_KEY", "test_api_key_12345")
    monkeypatch.setenv("DATA_DIR", str(temp_dir / "data"))
    monkeypatch.setenv("OUTPUT_DIR", str(temp_dir / "output"))
    monkeypatch.setenv("DATABASE_PATH", str(temp_dir / "voice.sqlite"))

    yield

    # Clean up
    Config.reset()
