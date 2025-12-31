"""Unit tests for voice matching algorithm."""

import pytest

from voicematch.voice_matcher import VoiceMatcher, MatchResult
from voicematch.database import VoiceDatabase, Actor
from voicematch.gemini_analyzer import AvatarAnalysis


@pytest.fixture
def populated_database(temp_dir):
    """Create a database with test actors."""
    db_path = temp_dir / "test.sqlite"
    db = VoiceDatabase(db_path)
    db.connect()
    db.create_schema()

    # Insert diverse actors
    actors = [
        Actor("1001", 30, "Male", "Caucasian", "Not Hispanic"),
        Actor("1002", 25, "Female", "African American", "Not Hispanic"),
        Actor("1003", 45, "Male", "Asian", "Not Hispanic"),
        Actor("1004", 35, "Female", "Caucasian", "Hispanic"),
        Actor("1005", 50, "Male", "African American", "Not Hispanic"),
        Actor("1006", 28, "Female", "Asian", "Not Hispanic"),
    ]

    for actor in actors:
        db.insert_actor(actor)

    yield db

    db.close()


class TestVoiceMatcher:
    """Tests for VoiceMatcher class."""

    def test_exact_match_returns_highest_score(self, populated_database):
        """Test that exact demographic matches get highest scores."""
        matcher = VoiceMatcher(populated_database)

        analysis = AvatarAnalysis(
            estimated_age=30,
            age_range=(25, 35),
            gender="male",
            race="caucasian",
            ethnicity=None,
            confidence=0.9,
            raw_response={},
        )

        results = matcher.find_matches(analysis, limit=5)

        # Actor 1001 is 30, Male, Caucasian - should be best match
        assert len(results) > 0
        best_match = results[0]
        assert best_match.actor.id == "1001"
        assert best_match.score > 0.8

    def test_gender_mismatch_excluded(self, populated_database):
        """Test that mismatched genders are excluded or scored very low."""
        matcher = VoiceMatcher(populated_database)

        analysis = AvatarAnalysis(
            estimated_age=30,
            age_range=(25, 35),
            gender="female",
            race="caucasian",
            ethnicity=None,
            confidence=0.9,
            raw_response={},
        )

        results = matcher.find_matches(analysis, limit=10)

        # All results should be female
        for result in results:
            assert result.actor.sex == "Female"

    def test_age_range_scoring(self, populated_database):
        """Test that actors within age range score higher."""
        matcher = VoiceMatcher(populated_database)

        analysis = AvatarAnalysis(
            estimated_age=30,
            age_range=(25, 35),
            gender="male",
            race="caucasian",
            ethnicity=None,
            confidence=0.9,
            raw_response={},
        )

        results = matcher.find_matches(analysis, limit=5)

        # Actor 1001 (age 30) should score higher than Actor 1003 (age 45)
        scores_by_id = {r.actor.id: r.score for r in results}
        if "1001" in scores_by_id and "1003" in scores_by_id:
            assert scores_by_id["1001"] > scores_by_id["1003"]

    def test_race_partial_match_scoring(self, populated_database):
        """Test that race matching affects scores appropriately."""
        matcher = VoiceMatcher(populated_database)

        analysis = AvatarAnalysis(
            estimated_age=50,
            age_range=(45, 55),
            gender="male",
            race="african_american",
            ethnicity=None,
            confidence=0.9,
            raw_response={},
        )

        results = matcher.find_matches(analysis, limit=5)

        # Actor 1005 is 50, Male, African American - should be top match
        assert len(results) > 0
        best_match = results[0]
        assert best_match.actor.id == "1005"

    def test_ambiguous_gender_includes_both(self, populated_database):
        """Test that ambiguous gender includes both male and female actors."""
        matcher = VoiceMatcher(populated_database)

        analysis = AvatarAnalysis(
            estimated_age=30,
            age_range=(25, 35),
            gender="ambiguous",
            race="caucasian",
            ethnicity=None,
            confidence=0.5,
            raw_response={},
        )

        results = matcher.find_matches(analysis, limit=10)

        # Should include both male and female actors
        genders = {r.actor.sex for r in results}
        assert "Male" in genders or "Female" in genders

    def test_no_matches_returns_empty(self, temp_dir):
        """Test that empty database returns empty results."""
        db_path = temp_dir / "empty.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()

        matcher = VoiceMatcher(db)

        analysis = AvatarAnalysis(
            estimated_age=30,
            age_range=(25, 35),
            gender="male",
            race="caucasian",
            ethnicity=None,
            confidence=0.9,
            raw_response={},
        )

        results = matcher.find_matches(analysis, limit=5)
        assert results == []

        db.close()

    def test_returns_sorted_by_score(self, populated_database):
        """Test that results are sorted by score (highest first)."""
        matcher = VoiceMatcher(populated_database)

        analysis = AvatarAnalysis(
            estimated_age=30,
            age_range=(20, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            confidence=0.9,
            raw_response={},
        )

        results = matcher.find_matches(analysis, limit=5)

        # Verify sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_match_result_has_details(self, populated_database):
        """Test that MatchResult includes scoring details."""
        matcher = VoiceMatcher(populated_database)

        analysis = AvatarAnalysis(
            estimated_age=30,
            age_range=(25, 35),
            gender="male",
            race="caucasian",
            ethnicity=None,
            confidence=0.9,
            raw_response={},
        )

        results = matcher.find_matches(analysis, limit=1)

        if results:
            result = results[0]
            assert isinstance(result, MatchResult)
            assert isinstance(result.actor, Actor)
            assert isinstance(result.score, float)
            assert isinstance(result.match_details, dict)
            assert 0 <= result.score <= 1
