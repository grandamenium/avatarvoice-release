"""Voice matching algorithm based on demographic analysis."""

from dataclasses import dataclass
from typing import List

from .database import VoiceDatabase, Actor
from .gemini_analyzer import AvatarAnalysis


@dataclass
class MatchResult:
    """A potential voice match with scoring."""

    actor: Actor
    score: float  # 0.0 to 1.0
    match_details: dict[str, object]  # Breakdown of scoring


class VoiceMatcher:
    """Matches avatar analysis to voice actors."""

    # Scoring weights
    GENDER_WEIGHT = 0.4
    RACE_WEIGHT = 0.35
    AGE_WEIGHT = 0.25

    # Race mapping from Gemini output to CREMA-D
    RACE_MAPPING = {
        "african_american": "African American",
        "asian": "Asian",
        "caucasian": "Caucasian",
        "hispanic": "Caucasian",  # CREMA-D uses ethnicity separately
        "mixed": None,  # Match any
        "ambiguous": None,
    }

    # Gender mapping from Gemini output to CREMA-D
    GENDER_MAPPING = {
        "male": "Male",
        "female": "Female",
        "ambiguous": None,  # Match any
    }

    def __init__(self, database: VoiceDatabase):
        """Initialize the voice matcher.

        Args:
            database: VoiceDatabase instance to query actors from.
        """
        self.database = database

    def find_matches(
        self,
        analysis: AvatarAnalysis,
        limit: int = 5,
    ) -> List[MatchResult]:
        """Find best matching voice actors for the analysis.

        Args:
            analysis: AvatarAnalysis from Gemini.
            limit: Maximum number of matches to return.

        Returns:
            List of MatchResult sorted by score (highest first).
        """
        # Get candidate actors based on gender
        candidates = self._get_candidates(analysis)

        if not candidates:
            return []

        # Score each candidate
        results = []
        for actor in candidates:
            score, details = self._score_actor(actor, analysis)
            results.append(MatchResult(actor=actor, score=score, match_details=details))

        # Sort by score (highest first) and return top N
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _get_candidates(self, analysis: AvatarAnalysis) -> List[Actor]:
        """Get candidate actors based on basic filters.

        Args:
            analysis: AvatarAnalysis to match against.

        Returns:
            List of candidate Actor objects.
        """
        crema_gender = self.GENDER_MAPPING.get(analysis.gender)

        if crema_gender is None:
            # Ambiguous - get all actors
            return self.database.query_actors()
        else:
            # Filter by gender
            return self.database.query_actors(sex=crema_gender)

    def _score_actor(self, actor: Actor, analysis: AvatarAnalysis) -> tuple[float, dict[str, object]]:
        """Calculate match score for a single actor.

        Args:
            actor: Actor to score.
            analysis: AvatarAnalysis to match against.

        Returns:
            Tuple of (score, details_dict).
        """
        details: dict[str, object] = {}

        # Gender score
        gender_score = self._score_gender(actor.sex, analysis.gender)
        details["gender_score"] = gender_score
        details["gender_match"] = actor.sex.lower() == analysis.gender or analysis.gender == "ambiguous"

        # Race score
        race_score = self._score_race(actor.race, analysis.race)
        details["race_score"] = race_score
        details["race_expected"] = self.RACE_MAPPING.get(analysis.race)
        details["race_actual"] = actor.race

        # Age score
        age_score = self._score_age(actor.age, analysis)
        details["age_score"] = age_score
        details["age_diff"] = abs(actor.age - analysis.estimated_age)

        # Weighted total
        total_score = (
            gender_score * self.GENDER_WEIGHT
            + race_score * self.RACE_WEIGHT
            + age_score * self.AGE_WEIGHT
        )

        details["total_score"] = total_score

        return total_score, details

    def _score_gender(self, actor_sex: str, analysis_gender: str) -> float:
        """Score gender match.

        Args:
            actor_sex: Actor's sex from CREMA-D ("Male" or "Female").
            analysis_gender: Gender from analysis ("male", "female", "ambiguous").

        Returns:
            Score from 0.0 to 1.0.
        """
        if analysis_gender == "ambiguous":
            return 0.5  # Neutral score for ambiguous

        expected = self.GENDER_MAPPING.get(analysis_gender)
        if expected is None:
            return 0.5

        return 1.0 if actor_sex == expected else 0.0

    def _score_race(self, actor_race: str, analysis_race: str) -> float:
        """Score race/ethnicity match.

        Args:
            actor_race: Actor's race from CREMA-D.
            analysis_race: Race from analysis.

        Returns:
            Score from 0.0 to 1.0.
        """
        if analysis_race in ("mixed", "ambiguous"):
            return 0.5  # Neutral score for ambiguous

        expected = self.RACE_MAPPING.get(analysis_race)
        if expected is None:
            return 0.5

        if actor_race == expected:
            return 1.0

        # Partial matches
        # Hispanic in Gemini might match Caucasian actors with Hispanic ethnicity
        if analysis_race == "hispanic" and actor_race == "Caucasian":
            return 0.7

        return 0.0

    def _score_age(self, actor_age: int, analysis: AvatarAnalysis) -> float:
        """Score age match using range.

        Args:
            actor_age: Actor's age.
            analysis: AvatarAnalysis with age range.

        Returns:
            Score from 0.0 to 1.0.
        """
        age_min, age_max = analysis.age_range
        estimated = analysis.estimated_age

        # Perfect match if within range
        if age_min <= actor_age <= age_max:
            # Score higher for closer to estimated age
            diff = abs(actor_age - estimated)
            range_size = age_max - age_min
            if range_size > 0:
                return 1.0 - (diff / range_size) * 0.3
            return 1.0

        # Outside range - score based on distance
        if actor_age < age_min:
            distance = age_min - actor_age
        else:
            distance = actor_age - age_max

        # Score decreases with distance
        # 10 years outside range = 0 score
        return max(0.0, 1.0 - (distance / 10))
