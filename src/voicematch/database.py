"""SQLite database operations for voice metadata."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


class DatabaseError(Exception):
    """Raised when a database operation fails."""

    pass


@dataclass
class Actor:
    """Represents a voice actor from CREMA-D."""

    id: str
    age: int
    sex: str  # "Male" | "Female"
    race: str  # "African American" | "Asian" | "Caucasian" | "Unknown"
    ethnicity: str  # "Hispanic" | "Not Hispanic"


@dataclass
class AudioClip:
    """Represents a single audio clip."""

    id: int
    actor_id: str
    filepath: Path
    sentence: str
    emotion: str
    level: str
    duration_ms: int


class VoiceDatabase:
    """SQLite database for voice metadata."""

    def __init__(self, db_path: Path):
        """Initialize database with path.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "VoiceDatabase":
        """Context manager entry - opens connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes connection."""
        self.close()

    def connect(self) -> None:
        """Open database connection.

        Raises:
            DatabaseError: If connection fails.
        """
        try:
            # Ensure parent directory exists
            if not self.db_path.parent.exists():
                raise DatabaseError(f"Directory does not exist: {self.db_path.parent}")

            # check_same_thread=False allows connection to be used across threads
            # (needed for Gradio which runs callbacks in separate threads)
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database: {e}") from e

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_schema(self) -> None:
        """Create database tables and indexes.

        Raises:
            DatabaseError: If schema creation fails.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.conn.cursor()

            # Create actors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actors (
                    id TEXT PRIMARY KEY,
                    age INTEGER NOT NULL,
                    sex TEXT NOT NULL,
                    race TEXT NOT NULL,
                    ethnicity TEXT NOT NULL,
                    embedding BLOB
                )
            """)

            # Create audio_clips table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audio_clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    sentence TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    level TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    FOREIGN KEY (actor_id) REFERENCES actors(id)
                )
            """)

            # Create indexes for efficient querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_actor_sex ON actors(sex)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_actor_race ON actors(race)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_actor_age ON actors(age)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clip_actor ON audio_clips(actor_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clip_emotion ON audio_clips(emotion)")

            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to create schema: {e}") from e

    def insert_actor(self, actor: Actor) -> None:
        """Insert an actor record.

        Args:
            actor: Actor to insert.

        Raises:
            DatabaseError: If insert fails (e.g., duplicate ID).
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO actors (id, age, sex, race, ethnicity)
                VALUES (?, ?, ?, ?, ?)
                """,
                (actor.id, actor.age, actor.sex, actor.race, actor.ethnicity),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise DatabaseError(f"Actor with ID {actor.id} already exists") from e
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert actor: {e}") from e

    def insert_audio_clip(self, clip: AudioClip) -> int:
        """Insert an audio clip record.

        Args:
            clip: AudioClip to insert.

        Returns:
            The auto-generated ID of the inserted clip.

        Raises:
            DatabaseError: If insert fails.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO audio_clips (actor_id, filepath, sentence, emotion, level, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    clip.actor_id,
                    str(clip.filepath),
                    clip.sentence,
                    clip.emotion,
                    clip.level,
                    clip.duration_ms,
                ),
            )
            self.conn.commit()
            return cursor.lastrowid or 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert audio clip: {e}") from e

    def query_actors(
        self,
        sex: Optional[str] = None,
        race: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
    ) -> List[Actor]:
        """Query actors by demographic criteria.

        Args:
            sex: Filter by sex ("Male" or "Female").
            race: Filter by race.
            min_age: Minimum age (inclusive).
            max_age: Maximum age (inclusive).

        Returns:
            List of matching Actor objects.

        Raises:
            DatabaseError: If query fails.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            query = "SELECT id, age, sex, race, ethnicity FROM actors WHERE 1=1"
            params: List = []

            if sex is not None:
                query += " AND sex = ?"
                params.append(sex)

            if race is not None:
                query += " AND race = ?"
                params.append(race)

            if min_age is not None:
                query += " AND age >= ?"
                params.append(min_age)

            if max_age is not None:
                query += " AND age <= ?"
                params.append(max_age)

            cursor = self.conn.cursor()
            cursor.execute(query, params)

            actors = []
            for row in cursor.fetchall():
                actors.append(
                    Actor(
                        id=row["id"],
                        age=row["age"],
                        sex=row["sex"],
                        race=row["race"],
                        ethnicity=row["ethnicity"],
                    )
                )

            return actors
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to query actors: {e}") from e

    def get_clips_for_actor(
        self,
        actor_id: str,
        emotion: Optional[str] = None,
    ) -> List[AudioClip]:
        """Get all audio clips for an actor.

        Args:
            actor_id: The actor's ID.
            emotion: Optional emotion filter.

        Returns:
            List of AudioClip objects for the actor.

        Raises:
            DatabaseError: If query fails.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            query = """
                SELECT id, actor_id, filepath, sentence, emotion, level, duration_ms
                FROM audio_clips
                WHERE actor_id = ?
            """
            params: List = [actor_id]

            if emotion is not None:
                query += " AND emotion = ?"
                params.append(emotion)

            cursor = self.conn.cursor()
            cursor.execute(query, params)

            clips = []
            for row in cursor.fetchall():
                clips.append(
                    AudioClip(
                        id=row["id"],
                        actor_id=row["actor_id"],
                        filepath=Path(row["filepath"]),
                        sentence=row["sentence"],
                        emotion=row["emotion"],
                        level=row["level"],
                        duration_ms=row["duration_ms"],
                    )
                )

            return clips
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get clips for actor: {e}") from e

    def get_actor_by_id(self, actor_id: str) -> Optional[Actor]:
        """Get a single actor by ID.

        Args:
            actor_id: The actor's ID.

        Returns:
            Actor object if found, None otherwise.

        Raises:
            DatabaseError: If query fails.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, age, sex, race, ethnicity FROM actors WHERE id = ?",
                (actor_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return Actor(
                id=row["id"],
                age=row["age"],
                sex=row["sex"],
                race=row["race"],
                ethnicity=row["ethnicity"],
            )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get actor: {e}") from e

    def get_all_actors(self) -> List[Actor]:
        """Get all actors in the database.

        Returns:
            List of all Actor objects.
        """
        return self.query_actors()

    def get_clip_count_for_actor(self, actor_id: str) -> int:
        """Get the number of clips for an actor.

        Args:
            actor_id: The actor's ID.

        Returns:
            Number of clips.

        Raises:
            DatabaseError: If query fails.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM audio_clips WHERE actor_id = ?",
                (actor_id,),
            )
            result = cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to count clips: {e}") from e

    def store_embedding(self, actor_id: str, embedding: np.ndarray) -> None:
        """Store actor embedding as BLOB.

        Args:
            actor_id: The actor's ID.
            embedding: NumPy array containing the speaker embedding.

        Raises:
            DatabaseError: If storage fails or actor doesn't exist.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            # Convert numpy array to bytes
            embedding_bytes = embedding.tobytes()

            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE actors SET embedding = ? WHERE id = ?",
                (embedding_bytes, actor_id),
            )

            if cursor.rowcount == 0:
                raise DatabaseError(f"Actor with ID {actor_id} not found")

            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to store embedding: {e}") from e

    def get_embedding(self, actor_id: str) -> Optional[np.ndarray]:
        """Retrieve actor embedding.

        Args:
            actor_id: The actor's ID.

        Returns:
            NumPy array containing the embedding, or None if no embedding stored.

        Raises:
            DatabaseError: If query fails.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT embedding FROM actors WHERE id = ?",
                (actor_id,),
            )
            result = cursor.fetchone()

            if result is None:
                raise DatabaseError(f"Actor with ID {actor_id} not found")

            embedding_bytes = result[0]
            if embedding_bytes is None:
                return None

            # Convert bytes back to numpy array
            # Resemblyzer embeddings are 256-dim float32
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            return embedding

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get embedding: {e}") from e

    def get_actors_with_embeddings(self) -> List[Tuple[Actor, np.ndarray]]:
        """Get all actors that have embeddings.

        Returns:
            List of (Actor, embedding) tuples.

        Raises:
            DatabaseError: If query fails.
        """
        if not self.conn:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, age, sex, race, ethnicity, embedding
                FROM actors
                WHERE embedding IS NOT NULL
            """)

            results = []
            for row in cursor.fetchall():
                actor = Actor(
                    id=row["id"],
                    age=row["age"],
                    sex=row["sex"],
                    race=row["race"],
                    ethnicity=row["ethnicity"],
                )
                embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                results.append((actor, embedding))

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get actors with embeddings: {e}") from e
