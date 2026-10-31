"""SQLite persistence for query answers and user feedback."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from app.errors import AnswerNotFoundError
from app.rag import QueryResult
from app.schemas import FeedbackRating


@dataclass(frozen=True, slots=True)
class StoredAnswer:
    answer_id: UUID
    question: str
    answer: str
    status: str
    rewrite_count: int


class SQLiteRepository:
    """Use short-lived SQLite connections for thread-safe local persistence."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS answers (
                    answer_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    rewrite_count INTEGER NOT NULL CHECK (rewrite_count >= 0),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    answer_id TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                    comment TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (answer_id) REFERENCES answers(answer_id)
                );

                CREATE INDEX IF NOT EXISTS feedback_answer_id_idx
                ON feedback(answer_id);
                """
            )

    def save_answer(self, question: str, result: QueryResult) -> UUID:
        answer_id = uuid4()
        sources_json = json.dumps(
            [asdict(source) for source in result.sources],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO answers (
                    answer_id, question, answer, status, sources_json,
                    rewrite_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(answer_id),
                    question,
                    result.answer,
                    result.status.value,
                    sources_json,
                    result.rewrite_count,
                    _utc_now(),
                ),
            )
        return answer_id

    def add_feedback(
        self,
        *,
        answer_id: UUID,
        rating: FeedbackRating,
        comment: str | None,
    ) -> UUID:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM answers WHERE answer_id = ?",
                (str(answer_id),),
            ).fetchone()
            if exists is None:
                raise AnswerNotFoundError(f"Answer {answer_id} was not found")
            feedback_id = uuid4()
            connection.execute(
                """
                INSERT INTO feedback (
                    feedback_id, answer_id, rating, comment, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(feedback_id),
                    str(answer_id),
                    rating.value,
                    comment,
                    _utc_now(),
                ),
            )
        return feedback_id

    def get_answer(self, answer_id: UUID) -> StoredAnswer | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT answer_id, question, answer, status, rewrite_count
                FROM answers
                WHERE answer_id = ?
                """,
                (str(answer_id),),
            ).fetchone()
        if row is None:
            return None
        return StoredAnswer(
            answer_id=UUID(row["answer_id"]),
            question=row["question"],
            answer=row["answer"],
            status=row["status"],
            rewrite_count=row["rewrite_count"],
        )

    def feedback_count(self, answer_id: UUID) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback WHERE answer_id = ?",
                (str(answer_id),),
            ).fetchone()
        return int(row["count"])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
