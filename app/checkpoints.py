"""Lifecycle management for durable LangGraph SQLite checkpoints."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

_ALLOWED_STATE_TYPES = [
    ("app.documents", "RetrievedChunk"),
    ("app.documents", "SourceKind"),
    ("app.rag", "AnswerSource"),
    ("app.rag", "AnswerStatus"),
    ("app.rag", "GradedChunk"),
    ("app.rag", "QueryKind"),
]


class SQLiteCheckpointManager:
    """Own a checkpointer connection and metadata used for retention cleanup."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(
            path,
            check_same_thread=False,
            timeout=10,
        )
        self._metadata_connection = sqlite3.connect(
            path,
            check_same_thread=False,
            timeout=10,
        )
        self._metadata_connection.execute("PRAGMA busy_timeout = 10000")
        self._lock = Lock()
        serializer = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_STATE_TYPES)
        self.saver = SqliteSaver(self._checkpoint_connection, serde=serializer)

    def initialize(self) -> None:
        with self._lock:
            self.saver.setup()
            self._metadata_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_threads (
                    thread_id TEXT PRIMARY KEY,
                    completed_at TEXT NOT NULL
                )
                """
            )
            self._metadata_connection.commit()

    def complete(self, thread_id: str) -> None:
        """Register a finished run for thread-level retention cleanup."""

        with self._lock:
            self._metadata_connection.execute(
                """
                INSERT INTO checkpoint_threads (thread_id, completed_at)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET completed_at = excluded.completed_at
                """,
                (thread_id, datetime.now(UTC).isoformat()),
            )
            self._metadata_connection.commit()

    def cleanup(self, retention_days: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        with self._lock:
            rows = self._metadata_connection.execute(
                "SELECT thread_id FROM checkpoint_threads WHERE completed_at < ?",
                (cutoff.isoformat(),),
            ).fetchall()
            thread_ids = [str(row[0]) for row in rows]
            for thread_id in thread_ids:
                self.saver.delete_thread(thread_id)
            if thread_ids:
                placeholders = ",".join("?" for _ in thread_ids)
                self._metadata_connection.execute(
                    f"DELETE FROM checkpoint_threads WHERE thread_id IN ({placeholders})",
                    thread_ids,
                )
                self._metadata_connection.commit()
        return len(thread_ids)

    def completed_thread_count(self) -> int:
        with self._lock:
            row = self._metadata_connection.execute(
                "SELECT COUNT(*) FROM checkpoint_threads"
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._metadata_connection.close()
        self._checkpoint_connection.close()
