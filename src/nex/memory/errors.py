"""Error pattern database backed by SQLite.

The database lives at .nex/errors.db and records every error the agent
encounters along with how it was fixed. Before generating new code the
agent queries this DB to surface similar past mistakes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rich.console import Console

from nex.exceptions import NexMemoryError

console: Final[Console] = Console(stderr=True)

_CREATE_TABLE_SQL: Final[str] = """\
CREATE TABLE IF NOT EXISTS error_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    task_summary TEXT NOT NULL,
    error_type TEXT NOT NULL,
    what_failed TEXT NOT NULL,
    what_fixed TEXT NOT NULL,
    file_path TEXT,
    language TEXT,
    code_context TEXT
);
"""

_CREATE_INDEXES_SQL: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_error_language ON error_patterns(language);",
    "CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type);",
)

_INSERT_SQL: Final[str] = """\
INSERT INTO error_patterns
    (task_summary, error_type, what_failed, what_fixed, file_path, language, code_context)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_FIND_SIMILAR_SQL: Final[str] = """\
SELECT id, timestamp, task_summary, error_type, what_failed, what_fixed,
       file_path, language, code_context
FROM error_patterns
WHERE language = ?
  AND (task_summary LIKE ? OR file_path LIKE ?)
ORDER BY timestamp DESC
LIMIT ?
"""

_FIND_ALL_SQL: Final[str] = """\
SELECT id, timestamp, task_summary, error_type, what_failed, what_fixed,
       file_path, language, code_context
FROM error_patterns
ORDER BY timestamp DESC
LIMIT ?
"""

_COUNT_SQL: Final[str] = "SELECT COUNT(*) FROM error_patterns"
_MAX_CODE_CONTEXT_LEN: Final[int] = 500


@dataclass
class ErrorPattern:
    """A single error pattern record.

    Attributes:
        id: Database primary key (None for unsaved records).
        timestamp: ISO-8601 datetime string.
        task_summary: What the user asked for when the error occurred.
        error_type: Category: syntax, runtime, logic, import, test_failure.
        what_failed: Human-readable description of the failure.
        what_fixed: Human-readable description of the fix applied.
        file_path: Path to the file that contained the error.
        language: Programming language of the affected code.
        code_context: Relevant code snippet, truncated to 500 chars.
    """

    id: int | None
    timestamp: str
    task_summary: str
    error_type: str
    what_failed: str
    what_fixed: str
    file_path: str | None = None
    language: str | None = None
    code_context: str | None = None


class ErrorPatternDB:
    """SQLite-backed error pattern database.

    Usage::

        db = ErrorPatternDB(project_dir=Path("/my/project"))
        record_id = db.log_error(ErrorPattern(...))
        similar = db.find_similar(language="python", task_summary="login")
        db.close()
    """

    def __init__(self, project_dir: Path) -> None:
        """Open or create the error pattern database.

        Args:
            project_dir: Project root containing the .nex folder.

        Raises:
            NexMemoryError: If the database cannot be opened.
        """
        self._project_dir = project_dir
        self._db_path = project_dir / ".nex" / "errors.db"
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn: sqlite3.Connection = sqlite3.connect(
                str(self._db_path), check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            self._init_db()
        except sqlite3.Error as exc:
            raise NexMemoryError(
                f"Failed to open error pattern DB at {self._db_path}: {exc}"
            ) from exc

    def __enter__(self) -> ErrorPatternDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    @property
    def db_path(self) -> Path:
        """Return the absolute path to the SQLite database file."""
        return self._db_path

    def _init_db(self) -> None:
        """Create the error_patterns table and indexes if absent."""
        cursor = self._conn.cursor()
        cursor.execute(_CREATE_TABLE_SQL)
        for index_sql in _CREATE_INDEXES_SQL:
            cursor.execute(index_sql)
        self._conn.commit()

    def log_error(self, pattern: ErrorPattern) -> int:
        """Insert an error pattern and return its database id.

        Args:
            pattern: The error record to persist.

        Returns:
            The auto-generated primary key of the new row.

        Raises:
            NexMemoryError: If the INSERT fails.
        """
        code_ctx = pattern.code_context
        if code_ctx and len(code_ctx) > _MAX_CODE_CONTEXT_LEN:
            code_ctx = code_ctx[:_MAX_CODE_CONTEXT_LEN]

        try:
            cursor = self._conn.cursor()
            cursor.execute(
                _INSERT_SQL,
                (
                    pattern.task_summary,
                    pattern.error_type,
                    pattern.what_failed,
                    pattern.what_fixed,
                    pattern.file_path,
                    pattern.language,
                    code_ctx,
                ),
            )
            self._conn.commit()
            row_id: int = cursor.lastrowid  # type: ignore[assignment]
            return row_id
        except sqlite3.Error as exc:
            raise NexMemoryError(f"Failed to log error pattern: {exc}") from exc

    def find_similar(
        self,
        language: str | None = None,
        task_summary: str = "",
        file_path: str = "",
        limit: int = 3,
    ) -> list[ErrorPattern]:
        """Find error patterns similar to the given criteria.

        Args:
            language: Programming language to filter on.
            task_summary: Substring to match against task_summary.
            file_path: Substring to match against file_path.
            limit: Maximum number of results.

        Returns:
            Matching ErrorPattern instances ordered by timestamp descending.

        Raises:
            NexMemoryError: If the query fails.
        """
        task_like = f"%{task_summary}%" if task_summary else "%%"
        file_like = f"%{file_path}%" if file_path else "%%"

        try:
            cursor = self._conn.cursor()
            if language is not None:
                cursor.execute(_FIND_SIMILAR_SQL, (language, task_like, file_like, limit))
            else:
                cursor.execute(_FIND_ALL_SQL, (limit,))
            return [self._row_to_pattern(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise NexMemoryError(f"Failed to query error patterns: {exc}") from exc

    def count(self) -> int:
        """Return the total number of error patterns in the database.

        Raises:
            NexMemoryError: If the query fails.
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(_COUNT_SQL)
            result: int = cursor.fetchone()[0]
            return result
        except sqlite3.Error as exc:
            raise NexMemoryError(f"Failed to count error patterns: {exc}") from exc

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    @staticmethod
    def _row_to_pattern(row: sqlite3.Row) -> ErrorPattern:
        """Convert a database row into an ErrorPattern."""
        return ErrorPattern(
            id=row["id"],
            timestamp=row["timestamp"],
            task_summary=row["task_summary"],
            error_type=row["error_type"],
            what_failed=row["what_failed"],
            what_fixed=row["what_fixed"],
            file_path=row["file_path"],
            language=row["language"],
            code_context=row["code_context"],
        )
