"""Tests for the error pattern database (uses real in-memory SQLite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nex.memory.errors import ErrorPattern, ErrorPatternDB


@pytest.fixture
def error_db(tmp_path: Path) -> ErrorPatternDB:
    (tmp_path / ".nex").mkdir()
    db = ErrorPatternDB(project_dir=tmp_path)
    yield db
    db.close()


def _make_pattern(**kwargs: str) -> ErrorPattern:
    defaults = {
        "id": None,
        "timestamp": "",
        "task_summary": "add login endpoint",
        "error_type": "import",
        "what_failed": "ModuleNotFoundError: bcrypt",
        "what_fixed": "pip install bcrypt",
        "file_path": "src/auth.py",
        "language": "python",
        "code_context": None,
    }
    defaults.update(kwargs)
    return ErrorPattern(**defaults)  # type: ignore[arg-type]


class TestErrorPatternDB:
    def test_log_and_count(self, error_db: ErrorPatternDB) -> None:
        assert error_db.count() == 0
        row_id = error_db.log_error(_make_pattern())
        assert row_id == 1
        assert error_db.count() == 1

    def test_find_similar_by_language(self, error_db: ErrorPatternDB) -> None:
        error_db.log_error(_make_pattern(language="python"))
        error_db.log_error(_make_pattern(language="javascript"))

        results = error_db.find_similar(language="python")
        assert len(results) == 1
        assert results[0].language == "python"

    def test_find_similar_by_task(self, error_db: ErrorPatternDB) -> None:
        error_db.log_error(_make_pattern(task_summary="add login", file_path="src/auth.py"))
        error_db.log_error(_make_pattern(task_summary="add payment", file_path="src/payment.py"))

        results = error_db.find_similar(language="python", task_summary="login", file_path="auth")
        assert len(results) == 1

    def test_find_similar_no_language(self, error_db: ErrorPatternDB) -> None:
        error_db.log_error(_make_pattern())
        error_db.log_error(_make_pattern())
        results = error_db.find_similar()
        assert len(results) == 2

    def test_code_context_truncation(self, error_db: ErrorPatternDB) -> None:
        long_context = "x" * 1000
        row_id = error_db.log_error(_make_pattern(code_context=long_context))
        results = error_db.find_similar()
        assert len(results[0].code_context or "") <= 500

    def test_context_manager(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir(exist_ok=True)
        with ErrorPatternDB(project_dir=tmp_path) as db:
            db.log_error(_make_pattern())
            assert db.count() == 1

    def test_limit(self, error_db: ErrorPatternDB) -> None:
        for i in range(10):
            error_db.log_error(_make_pattern(task_summary=f"task {i}"))
        results = error_db.find_similar(limit=3)
        assert len(results) == 3
