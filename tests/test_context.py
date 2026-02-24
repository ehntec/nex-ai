"""Tests for the context assembler."""

from __future__ import annotations

from pathlib import Path

from nex.context import ContextAssembler
from nex.indexer.index import IndexBuilder


class TestEstimateTokens:
    def test_estimate_tokens_code(self) -> None:
        code = "def hello():\n    return 42\n"
        tokens = ContextAssembler.estimate_tokens(code)
        assert tokens > 0
        # Code lines at ~3.5 chars/token should give more tokens than len//4
        assert tokens >= len(code) // 5

    def test_estimate_tokens_comments(self) -> None:
        text = "# This is a long comment line explaining the algorithm\n"
        tokens = ContextAssembler.estimate_tokens(text)
        assert tokens > 0

    def test_estimate_tokens_empty(self) -> None:
        assert ContextAssembler.estimate_tokens("") == 0

    def test_estimate_tokens_blank_lines(self) -> None:
        text = "\n\n\n"
        tokens = ContextAssembler.estimate_tokens(text)
        assert tokens == 3  # 3 blank lines = 3 tokens


class TestRelevanceThreshold:
    def test_relevance_threshold(self) -> None:
        threshold = ContextAssembler._relevance_threshold(50.0)
        assert threshold == 5.0  # 50.0 * 0.10

    def test_relevance_threshold_floor(self) -> None:
        # When 10% of top_score < 1.0, threshold should be 1.0
        threshold = ContextAssembler._relevance_threshold(5.0)
        assert threshold == 1.0  # max(0.5, 1.0) = 1.0

    def test_relevance_threshold_zero(self) -> None:
        threshold = ContextAssembler._relevance_threshold(0.0)
        assert threshold == 1.0


class TestExtractSignatures:
    def test_extract_signatures(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        (tmp_path / "utils.py").write_text(
            'def greet(name: str) -> str:\n    """Say hello."""\n    return f"Hello {name}"\n\n'
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            encoding="utf-8",
        )

        builder = IndexBuilder(tmp_path)
        index = builder.build()

        sigs = ContextAssembler._extract_signatures("utils.py", builder, index)
        assert "greet" in sigs
        assert "add" in sigs

    def test_extract_signatures_no_symbols(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        builder = IndexBuilder(tmp_path)
        index = builder.build()

        sigs = ContextAssembler._extract_signatures("nonexistent.py", builder, index)
        assert sigs == ""


class TestSelectRelevantCode:
    def test_select_relevant_code_with_threshold(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        # Create a highly relevant file
        (tmp_path / "auth.py").write_text(
            "def authenticate_user(username, password):\n    pass\n",
            encoding="utf-8",
        )
        # Create a less relevant file
        (tmp_path / "utils.py").write_text(
            "def format_date(date):\n    pass\n",
            encoding="utf-8",
        )

        builder = IndexBuilder(tmp_path)
        index = builder.build()
        assembler = ContextAssembler(tmp_path)

        results = assembler.select_relevant_code("authenticate user login", index)
        # auth.py should be included, utils.py likely filtered by threshold
        file_paths = [fp for fp, _ in results]
        assert "auth.py" in file_paths

    def test_select_relevant_code_no_index(self, tmp_path: Path) -> None:
        assembler = ContextAssembler(tmp_path)
        results = assembler.select_relevant_code("anything", None)
        assert results == []

    def test_select_relevant_code_empty_query(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        (tmp_path / "app.py").write_text("def main(): pass\n", encoding="utf-8")

        builder = IndexBuilder(tmp_path)
        index = builder.build()
        assembler = ContextAssembler(tmp_path)

        # All stop words -> empty tokens -> no results
        results = assembler.select_relevant_code("get set init", index)
        assert results == []
