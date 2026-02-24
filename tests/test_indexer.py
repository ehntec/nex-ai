"""Tests for the codebase indexer."""

from __future__ import annotations

from pathlib import Path

import pytest

from nex.indexer.index import IndexBuilder
from nex.indexer.parser import ASTParser
from nex.indexer.scanner import FileScanner


class TestFileScanner:
    def test_scan_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def foo(): pass", encoding="utf-8")
        (tmp_path / "readme.txt").write_text("not code", encoding="utf-8")

        scanner = FileScanner(tmp_path)
        files = scanner.scan()

        assert len(files) == 2
        assert all(f.language == "python" for f in files)

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("pass", encoding="utf-8")
        (tmp_path / "visible.py").write_text("pass", encoding="utf-8")

        scanner = FileScanner(tmp_path)
        files = scanner.scan()

        assert len(files) == 1
        assert files[0].path == Path("visible.py")

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "pkg.js").write_text("module.exports = {}", encoding="utf-8")
        (tmp_path / "app.js").write_text("console.log('hi')", encoding="utf-8")

        scanner = FileScanner(tmp_path)
        files = scanner.scan()

        assert len(files) == 1

    def test_language_detection(self, tmp_path: Path) -> None:
        (tmp_path / "app.ts").write_text("const x = 1", encoding="utf-8")
        (tmp_path / "main.go").write_text("package main", encoding="utf-8")
        (tmp_path / "lib.rs").write_text("fn main() {}", encoding="utf-8")

        scanner = FileScanner(tmp_path)
        files = scanner.scan()
        langs = {f.language for f in files}

        assert "typescript" in langs
        assert "go" in langs
        assert "rust" in langs


class TestASTParser:
    def test_parse_python_function(self, tmp_path: Path) -> None:
        code = 'def hello(name: str) -> str:\n    """Greet someone."""\n    return f"Hello {name}"'
        test_file = tmp_path / "test.py"
        test_file.write_text(code, encoding="utf-8")

        parser = ASTParser()
        symbols = parser.parse_file(test_file, "python")

        funcs = [s for s in symbols if s.kind == "function"]
        assert len(funcs) >= 1
        assert funcs[0].name == "hello"

    def test_parse_python_class(self, tmp_path: Path) -> None:
        code = "class MyClass:\n    def method(self):\n        pass"
        test_file = tmp_path / "test.py"
        test_file.write_text(code, encoding="utf-8")

        parser = ASTParser()
        symbols = parser.parse_file(test_file, "python")

        classes = [s for s in symbols if s.kind == "class"]
        assert len(classes) >= 1
        assert classes[0].name == "MyClass"


class TestIndexBuilder:
    def test_build_and_load(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        (tmp_path / "main.py").write_text("def hello():\n    pass\n", encoding="utf-8")

        builder = IndexBuilder(tmp_path)
        index = builder.build()

        assert len(index.files) == 1
        assert len(index.symbols) >= 1

        # Load from disk
        loaded = builder.load()
        assert loaded is not None
        assert len(loaded.symbols) == len(index.symbols)

    def test_search_symbols(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        (tmp_path / "auth.py").write_text(
            "def authenticate_user(username, password):\n    pass\n\n"
            "def get_profile(user_id):\n    pass\n",
            encoding="utf-8",
        )

        builder = IndexBuilder(tmp_path)
        index = builder.build()
        results = builder.search_symbols("authenticate", index)

        assert len(results) > 0
        assert results[0].name == "authenticate_user"

    def test_no_index_raises(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        builder = IndexBuilder(tmp_path)

        from nex.exceptions import IndexerError

        with pytest.raises(IndexerError):
            builder.search_symbols("anything")


class TestIndexBuilderSearchFiles:
    def test_search_files_groups_by_file(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        (tmp_path / "auth.py").write_text(
            "def authenticate_user(username, password):\n    pass\n\n"
            "def verify_token(token):\n    pass\n",
            encoding="utf-8",
        )
        (tmp_path / "db.py").write_text(
            "def connect_database():\n    pass\n",
            encoding="utf-8",
        )

        builder = IndexBuilder(tmp_path)
        index = builder.build()
        results = builder.search_files("authenticate user token", index)

        # auth.py should score higher (more relevant symbols)
        assert len(results) > 0
        file_paths = [fp for fp, _ in results]
        assert "auth.py" in file_paths

    def test_search_files_empty_query(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        (tmp_path / "app.py").write_text("def main(): pass\n", encoding="utf-8")

        builder = IndexBuilder(tmp_path)
        index = builder.build()
        # All stop words => empty tokens
        results = builder.search_files("get set init", index)
        assert results == []

    def test_search_files_returns_scores(self, tmp_path: Path) -> None:
        (tmp_path / ".nex").mkdir()
        (tmp_path / "math_ops.py").write_text(
            "def calculate_sum(numbers):\n    return sum(numbers)\n",
            encoding="utf-8",
        )

        builder = IndexBuilder(tmp_path)
        index = builder.build()
        results = builder.search_files("calculate sum", index)

        assert len(results) > 0
        for file_path, score in results:
            assert isinstance(file_path, str)
            assert isinstance(score, float)
            assert score > 0


class TestTokenizeStopWords:
    def test_tokenize_filters_stop_words(self) -> None:
        tokens = IndexBuilder._tokenize("get config file path")
        # "get", "config", "file", "path" are all stop words
        assert tokens == []

    def test_tokenize_keeps_meaningful_words(self) -> None:
        tokens = IndexBuilder._tokenize("authenticate user login")
        assert "authenticate" in tokens
        assert "user" in tokens  # "user" is not a stop word
        assert "login" in tokens

    def test_tokenize_mixed_stop_and_meaningful(self) -> None:
        tokens = IndexBuilder._tokenize("get authenticate config")
        # "get" and "config" are stop words, "authenticate" is not
        assert "authenticate" in tokens
        assert "get" not in tokens
        assert "config" not in tokens
