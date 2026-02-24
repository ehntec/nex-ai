"""AST-based symbol extraction from source files."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from nex.exceptions import IndexerError

console = Console(stderr=True)


@dataclass(frozen=True, slots=True)
class Symbol:
    """A single code symbol extracted from a source file.

    Attributes:
        name: Symbol identifier (e.g. function/class name).
        kind: One of "function", "class", "method", "import".
        file_path: Path to the source file (relative to project root).
        line_start: 1-based starting line number.
        line_end: 1-based ending line number.
        signature: Full declaration signature text.
        docstring: Extracted docstring, or None.
    """

    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: str | None


class ASTParser:
    """Extracts code symbols using tree-sitter with regex fallback."""

    def __init__(self) -> None:
        """Initialize tree-sitter parsers for supported languages."""
        self._ts_languages: dict[str, Any] = {}
        self._ts_parser: Any | None = None
        self._init_tree_sitter()

    def parse_file(self, file_path: Path, language: str) -> list[Symbol]:
        """Parse a file and return extracted symbols.

        Args:
            file_path: Absolute path to a source file.
            language: Language identifier (e.g. "python").

        Returns:
            A list of Symbol instances found in the file.

        Raises:
            IndexerError: If the file cannot be read.
        """
        try:
            content = file_path.read_bytes()
        except OSError as exc:
            raise IndexerError(f"Cannot read {file_path}: {exc}") from exc

        file_str = str(file_path)

        if language in self._ts_languages and self._ts_parser is not None:
            dispatch: dict[str, Any] = {
                "python": self._parse_python,
                "javascript": self._parse_javascript,
                "typescript": self._parse_typescript,
            }
            handler = dispatch.get(language)
            if handler is not None:
                try:
                    result: list[Symbol] = handler(content, file_str)
                    return result
                except Exception:
                    pass  # Fall through to regex

        return self._parse_regex(content, file_str, language)

    def _init_tree_sitter(self) -> None:
        """Attempt to load tree-sitter and language grammars."""
        try:
            import tree_sitter as ts
        except ImportError:
            return

        self._ts_parser = ts.Parser()

        lang_modules = {
            "python": "tree_sitter_python",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
        }

        for lang_name, mod_name in lang_modules.items():
            try:
                mod = __import__(mod_name)
                if hasattr(mod, "language"):
                    ts_lang = ts.Language(mod.language())
                    self._ts_languages[lang_name] = ts_lang
            except Exception:
                pass

    def _parse_python(self, content: bytes, file_path: str) -> list[Symbol]:
        """Extract symbols from Python source using tree-sitter."""
        tree = self._get_tree(content, "python")
        if tree is None:
            return self._parse_regex(content, file_path, "python")

        symbols: list[Symbol] = []
        for node in tree.root_node.children:
            if node.type == "function_definition":
                symbols.append(self._ts_function(node, content, file_path, "function"))
            elif node.type == "class_definition":
                symbols.append(self._ts_function(node, content, file_path, "class"))
                body = self._child_by_type(node, "block")
                if body is not None:
                    for child in body.children:
                        if child.type == "function_definition":
                            symbols.append(self._ts_function(child, content, file_path, "method"))
            elif node.type in ("import_statement", "import_from_statement"):
                text = content[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                symbols.append(
                    Symbol(
                        name=text.strip(),
                        kind="import",
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=text.strip(),
                        docstring=None,
                    )
                )
        return symbols

    def _parse_javascript(self, content: bytes, file_path: str) -> list[Symbol]:
        """Extract symbols from JavaScript source using tree-sitter."""
        tree = self._get_tree(content, "javascript")
        if tree is None:
            return self._parse_regex(content, file_path, "javascript")

        symbols: list[Symbol] = []
        for node in tree.root_node.children:
            if node.type == "function_declaration":
                symbols.append(self._ts_function(node, content, file_path, "function"))
            elif node.type == "class_declaration":
                symbols.append(self._ts_function(node, content, file_path, "class"))
            elif node.type in ("lexical_declaration", "variable_declaration"):
                sym = self._ts_arrow_function(node, content, file_path)
                if sym is not None:
                    symbols.append(sym)
        return symbols

    def _parse_typescript(self, content: bytes, file_path: str) -> list[Symbol]:
        """Extract symbols from TypeScript source using tree-sitter."""
        tree = self._get_tree(content, "typescript")
        if tree is None:
            return self._parse_regex(content, file_path, "typescript")

        symbols: list[Symbol] = []
        for node in tree.root_node.children:
            if node.type in ("function_declaration", "function_signature"):
                symbols.append(self._ts_function(node, content, file_path, "function"))
            elif node.type == "class_declaration":
                symbols.append(self._ts_function(node, content, file_path, "class"))
            elif node.type == "interface_declaration":
                name_node = self._child_by_type(node, "type_identifier")
                name = self._node_text(name_node, content) if name_node else "<anonymous>"
                sig = self._first_line(node, content)
                symbols.append(
                    Symbol(
                        name=name,
                        kind="class",
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=sig,
                        docstring=None,
                    )
                )
            elif node.type in ("lexical_declaration", "variable_declaration"):
                sym = self._ts_arrow_function(node, content, file_path)
                if sym is not None:
                    symbols.append(sym)
        return symbols

    def _parse_regex(self, content: bytes, file_path: str, language: str) -> list[Symbol]:
        """Fallback regex-based symbol extraction."""
        text = content.decode("utf-8", errors="replace")
        lines = text.splitlines()

        patterns: dict[str, re.Pattern[str]] = {
            "python": re.compile(r"^( *)(class|def|async\s+def)\s+(\w+)"),
            "javascript": re.compile(
                r"^(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+))"
            ),
            "typescript": re.compile(
                r"^(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+)|interface\s+(\w+))"
            ),
            "go": re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)"),
        }

        pattern = patterns.get(language, patterns["python"])
        symbols: list[Symbol] = []

        for idx, line in enumerate(lines, start=1):
            m = pattern.match(line if language == "python" else line.strip())
            if m is None:
                continue

            groups = [g for g in m.groups() if g is not None]
            name = groups[-1] if groups else None
            if name is None:
                continue

            stripped = line.strip()
            kind = "class" if "class " in stripped or "interface " in stripped else "function"

            docstring: str | None = None
            if language == "python" and idx < len(lines):
                docstring = self._extract_python_docstring(lines, idx - 1)

            symbols.append(
                Symbol(
                    name=name,
                    kind=kind,
                    file_path=file_path,
                    line_start=idx,
                    line_end=idx,
                    signature=stripped,
                    docstring=docstring,
                )
            )
        return symbols

    @staticmethod
    def _extract_python_docstring(lines: list[str], def_idx: int) -> str | None:
        """Extract Python docstring from lines after a def/class."""
        start = def_idx + 1
        while start < len(lines) and lines[start].strip() == "":
            start += 1
        if start >= len(lines):
            return None

        first = lines[start].strip()
        if not (first.startswith('"""') or first.startswith("'''")):
            return None

        quote = first[:3]
        if first.count(quote) >= 2:
            return first.strip(quote).strip()

        doc_lines = [first[3:]]
        for i in range(start + 1, min(start + 100, len(lines))):
            if quote in lines[i]:
                doc_lines.append(lines[i][: lines[i].index(quote)])
                break
            doc_lines.append(lines[i])
        return textwrap.dedent("\n".join(doc_lines)).strip()

    # Tree-sitter helpers

    def _get_tree(self, content: bytes, language: str) -> Any | None:
        """Parse content with tree-sitter."""
        ts_lang = self._ts_languages.get(language)
        if ts_lang is None or self._ts_parser is None:
            return None
        self._ts_parser.language = ts_lang
        return self._ts_parser.parse(content)

    def _ts_function(self, node: Any, content: bytes, file_path: str, kind: str) -> Symbol:
        """Build a Symbol from a tree-sitter node."""
        name_node = self._child_by_type(node, "identifier")
        name = self._node_text(name_node, content) if name_node else "<anonymous>"
        sig = self._first_line(node, content)
        docstring = self._ts_docstring(node, content)
        return Symbol(
            name=name,
            kind=kind,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=sig,
            docstring=docstring,
        )

    def _ts_arrow_function(self, node: Any, content: bytes, file_path: str) -> Symbol | None:
        """Extract a Symbol if a variable holds an arrow/function expression."""
        for child in node.children:
            if child.type == "variable_declarator":
                init = self._child_by_type(child, "arrow_function") or self._child_by_type(
                    child, "function"
                )
                if init is not None:
                    name_node = self._child_by_type(child, "identifier")
                    name = self._node_text(name_node, content) if name_node else "<anonymous>"
                    sig = self._first_line(node, content)
                    return Symbol(
                        name=name,
                        kind="function",
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=sig,
                        docstring=None,
                    )
        return None

    def _ts_docstring(self, node: Any, content: bytes) -> str | None:
        """Extract docstring from first expression_statement in a block."""
        body = self._child_by_type(node, "block")
        if body is None:
            return None
        for child in body.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw = self._node_text(sub, content)
                        for q in ('"""', "'''", '"', "'"):
                            if raw.startswith(q) and raw.endswith(q):
                                raw = raw[len(q) : -len(q)]
                                break
                        return textwrap.dedent(raw).strip()
                break
            if child.type != "comment":
                break
        return None

    @staticmethod
    def _child_by_type(node: Any, type_name: str) -> Any | None:
        """Return first child of node with the given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    @staticmethod
    def _node_text(node: Any, content: bytes) -> str:
        """Extract source text for a tree-sitter node."""
        return content[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _first_line(node: Any, content: bytes) -> str:
        """Extract the first source line for a tree-sitter node."""
        text = content[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        return text.split("\n", 1)[0].strip()
