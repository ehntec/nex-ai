"""Code index builder and query engine."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from nex.exceptions import IndexerError
from nex.indexer.parser import ASTParser, Symbol
from nex.indexer.scanner import FileInfo, FileScanner

console = Console(stderr=True)


@dataclass
class CodeIndex:
    """Queryable code index persisted as .nex/index.json.

    Attributes:
        symbols: All extracted code symbols.
        files: All discovered source files.
        last_updated: ISO-8601 timestamp of the last build.
    """

    symbols: list[Symbol] = field(default_factory=list)
    files: list[FileInfo] = field(default_factory=list)
    last_updated: str = ""


class IndexBuilder:
    """Builds, persists, and queries the codebase index."""

    def __init__(self, project_dir: Path) -> None:
        """Initialize the builder.

        Args:
            project_dir: Absolute path to the project root.
        """
        self._project_dir = project_dir.resolve()
        self._scanner = FileScanner(self._project_dir)
        self._parser = ASTParser()

    @property
    def index_path(self) -> Path:
        """Path to the persisted index file."""
        return self._project_dir / ".nex" / "index.json"

    def build(self) -> CodeIndex:
        """Full scan + parse, persist to index.json.

        Returns:
            The newly built CodeIndex.
        """
        console.print("[bold blue]Indexer[/bold blue] building codebase index...")

        files = self._scanner.scan()
        all_symbols: list[Symbol] = []

        for fi in files:
            abs_path = self._project_dir / fi.path
            try:
                symbols = self._parser.parse_file(abs_path, fi.language)
                symbols = [
                    Symbol(
                        name=s.name, kind=s.kind,
                        file_path=str(fi.path).replace("\\", "/"),
                        line_start=s.line_start, line_end=s.line_end,
                        signature=s.signature, docstring=s.docstring,
                    )
                    for s in symbols
                ]
                all_symbols.extend(symbols)
            except IndexerError as exc:
                console.print(f"[yellow]Warning[/yellow]: Skipping {fi.path}: {exc}")

        index = CodeIndex(
            symbols=all_symbols,
            files=files,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        self.save(index)
        console.print(
            f"[green]Indexer[/green] indexed [bold]{len(all_symbols)}[/bold] "
            f"symbols across [bold]{len(files)}[/bold] files"
        )
        return index

    def load(self) -> CodeIndex | None:
        """Load a previously persisted index from disk.

        Returns:
            A CodeIndex, or None if no index file exists.
        """
        if not self.index_path.is_file():
            return None

        try:
            raw = self.index_path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)

            symbols = [
                Symbol(
                    name=s["name"], kind=s["kind"], file_path=s["file_path"],
                    line_start=s["line_start"], line_end=s["line_end"],
                    signature=s["signature"], docstring=s.get("docstring"),
                )
                for s in data.get("symbols", [])
            ]
            files = [
                FileInfo(path=Path(f["path"]), language=f["language"], size=f["size"])
                for f in data.get("files", [])
            ]
            return CodeIndex(
                symbols=symbols, files=files,
                last_updated=data.get("last_updated", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise IndexerError(f"Corrupt index at {self.index_path}: {exc}") from exc

    def save(self, index: CodeIndex) -> None:
        """Serialize index to .nex/index.json."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {
                "last_updated": index.last_updated,
                "symbols": [asdict(s) for s in index.symbols],
                "files": [
                    {"path": str(f.path).replace("\\", "/"), "language": f.language, "size": f.size}
                    for f in index.files
                ],
            }
            self.index_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            raise IndexerError(f"Cannot write index: {exc}") from exc

    def search_symbols(self, query: str, index: CodeIndex | None = None) -> list[Symbol]:
        """Search symbols using TF-IDF-like relevance scoring.

        Args:
            query: Free-text search query.
            index: Pre-loaded index. If None, loads from disk.

        Returns:
            Symbols sorted by descending relevance score.
        """
        idx = self._ensure_index(index)
        tokens = self._tokenize(query)
        if not tokens:
            return []

        doc_count = len(idx.symbols) or 1
        df: Counter[str] = Counter()
        for sym in idx.symbols:
            for t in set(self._symbol_tokens(sym)):
                df[t] += 1

        scored: list[tuple[float, Symbol]] = []
        for sym in idx.symbols:
            score = self._score(sym, tokens, df, doc_count)
            if score > 0.0:
                scored.append((score, sym))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [sym for _, sym in scored]

    def get_file_symbols(self, file_path: str, index: CodeIndex | None = None) -> list[Symbol]:
        """Return all symbols defined in a file.

        Args:
            file_path: Relative file path.
            index: Pre-loaded index.

        Returns:
            Symbols in the file, ordered by line number.
        """
        idx = self._ensure_index(index)
        normalized = file_path.replace("\\", "/")
        results = [s for s in idx.symbols if s.file_path.replace("\\", "/") == normalized]
        results.sort(key=lambda s: s.line_start)
        return results

    def _ensure_index(self, index: CodeIndex | None) -> CodeIndex:
        """Return index or load from disk."""
        if index is not None:
            return index
        loaded = self.load()
        if loaded is None:
            raise IndexerError("No index available. Run 'nex init' or builder.build() first.")
        return loaded

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase tokens, splitting camelCase and snake_case."""
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        spaced = re.sub(r"[_\W]+", " ", spaced)
        return [t.lower() for t in spaced.split() if len(t) >= 2]

    @staticmethod
    def _symbol_tokens(sym: Symbol) -> list[str]:
        """Extract searchable tokens from a symbol."""
        parts = f"{sym.name} {sym.signature}"
        if sym.docstring:
            parts += f" {sym.docstring}"
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", parts)
        spaced = re.sub(r"[_\W]+", " ", spaced)
        return [t.lower() for t in spaced.split() if len(t) >= 2]

    @staticmethod
    def _score(
        sym: Symbol, query_tokens: list[str], df: Counter[str], doc_count: int
    ) -> float:
        """Compute TF-IDF-like relevance score."""
        sym_bag = Counter(IndexBuilder._symbol_tokens(sym))
        name_tokens = {
            t.lower()
            for t in re.sub(r"([a-z])([A-Z])", r"\1 \2", sym.name).replace("_", " ").split()
            if len(t) >= 2
        }

        score = 0.0
        for qt in query_tokens:
            tf = sym_bag.get(qt, 0)
            if tf == 0:
                continue
            idf = math.log((doc_count + 1) / (df.get(qt, 0) + 1)) + 1.0
            contribution = tf * idf
            if qt in name_tokens:
                contribution *= 3.0
            score += contribution

        # Exact name match bonus
        if " ".join(query_tokens) == sym.name.lower():
            score += 20.0
        return score
