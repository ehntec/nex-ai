"""Codebase indexer — file discovery, AST parsing, and index building."""

from __future__ import annotations

from nex.indexer.index import CodeIndex, IndexBuilder
from nex.indexer.parser import ASTParser, Symbol
from nex.indexer.scanner import FileInfo, FileScanner

__all__ = [
    "ASTParser",
    "CodeIndex",
    "FileInfo",
    "FileScanner",
    "IndexBuilder",
    "Symbol",
]
