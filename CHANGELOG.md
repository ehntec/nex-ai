# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-02-24

### Added

- Core agent loop with 5 tools (read_file, write_file, run_command, search_files, list_directory)
- Safety layer with destructive operation detection and user approval prompts
- Project memory system (.nex/memory.md)
- Error pattern database (.nex/errors.db) with auto-logging and similarity search
- Codebase indexer with tree-sitter AST parsing and `nex index` command
- Context assembler with TF-IDF relevance ranking
- Git integration (branch per task, auto-commit, rollback)
- Interactive chat mode (`nex chat`)
- Test runner auto-detection (pytest, npm test, go test, cargo test, maven, gradle)
- CLI commands: nex, nex init, nex status, nex memory, nex rollback, nex auth, nex chat, nex index
- Dry-run mode (--dry-run flag)
- Task decomposition via Claude Haiku
- Independent code review step
