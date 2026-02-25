# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-02-25

### Changed

- Agent now always decomposes tasks into subtasks via the planner, regardless of rate limit settings
- `_run_single()` is now a fallback path, used only when the planner fails
- `_run_subtask_loop()` returns `SubtaskResult` (with text, files_touched, iterations) instead of a bare string

### Added

- Memory updates after every subtask — Haiku summarizes what was done and appends to `## Session Log` in `.nex/memory.md`
- Mid-subtask memory checkpoint at iteration 8 for long-running subtasks (no API call, lightweight progress note)
- `ProjectMemory.prune_section()` to prevent Session Log from growing unbounded (trims oldest entries beyond 30 lines)
- Tests for planner failure fallback, memory updates after subtasks, always-subtask behavior, and section pruning

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
