# Nex AI — The Coding Agent That Remembers

## Overview

Nex AI is a CLI-based AI coding agent that wraps the Anthropic Claude API with persistent project memory, error pattern learning, and codebase indexing. It makes AI coding assistants more reliable by remembering your project context, avoiding past mistakes, and verifying its own work.

**What it is:** A Python CLI tool installed via `pip install nexcoder`. Users run `nex "add a login endpoint"` and the agent reads the codebase, recalls project conventions and past errors, generates code, runs tests, and commits — all with context it retains across sessions.

**What it is NOT:** An IDE. A Cursor/Windsurf competitor. A web app. A multi-model orchestrator (yet). Keep it simple.

## Tech Stack

- **Language:** Python 3.12+
- **CLI Framework:** Typer + Rich (beautiful terminal UI)
- **AI Provider:** Anthropic API (Claude Sonnet 4 for reasoning, Claude Haiku for planning/lightweight tasks)
- **Database:** SQLite via stdlib sqlite3 (error pattern DB)
- **AST Parsing:** py-tree-sitter (codebase indexing)
- **Git:** GitPython (branch, commit, diff operations)
- **HTTP Client:** httpx (async Anthropic API calls)
- **Testing:** pytest + pytest-asyncio
- **Linting:** ruff + mypy (strict mode)
- **Packaging:** PyPI distribution via pyproject.toml + hatchling
- **CI/CD:** GitHub Actions

## Project Structure

```
nex-ai/
├── src/nex/
│   ├── __init__.py              # Version, package metadata
│   ├── cli.py                   # Typer CLI: entry point, all commands
│   ├── agent.py                 # Core agent loop (prompt → API → tools → observe → loop)
│   ├── planner.py               # Task decomposition (single Haiku call, returns subtask list)
│   ├── reviewer.py              # Independent code review (separate API call, no access to plan)
│   ├── context.py               # Context assembly: selects relevant code for API context window
│   ├── api_client.py            # Anthropic API wrapper (handles retries, token tracking, model routing)
│   ├── config.py                # Settings: API keys, model prefs, .nex/config.toml parsing
│   ├── safety.py                # Destructive op detection, user approval prompts, dry-run mode
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── project.py           # Read/write .nex/memory.md (project conventions, architecture)
│   │   ├── errors.py            # Error pattern DB: log failures, query similar past errors
│   │   └── decisions.py         # Decision log: append-only .nex/decisions.md
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── scanner.py           # File discovery, .gitignore respect, language detection
│   │   ├── parser.py            # tree-sitter AST: extract functions, classes, imports
│   │   └── index.py             # Build/query .nex/index.json (function sigs, deps, structure)
│   └── tools/
│       ├── __init__.py
│       ├── file_ops.py          # read_file, write_file, list_directory
│       ├── shell.py             # run_command (with safety layer approval)
│       ├── search.py            # grep/ripgrep wrapper for codebase search
│       └── git_ops.py           # branch, commit, diff, status, rollback
├── tests/
│   ├── conftest.py              # Shared fixtures, mock API client
│   ├── test_agent.py            # Agent loop tests
│   ├── test_memory.py           # Memory read/write tests
│   ├── test_errors.py           # Error pattern DB tests
│   ├── test_indexer.py          # Codebase indexing tests
│   ├── test_tools.py            # Tool execution tests
│   ├── test_safety.py           # Safety layer tests
│   └── test_cli.py              # CLI integration tests
├── pyproject.toml               # Package config, dependencies, scripts
├── README.md                    # User-facing docs
├── CLAUDE.md                    # This file
├── CHANGELOG.md                 # Keep updated with every release
├── LICENSE                      # MIT
└── .github/
    └── workflows/
        └── ci.yml               # Test + lint on every push/PR
```

## Key Commands

```bash
# Development
pip install -e ".[dev]"          # Install in editable mode with dev deps
pytest                           # Run all tests
pytest -x --tb=short             # Run tests, stop on first failure
ruff check src/ tests/           # Lint
ruff format src/ tests/          # Format
mypy src/nex/                  # Type check

# User-facing CLI
nex "your task here"           # Run a task
nex init                       # Initialize .nex/ directory for a project
nex status                     # Show project memory, error count, index stats
nex memory edit                # Interactive memory editor
nex rollback                   # Undo last agent change (git revert)
nex --dry-run "your task"      # Show what would happen without executing
nex auth                       # Set up API key
```

## Architecture Decisions

### Agent Loop (Most Critical Component)
The agent follows the same agentic REPL pattern as Claude Code:
1. Assemble context: system prompt + project memory + relevant code from index + error patterns
2. Call Claude API with tool definitions
3. Parse response: if `tool_use` blocks, execute the tools
4. Feed tool results back to Claude
5. Repeat until Claude returns a text-only response (task complete) or max 25 iterations
6. Log any errors encountered + fixes applied to error pattern DB
7. If code was modified: run tests, show diff, ask to commit

### Memory System
- **Project memory** (`.nex/memory.md`): Human-readable markdown. Contains project overview, tech stack, coding conventions, architecture notes. Loaded into system prompt on every invocation. Users can edit directly.
- **Error patterns** (`.nex/errors.db`): SQLite. Schema: `(id, timestamp, task_summary, error_type, what_failed, what_fixed, file_path, language)`. Queried before code generation — top 3 similar errors injected into context.
- **Decision log** (`.nex/decisions.md`): Append-only. Records architectural choices made during sessions (e.g., "Chose Express over Fastify because existing codebase uses Express middleware").

### Tool System
Exactly 5 tools (kept minimal for safety + token efficiency):
- `read_file(path)` → returns file content
- `write_file(path, content)` → writes content, creates dirs if needed
- `run_command(command)` → executes shell command (safety layer checks first)
- `search_files(pattern, path?)` → ripgrep wrapper, returns matches with context
- `list_directory(path?, depth?)` → recursive directory listing

New tools require explicit justification and approval. Every tool schema adds ~200 tokens per API call.

### Safety Layer
- **Destructive command detection:** regex patterns for `rm -rf`, `DROP TABLE`, `DELETE FROM`, `git push --force`, `git clean`, etc. These ALWAYS require explicit user confirmation.
- **Dry-run mode:** `--dry-run` flag shows all planned actions without executing. Default for first-time users.
- **Git isolation:** Every task creates a branch (`nex/<task-slug>`). Changes are never made on main/master directly.
- **Max iterations:** Hard cap at 25 tool calls per task. Prevents runaway loops.
- **Cost cap:** Warn user if a single task exceeds $1 in API costs.

### Context Assembly
- Token budget: reserve 150K of 200K window. 50K for response.
- Priority order: system prompt (2K) → project memory (1-3K) → error patterns (1-2K) → relevant code from index (variable) → user's task
- Relevance ranking: TF-IDF match between task description and function signatures/docstrings from index. Include top N files until budget is reached.
- If index doesn't exist yet, fall back to including the file tree + README.

### Model Routing
- **Claude Sonnet 4:** All code generation, reasoning, review steps
- **Claude Haiku:** Planning/decomposition, context relevance scoring, simple file operations
- Selection happens in `api_client.py` based on task type, NOT user-configurable in Phase 1

## Conventions

### Code Style
- All code is typed. `mypy --strict` must pass.
- Use `async/await` for all API calls (httpx async client)
- Use dataclasses or Pydantic models for structured data (prefer dataclasses for simplicity)
- Named exports only. No wildcard imports.
- Docstrings on all public functions (Google style)
- No `print()` statements — use Rich console for all output
- Error handling: custom exception hierarchy (`NexError` base, `APIError`, `ToolError`, `SafetyError`, etc.)
- Configuration: `.nex/config.toml` for project settings, `~/.config/nex/config.toml` for global settings
- Secrets: NEVER log or store API keys in files. Use environment variables or keyring.

### Git Conventions
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Branch naming: `nex/<task-slug>` for agent-created branches
- Never commit `.nex/errors.db` to the user's repo (add to .gitignore template)
- Always commit `.nex/memory.md` and `.nex/decisions.md` (these are valuable project context)

### Testing
- Every module has a corresponding test file
- Mock the Anthropic API client in all tests (never make real API calls in tests)
- Use `tmp_path` fixture for file system tests
- Test the safety layer exhaustively — this is the highest-risk component
- Integration tests: use a real SQLite DB (in-memory) for error pattern tests

### Dependencies Policy
- Minimize dependencies. Every new dep needs justification.
- Core deps: anthropic, typer, rich, httpx, gitpython, tree-sitter
- Dev deps: pytest, pytest-asyncio, ruff, mypy, hatchling
- No LangChain, no LlamaIndex, no heavy frameworks. We own the orchestration logic.

## Error Pattern DB Schema

```sql
CREATE TABLE IF NOT EXISTS error_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    task_summary TEXT NOT NULL,          -- what the user asked for
    error_type TEXT NOT NULL,            -- category: syntax, runtime, logic, import, test_failure
    what_failed TEXT NOT NULL,           -- what went wrong
    what_fixed TEXT NOT NULL,            -- how it was fixed
    file_path TEXT,                      -- which file had the error
    language TEXT,                       -- python, javascript, typescript, go, etc.
    code_context TEXT                    -- relevant code snippet (truncated to 500 chars)
);

CREATE INDEX IF NOT EXISTS idx_error_language ON error_patterns(language);
CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type);
```

Query for similar errors: `SELECT * FROM error_patterns WHERE language = ? AND (task_summary LIKE ? OR file_path LIKE ?) ORDER BY timestamp DESC LIMIT 3`

## System Prompt Template

The system prompt is assembled dynamically. Here's the structure:

```
You are Nex, an AI coding agent that works on the user's codebase. You have access to tools for reading files, writing files, running commands, searching code, and listing directories.

## Project Context
{contents of .nex/memory.md}

## Past Errors to Avoid
{top 3 relevant error patterns from errors.db, if any}

## Relevant Code
{function signatures and code snippets selected by context assembler}

## Rules
- Always read existing code before modifying it. Match the project's style.
- Run tests after making changes. If tests fail, fix them before reporting success.
- Never execute destructive commands without user approval.
- When you make an architectural decision, explain why briefly.
- If you're unsure about something, ask the user rather than guessing.
- Keep changes minimal — don't refactor code that isn't related to the task.
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...      # Required. User's Anthropic API key.
NEX_MODEL=claude-sonnet-4-20250514  # Optional. Override default model.
NEX_MAX_ITERATIONS=25           # Optional. Max tool calls per task.
NEX_DRY_RUN=false               # Optional. Default dry-run mode.
NEX_LOG_LEVEL=INFO              # Optional. DEBUG for development.
```

## Development Workflow

1. Create a feature branch: `git checkout -b feat/feature-name`
2. Write failing tests first (TDD encouraged)
3. Implement the feature
4. Run `ruff check src/ tests/ && mypy src/nex/ && pytest`
5. All must pass before committing
6. Write a clear commit message: `feat: add error pattern lookup before code generation`
7. Open a PR (when we have more than one contributor)

## What NOT to Build (Phase 1 Constraints)

These are explicitly out of scope. Do not build them regardless of how useful they seem:
- ❌ Web dashboard or web UI
- ❌ Multi-model support (GPT-4o, Ollama, etc.)
- ❌ Graph database (Neo4j, etc.) — use JSON index
- ❌ Vector database (Qdrant, Chroma, etc.) — use TF-IDF
- ❌ LangChain or LangGraph integration
- ❌ Docker sandboxing for code execution
- ❌ VS Code extension or IDE integration
- ❌ Team features or collaboration
- ❌ Billing, subscriptions, or payment processing
- ❌ MCP server support
- ❌ More than 5 tools
- ❌ Custom fine-tuned models

These may come in Phase 2/3 when validated by user demand and revenue.

## Current Phase: Phase 1 — Build MVP

We are in Phase 1 (Weeks 5–14 of the implementation plan). Focus areas in priority order:
1. Core agent loop with 5 tools ← START HERE
2. Safety layer (destructive op detection + approval)
3. Project memory system (.nex/memory.md)
4. Error pattern DB (.nex/errors.db) with auto-logging
5. Codebase indexer (tree-sitter + JSON index)
6. Context assembler (relevance ranking)
7. Git integration (branch per task, auto-commit)
8. Test runner detection (pytest/jest/go test)
9. CLI polish (nex init, nex status, nex memory edit)
10. PyPI publication
