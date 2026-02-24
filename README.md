# Nex AI — The Coding Agent That Remembers

A CLI-based AI coding agent that wraps the Anthropic Claude API with persistent project memory, error pattern learning, and codebase indexing.

## Features

- **Persistent Memory** — Remembers your project context, conventions, and architecture across sessions
- **Error Learning** — Logs past mistakes and queries them before generating code to avoid repeating errors
- **Codebase Indexing** — Uses tree-sitter to parse your codebase and select relevant context for each task
- **Interactive Chat** — Multi-turn conversation mode with full tool access and persistent history
- **Smart Context Selection** — Two-phase token budgeting with file-level TF-IDF ranking and signature-only fallback
- **Safety Layer** — Detects destructive operations (rm -rf, DROP TABLE, force push) and requires confirmation
- **Git Integration** — Creates isolated branches per task, shows diffs, and offers to commit

## Installation

```bash
pip install nexcoder
```

## Quick Start

```bash
# Initialize Nex in your project
nex init

# Set up your API key
nex auth

# Build the codebase index
nex index

# Run a one-shot task
nex "add a health check endpoint"

# Or start an interactive session
nex chat

# Check project status
nex status
```

## Commands

| Command | Description |
|---------|-------------|
| `nex "task"` | Run a coding task |
| `nex init` | Initialize .nex/ directory |
| `nex index` | Build codebase index (.nex/index.json) |
| `nex chat` | Start interactive chat session |
| `nex status` | Show project stats |
| `nex auth` | Configure API key |
| `nex memory show` | View project memory |
| `nex memory edit` | Edit project memory |
| `nex rollback` | Undo last agent change |
| `nex --dry-run "task"` | Preview without executing |

## How It Works

1. **Context Assembly** — Loads project memory, error patterns, and relevant code into the prompt. Files are ranked by TF-IDF relevance, with top files included in full and remaining files as signature summaries.
2. **Agent Loop** — Iterates: call Claude API → execute tools → feed results back, up to 25 iterations.
3. **Interactive Chat** — `nex chat` maintains a persistent conversation with the agent, accumulating context across turns. Useful for exploratory work, debugging, or multi-step tasks.
4. **Codebase Index** — `nex index` parses your source files with tree-sitter, extracting function/class signatures for fast relevance search.
5. **Git Commit** — Shows diff and offers to commit on an isolated branch.

## Tools

The agent has access to 5 tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers |
| `write_file` | Write content, creating directories if needed |
| `run_command` | Execute shell commands (safety-checked) |
| `search_files` | Regex search across the codebase |
| `list_directory` | Recursive directory listing |

## Development

```bash
# Clone and install in editable mode
git clone https://github.com/nex-ai/nex-ai.git
cd nex-ai
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/nex/
```

## Architecture

```
src/nex/
├── cli.py          # Typer CLI entry point
├── agent.py        # Core agent loop + ChatSession
├── api_client.py   # Anthropic API wrapper
├── planner.py      # Task decomposition (Haiku)
├── reviewer.py     # Independent code review
├── context.py      # Context assembly + token budgeting
├── safety.py       # Destructive operation detection
├── config.py       # Configuration management
├── memory/         # Persistent memory system
│   ├── project.py  # .nex/memory.md
│   ├── errors.py   # .nex/errors.db (SQLite)
│   └── decisions.py # .nex/decisions.md
├── indexer/        # Codebase indexing
│   ├── scanner.py  # File discovery
│   ├── parser.py   # tree-sitter AST parsing
│   └── index.py    # Index builder + TF-IDF search
└── tools/          # Agent tools (5 total)
    ├── file_ops.py # read_file, write_file
    ├── shell.py    # run_command
    ├── search.py   # search_files
    └── git_ops.py  # Git operations
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (required) | — |
| `NEX_MODEL` | Override default model | `claude-sonnet-4-20250514` |
| `NEX_MAX_ITERATIONS` | Max tool calls per task | `25` |
| `NEX_DRY_RUN` | Default dry-run mode | `false` |
| `NEX_LOG_LEVEL` | Logging verbosity | `INFO` |

## License

MIT
