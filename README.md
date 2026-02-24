# Nex AI — The Coding Agent That Remembers

A CLI-based AI coding agent that wraps the Anthropic Claude API with persistent project memory, error pattern learning, and codebase indexing.

## Features

- **Persistent Memory** — Remembers your project context, conventions, and architecture across sessions
- **Error Learning** — Logs past mistakes and queries them before generating code to avoid repeating errors
- **Codebase Indexing** — Uses tree-sitter to parse your codebase and select relevant context for each task
- **Safety Layer** — Detects destructive operations (rm -rf, DROP TABLE, force push) and requires confirmation
- **Git Integration** — Creates isolated branches per task, shows diffs, and offers to commit

## Installation

```bash
pip install nex-ai
```

## Quick Start

```bash
# Initialize Nex in your project
nex init

# Set up your API key
nex auth

# Run your first task
nex "add a health check endpoint"

# Check project status
nex status

# View project memory
nex memory show
```

## Commands

| Command | Description |
|---------|-------------|
| `nex "task"` | Run a coding task |
| `nex init` | Initialize .nex/ directory |
| `nex status` | Show project stats |
| `nex auth` | Configure API key |
| `nex memory show` | View project memory |
| `nex memory edit` | Edit project memory |
| `nex rollback` | Undo last agent change |
| `nex --dry-run "task"` | Preview without executing |

## How It Works

1. **Context Assembly** — Loads project memory, error patterns, and relevant code into the prompt
2. **Task Planning** — Decomposes complex tasks using Claude Haiku
3. **Agent Loop** — Iterates: call Claude API → execute tools → feed results back
4. **Code Review** — Independent review step checks the changes
5. **Git Commit** — Shows diff and offers to commit on an isolated branch

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
├── agent.py        # Core agent loop
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

## License

MIT
