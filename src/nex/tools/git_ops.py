"""Git operations for the agent."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from nex.exceptions import ToolError

console = Console(stderr=True)


class GitOperations:
    """Git operations using GitPython.

    All branch operations create nex/* branches to isolate agent changes.
    """

    def __init__(self, project_dir: Path) -> None:
        """Initialize git operations for a project.

        Args:
            project_dir: Project root directory.

        Raises:
            ToolError: If GitPython is not installed.
        """
        self._project_dir = project_dir
        try:
            from git import Repo  # type: ignore[import-untyped]

            self._repo_class = Repo
        except ImportError as exc:
            raise ToolError("GitPython is required: pip install gitpython") from exc

        self._repo = self._open_repo()

    def _open_repo(self) -> object | None:
        """Open the git repo, returning None if not a repo."""
        try:
            return self._repo_class(self._project_dir)
        except Exception:  # noqa: BLE001
            return None

    def is_repo(self) -> bool:
        """Check if the project directory is a git repository."""
        return self._repo is not None

    def status(self) -> str:
        """Return the git status summary.

        Returns:
            A formatted string showing modified, staged, and untracked files.

        Raises:
            ToolError: If not a git repository.
        """
        repo = self._ensure_repo()
        lines: list[str] = []

        # Staged changes
        staged = repo.index.diff("HEAD")  # type: ignore[union-attr]
        if staged:
            lines.append("Staged:")
            for diff in staged:
                lines.append(f"  {diff.change_type}: {diff.a_path}")

        # Unstaged changes
        unstaged = repo.index.diff(None)  # type: ignore[union-attr]
        if unstaged:
            lines.append("Modified:")
            for diff in unstaged:
                lines.append(f"  {diff.change_type}: {diff.a_path}")

        # Untracked
        untracked = repo.untracked_files  # type: ignore[union-attr]
        if untracked:
            lines.append("Untracked:")
            for f in untracked[:20]:
                lines.append(f"  {f}")
            if len(untracked) > 20:
                lines.append(f"  ... and {len(untracked) - 20} more")

        return "\n".join(lines) if lines else "Working tree clean"

    def diff(self, staged: bool = False) -> str:
        """Return the git diff.

        Args:
            staged: If True, show staged changes. Otherwise show unstaged.

        Returns:
            The diff text.

        Raises:
            ToolError: If not a git repository.
        """
        repo = self._ensure_repo()
        try:
            if staged:
                return repo.git.diff("--cached")  # type: ignore[union-attr]
            return repo.git.diff()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Failed to get diff: {exc}") from exc

    def create_branch(self, name: str) -> None:
        """Create and checkout a nex/* branch.

        Args:
            name: Branch name suffix (will be prefixed with nex/).

        Raises:
            ToolError: If branch creation fails.
        """
        repo = self._ensure_repo()
        branch_name = f"nex/{name}"
        try:
            repo.git.checkout("-b", branch_name)  # type: ignore[union-attr]
            console.print(f"[green]Created branch[/green] {branch_name}")
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Failed to create branch {branch_name}: {exc}") from exc

    def commit(self, message: str) -> str:
        """Stage all changes and commit.

        Args:
            message: Commit message.

        Returns:
            The commit hash.

        Raises:
            ToolError: If the commit fails.
        """
        repo = self._ensure_repo()
        try:
            repo.git.add("-A")  # type: ignore[union-attr]
            repo.git.commit("-m", message)  # type: ignore[union-attr]
            commit_hash: str = repo.head.commit.hexsha[:8]  # type: ignore[union-attr]
            console.print(f"[green]Committed[/green] {commit_hash}: {message}")
            return commit_hash
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Failed to commit: {exc}") from exc

    def rollback(self) -> None:
        """Revert the last commit.

        Raises:
            ToolError: If the revert fails or not on a nex/* branch.
        """
        repo = self._ensure_repo()
        branch = self.current_branch()
        if not branch.startswith("nex/"):
            raise ToolError(f"Rollback only works on nex/* branches (current: {branch})")

        try:
            repo.git.revert("HEAD", "--no-edit")  # type: ignore[union-attr]
            console.print("[green]Rolled back[/green] last commit")
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Failed to rollback: {exc}") from exc

    def current_branch(self) -> str:
        """Return the current branch name.

        Returns:
            Branch name string.

        Raises:
            ToolError: If not a git repository.
        """
        repo = self._ensure_repo()
        try:
            return str(repo.active_branch)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Failed to get current branch: {exc}") from exc

    def _ensure_repo(self) -> object:
        """Return the repo object, raising if not available."""
        if self._repo is None:
            raise ToolError("Not a git repository")
        return self._repo
