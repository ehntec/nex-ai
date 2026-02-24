"""Test runner detection and execution.

Auto-detects the project's test framework and runs tests after the agent
modifies files. Supports Python (pytest), JavaScript/TypeScript (npm test),
Go, Rust, and Java (Maven/Gradle).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nex.tools import ToolResult
from nex.tools.shell import run_command


@dataclass
class TestRunnerResult:
    """Result from running the project's test suite.

    Attributes:
        success: Whether tests passed.
        output: Combined stdout/stderr from the test run.
        command: The command that was executed.
        framework: Detected framework name (e.g. "pytest", "npm test").
    """

    success: bool
    output: str
    command: str
    framework: str


class TestRunner:
    """Detects and runs the project's test framework.

    Detection checks multiple languages in priority order and uses
    the first match found.

    Args:
        project_dir: Root directory of the project.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = project_dir

    def detect(self) -> str | None:
        """Detect the project's test command.

        Checks for test frameworks in order: Python, JavaScript/TypeScript,
        Go, Rust, Java (Maven), Java (Gradle). Returns the first match.

        Returns:
            The test command string, or None if no framework detected.
        """
        detectors = [
            self._detect_python,
            self._detect_javascript,
            self._detect_go,
            self._detect_rust,
            self._detect_java_maven,
            self._detect_java_gradle,
        ]
        for detector in detectors:
            result = detector()
            if result is not None:
                return result
        return None

    async def run(
        self,
        command: str | None = None,
        timeout: int = 120,
    ) -> TestRunnerResult:
        """Run the test suite.

        Args:
            command: Explicit test command. If None, auto-detects.
            timeout: Maximum seconds to wait for tests.

        Returns:
            TestRunnerResult with pass/fail status and output.
        """
        if command is None:
            command = self.detect()

        if command is None:
            return TestRunnerResult(
                success=True,
                output="No test runner detected",
                command="",
                framework="",
            )

        framework = _derive_framework(command)
        result: ToolResult = await run_command(command, self._project_dir, timeout=timeout)

        return TestRunnerResult(
            success=result.success,
            output=result.output if result.success else f"{result.output}\n{result.error or ''}",
            command=command,
            framework=framework,
        )

    def _detect_python(self) -> str | None:
        """Check for Python test frameworks (pytest)."""
        markers = ["pyproject.toml", "pytest.ini", "setup.cfg"]
        for marker in markers:
            if (self._project_dir / marker).is_file():
                return "python -m pytest"

        # Fall back to checking for test files
        for path in self._project_dir.rglob("test_*.py"):
            # Skip hidden dirs and common non-project dirs
            parts = path.relative_to(self._project_dir).parts
            if any(p.startswith(".") or p in ("node_modules", "__pycache__") for p in parts):
                continue
            return "python -m pytest"

        return None

    def _detect_javascript(self) -> str | None:
        """Check for JavaScript/TypeScript test frameworks (npm test)."""
        pkg_path = self._project_dir / "package.json"
        if not pkg_path.is_file():
            return None

        try:
            data = json.loads(pkg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        scripts = data.get("scripts", {})
        test_script = scripts.get("test", "")

        # npm init creates a placeholder: 'echo "Error: no test specified" && exit 1'
        if not test_script or "no test specified" in test_script:
            return None

        return "npm test"

    def _detect_go(self) -> str | None:
        """Check for Go test files."""
        if not (self._project_dir / "go.mod").is_file():
            return None

        # Need at least one *_test.go file
        for path in self._project_dir.rglob("*_test.go"):
            parts = path.relative_to(self._project_dir).parts
            if any(p.startswith(".") for p in parts):
                continue
            return "go test ./..."

        return None

    def _detect_rust(self) -> str | None:
        """Check for Rust project (Cargo.toml)."""
        if (self._project_dir / "Cargo.toml").is_file():
            return "cargo test"
        return None

    def _detect_java_maven(self) -> str | None:
        """Check for Maven project (pom.xml)."""
        if (self._project_dir / "pom.xml").is_file():
            return "mvn test -q"
        return None

    def _detect_java_gradle(self) -> str | None:
        """Check for Gradle project (build.gradle or build.gradle.kts)."""
        if (self._project_dir / "build.gradle").is_file():
            return "gradle test"
        if (self._project_dir / "build.gradle.kts").is_file():
            return "gradle test"
        return None


def _derive_framework(command: str) -> str:
    """Derive a human-readable framework name from the test command.

    Args:
        command: The test command string.

    Returns:
        A short framework name.
    """
    if "pytest" in command:
        return "pytest"
    if "npm test" in command:
        return "npm test"
    if "go test" in command:
        return "go test"
    if "cargo test" in command:
        return "cargo test"
    if "mvn" in command:
        return "maven"
    if "gradle" in command:
        return "gradle"
    return command
