"""Shell command execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nex.tools import ToolResult

_MAX_OUTPUT_LEN = 10_000


async def run_command(
    command: str,
    project_dir: Path,
    timeout: int = 30,
) -> ToolResult:
    """Execute a shell command with timeout.

    Safety checks are NOT performed here — they happen in the safety layer
    before this function is called.

    Args:
        command: The shell command to execute.
        project_dir: Working directory for the command.
        timeout: Maximum seconds to wait (default 30).

    Returns:
        ToolResult with stdout+stderr or error message.
    """
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_dir),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s: {command}",
            )

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        combined = ""
        if stdout_str:
            combined += stdout_str
        if stderr_str:
            if combined:
                combined += "\n--- stderr ---\n"
            combined += stderr_str

        if len(combined) > _MAX_OUTPUT_LEN:
            combined = combined[:_MAX_OUTPUT_LEN] + f"\n... (truncated, total {len(combined)} chars)"

        return ToolResult(
            success=process.returncode == 0,
            output=combined,
            error=f"Exit code {process.returncode}" if process.returncode != 0 else None,
        )

    except OSError as exc:
        return ToolResult(success=False, output="", error=f"Failed to run command: {exc}")
