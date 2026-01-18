"""Centralized console management for ai-infra CLI.

Phase 16.6.1 of EXECUTOR_6.md: Rich Console Foundation.

This module provides:
- ExecutorTheme: Consistent color theming across all CLI output
- get_console(): Factory function for themed console instances
- Semantic output helpers: print_header, print_success, print_error, etc.
- Spinner configurations for different operations
- Box styles for panels

Example:
    ```python
    from ai_infra.cli.console import get_console, print_success, print_error

    console = get_console()
    console.print("[info]Processing...[/info]")

    print_success("Task completed", details="Created 3 files")
    print_error("Connection failed", hint="Check your API key")
    ```
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from rich.box import DOUBLE, HEAVY, MINIMAL, ROUNDED, Box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from collections.abc import Iterator


# =============================================================================
# Brand Colors
# =============================================================================

# nfrax brand colors (dark blue theme)
BRAND_PRIMARY = "#1e3a5f"  # Dark navy
BRAND_ACCENT = "#3b82f6"  # Accent blue
BRAND_MUTED = "#64748b"  # Slate


# =============================================================================
# Theme Definition
# =============================================================================

EXECUTOR_THEME = Theme(
    {
        # Brand colors
        "brand": BRAND_ACCENT,
        "brand.bold": f"bold {BRAND_ACCENT}",
        "brand.dim": f"dim {BRAND_ACCENT}",
        # Semantic colors
        "info": BRAND_ACCENT,
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "critical": "bold white on red",
        # Status indicators
        "status.pending": "dim white",
        "status.running": "bold yellow",
        "status.complete": "bold green",
        "status.failed": "bold red",
        "status.skipped": "dim yellow",
        # Task display
        "task.id": f"dim {BRAND_ACCENT}",
        "task.title": "bold white",
        "task.description": "dim white",
        "task.duration": "dim green",
        # Agent types
        "agent.coder": "bold blue",
        "agent.tester": "bold green",
        "agent.reviewer": "bold magenta",
        "agent.debugger": "bold red",
        "agent.researcher": f"bold {BRAND_ACCENT}",
        "agent.orchestrator": "bold yellow",
        # Code and files
        "code": BRAND_ACCENT,
        "path": f"{BRAND_ACCENT} underline",
        "command": "bold white on grey23",
        # Metrics
        "metric.label": "dim white",
        "metric.value": f"bold {BRAND_ACCENT}",
        "metric.unit": f"dim {BRAND_ACCENT}",
        # Hints and secondary text
        "hint": "dim italic",
        "muted": "dim",
        # Progress indicators
        "progress.bar": BRAND_ACCENT,
        "progress.percentage": f"bold {BRAND_ACCENT}",
        "progress.remaining": f"dim {BRAND_ACCENT}",
    }
)


# =============================================================================
# Color Support Detection
# =============================================================================


class ColorSupport(str, Enum):
    """Terminal color support levels."""

    NONE = "none"
    BASIC = "basic"  # 16 colors
    EXTENDED = "256"  # 256 colors
    TRUECOLOR = "truecolor"  # 24-bit color


@dataclass(frozen=True)
class TerminalCapabilities:
    """Detected terminal capabilities.

    Attributes:
        color_support: Level of color support.
        unicode_support: Whether unicode characters are supported.
        interactive: Whether running in an interactive TTY.
        width: Terminal width in columns.
        ci_environment: Whether running in a CI environment.
    """

    color_support: ColorSupport
    unicode_support: bool
    interactive: bool
    width: int
    ci_environment: bool


def _detect_ci_environment() -> bool:
    """Detect if running in a CI environment."""
    ci_vars = [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "JENKINS_URL",
        "BUILDKITE",
        "DRONE",
        "AZURE_PIPELINES",
        "TF_BUILD",
    ]
    return any(os.environ.get(var) for var in ci_vars)


def _detect_color_support() -> ColorSupport:
    """Detect terminal color support level."""
    # Check for explicit disable
    if os.environ.get("NO_COLOR"):
        return ColorSupport.NONE

    # Check for explicit enable
    force_color = os.environ.get("FORCE_COLOR")
    if force_color:
        return ColorSupport.TRUECOLOR

    # Check COLORTERM for truecolor
    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
        return ColorSupport.TRUECOLOR

    # Check TERM for capabilities
    term = os.environ.get("TERM", "").lower()
    if not term or term == "dumb":
        return ColorSupport.NONE

    if "256color" in term or "256-color" in term:
        return ColorSupport.EXTENDED

    if any(t in term for t in ("xterm", "screen", "vt100", "linux", "ansi")):
        return ColorSupport.BASIC

    # Check for known terminals that support colors
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if term_program in ("iterm.app", "apple_terminal", "vscode", "alacritty"):
        return ColorSupport.TRUECOLOR

    # Default to basic if we have a TTY
    if sys.stdout.isatty():
        return ColorSupport.BASIC

    return ColorSupport.NONE


def _detect_unicode_support() -> bool:
    """Detect if terminal supports unicode."""
    # Check for explicit UTF-8 encoding
    encoding = sys.stdout.encoding or ""
    if "utf" in encoding.lower():
        return True

    # Check LANG/LC_ALL for UTF-8
    for var in ("LC_ALL", "LC_CTYPE", "LANG"):
        value = os.environ.get(var, "").lower()
        if "utf-8" in value or "utf8" in value:
            return True

    # Windows Terminal and modern terminals support unicode
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if term_program in ("iterm.app", "vscode", "alacritty", "windows-terminal"):
        return True

    # Check for Windows Terminal via WT_SESSION
    if os.environ.get("WT_SESSION"):
        return True

    return False


@lru_cache(maxsize=1)
def detect_terminal_capabilities() -> TerminalCapabilities:
    """Detect terminal capabilities.

    Returns cached result for performance.
    """
    return TerminalCapabilities(
        color_support=_detect_color_support(),
        unicode_support=_detect_unicode_support(),
        interactive=sys.stdout.isatty(),
        width=os.get_terminal_size().columns if sys.stdout.isatty() else 80,
        ci_environment=_detect_ci_environment(),
    )


# =============================================================================
# Console Factory
# =============================================================================

# Global console instance (created lazily)
_console: Console | None = None


def get_console(
    *,
    force_terminal: bool | None = None,
    force_color: bool | None = None,
    width: int | None = None,
) -> Console:
    """Get the themed console instance.

    Creates a singleton Console with the executor theme. The console
    is configured based on detected terminal capabilities.

    Args:
        force_terminal: Force terminal mode (for testing).
        force_color: Force color output (for testing).
        width: Override terminal width.

    Returns:
        Themed Console instance.

    Example:
        ```python
        console = get_console()
        console.print("[success]Done![/success]")
        console.print("[error]Failed[/error]")
        ```
    """
    global _console

    if _console is None:
        caps = detect_terminal_capabilities()

        # Determine color system based on detection
        color_system: Literal["auto", "standard", "256", "truecolor", None]
        if force_color is False or caps.color_support == ColorSupport.NONE:
            color_system = None
        elif caps.color_support == ColorSupport.TRUECOLOR:
            color_system = "truecolor"
        elif caps.color_support == ColorSupport.EXTENDED:
            color_system = "256"
        elif caps.color_support == ColorSupport.BASIC:
            color_system = "standard"
        else:
            color_system = "auto"

        _console = Console(
            theme=EXECUTOR_THEME,
            force_terminal=force_terminal,
            color_system=color_system,
            width=width or (caps.width if caps.interactive else 80),
            highlight=False,
        )

    return _console


def reset_console() -> None:
    """Reset the console singleton.

    Useful for testing or when terminal capabilities change.
    """
    global _console
    _console = None
    detect_terminal_capabilities.cache_clear()


# =============================================================================
# Spinner Configurations
# =============================================================================


@dataclass(frozen=True)
class SpinnerConfig:
    """Configuration for a spinner.

    Attributes:
        spinner: Rich spinner name (e.g., "dots", "line").
        text: Default text to display.
        style: Style for the spinner.
    """

    spinner: str
    text: str
    style: str = BRAND_ACCENT


SPINNERS: dict[str, SpinnerConfig] = {
    "thinking": SpinnerConfig(spinner="dots", text="Analyzing...", style=BRAND_ACCENT),
    "executing": SpinnerConfig(spinner="line", text="Executing...", style="yellow"),
    "verifying": SpinnerConfig(spinner="dots2", text="Verifying...", style="green"),
    "loading": SpinnerConfig(spinner="dots12", text="Loading...", style=BRAND_ACCENT),
    "connecting": SpinnerConfig(spinner="dots", text="Connecting...", style="blue"),
    "processing": SpinnerConfig(spinner="dots8Bit", text="Processing...", style=BRAND_ACCENT),
}


def get_spinner(name: str) -> SpinnerConfig:
    """Get spinner configuration by name.

    Args:
        name: Spinner name (thinking, executing, verifying, etc.).

    Returns:
        SpinnerConfig for the requested spinner.

    Raises:
        KeyError: If spinner name is not found.
    """
    if name not in SPINNERS:
        raise KeyError(f"Unknown spinner: {name}. Available: {list(SPINNERS.keys())}")
    return SPINNERS[name]


# =============================================================================
# Box Styles
# =============================================================================

BOX_STYLES: dict[str, Box] = {
    "default": ROUNDED,
    "error": HEAVY,
    "success": DOUBLE,
    "minimal": MINIMAL,
}


def get_box_style(name: str) -> Box:
    """Get box style by name.

    Args:
        name: Box style name (default, error, success, minimal).

    Returns:
        Rich Box instance.

    Raises:
        KeyError: If box style name is not found.
    """
    if name not in BOX_STYLES:
        raise KeyError(f"Unknown box style: {name}. Available: {list(BOX_STYLES.keys())}")
    return BOX_STYLES[name]


# =============================================================================
# Semantic Output Helpers
# =============================================================================


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print a styled header.

    Args:
        title: Main header text.
        subtitle: Optional subtitle.

    Example:
        ```python
        print_header("Executor Run", subtitle="Processing 12 tasks")
        ```
    """
    console = get_console()
    header_text = Text(title, style="bold white")

    if subtitle:
        content = Text()
        content.append(title, style="bold white")
        content.append("\n")
        content.append(subtitle, style="dim white")
        console.print(
            Panel(
                content,
                box=ROUNDED,
                padding=(0, 2),
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                header_text,
                box=ROUNDED,
                padding=(0, 2),
                expand=False,
            )
        )


def print_success(message: str, details: str | None = None) -> None:
    """Print a success message.

    Args:
        message: Success message.
        details: Optional additional details.

    Example:
        ```python
        print_success("Task completed", details="Created 3 files")
        ```
    """
    console = get_console()
    text = Text()
    text.append("[OK] ", style="bold green")
    text.append(message, style="success")

    if details:
        text.append("\n     ", style="")
        text.append(details, style="dim")

    console.print(text)


def print_error(message: str, hint: str | None = None) -> None:
    """Print an error message.

    Args:
        message: Error message.
        hint: Optional hint for resolution.

    Example:
        ```python
        print_error("Connection failed", hint="Check your API key")
        ```
    """
    console = get_console()
    text = Text()
    text.append("[X] ", style="bold red")
    text.append(message, style="error")

    if hint:
        text.append("\n    ", style="")
        text.append("Hint: ", style="dim")
        text.append(hint, style="hint")

    console.print(text)


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message.

    Example:
        ```python
        print_warning("Token usage approaching limit")
        ```
    """
    console = get_console()
    text = Text()
    text.append("[!] ", style="bold yellow")
    text.append(message, style="warning")
    console.print(text)


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Info message.

    Example:
        ```python
        print_info("Loading configuration...")
        ```
    """
    console = get_console()
    text = Text()
    text.append("[*] ", style=f"bold {BRAND_ACCENT}")
    text.append(message, style="info")
    console.print(text)


def print_step(step: int, total: int, message: str) -> None:
    """Print a step indicator.

    Args:
        step: Current step number (1-indexed).
        total: Total number of steps.
        message: Step description.

    Example:
        ```python
        print_step(1, 5, "Initializing project")
        print_step(2, 5, "Installing dependencies")
        ```
    """
    console = get_console()
    text = Text()
    text.append(f"[{step}/{total}] ", style=f"dim {BRAND_ACCENT}")
    text.append(message, style="white")
    console.print(text)


def print_task_status(
    task_id: str,
    title: str,
    status: str,
    duration: float | None = None,
) -> None:
    """Print a task status line.

    Args:
        task_id: Task identifier (e.g., "1.1.1").
        title: Task title.
        status: Status string (pending, running, complete, failed, skipped).
        duration: Optional duration in seconds.

    Example:
        ```python
        print_task_status("1.1.1", "Create project", "complete", duration=0.8)
        print_task_status("1.1.2", "Install deps", "running")
        ```
    """
    console = get_console()
    caps = detect_terminal_capabilities()

    # Status icons with unicode/ASCII fallback
    status_icons = {
        "pending": ("[ ]", "○") if not caps.unicode_support else ("○", "○"),
        "running": ("[~]", "◐") if not caps.unicode_support else ("◐", "◐"),
        "complete": ("[x]", "●") if not caps.unicode_support else ("●", "●"),
        "failed": ("[!]", "✗") if not caps.unicode_support else ("✗", "✗"),
        "skipped": ("[-]", "○") if not caps.unicode_support else ("○", "○"),
    }

    icon = status_icons.get(status, ("[ ]", "○"))[0 if not caps.unicode_support else 1]

    text = Text()
    text.append(f"  {icon} ", style=f"status.{status}")
    text.append(f"{task_id}  ", style="task.id")
    text.append(title, style="task.title")

    if duration is not None:
        duration_str = _format_duration(duration)
        # Right-align duration
        padding = max(1, console.width - len(task_id) - len(title) - len(duration_str) - 10)
        text.append(" " * padding)
        text.append(duration_str, style="task.duration")
    elif status == "running":
        padding = max(1, console.width - len(task_id) - len(title) - 15)
        text.append(" " * padding)
        text.append("running", style="status.running")

    console.print(text)


def _format_duration(seconds: float) -> str:
    """Format duration to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string (e.g., "0.3s", "1m 23s", "1h 5m").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_divider(style: str = "dim") -> None:
    """Print a horizontal divider line.

    Args:
        style: Style for the divider line.
    """
    console = get_console()
    console.print(Text("─" * min(console.width, 72), style=style))


def print_section(title: str) -> None:
    """Print a section header.

    Args:
        title: Section title.

    Example:
        ```python
        print_section("FILES MODIFIED")
        ```
    """
    console = get_console()
    console.print()
    console.print(Text(title.upper(), style="bold white"))
    print_divider()


@contextmanager
def status_spinner(
    message: str,
    spinner_name: str = "thinking",
) -> Iterator[None]:
    """Context manager for showing a spinner during an operation.

    Args:
        message: Message to display with spinner.
        spinner_name: Name of spinner configuration to use.

    Example:
        ```python
        with status_spinner("Connecting to API...", "connecting"):
            await connect()
        ```
    """
    console = get_console()
    config = get_spinner(spinner_name)

    with console.status(
        message or config.text,
        spinner=config.spinner,
        spinner_style=config.style,
    ):
        yield


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Theme
    "EXECUTOR_THEME",
    # Console
    "get_console",
    "reset_console",
    # Capabilities
    "ColorSupport",
    "TerminalCapabilities",
    "detect_terminal_capabilities",
    # Spinners
    "SpinnerConfig",
    "SPINNERS",
    "get_spinner",
    # Box styles
    "BOX_STYLES",
    "get_box_style",
    # Output helpers
    "print_header",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_step",
    "print_task_status",
    "print_divider",
    "print_section",
    "status_spinner",
]
