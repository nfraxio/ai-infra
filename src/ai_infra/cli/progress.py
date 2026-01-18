"""Task progress display components for ai-infra CLI.

Phase 16.6.2 of EXECUTOR_6.md: Task Progress Display.

This module provides:
- TaskProgressPanel: Rich renderable for displaying task progress
- PhaseSection: Collapsible phase sections for long roadmaps
- TokenCounter: Live token usage visualization
- ETA calculation based on task history

Example:
    ```python
    from ai_infra.cli.progress import TaskProgressPanel, TaskItem, PhaseSection

    tasks = [
        TaskItem(id="1.1.1", title="Create project", status="complete", duration=0.8),
        TaskItem(id="1.1.2", title="Init git", status="complete", duration=0.3),
        TaskItem(id="1.1.3", title="Configure deps", status="running"),
        TaskItem(id="1.1.4", title="Add README", status="pending"),
    ]

    panel = TaskProgressPanel(
        phase_title="Phase 1: Project Setup",
        tasks=tasks,
    )
    console.print(panel)
    ```
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ai_infra.cli.console import (
    BOX_STYLES,
    BRAND_ACCENT,
    detect_terminal_capabilities,
    get_console,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Status Icons
# =============================================================================


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StatusIcon:
    """Status icon with ASCII and Unicode variants.

    Attributes:
        ascii: ASCII representation for basic terminals.
        unicode: Unicode representation for modern terminals.
    """

    ascii: str
    unicode: str


STATUS_ICONS: dict[TaskStatus, StatusIcon] = {
    TaskStatus.PENDING: StatusIcon(ascii="[ ]", unicode="○"),
    TaskStatus.RUNNING: StatusIcon(ascii="[~]", unicode="◐"),
    TaskStatus.COMPLETE: StatusIcon(ascii="[x]", unicode="●"),
    TaskStatus.FAILED: StatusIcon(ascii="[!]", unicode="✗"),
    TaskStatus.SKIPPED: StatusIcon(ascii="[-]", unicode="○"),
}


def get_status_icon(status: TaskStatus | str, use_unicode: bool = True) -> str:
    """Get the appropriate status icon.

    Args:
        status: Task status.
        use_unicode: Whether to use unicode icons.

    Returns:
        Status icon string.
    """
    if isinstance(status, str):
        status = TaskStatus(status)

    icon = STATUS_ICONS.get(status, STATUS_ICONS[TaskStatus.PENDING])
    return icon.unicode if use_unicode else icon.ascii


# =============================================================================
# Duration Formatting
# =============================================================================


def format_duration(seconds: float) -> str:
    """Format duration to human-readable string.

    Follows smart formatting rules:
    - < 1s: "0.3s"
    - < 60s: "12.5s"
    - < 1h: "2m 34s"
    - >= 1h: "1h 23m"

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.
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


# =============================================================================
# Task Item
# =============================================================================


@dataclass
class TaskItem:
    """Represents a single task in the progress display.

    Attributes:
        id: Task identifier (e.g., "1.1.1").
        title: Task title.
        status: Current status.
        duration: Duration in seconds (for completed tasks).
        start_time: When the task started (for running tasks).
    """

    id: str
    title: str
    status: TaskStatus | str = TaskStatus.PENDING
    duration: float | None = None
    start_time: float | None = None

    def __post_init__(self) -> None:
        """Convert string status to enum."""
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)

    def get_elapsed(self) -> float | None:
        """Get elapsed time for running tasks.

        Returns:
            Elapsed seconds or None if not running.
        """
        if self.status == TaskStatus.RUNNING and self.start_time is not None:
            return time.time() - self.start_time
        return self.duration


# =============================================================================
# ETA Calculation
# =============================================================================


@dataclass
class ETACalculator:
    """Calculates estimated time of arrival based on task history.

    Attributes:
        completed_durations: List of completed task durations.
        task_weights: Optional weights for remaining tasks.
    """

    completed_durations: list[float] = field(default_factory=list)
    task_weights: dict[str, float] = field(default_factory=dict)

    def add_completed(self, duration: float) -> None:
        """Record a completed task duration.

        Args:
            duration: Task duration in seconds.
        """
        self.completed_durations.append(duration)

    def get_average_duration(self) -> float:
        """Get average duration of completed tasks.

        Returns:
            Average duration or 0 if no completed tasks.
        """
        if not self.completed_durations:
            return 0.0
        return sum(self.completed_durations) / len(self.completed_durations)

    def estimate_remaining(
        self,
        remaining_count: int,
        remaining_task_ids: Sequence[str] | None = None,
    ) -> float:
        """Estimate time remaining for pending tasks.

        Args:
            remaining_count: Number of remaining tasks.
            remaining_task_ids: Optional task IDs for weighted estimation.

        Returns:
            Estimated seconds remaining.
        """
        avg_duration = self.get_average_duration()

        if remaining_task_ids and self.task_weights:
            # Use weighted estimation if weights are available
            total_weight = sum(self.task_weights.get(tid, 1.0) for tid in remaining_task_ids)
            return avg_duration * total_weight

        # Simple estimation based on average
        return avg_duration * remaining_count

    def format_eta(self, remaining_count: int) -> str:
        """Get formatted ETA string.

        Args:
            remaining_count: Number of remaining tasks.

        Returns:
            Formatted ETA (e.g., "~3m 20s") or empty string if unknown.
        """
        if not self.completed_durations:
            return ""

        estimated = self.estimate_remaining(remaining_count)
        if estimated <= 0:
            return ""

        return f"~{format_duration(estimated)}"


# =============================================================================
# Task Progress Panel
# =============================================================================


class TaskProgressPanel:
    """Rich renderable for displaying task progress within a phase.

    Displays tasks with status icons, titles, and duration/status indicators
    in a styled panel.
    """

    def __init__(
        self,
        phase_title: str,
        tasks: Sequence[TaskItem],
        *,
        show_progress_bar: bool = True,
        show_eta: bool = True,
        elapsed: float | None = None,
        eta_calculator: ETACalculator | None = None,
    ) -> None:
        """Initialize the panel.

        Args:
            phase_title: Title for the phase (e.g., "Phase 1: Project Setup").
            tasks: Sequence of TaskItem objects.
            show_progress_bar: Whether to show progress bar footer.
            show_eta: Whether to show ETA.
            elapsed: Total elapsed time in seconds.
            eta_calculator: Optional ETA calculator for estimation.
        """
        self.phase_title = phase_title
        self.tasks = list(tasks)
        self.show_progress_bar = show_progress_bar
        self.show_eta = show_eta
        self.elapsed = elapsed
        self.eta_calculator = eta_calculator or ETACalculator()

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the panel to the console."""
        caps = detect_terminal_capabilities()
        use_unicode = caps.unicode_support
        _ = options.max_width  # Available for future use

        # Build task table
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("icon", width=3, no_wrap=True)
        table.add_column("id", width=8, no_wrap=True)
        table.add_column("title", ratio=1)
        table.add_column("duration", width=12, justify="right", no_wrap=True)

        completed_count = 0
        total_count = len(self.tasks)

        for task in self.tasks:
            status = task.status if isinstance(task.status, TaskStatus) else TaskStatus(task.status)
            icon = get_status_icon(status, use_unicode)

            # Style based on status
            icon_style = f"status.{status.value}"
            id_style = "task.id"
            title_style = "task.title" if status != TaskStatus.FAILED else "error"

            # Duration or status text
            if status == TaskStatus.COMPLETE:
                completed_count += 1
                duration_text = format_duration(task.duration) if task.duration else ""
                duration_style = "task.duration"
            elif status == TaskStatus.RUNNING:
                elapsed = task.get_elapsed()
                if elapsed is not None:
                    duration_text = f"{format_duration(elapsed)}"
                else:
                    duration_text = "running"
                duration_style = "status.running"
            elif status == TaskStatus.FAILED:
                completed_count += 1  # Count as processed
                duration_text = "FAILED"
                duration_style = "status.failed"
            elif status == TaskStatus.SKIPPED:
                completed_count += 1  # Count as processed
                duration_text = "skipped"
                duration_style = "status.skipped"
            else:
                duration_text = ""
                duration_style = "dim"

            table.add_row(
                Text(icon, style=icon_style),
                Text(task.id, style=id_style),
                Text(task.title, style=title_style),
                Text(duration_text, style=duration_style),
            )

        # Build content with optional footer
        content_parts: list[RenderableType] = [table]

        if self.show_progress_bar:
            # Progress footer
            remaining = total_count - completed_count
            eta_str = ""
            if self.show_eta and remaining > 0:
                eta_str = self.eta_calculator.format_eta(remaining)

            elapsed_str = format_duration(self.elapsed) if self.elapsed else ""

            footer_parts = [f"Progress: {completed_count}/{total_count} tasks"]
            if elapsed_str:
                footer_parts.append(f"Elapsed: {elapsed_str}")
            if eta_str:
                footer_parts.append(f"ETA: {eta_str}")

            footer = Text("  " + "    ".join(footer_parts), style="dim")
            content_parts.append(Text(""))  # Blank line
            content_parts.append(footer)

        # Combine content
        from rich.console import Group

        content = Group(*content_parts)

        # Create panel
        yield Panel(
            content,
            title=self.phase_title,
            title_align="left",
            box=BOX_STYLES["default"],
            padding=(1, 2),
            expand=True,
        )


# =============================================================================
# Collapsible Phase Section
# =============================================================================


class PhaseState(str, Enum):
    """State of a collapsible phase section."""

    EXPANDED = "expanded"
    COLLAPSED = "collapsed"


@dataclass
class PhaseSection:
    """Collapsible phase section for long roadmaps.

    Attributes:
        title: Phase title.
        completed: Number of completed tasks.
        total: Total number of tasks.
        state: Expanded or collapsed.
        status: Overall phase status.
    """

    title: str
    completed: int
    total: int
    state: PhaseState = PhaseState.COLLAPSED
    status: str | None = None

    def get_status_text(self) -> str:
        """Get status text for the phase.

        Returns:
            Status string (e.g., "COMPLETE", "IN PROGRESS", "PENDING").
        """
        if self.status:
            return self.status
        if self.completed == self.total:
            return "COMPLETE"
        elif self.completed > 0:
            return "IN PROGRESS"
        return "PENDING"

    def render(self, use_unicode: bool = True) -> Text:
        """Render the phase section header.

        Args:
            use_unicode: Whether to use unicode characters.

        Returns:
            Styled Text object.
        """
        # State indicator
        if self.state == PhaseState.EXPANDED:
            indicator = "v" if not use_unicode else "▼"
        else:
            indicator = ">" if not use_unicode else "▶"

        # Status styling
        status_text = self.get_status_text()
        if status_text == "COMPLETE":
            status_style = "status.complete"
        elif status_text == "IN PROGRESS":
            status_style = "status.running"
        else:
            status_style = "status.pending"

        text = Text()
        text.append(f"[{indicator}] ", style="dim")
        text.append(self.title, style="bold white")
        text.append(f"  [{self.completed}/{self.total}] ", style=f"dim {BRAND_ACCENT}")
        text.append(status_text, style=status_style)

        return text


def render_phase_list(
    phases: Sequence[PhaseSection],
    use_unicode: bool | None = None,
) -> list[Text]:
    """Render a list of phase sections.

    Args:
        phases: Sequence of PhaseSection objects.
        use_unicode: Override unicode detection.

    Returns:
        List of Text objects for each phase.
    """
    if use_unicode is None:
        caps = detect_terminal_capabilities()
        use_unicode = caps.unicode_support

    return [phase.render(use_unicode) for phase in phases]


# =============================================================================
# Token Counter
# =============================================================================


@dataclass
class TokenCounter:
    """Token usage visualization component.

    Attributes:
        used: Number of tokens used.
        limit: Token limit.
        width: Width of the progress bar in characters.
    """

    used: int
    limit: int
    width: int = 24

    @property
    def percentage(self) -> float:
        """Get usage percentage.

        Returns:
            Percentage (0.0-100.0).
        """
        if self.limit <= 0:
            return 0.0
        return min(100.0, (self.used / self.limit) * 100)

    @property
    def level(self) -> str:
        """Get usage level for styling.

        Returns:
            "low" (0-50%), "medium" (50-80%), or "high" (80-100%).
        """
        pct = self.percentage
        if pct >= 80:
            return "high"
        elif pct >= 50:
            return "medium"
        return "low"

    def _format_count(self, count: int) -> str:
        """Format token count with k/M suffix.

        Args:
            count: Token count.

        Returns:
            Formatted string (e.g., "12,431", "1.2M").
        """
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 10_000:
            return f"{count / 1_000:.1f}k"
        return f"{count:,}"

    def render(self) -> Text:
        """Render the token counter.

        Returns:
            Styled Text object.
        """
        pct = self.percentage
        filled = int((pct / 100) * self.width)
        empty = self.width - filled

        # Color based on level
        if self.level == "high":
            bar_style = "bold red"
            pct_style = "bold red"
        elif self.level == "medium":
            bar_style = "bold yellow"
            pct_style = "bold yellow"
        else:
            bar_style = BRAND_ACCENT
            pct_style = f"bold {BRAND_ACCENT}"

        text = Text()
        text.append("Tokens: ", style="dim")
        text.append(self._format_count(self.used), style="bold")
        text.append(" / ", style="dim")
        text.append(self._format_count(self.limit), style="dim")
        text.append("  [", style="dim")
        text.append("=" * filled, style=bar_style)
        text.append("." * empty, style="dim")
        text.append("] ", style="dim")
        text.append(f"{pct:.0f}%", style=pct_style)

        return text


# =============================================================================
# Animated Running Task Spinner
# =============================================================================


class TaskSpinner:
    """Non-blocking spinner for running tasks.

    Uses Rich Live for non-blocking updates.
    """

    def __init__(
        self,
        task_id: str,
        title: str,
        spinner_name: str = "dots",
    ) -> None:
        """Initialize the spinner.

        Args:
            task_id: Task identifier.
            title: Task title.
            spinner_name: Rich spinner name.
        """
        self.task_id = task_id
        self.title = title
        self.spinner = Spinner(spinner_name, style="status.running")
        self._live: Live | None = None
        self._start_time: float | None = None

    def _render(self) -> Text:
        """Render the current spinner state.

        Returns:
            Styled Text object.
        """
        elapsed = ""
        if self._start_time is not None:
            elapsed = format_duration(time.time() - self._start_time)

        text = Text()
        text.append("  ")
        spinner_text = self.spinner.render(time.time())
        if isinstance(spinner_text, Text):
            text.append_text(spinner_text)
        else:
            text.append(str(spinner_text))
        text.append(" ", style="")
        text.append(self.task_id, style="task.id")
        text.append("  ", style="")
        text.append(self.title, style="task.title")
        if elapsed:
            text.append(f"  {elapsed}", style="status.running")

        return text

    def start(self, console: Console | None = None) -> Live:
        """Start the spinner animation.

        Args:
            console: Optional console to use.

        Returns:
            Rich Live instance.
        """
        if console is None:
            console = get_console()

        self._start_time = time.time()
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()
        return self._live

    def update(self) -> None:
        """Update the spinner display."""
        if self._live is not None:
            self._live.update(self._render())

    def stop(self) -> float:
        """Stop the spinner and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time

        if self._live is not None:
            self._live.stop()
            self._live = None

        return elapsed


# =============================================================================
# Live Progress Display
# =============================================================================


class LiveProgressDisplay:
    """Live-updating progress display for task execution.

    Combines TaskProgressPanel with Rich Live for real-time updates.
    """

    def __init__(
        self,
        phase_title: str,
        tasks: list[TaskItem],
        console: Console | None = None,
    ) -> None:
        """Initialize the display.

        Args:
            phase_title: Phase title.
            tasks: List of task items (will be mutated during updates).
            console: Optional console to use.
        """
        self.phase_title = phase_title
        self.tasks = tasks
        self.console = console or get_console()
        self.eta_calculator = ETACalculator()
        self._live: Live | None = None
        self._start_time: float | None = None

    def _render(self) -> TaskProgressPanel:
        """Render current state.

        Returns:
            TaskProgressPanel instance.
        """
        elapsed = None
        if self._start_time is not None:
            elapsed = time.time() - self._start_time

        return TaskProgressPanel(
            phase_title=self.phase_title,
            tasks=self.tasks,
            elapsed=elapsed,
            eta_calculator=self.eta_calculator,
        )

    def start(self) -> None:
        """Start the live display."""
        self._start_time = time.time()
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.start()

    def update(self) -> None:
        """Update the display."""
        if self._live is not None:
            self._live.update(self._render())

    def mark_running(self, task_id: str) -> None:
        """Mark a task as running.

        Args:
            task_id: Task identifier.
        """
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()
                break
        self.update()

    def mark_complete(self, task_id: str, duration: float | None = None) -> None:
        """Mark a task as complete.

        Args:
            task_id: Task identifier.
            duration: Override duration (uses elapsed if None).
        """
        for task in self.tasks:
            if task.id == task_id:
                if duration is None and task.start_time is not None:
                    duration = time.time() - task.start_time
                task.status = TaskStatus.COMPLETE
                task.duration = duration
                if duration is not None:
                    self.eta_calculator.add_completed(duration)
                break
        self.update()

    def mark_failed(self, task_id: str) -> None:
        """Mark a task as failed.

        Args:
            task_id: Task identifier.
        """
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.FAILED
                break
        self.update()

    def stop(self) -> float:
        """Stop the live display.

        Returns:
            Total elapsed time in seconds.
        """
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time

        if self._live is not None:
            self._live.stop()
            self._live = None

        return elapsed


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Status
    "TaskStatus",
    "StatusIcon",
    "STATUS_ICONS",
    "get_status_icon",
    # Duration
    "format_duration",
    # Task
    "TaskItem",
    # ETA
    "ETACalculator",
    # Progress Panel
    "TaskProgressPanel",
    # Phase sections
    "PhaseState",
    "PhaseSection",
    "render_phase_list",
    # Token counter
    "TokenCounter",
    # Spinners
    "TaskSpinner",
    # Live display
    "LiveProgressDisplay",
]
