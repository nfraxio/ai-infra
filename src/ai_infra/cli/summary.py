"""Execution summary display for ai-infra CLI.

Phase 16.6.4 of EXECUTOR_6.md: Execution Summary.

This module provides:
- ExecutionSummary: Comprehensive execution summary display
- FileChangeSummary: File modification statistics
- GitCheckpointSummary: Git commit checkpoint display
- CostEstimator: Token usage cost estimation
- FailureSummary: Actionable failure diagnostics

Example:
    ```python
    from ai_infra.cli.summary import ExecutionSummary, ExecutionResult

    result = ExecutionResult(
        status="completed",
        completed_tasks=12,
        total_tasks=12,
        duration_seconds=154.0,
        tokens_input=38102,
        tokens_output=7129,
        model_name="claude-sonnet-4",
    )

    summary = ExecutionSummary(result)
    console.print(summary)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from rich.box import ROUNDED
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ai_infra.cli.console import BRAND_ACCENT, get_console
from ai_infra.cli.progress import format_duration

if TYPE_CHECKING:
    pass


# =============================================================================
# Execution Status
# =============================================================================


class ExecutionStatus(str, Enum):
    """Execution completion status."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


# =============================================================================
# File Change Types
# =============================================================================


class FileChangeType(str, Enum):
    """Type of file modification."""

    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileChange:
    """Represents a file modification.

    Attributes:
        path: File path relative to project root.
        change_type: Type of change (new, modified, deleted).
        lines_added: Number of lines added.
        lines_removed: Number of lines removed.
    """

    path: str
    change_type: FileChangeType
    lines_added: int = 0
    lines_removed: int = 0

    def get_icon(self) -> str:
        """Get icon for change type."""
        icons = {
            FileChangeType.NEW: "+",
            FileChangeType.MODIFIED: "~",
            FileChangeType.DELETED: "-",
        }
        return icons.get(self.change_type, "?")

    def get_style(self) -> str:
        """Get style for change type."""
        styles = {
            FileChangeType.NEW: "bold green",
            FileChangeType.MODIFIED: "bold yellow",
            FileChangeType.DELETED: "bold red",
        }
        return styles.get(self.change_type, "white")

    def get_stats(self) -> str:
        """Get line statistics string."""
        if self.change_type == FileChangeType.NEW:
            return f"+{self.lines_added} lines"
        elif self.change_type == FileChangeType.DELETED:
            return f"-{self.lines_removed} lines"
        else:
            return f"+{self.lines_added}/-{self.lines_removed} lines"


# =============================================================================
# File Change Summary
# =============================================================================


@dataclass
class FileChangeSummary:
    """Summary of file changes.

    Attributes:
        changes: List of file changes.
    """

    changes: list[FileChange] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        """Total number of files changed."""
        return len(self.changes)

    @property
    def files_added(self) -> int:
        """Number of new files."""
        return sum(1 for c in self.changes if c.change_type == FileChangeType.NEW)

    @property
    def files_modified(self) -> int:
        """Number of modified files."""
        return sum(1 for c in self.changes if c.change_type == FileChangeType.MODIFIED)

    @property
    def files_deleted(self) -> int:
        """Number of deleted files."""
        return sum(1 for c in self.changes if c.change_type == FileChangeType.DELETED)

    @property
    def total_lines_added(self) -> int:
        """Total lines added."""
        return sum(c.lines_added for c in self.changes)

    @property
    def total_lines_removed(self) -> int:
        """Total lines removed."""
        return sum(c.lines_removed for c in self.changes)

    def render(self) -> Table:
        """Render the file changes table.

        Returns:
            Rich Table object.
        """
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("icon", width=2, no_wrap=True)
        table.add_column("path", ratio=1)
        table.add_column("type", width=12, justify="center")
        table.add_column("stats", width=18, justify="right")

        for change in self.changes:
            table.add_row(
                Text(change.get_icon(), style=change.get_style()),
                Text(change.path, style="path"),
                Text(change.change_type.value, style="dim"),
                Text(change.get_stats(), style=f"dim {BRAND_ACCENT}"),
            )

        return table


# =============================================================================
# Git Checkpoint
# =============================================================================


@dataclass
class GitCheckpoint:
    """Represents a git checkpoint commit.

    Attributes:
        commit_hash: Short commit hash.
        message: Commit message.
        task_id: Associated task ID.
    """

    commit_hash: str
    message: str
    task_id: str | None = None


@dataclass
class GitCheckpointSummary:
    """Summary of git checkpoints.

    Attributes:
        checkpoints: List of checkpoints created.
    """

    checkpoints: list[GitCheckpoint] = field(default_factory=list)

    @property
    def total_checkpoints(self) -> int:
        """Number of checkpoints created."""
        return len(self.checkpoints)

    def render(self) -> Text:
        """Render the checkpoint summary.

        Returns:
            Styled Text object.
        """
        if not self.checkpoints:
            return Text("No checkpoints created", style="dim")

        text = Text()
        text.append(
            f"{self.total_checkpoints} checkpoint(s) created ",
            style="dim",
        )
        text.append(
            "(use 'git log --oneline -5' to view)",
            style="dim italic",
        )
        return text


# =============================================================================
# Cost Estimation
# =============================================================================

# Pricing per 1M tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # OpenAI
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "o1": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
    "o3-mini": {"input": 1.1, "output": 4.4},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # Mistral
    "mistral-large": {"input": 2.0, "output": 6.0},
    "mistral-small": {"input": 0.2, "output": 0.6},
    "codestral": {"input": 0.3, "output": 0.9},
}

# Default pricing for unknown models
DEFAULT_PRICING: dict[str, float] = {"input": 3.0, "output": 15.0}


@dataclass
class CostEstimator:
    """Estimates cost based on token usage and model.

    Attributes:
        model_name: Model name for pricing lookup.
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.
    """

    model_name: str
    tokens_input: int
    tokens_output: int

    def _get_pricing(self) -> dict[str, float]:
        """Get pricing for the model."""
        # Try exact match first
        if self.model_name in MODEL_PRICING:
            return MODEL_PRICING[self.model_name]

        # Try prefix match
        model_lower = self.model_name.lower()
        for key in MODEL_PRICING:
            if model_lower.startswith(key.lower()):
                return MODEL_PRICING[key]

        return DEFAULT_PRICING

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.tokens_input + self.tokens_output

    def estimate_cost(self) -> float:
        """Estimate cost in USD.

        Returns:
            Estimated cost in USD.
        """
        pricing = self._get_pricing()
        input_cost = (self.tokens_input / 1_000_000) * pricing["input"]
        output_cost = (self.tokens_output / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def format_cost(self) -> str:
        """Format cost as string.

        Returns:
            Formatted cost string (e.g., "~$0.14").
        """
        cost = self.estimate_cost()
        if cost < 0.01:
            return f"~${cost:.3f}"
        return f"~${cost:.2f}"

    def format_tokens(self) -> str:
        """Format token breakdown.

        Returns:
            Formatted token string.
        """
        return (
            f"{self.total_tokens:,} (input: {self.tokens_input:,} / output: {self.tokens_output:,})"
        )


# =============================================================================
# Test Results
# =============================================================================


@dataclass
class TestResults:
    """Test execution results.

    Attributes:
        passed: Number of passed tests.
        failed: Number of failed tests.
        skipped: Number of skipped tests.
        coverage: Code coverage percentage (0-100).
    """

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    coverage: float | None = None

    @property
    def total(self) -> int:
        """Total number of tests."""
        return self.passed + self.failed + self.skipped

    @property
    def success_rate(self) -> float:
        """Success rate percentage."""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100

    def render(self) -> Text:
        """Render test results.

        Returns:
            Styled Text object.
        """
        text = Text()

        # Passed
        text.append("Passed: ", style="dim")
        text.append(str(self.passed), style="bold green")
        text.append("    ", style="")

        # Failed
        text.append("Failed: ", style="dim")
        failed_style = "bold red" if self.failed > 0 else "dim"
        text.append(str(self.failed), style=failed_style)
        text.append("    ", style="")

        # Skipped
        text.append("Skipped: ", style="dim")
        text.append(str(self.skipped), style="dim yellow")

        # Coverage
        if self.coverage is not None:
            text.append("    ", style="")
            text.append("Coverage: ", style="dim")
            coverage_style = "bold green" if self.coverage >= 80 else "bold yellow"
            text.append(f"{self.coverage:.0f}%", style=coverage_style)

        return text


# =============================================================================
# Next Steps
# =============================================================================


@dataclass
class NextStep:
    """A suggested next step.

    Attributes:
        description: Step description.
        command: Optional command to run.
    """

    description: str
    command: str | None = None


def generate_next_steps(
    status: ExecutionStatus,
    has_failures: bool = False,
    has_git_changes: bool = False,
    failed_task_id: str | None = None,
) -> list[NextStep]:
    """Generate contextual next steps based on execution state.

    Args:
        status: Execution status.
        has_failures: Whether there were test failures.
        has_git_changes: Whether there are uncommitted changes.
        failed_task_id: ID of failed task (if any).

    Returns:
        List of suggested next steps.
    """
    steps: list[NextStep] = []

    if status == ExecutionStatus.COMPLETED:
        if has_git_changes:
            steps.append(
                NextStep(
                    "Review generated code",
                    "git diff HEAD~3",
                )
            )
        steps.append(
            NextStep(
                "Run full test suite",
                "pytest -v",
            )
        )
        steps.append(
            NextStep(
                "Update documentation if needed",
                None,
            )
        )
    elif status == ExecutionStatus.FAILED:
        if failed_task_id:
            steps.append(
                NextStep(
                    "Retry from failed task",
                    f"executor run --resume --from {failed_task_id}",
                )
            )
        steps.append(
            NextStep(
                "Check error logs for details",
                None,
            )
        )
        if has_failures:
            steps.append(
                NextStep(
                    "Fix failing tests and retry",
                    "pytest -v --tb=short",
                )
            )
    elif status == ExecutionStatus.PARTIAL:
        steps.append(
            NextStep(
                "Resume execution",
                "executor run --resume",
            )
        )
        steps.append(
            NextStep(
                "Review completed tasks",
                None,
            )
        )
    elif status == ExecutionStatus.CANCELLED:
        steps.append(
            NextStep(
                "Resume from checkpoint",
                "executor run --resume",
            )
        )
        steps.append(
            NextStep(
                "Start fresh",
                "executor run",
            )
        )

    return steps


def render_next_steps(steps: list[NextStep]) -> Table:
    """Render next steps as numbered list.

    Args:
        steps: List of next steps.

    Returns:
        Rich Table object.
    """
    table = Table(
        show_header=False,
        show_edge=False,
        box=None,
        padding=(0, 1),
    )
    table.add_column("num", width=3, no_wrap=True)
    table.add_column("step", ratio=1)

    for i, step in enumerate(steps, 1):
        step_text = Text()
        step_text.append(step.description, style="white")
        if step.command:
            step_text.append(": ", style="dim")
            step_text.append(step.command, style="command")

        table.add_row(
            Text(f"{i}.", style=f"dim {BRAND_ACCENT}"),
            step_text,
        )

    return table


# =============================================================================
# Failure Summary
# =============================================================================


@dataclass
class FailureInfo:
    """Information about a task failure.

    Attributes:
        task_id: ID of the failed task.
        task_title: Title of the failed task.
        error_type: Type of error.
        error_message: Error message.
        suggestion: Suggested fix.
        retry_command: Command to retry.
    """

    task_id: str
    task_title: str
    error_type: str
    error_message: str
    suggestion: str | None = None
    retry_command: str | None = None


class FailureSummary:
    """Actionable failure diagnostics display."""

    def __init__(self, failure: FailureInfo) -> None:
        """Initialize the failure summary.

        Args:
            failure: Failure information.
        """
        self.failure = failure

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the failure summary."""
        content_parts = []

        # Failed task info
        task_text = Text()
        task_text.append("Failed Task: ", style="dim")
        task_text.append(f"{self.failure.task_id} ", style="bold red")
        task_text.append(self.failure.task_title, style="white")
        content_parts.append(task_text)
        content_parts.append(Text(""))

        # Error info
        error_text = Text()
        error_text.append("Error: ", style="dim")
        error_text.append(self.failure.error_type, style="bold red")
        content_parts.append(error_text)
        content_parts.append(Text(""))

        # Error message
        content_parts.append(Text(self.failure.error_message, style="dim"))
        content_parts.append(Text(""))

        # Suggestion
        if self.failure.suggestion:
            suggestion_text = Text()
            suggestion_text.append("Suggestion: ", style="bold yellow")
            suggestion_text.append(self.failure.suggestion, style="white")
            content_parts.append(suggestion_text)
            content_parts.append(Text(""))

        # Retry command
        if self.failure.retry_command:
            retry_text = Text()
            retry_text.append("To retry: ", style="dim")
            retry_text.append(self.failure.retry_command, style="command")
            content_parts.append(retry_text)

        yield Panel(
            Group(*content_parts),
            title="EXECUTION FAILED",
            title_align="left",
            border_style="red",
            box=ROUNDED,
            padding=(1, 2),
        )


# =============================================================================
# Execution Result
# =============================================================================


@dataclass
class ExecutionResult:
    """Complete execution result data.

    Attributes:
        status: Execution status.
        completed_tasks: Number of completed tasks.
        total_tasks: Total number of tasks.
        duration_seconds: Total execution duration.
        tokens_input: Input tokens used.
        tokens_output: Output tokens used.
        model_name: Model used for execution.
        file_changes: File change summary.
        git_checkpoints: Git checkpoint summary.
        test_results: Test results.
        failure_info: Failure information (if failed).
    """

    status: ExecutionStatus | str
    completed_tasks: int = 0
    total_tasks: int = 0
    duration_seconds: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    model_name: str = "unknown"
    file_changes: FileChangeSummary | None = None
    git_checkpoints: GitCheckpointSummary | None = None
    test_results: TestResults | None = None
    failure_info: FailureInfo | None = None

    def __post_init__(self) -> None:
        """Convert string status to enum."""
        if isinstance(self.status, str):
            self.status = ExecutionStatus(self.status)


# =============================================================================
# Execution Summary
# =============================================================================


class ExecutionSummary:
    """Comprehensive execution summary display.

    Renders a complete summary of execution including status, statistics,
    file changes, test results, and next steps.
    """

    def __init__(self, result: ExecutionResult) -> None:
        """Initialize the summary.

        Args:
            result: Execution result data.
        """
        self.result = result

    def _render_header(self) -> Panel:
        """Render the header panel."""
        status = self.result.status
        if status == ExecutionStatus.COMPLETED:
            title = "EXECUTION COMPLETE"
            style = "bold green"
        elif status == ExecutionStatus.FAILED:
            title = "EXECUTION FAILED"
            style = "bold red"
        elif status == ExecutionStatus.PARTIAL:
            title = "EXECUTION PARTIAL"
            style = "bold yellow"
        else:
            title = "EXECUTION CANCELLED"
            style = "bold yellow"

        return Panel(
            Text(title, style=style, justify="center"),
            box=ROUNDED,
            padding=(0, 2),
            expand=True,
        )

    def _render_summary_table(self) -> Table:
        """Render the main summary table."""
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("label", width=18)
        table.add_column("value", ratio=1)

        # Status
        status = self.result.status
        if status == ExecutionStatus.COMPLETED:
            status_text = Text("Completed successfully", style="bold green")
        elif status == ExecutionStatus.FAILED:
            status_text = Text("Failed", style="bold red")
        elif status == ExecutionStatus.PARTIAL:
            status_text = Text("Partially completed", style="bold yellow")
        else:
            status_text = Text("Cancelled by user", style="bold yellow")

        table.add_row(
            Text("Status", style="dim"),
            status_text,
        )

        # Tasks with progress bar
        completed = self.result.completed_tasks
        total = self.result.total_tasks
        pct = (completed / total * 100) if total > 0 else 0

        # Create simple progress bar
        bar_width = 20
        filled = int((pct / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        tasks_text = Text()
        tasks_text.append(f"{completed}/{total} completed  ", style="white")
        tasks_text.append(bar, style=BRAND_ACCENT)
        tasks_text.append(f"  {pct:.0f}%", style=f"bold {BRAND_ACCENT}")

        table.add_row(
            Text("Tasks", style="dim"),
            tasks_text,
        )

        # Duration
        table.add_row(
            Text("Duration", style="dim"),
            Text(format_duration(self.result.duration_seconds), style="white"),
        )

        # Tokens
        cost_estimator = CostEstimator(
            self.result.model_name,
            self.result.tokens_input,
            self.result.tokens_output,
        )
        table.add_row(
            Text("Tokens", style="dim"),
            Text(cost_estimator.format_tokens(), style="white"),
        )

        # Model
        table.add_row(
            Text("Model", style="dim"),
            Text(self.result.model_name, style=BRAND_ACCENT),
        )

        # Cost
        table.add_row(
            Text("Cost", style="dim"),
            Text(f"{cost_estimator.format_cost()} (estimated)", style="dim"),
        )

        return table

    def _render_section_header(self, title: str) -> Text:
        """Render a section header."""
        return Text(title, style="bold white")

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the complete summary."""
        # Header
        yield self._render_header()
        yield Text("")

        # Summary section
        yield self._render_section_header("SUMMARY")
        yield Text("─" * min(options.max_width, 72), style="dim")
        yield self._render_summary_table()
        yield Text("")

        # File changes section
        if self.result.file_changes and self.result.file_changes.total_files > 0:
            header = f"FILES MODIFIED ({self.result.file_changes.total_files})"
            yield self._render_section_header(header)
            yield Text("─" * min(options.max_width, 72), style="dim")
            yield self.result.file_changes.render()
            yield Text("")

        # Test results section
        if self.result.test_results:
            yield self._render_section_header("TEST RESULTS")
            yield Text("─" * min(options.max_width, 72), style="dim")
            yield self.result.test_results.render()
            yield Text("")

        # Git checkpoints section
        if self.result.git_checkpoints and self.result.git_checkpoints.total_checkpoints > 0:
            yield self._render_section_header("GIT CHECKPOINTS")
            yield Text("─" * min(options.max_width, 72), style="dim")
            yield self.result.git_checkpoints.render()
            yield Text("")

        # Failure details (if failed)
        if self.result.status == ExecutionStatus.FAILED and self.result.failure_info:
            yield FailureSummary(self.result.failure_info)
            yield Text("")

        # Next steps section
        next_steps = generate_next_steps(
            status=self.result.status
            if isinstance(self.result.status, ExecutionStatus)
            else ExecutionStatus(self.result.status),
            has_failures=self.result.test_results.failed > 0 if self.result.test_results else False,
            has_git_changes=self.result.git_checkpoints.total_checkpoints > 0
            if self.result.git_checkpoints
            else False,
            failed_task_id=self.result.failure_info.task_id if self.result.failure_info else None,
        )
        if next_steps:
            yield self._render_section_header("NEXT STEPS")
            yield Text("─" * min(options.max_width, 72), style="dim")
            yield render_next_steps(next_steps)


# =============================================================================
# Convenience Functions
# =============================================================================


def print_execution_summary(result: ExecutionResult) -> None:
    """Print an execution summary to the console.

    Args:
        result: Execution result data.
    """
    console = get_console()
    summary = ExecutionSummary(result)
    console.print(summary)


def print_failure_summary(failure: FailureInfo) -> None:
    """Print a failure summary to the console.

    Args:
        failure: Failure information.
    """
    console = get_console()
    summary = FailureSummary(failure)
    console.print(summary)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Status
    "ExecutionStatus",
    # File changes
    "FileChangeType",
    "FileChange",
    "FileChangeSummary",
    # Git checkpoints
    "GitCheckpoint",
    "GitCheckpointSummary",
    # Cost estimation
    "MODEL_PRICING",
    "CostEstimator",
    # Test results
    "TestResults",
    # Next steps
    "NextStep",
    "generate_next_steps",
    "render_next_steps",
    # Failure summary
    "FailureInfo",
    "FailureSummary",
    # Execution result
    "ExecutionResult",
    # Summary
    "ExecutionSummary",
    # Convenience functions
    "print_execution_summary",
    "print_failure_summary",
]
