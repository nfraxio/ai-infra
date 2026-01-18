"""Interactive task review mode for ai-infra CLI.

Phase 16.7.1 of EXECUTOR_6.md: Interactive Task Review Mode.

This module provides:
- InteractiveSession: Main class for managing interactive execution
- TaskPreview: Rich renderable for task preview panel
- ContextPreview: LLM context preview panel
- DependencyGraph: Visual task dependency display
- Batch operations for efficient task management

Example:
    ```python
    from ai_infra.cli.interactive import InteractiveSession, TaskPreview

    session = InteractiveSession(tasks=task_list, console=console)
    result = await session.run()  # Returns action taken

    # Or use TaskPreview directly
    preview = TaskPreview(task=current_task)
    console.print(preview)
    ```
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ai_infra.cli.console import (
    BOX_STYLES,
    BRAND_ACCENT,
    detect_terminal_capabilities,
    get_console,
    print_info,
    print_success,
    print_warning,
)
from ai_infra.cli.progress import TaskStatus, get_status_icon

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Enums and Constants
# =============================================================================


class TaskAction(str, Enum):
    """Actions a user can take on a task."""

    EXECUTE = "execute"
    SKIP = "skip"
    EDIT = "edit"
    VIEW_CONTEXT = "view_context"
    VIEW_DEPENDENCIES = "view_dependencies"
    QUIT = "quit"
    EXECUTE_ALL = "execute_all"
    NEXT_PHASE = "next_phase"
    REORDER = "reorder"
    HELP = "help"


class SessionState(str, Enum):
    """State of the interactive session."""

    IDLE = "idle"
    PREVIEWING = "previewing"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# Default key bindings
DEFAULT_KEY_BINDINGS: dict[str, TaskAction] = {
    "enter": TaskAction.EXECUTE,
    "s": TaskAction.SKIP,
    "e": TaskAction.EDIT,
    "v": TaskAction.VIEW_CONTEXT,
    "d": TaskAction.VIEW_DEPENDENCIES,
    "q": TaskAction.QUIT,
    "a": TaskAction.EXECUTE_ALL,
    "n": TaskAction.NEXT_PHASE,
    "r": TaskAction.REORDER,
    "?": TaskAction.HELP,
}


# Complexity levels with estimated durations
COMPLEXITY_ESTIMATES: dict[str, tuple[str, str]] = {
    "trivial": ("Trivial", "< 30 seconds"),
    "low": ("Low", "30-60 seconds"),
    "medium": ("Medium", "1-3 minutes"),
    "high": ("High", "3-5 minutes"),
    "complex": ("Complex", "5+ minutes"),
}


# =============================================================================
# Task Data Structures
# =============================================================================


@dataclass
class TaskInfo:
    """Information about a task for interactive display.

    Attributes:
        id: Task identifier (e.g., "1.1.3").
        title: Task title.
        description: Detailed task description.
        phase: Phase this task belongs to.
        status: Current task status.
        complexity: Estimated complexity level.
        dependencies: List of dependency task IDs.
        completed_dependencies: Subset of dependencies that are complete.
    """

    id: str
    title: str
    description: str = ""
    phase: str = ""
    status: TaskStatus | str = TaskStatus.PENDING
    complexity: str = "medium"
    dependencies: list[str] = field(default_factory=list)
    completed_dependencies: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert string status to enum."""
        if isinstance(self.status, str):
            try:
                self.status = TaskStatus(self.status)
            except ValueError:
                self.status = TaskStatus.PENDING


@dataclass
class ContextInfo:
    """Information about LLM context for a task.

    Attributes:
        system_tokens: Tokens in system prompt.
        task_tokens: Tokens in task context.
        file_tokens: Tokens from file context.
        memory_tokens: Tokens from run memory.
        files_included: List of included file paths.
        context_window: Total context window size.
    """

    system_tokens: int = 0
    task_tokens: int = 0
    file_tokens: int = 0
    memory_tokens: int = 0
    files_included: list[tuple[str, str]] = field(default_factory=list)
    context_window: int = 200_000

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.system_tokens + self.task_tokens + self.file_tokens + self.memory_tokens

    @property
    def usage_percentage(self) -> float:
        """Percentage of context window used."""
        if self.context_window <= 0:
            return 0.0
        return (self.total_tokens / self.context_window) * 100


@dataclass
class DependencyNode:
    """Node in the dependency graph.

    Attributes:
        task_id: Task identifier.
        title: Task title.
        status: Task status.
        is_current: Whether this is the current task.
    """

    task_id: str
    title: str
    status: TaskStatus | str = TaskStatus.PENDING
    is_current: bool = False


# =============================================================================
# Task Preview Panel
# =============================================================================


class TaskPreview:
    """Rich renderable for task preview panel.

    Displays task information with action hints for interactive mode.
    """

    def __init__(
        self,
        task: TaskInfo,
        *,
        show_actions: bool = True,
        compact: bool = False,
    ) -> None:
        """Initialize the task preview.

        Args:
            task: Task information to display.
            show_actions: Whether to show action key hints.
            compact: Use compact layout for narrow terminals.
        """
        self.task = task
        self.show_actions = show_actions
        self.compact = compact

    def _get_complexity_display(self) -> tuple[str, str]:
        """Get complexity label and estimate."""
        return COMPLEXITY_ESTIMATES.get(
            self.task.complexity.lower(),
            COMPLEXITY_ESTIMATES["medium"],
        )

    def _render_description(self) -> Text:
        """Render task description with word wrapping."""
        text = Text()
        if self.task.description:
            # Word wrap description
            lines = self.task.description.strip().split("\n")
            for line in lines:
                text.append(f"  {line}\n", style="task.description")
        return text

    def _render_metadata(self) -> Table:
        """Render task metadata (complexity, dependencies)."""
        _ = detect_terminal_capabilities()  # For future use

        table = Table(
            show_header=False,
            show_edge=False,
            pad_edge=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("Label", style="dim")
        table.add_column("Value", style="bold")

        # Complexity
        complexity_label, estimate = self._get_complexity_display()
        table.add_row("Complexity:", f"{complexity_label}    Estimated: {estimate}")

        # Dependencies
        if self.task.dependencies:
            completed = len(self.task.completed_dependencies)
            total = len(self.task.dependencies)
            dep_status = "completed" if completed == total else f"{completed}/{total} completed"
            dep_ids = ", ".join(self.task.dependencies[:3])
            if len(self.task.dependencies) > 3:
                dep_ids += f" (+{len(self.task.dependencies) - 3} more)"
            table.add_row("Dependencies:", f"{dep_ids} ({dep_status})")

        return table

    def _render_actions(self) -> Text:
        """Render action key hints."""
        text = Text()
        actions = [
            ("[Enter]", "Execute"),
            ("[s]", "Skip"),
            ("[e]", "Edit"),
            ("[v]", "Context"),
            ("[d]", "Deps"),
            ("[q]", "Quit"),
        ]

        for i, (key, label) in enumerate(actions):
            if i > 0:
                text.append("   ", style="dim")
            text.append(key, style=f"bold {BRAND_ACCENT}")
            text.append(f" {label}", style="dim")

        return text

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the task preview panel."""
        # Title section
        title_text = Text()
        title_text.append(f"Task {self.task.id}: ", style="task.id")
        title_text.append(self.task.title, style="task.title")

        # Build content
        content_parts: list[Any] = []

        # Task title first
        content_parts.append(title_text)

        # Description
        if self.task.description:
            content_parts.append(Text())  # Spacer
            content_parts.append(Text("Description:", style="dim"))
            content_parts.append(self._render_description())

        # Metadata
        content_parts.append(self._render_metadata())

        # Group content
        content = Group(*content_parts)

        # Build panel
        panel = Panel(
            content,
            title="[bold]NEXT TASK[/]",
            title_align="left",
            border_style=BRAND_ACCENT,
            box=BOX_STYLES.get("default", BOX_STYLES["default"]),
            padding=(1, 2),
        )

        yield panel

        # Action hints below panel
        if self.show_actions:
            yield Text()
            yield self._render_actions()


# =============================================================================
# Context Preview Panel
# =============================================================================


class ContextPreview:
    """Rich renderable for LLM context preview.

    Shows token breakdown and included files.
    """

    def __init__(self, context: ContextInfo) -> None:
        """Initialize the context preview.

        Args:
            context: Context information to display.
        """
        self.context = context

    def _format_tokens(self, count: int) -> str:
        """Format token count with comma separators."""
        return f"{count:,}"

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the context preview panel."""
        ctx = self.context

        # Token breakdown table
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("Component", style="dim")
        table.add_column("Tokens", style=f"bold {BRAND_ACCENT}", justify="right")

        table.add_row("System Prompt:", self._format_tokens(ctx.system_tokens))
        table.add_row("Task Context:", self._format_tokens(ctx.task_tokens))
        file_count = len(ctx.files_included)
        table.add_row(
            "File Context:",
            f"{self._format_tokens(ctx.file_tokens)} ({file_count} files)",
        )
        table.add_row("Run Memory:", self._format_tokens(ctx.memory_tokens))

        # Separator and total
        separator = Text("─" * 25, style="dim")
        total_text = Text()
        total_text.append("Total: ", style="dim")
        total_text.append(self._format_tokens(ctx.total_tokens), style="bold white")
        total_text.append(f" (~{ctx.usage_percentage:.0f}% of context window)", style="dim")

        # Files list
        files_text = Text()
        if ctx.files_included:
            files_text.append("\nFiles included:\n", style="dim")
            for path, scope in ctx.files_included[:5]:
                files_text.append(f"  - {path}", style="path")
                files_text.append(f" ({scope})\n", style="dim")
            if len(ctx.files_included) > 5:
                files_text.append(
                    f"  ... and {len(ctx.files_included) - 5} more\n",
                    style="dim",
                )

        content = Group(table, separator, total_text, files_text)

        panel = Panel(
            content,
            title="[bold]CONTEXT PREVIEW[/]",
            title_align="left",
            border_style="blue",
            box=BOX_STYLES.get("default", BOX_STYLES["default"]),
            padding=(1, 2),
        )

        yield panel


# =============================================================================
# Dependency Graph Display
# =============================================================================


class DependencyGraph:
    """Rich renderable for task dependency visualization.

    Shows the dependency chain leading to the current task.
    """

    def __init__(
        self,
        task_id: str,
        dependencies: list[DependencyNode],
    ) -> None:
        """Initialize the dependency graph.

        Args:
            task_id: Current task ID.
            dependencies: List of dependency nodes in order.
        """
        self.task_id = task_id
        self.dependencies = dependencies

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the dependency graph."""
        caps = detect_terminal_capabilities()
        use_unicode = caps.unicode_support

        # Connectors
        pipe = "│" if use_unicode else "|"

        content = Text()
        content.append(f"Task Dependencies for {self.task_id}:\n\n", style="bold")

        for i, node in enumerate(self.dependencies):
            # Status icon
            status = node.status if isinstance(node.status, TaskStatus) else TaskStatus(node.status)
            icon = get_status_icon(status, use_unicode)

            # Style based on status
            if node.is_current:
                style = "bold yellow"
                suffix = " [current]"
            elif status == TaskStatus.COMPLETE:
                style = "dim green"
                suffix = " [complete]"
            elif status == TaskStatus.FAILED:
                style = "bold red"
                suffix = " [failed]"
            else:
                style = "dim"
                suffix = ""

            # Task line with dots
            task_text = f"{node.task_id} {node.title}"
            padding = 50 - len(task_text)
            dots = "." * max(1, padding)

            content.append(f"{icon} {task_text} ", style=style)
            content.append(dots, style="dim")
            content.append(f"{suffix}\n", style=style)

            # Connector to next
            if i < len(self.dependencies) - 1:
                content.append(f"  {pipe}\n", style="dim")

        panel = Panel(
            content,
            title="[bold]DEPENDENCIES[/]",
            title_align="left",
            border_style="magenta",
            box=BOX_STYLES.get("default", BOX_STYLES["default"]),
            padding=(1, 2),
        )

        yield panel


# =============================================================================
# Help Display
# =============================================================================


def render_help() -> Panel:
    """Render help panel with all available commands.

    Returns:
        Panel with help content.
    """
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Key", style=f"bold {BRAND_ACCENT}")
    table.add_column("Action", style="white")
    table.add_column("Description", style="dim")

    commands = [
        ("Enter", "Execute", "Run the current task"),
        ("s", "Skip", "Skip this task and move to next"),
        ("e", "Edit", "Edit task description before execution"),
        ("v", "Context", "Preview LLM context (tokens, files)"),
        ("d", "Dependencies", "View task dependency graph"),
        ("a", "Execute All", "Run all remaining tasks without prompts"),
        ("n", "Next Phase", "Skip to the next phase"),
        ("r", "Reorder", "Change task execution order"),
        ("q", "Quit", "Stop execution and exit"),
        ("?", "Help", "Show this help message"),
    ]

    for key, action, desc in commands:
        table.add_row(key, action, desc)

    return Panel(
        table,
        title="[bold]INTERACTIVE MODE HELP[/]",
        title_align="left",
        border_style="green",
        box=BOX_STYLES.get("default", BOX_STYLES["default"]),
        padding=(1, 2),
    )


# =============================================================================
# Task Editor
# =============================================================================


def edit_task_description(
    task: TaskInfo,
    editor: str | None = None,
) -> str | None:
    """Open task description in editor for modification.

    Args:
        task: Task to edit.
        editor: Editor command (defaults to $EDITOR or 'nano').

    Returns:
        Modified description or None if cancelled.
    """
    # Determine editor
    if editor is None:
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))

    # Create temp file with current description
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix=f"task_{task.id}_",
        delete=False,
    ) as f:
        f.write(f"# Task {task.id}: {task.title}\n\n")
        f.write("## Description\n\n")
        f.write(task.description or "(No description)")
        f.write("\n\n")
        f.write("<!-- Edit the description above and save to apply changes -->\n")
        f.write("<!-- Delete all content to cancel edit -->\n")
        temp_path = f.name

    try:
        # Open editor
        subprocess.run(
            [editor, temp_path],
            check=True,
        )

        # Read modified content
        with open(temp_path) as f:
            content = f.read()

        # Parse modified description
        lines = content.strip().split("\n")
        # Skip header lines and comments
        description_lines = []
        in_description = False
        for line in lines:
            if line.startswith("## Description"):
                in_description = True
                continue
            if line.startswith("<!--"):
                continue
            if in_description:
                description_lines.append(line)

        new_description = "\n".join(description_lines).strip()

        # Return None if empty (cancelled)
        if not new_description or new_description == "(No description)":
            return None

        return new_description

    except subprocess.CalledProcessError:
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# =============================================================================
# Interactive Session
# =============================================================================


@dataclass
class SessionResult:
    """Result of an interactive session action.

    Attributes:
        action: Action taken.
        task_id: Task ID affected.
        modified_description: New description if edited.
        execute_remaining: Whether to execute all remaining.
        skip_to_phase: Phase to skip to (if next_phase).
    """

    action: TaskAction
    task_id: str | None = None
    modified_description: str | None = None
    execute_remaining: bool = False
    skip_to_phase: str | None = None


class InteractiveSession:
    """Manages interactive task review and execution.

    Provides a REPL-like interface for reviewing and controlling
    task execution one at a time.
    """

    def __init__(
        self,
        tasks: Sequence[TaskInfo],
        *,
        console: Console | None = None,
        context_provider: Callable[[TaskInfo], ContextInfo] | None = None,
        dependency_provider: Callable[[str], list[DependencyNode]] | None = None,
    ) -> None:
        """Initialize the interactive session.

        Args:
            tasks: Sequence of tasks to process.
            console: Rich console instance.
            context_provider: Callable to get context info for a task.
            dependency_provider: Callable to get dependencies for a task.
        """
        self.tasks = list(tasks)
        self.console = console or get_console()
        self.context_provider = context_provider
        self.dependency_provider = dependency_provider

        self._current_index = 0
        self._state = SessionState.IDLE
        self._interrupted = False

    @property
    def current_task(self) -> TaskInfo | None:
        """Get the current task."""
        if 0 <= self._current_index < len(self.tasks):
            return self.tasks[self._current_index]
        return None

    @property
    def remaining_tasks(self) -> int:
        """Number of remaining tasks."""
        return len(self.tasks) - self._current_index

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    def _read_key(self) -> str:
        """Read a single keypress from the user.

        Returns:
            Key pressed as string.
        """
        # Try to use prompt_toolkit for better key handling
        try:
            from prompt_toolkit import prompt  # type: ignore[import-not-found]
            from prompt_toolkit.key_binding import KeyBindings  # type: ignore[import-not-found]
            from prompt_toolkit.keys import Keys  # type: ignore[import-not-found]

            bindings = KeyBindings()
            result: list[str] = []

            @bindings.add(Keys.Enter)
            def _(event: Any) -> None:
                result.append("enter")
                event.app.exit()

            @bindings.add(Keys.Any)
            def _(event: Any) -> None:
                result.append(event.data)
                event.app.exit()

            prompt("", key_bindings=bindings)
            return result[0] if result else ""

        except ImportError:
            # Fallback to simple input
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = sys.stdin.read(1)
                    if ch == "\r" or ch == "\n":
                        return "enter"
                    return ch.lower()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except (ImportError, OSError, termios.error):
                # Ultimate fallback - simple input
                user_input = input("\nAction: ").strip().lower()
                if not user_input:
                    return "enter"
                return user_input[0]

    def _show_task_preview(self, task: TaskInfo) -> None:
        """Display task preview panel."""
        preview = TaskPreview(task)
        self.console.print()
        self.console.print(preview)

    def _show_context_preview(self, task: TaskInfo) -> None:
        """Display context preview for a task."""
        if self.context_provider:
            context = self.context_provider(task)
        else:
            # Default mock context
            context = ContextInfo(
                system_tokens=847,
                task_tokens=1234,
                file_tokens=2456,
                memory_tokens=512,
                files_included=[
                    ("src/main.py", "full"),
                    ("src/models.py", "full"),
                    ("tests/conftest.py", "partial: lines 1-50"),
                ],
            )

        preview = ContextPreview(context)
        self.console.print()
        self.console.print(preview)
        self.console.print()
        self.console.print("[dim]Press any key to continue...[/]")
        self._read_key()

    def _show_dependency_graph(self, task: TaskInfo) -> None:
        """Display dependency graph for a task."""
        if self.dependency_provider:
            nodes = self.dependency_provider(task.id)
        else:
            # Default mock dependencies based on task info
            nodes = []
            for dep_id in task.completed_dependencies:
                nodes.append(
                    DependencyNode(
                        task_id=dep_id,
                        title=f"Dependency {dep_id}",
                        status=TaskStatus.COMPLETE,
                    )
                )
            for dep_id in task.dependencies:
                if dep_id not in task.completed_dependencies:
                    nodes.append(
                        DependencyNode(
                            task_id=dep_id,
                            title=f"Dependency {dep_id}",
                            status=TaskStatus.PENDING,
                        )
                    )
            nodes.append(
                DependencyNode(
                    task_id=task.id,
                    title=task.title,
                    status=task.status,
                    is_current=True,
                )
            )

        graph = DependencyGraph(task.id, nodes)
        self.console.print()
        self.console.print(graph)
        self.console.print()
        self.console.print("[dim]Press any key to continue...[/]")
        self._read_key()

    def _show_help(self) -> None:
        """Display help panel."""
        self.console.print()
        self.console.print(render_help())
        self.console.print()
        self.console.print("[dim]Press any key to continue...[/]")
        self._read_key()

    def _edit_task(self, task: TaskInfo) -> str | None:
        """Edit task description in external editor."""
        print_info("Opening task in editor...")
        new_desc = edit_task_description(task)
        if new_desc:
            print_success("Task description updated")
        else:
            print_warning("Edit cancelled or no changes made")
        return new_desc

    def _handle_action(self, key: str, task: TaskInfo) -> SessionResult | None:
        """Handle a user action.

        Args:
            key: Key pressed.
            task: Current task.

        Returns:
            SessionResult if action completes the interaction, None to continue.
        """
        action = DEFAULT_KEY_BINDINGS.get(key)

        if action == TaskAction.EXECUTE:
            return SessionResult(action=TaskAction.EXECUTE, task_id=task.id)

        elif action == TaskAction.SKIP:
            return SessionResult(action=TaskAction.SKIP, task_id=task.id)

        elif action == TaskAction.EDIT:
            new_desc = self._edit_task(task)
            if new_desc:
                return SessionResult(
                    action=TaskAction.EDIT,
                    task_id=task.id,
                    modified_description=new_desc,
                )
            # Continue previewing after cancelled edit
            return None

        elif action == TaskAction.VIEW_CONTEXT:
            self._show_context_preview(task)
            return None

        elif action == TaskAction.VIEW_DEPENDENCIES:
            self._show_dependency_graph(task)
            return None

        elif action == TaskAction.QUIT:
            return SessionResult(action=TaskAction.QUIT)

        elif action == TaskAction.EXECUTE_ALL:
            return SessionResult(
                action=TaskAction.EXECUTE_ALL,
                execute_remaining=True,
            )

        elif action == TaskAction.NEXT_PHASE:
            # Find next phase
            current_phase = task.phase
            for t in self.tasks[self._current_index :]:
                if t.phase != current_phase:
                    return SessionResult(
                        action=TaskAction.NEXT_PHASE,
                        skip_to_phase=t.phase,
                    )
            print_warning("Already on the last phase")
            return None

        elif action == TaskAction.REORDER:
            print_warning("Reorder not yet implemented")
            return None

        elif action == TaskAction.HELP:
            self._show_help()
            return None

        else:
            # Unknown key
            print_warning(f"Unknown key: {key!r} - press ? for help")
            return None

    def preview_task(self, task: TaskInfo) -> SessionResult:
        """Preview a single task and get user action.

        Args:
            task: Task to preview.

        Returns:
            SessionResult with user's chosen action.
        """
        self._state = SessionState.PREVIEWING

        while True:
            self._show_task_preview(task)
            self.console.print()

            # Get user input
            key = self._read_key()

            result = self._handle_action(key, task)
            if result is not None:
                return result

    async def run_async(self) -> SessionResult:
        """Run the interactive session asynchronously.

        Returns:
            Final session result.
        """
        self._state = SessionState.IDLE

        while self._current_index < len(self.tasks):
            task = self.tasks[self._current_index]

            # Skip completed/skipped tasks
            if task.status in (TaskStatus.COMPLETE, TaskStatus.SKIPPED):
                self._current_index += 1
                continue

            result = self.preview_task(task)

            if result.action == TaskAction.QUIT:
                self._state = SessionState.CANCELLED
                return result

            elif result.action == TaskAction.EXECUTE_ALL:
                self._state = SessionState.EXECUTING
                return result

            elif result.action in (TaskAction.EXECUTE, TaskAction.SKIP, TaskAction.EDIT):
                # Return result for caller to handle execution
                return result

            elif result.action == TaskAction.NEXT_PHASE:
                # Skip to next phase
                if result.skip_to_phase:
                    while (
                        self._current_index < len(self.tasks)
                        and self.tasks[self._current_index].phase != result.skip_to_phase
                    ):
                        self._current_index += 1
                continue

        self._state = SessionState.COMPLETED
        return SessionResult(action=TaskAction.QUIT)

    def run(self) -> SessionResult:
        """Run the interactive session synchronously.

        Returns:
            Final session result.
        """
        return asyncio.get_event_loop().run_until_complete(self.run_async())

    def advance(self) -> None:
        """Move to the next task."""
        self._current_index += 1

    def interrupt(self) -> None:
        """Signal session interruption."""
        self._interrupted = True
        self._state = SessionState.PAUSED


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "TaskAction",
    "SessionState",
    # Data classes
    "TaskInfo",
    "ContextInfo",
    "DependencyNode",
    "SessionResult",
    # Renderables
    "TaskPreview",
    "ContextPreview",
    "DependencyGraph",
    # Functions
    "render_help",
    "edit_task_description",
    # Session
    "InteractiveSession",
    # Constants
    "DEFAULT_KEY_BINDINGS",
    "COMPLEXITY_ESTIMATES",
]
