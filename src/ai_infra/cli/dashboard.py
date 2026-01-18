"""Live status dashboard for ai-infra CLI.

Phase 16.6.3 of EXECUTOR_6.md: Live Status Dashboard.

This module provides:
- StatusBar: Persistent status bar at terminal bottom
- ModelIndicator: Model name with provider abbreviation
- SubagentActivity: Activity indicators for running subagents
- KeyboardHints: Shortcut hints display

Example:
    ```python
    from ai_infra.cli.dashboard import StatusBar, StatusBarState

    state = StatusBarState(
        model_name="claude-sonnet-4",
        tokens_used=12400,
        tokens_limit=100000,
        elapsed_seconds=154.0,
        current_phase=2,
        total_phases=4,
        completed_tasks=5,
        total_tasks=12,
        active_agent="coder",
        agent_status="writing tests...",
    )

    bar = StatusBar(state)
    console.print(bar)
    ```
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from rich.box import ROUNDED, SQUARE
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ai_infra.cli.console import (
    BRAND_ACCENT,
    get_console,
)
from ai_infra.cli.progress import format_duration

if TYPE_CHECKING:
    pass


# =============================================================================
# Model Provider Mapping
# =============================================================================


class ModelProvider(str, Enum):
    """Known model providers with abbreviations."""

    ANTHROPIC = "ANT"
    OPENAI = "OAI"
    GOOGLE = "GCP"
    MISTRAL = "MIS"
    COHERE = "COH"
    META = "META"
    AMAZON = "AWS"
    UNKNOWN = "???"


# Model name prefixes to provider mapping
MODEL_PROVIDER_PREFIXES: dict[str, ModelProvider] = {
    "claude": ModelProvider.ANTHROPIC,
    "gpt": ModelProvider.OPENAI,
    "o1": ModelProvider.OPENAI,
    "o3": ModelProvider.OPENAI,
    "gemini": ModelProvider.GOOGLE,
    "palm": ModelProvider.GOOGLE,
    "mistral": ModelProvider.MISTRAL,
    "mixtral": ModelProvider.MISTRAL,
    "codestral": ModelProvider.MISTRAL,
    "command": ModelProvider.COHERE,
    "llama": ModelProvider.META,
    "titan": ModelProvider.AMAZON,
    "nova": ModelProvider.AMAZON,
}


def get_model_provider(model_name: str) -> ModelProvider:
    """Get the provider for a model name.

    Args:
        model_name: Full model name (e.g., "claude-sonnet-4").

    Returns:
        ModelProvider enum value.
    """
    model_lower = model_name.lower()
    for prefix, provider in MODEL_PROVIDER_PREFIXES.items():
        if model_lower.startswith(prefix):
            return provider
    return ModelProvider.UNKNOWN


# =============================================================================
# Model Indicator
# =============================================================================


@dataclass(frozen=True)
class ModelIndicator:
    """Model name with provider abbreviation.

    Attributes:
        model_name: Full model name.
        show_provider: Whether to show provider abbreviation.
    """

    model_name: str
    show_provider: bool = True

    def render(self) -> Text:
        """Render the model indicator.

        Returns:
            Styled Text object.
        """
        text = Text()
        text.append("Model: ", style="dim")
        text.append(self.model_name, style=f"bold {BRAND_ACCENT}")

        if self.show_provider:
            provider = get_model_provider(self.model_name)
            text.append(" (", style="dim")
            text.append(provider.value, style=f"dim {BRAND_ACCENT}")
            text.append(")", style="dim")

        return text

    def render_compact(self) -> Text:
        """Render compact version for narrow terminals.

        Returns:
            Styled Text object with abbreviated model name.
        """
        # Abbreviate long model names
        name = self.model_name
        if len(name) > 20:
            # Keep first 17 chars + "..."
            name = name[:17] + "..."

        text = Text()
        text.append(name, style=BRAND_ACCENT)
        return text


# =============================================================================
# Token Budget Visualization
# =============================================================================


class TokenBudgetLevel(str, Enum):
    """Token budget usage level for styling."""

    LOW = "low"  # 0-50%: Green
    MEDIUM = "medium"  # 50-80%: Yellow
    HIGH = "high"  # 80-100%: Red


@dataclass
class TokenBudget:
    """Token budget visualization.

    Attributes:
        used: Tokens used.
        limit: Token limit.
    """

    used: int
    limit: int

    @property
    def percentage(self) -> float:
        """Get usage percentage."""
        if self.limit <= 0:
            return 0.0
        return min(100.0, (self.used / self.limit) * 100)

    @property
    def level(self) -> TokenBudgetLevel:
        """Get usage level for styling."""
        pct = self.percentage
        if pct >= 80:
            return TokenBudgetLevel.HIGH
        elif pct >= 50:
            return TokenBudgetLevel.MEDIUM
        return TokenBudgetLevel.LOW

    def _format_count(self, count: int) -> str:
        """Format token count with k/M suffix."""
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1000:
            return f"{count / 1000:.1f}k"
        return str(count)

    def get_style(self) -> str:
        """Get Rich style based on level."""
        if self.level == TokenBudgetLevel.HIGH:
            return "bold red"
        elif self.level == TokenBudgetLevel.MEDIUM:
            return "bold yellow"
        return "bold green"

    def render(self) -> Text:
        """Render token budget display.

        Returns:
            Styled Text object.
        """
        style = self.get_style()

        text = Text()
        text.append("Tokens: ", style="dim")
        text.append(self._format_count(self.used), style=style)
        text.append("/", style="dim")
        text.append(self._format_count(self.limit), style="dim")

        return text

    def render_compact(self) -> Text:
        """Render compact version.

        Returns:
            Styled Text with just the count.
        """
        style = self.get_style()
        text = Text()
        text.append(self._format_count(self.used), style=style)
        text.append("/", style="dim")
        text.append(self._format_count(self.limit), style="dim")
        return text


# =============================================================================
# Subagent Activity
# =============================================================================


@dataclass
class SubagentActivity:
    """Subagent activity indicator.

    Attributes:
        agent_type: Type of agent (coder, tester, reviewer, etc.).
        status: Current activity description.
    """

    agent_type: str
    status: str

    def get_agent_style(self) -> str:
        """Get style for agent type."""
        styles = {
            "coder": "agent.coder",
            "tester": "agent.tester",
            "reviewer": "agent.reviewer",
            "debugger": "agent.debugger",
            "researcher": "agent.researcher",
            "orchestrator": "agent.orchestrator",
        }
        return styles.get(self.agent_type.lower(), "bold white")

    def render(self) -> Text:
        """Render the activity indicator.

        Returns:
            Styled Text object.
        """
        text = Text()
        text.append("[", style="dim")
        text.append(self.agent_type, style=self.get_agent_style())
        text.append(": ", style="dim")
        text.append(self.status, style="dim italic")
        text.append("]", style="dim")
        return text

    def render_compact(self) -> Text:
        """Render compact version.

        Returns:
            Styled Text with abbreviated status.
        """
        status = self.status
        if len(status) > 15:
            status = status[:12] + "..."

        text = Text()
        text.append("[", style="dim")
        text.append(self.agent_type[:4], style=self.get_agent_style())
        text.append(": ", style="dim")
        text.append(status, style="dim italic")
        text.append("]", style="dim")
        return text


# =============================================================================
# Keyboard Hints
# =============================================================================


@dataclass(frozen=True)
class KeyboardShortcut:
    """Single keyboard shortcut.

    Attributes:
        key: Key character.
        action: Action description.
    """

    key: str
    action: str


DEFAULT_SHORTCUTS: list[KeyboardShortcut] = [
    KeyboardShortcut("q", "quit"),
    KeyboardShortcut("p", "pause"),
    KeyboardShortcut("s", "skip"),
    KeyboardShortcut("c", "checkpoint"),
    KeyboardShortcut("?", "help"),
]


def render_keyboard_hints(
    shortcuts: list[KeyboardShortcut] | None = None,
    compact: bool = False,
) -> Text:
    """Render keyboard shortcut hints.

    Args:
        shortcuts: List of shortcuts to display. Defaults to DEFAULT_SHORTCUTS.
        compact: Use compact format.

    Returns:
        Styled Text object.
    """
    if shortcuts is None:
        shortcuts = DEFAULT_SHORTCUTS

    if compact:
        # Show only essential shortcuts
        shortcuts = shortcuts[:3]

    text = Text()
    for i, shortcut in enumerate(shortcuts):
        if i > 0:
            text.append("  ", style="dim")
        text.append("[", style="dim")
        text.append(shortcut.key, style=f"bold {BRAND_ACCENT}")
        text.append("] ", style="dim")
        text.append(shortcut.action, style="dim")

    return text


# =============================================================================
# Status Bar State
# =============================================================================


@dataclass
class StatusBarState:
    """State for the status bar.

    Attributes:
        model_name: Current model name.
        tokens_used: Tokens used.
        tokens_limit: Token limit.
        elapsed_seconds: Elapsed time in seconds.
        current_phase: Current phase number.
        total_phases: Total number of phases.
        completed_tasks: Number of completed tasks.
        total_tasks: Total number of tasks.
        active_agent: Currently active agent type.
        agent_status: Current agent activity.
        show_shortcuts: Whether to show keyboard shortcuts.
    """

    model_name: str = "unknown"
    tokens_used: int = 0
    tokens_limit: int = 100_000
    elapsed_seconds: float = 0.0
    current_phase: int = 1
    total_phases: int = 1
    completed_tasks: int = 0
    total_tasks: int = 0
    active_agent: str | None = None
    agent_status: str | None = None
    show_shortcuts: bool = True


# =============================================================================
# Status Bar
# =============================================================================


class StatusBar:
    """Persistent status bar for terminal bottom.

    Displays execution state including model, tokens, phase, tasks,
    and subagent activity in a compact format.
    """

    def __init__(
        self,
        state: StatusBarState | None = None,
        *,
        title: str = "ai-infra executor",
    ) -> None:
        """Initialize the status bar.

        Args:
            state: Initial state.
            title: Title to display.
        """
        self.state = state or StatusBarState()
        self.title = title

    def _is_narrow(self, width: int) -> bool:
        """Check if terminal is narrow (<80 cols)."""
        return width < 80

    def _render_line1(self, width: int) -> Text:
        """Render first line of status bar."""
        narrow = self._is_narrow(width)

        text = Text()

        # Title
        if not narrow:
            text.append(f" {self.title}  ", style="bold white")
        else:
            text.append(" ai-infra  ", style="bold white")

        # Model
        model = ModelIndicator(self.state.model_name)
        if narrow:
            text.append_text(model.render_compact())
        else:
            text.append_text(model.render())

        text.append("  ", style="")

        # Tokens
        tokens = TokenBudget(self.state.tokens_used, self.state.tokens_limit)
        if narrow:
            text.append_text(tokens.render_compact())
        else:
            text.append_text(tokens.render())

        text.append("  ", style="")

        # Elapsed time
        elapsed = format_duration(self.state.elapsed_seconds)
        text.append(elapsed, style="dim")

        return text

    def _render_line2(self, width: int) -> Text:
        """Render second line of status bar."""
        narrow = self._is_narrow(width)

        text = Text()
        text.append(" ", style="")

        # Phase progress
        text.append("Phase ", style="dim")
        text.append(
            f"{self.state.current_phase}/{self.state.total_phases}",
            style=f"bold {BRAND_ACCENT}",
        )
        text.append("  ", style="")

        # Task progress
        text.append("Tasks ", style="dim")
        text.append(
            f"{self.state.completed_tasks}/{self.state.total_tasks}",
            style=f"bold {BRAND_ACCENT}",
        )
        text.append("  ", style="")

        # Agent activity (if any)
        if self.state.active_agent and self.state.agent_status:
            activity = SubagentActivity(
                self.state.active_agent,
                self.state.agent_status,
            )
            if narrow:
                text.append_text(activity.render_compact())
            else:
                text.append_text(activity.render())
            text.append("  ", style="")

        # Keyboard hints (if space permits and enabled)
        if self.state.show_shortcuts and not narrow:
            hints = render_keyboard_hints(compact=narrow)
            text.append_text(hints)

        return text

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the status bar."""
        width = options.max_width
        narrow = self._is_narrow(width)

        # Build content table
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=0,
            expand=True,
        )
        table.add_column("content", ratio=1)

        table.add_row(self._render_line1(width))
        table.add_row(self._render_line2(width))

        # Use SQUARE box for narrow, ROUNDED for normal
        box_style = SQUARE if narrow else ROUNDED

        yield Panel(
            table,
            box=box_style,
            padding=(0, 1),
            expand=True,
        )


# =============================================================================
# Live Status Bar
# =============================================================================


class LiveStatusBar:
    """Live-updating status bar.

    Uses Rich Live for real-time updates without screen flicker.
    """

    def __init__(
        self,
        state: StatusBarState | None = None,
        console: Console | None = None,
        *,
        title: str = "ai-infra executor",
        refresh_rate: float = 4.0,
    ) -> None:
        """Initialize the live status bar.

        Args:
            state: Initial state.
            console: Console to use.
            title: Status bar title.
            refresh_rate: Refresh rate in Hz.
        """
        self.state = state or StatusBarState()
        self.console = console or get_console()
        self.title = title
        self.refresh_rate = refresh_rate
        self._status_bar = StatusBar(self.state, title=title)
        self._live: Live | None = None
        self._start_time: float | None = None

    def start(self) -> None:
        """Start the live display."""
        self._start_time = time.time()
        self._live = Live(
            self._status_bar,
            console=self.console,
            refresh_per_second=self.refresh_rate,
            vertical_overflow="visible",
        )
        self._live.start()

    def update(self, **kwargs: object) -> None:
        """Update state and refresh display.

        Args:
            **kwargs: State attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

        # Update elapsed time
        if self._start_time is not None:
            self.state.elapsed_seconds = time.time() - self._start_time

        if self._live is not None:
            self._live.update(self._status_bar)

    def set_agent_activity(self, agent_type: str, status: str) -> None:
        """Set current agent activity.

        Args:
            agent_type: Agent type (coder, tester, etc.).
            status: Activity description.
        """
        self.update(active_agent=agent_type, agent_status=status)

    def clear_agent_activity(self) -> None:
        """Clear agent activity indicator."""
        self.update(active_agent=None, agent_status=None)

    def increment_tasks(self) -> None:
        """Increment completed task count."""
        self.update(completed_tasks=self.state.completed_tasks + 1)

    def set_phase(self, current: int, total: int | None = None) -> None:
        """Set current phase.

        Args:
            current: Current phase number.
            total: Total phases (optional).
        """
        if total is not None:
            self.update(current_phase=current, total_phases=total)
        else:
            self.update(current_phase=current)

    def set_tokens(self, used: int, limit: int | None = None) -> None:
        """Update token counts.

        Args:
            used: Tokens used.
            limit: Token limit (optional).
        """
        if limit is not None:
            self.update(tokens_used=used, tokens_limit=limit)
        else:
            self.update(tokens_used=used)

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
    # Provider
    "ModelProvider",
    "get_model_provider",
    # Model indicator
    "ModelIndicator",
    # Token budget
    "TokenBudgetLevel",
    "TokenBudget",
    # Subagent activity
    "SubagentActivity",
    # Keyboard hints
    "KeyboardShortcut",
    "DEFAULT_SHORTCUTS",
    "render_keyboard_hints",
    # Status bar
    "StatusBarState",
    "StatusBar",
    "LiveStatusBar",
]
