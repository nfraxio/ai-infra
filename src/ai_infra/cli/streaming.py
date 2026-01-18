"""Real-time streaming output components for ai-infra CLI.

Phase 16.7.2 of EXECUTOR_6.md: Real-time Streaming Output.

This module provides:
- StreamingOutput: Displays streaming LLM output with syntax highlighting
- ToolCallPanel: Collapsible panels for tool call visualization
- CodeBlockHighlighter: Syntax highlighting for code blocks
- DiffViewer: Visualizes file modifications with unified diff format
- FileTreeDisplay: Real-time file tree updates

Example:
    ```python
    from ai_infra.cli.streaming import (
        StreamingOutput,
        ToolCallPanel,
        DiffViewer,
        FileTreeDisplay,
    )

    # Stream LLM output
    streamer = StreamingOutput(console=console)
    for chunk in llm_response:
        streamer.append(chunk)
    streamer.finalize()

    # Show tool call
    panel = ToolCallPanel(
        tool_name="write_file",
        args={"path": "src/auth.py"},
        output="File created successfully",
    )
    console.print(panel)
    ```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ai_infra.cli.console import (
    BOX_STYLES,
    BRAND_ACCENT,
    detect_terminal_capabilities,
    get_console,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Constants and Enums
# =============================================================================


class ToolCallState(str, Enum):
    """State of a tool call."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    COLLAPSED = "collapsed"


class FileChangeType(str, Enum):
    """Type of file change."""

    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


# Supported languages for syntax highlighting
SUPPORTED_LANGUAGES: dict[str, list[str]] = {
    "python": ["py", "pyi", "pyw"],
    "typescript": ["ts", "tsx"],
    "javascript": ["js", "jsx", "mjs", "cjs"],
    "sql": ["sql"],
    "bash": ["sh", "bash", "zsh"],
    "yaml": ["yaml", "yml"],
    "json": ["json"],
    "toml": ["toml"],
    "markdown": ["md", "markdown"],
    "rust": ["rs"],
    "go": ["go"],
    "dockerfile": ["dockerfile"],
    "makefile": ["makefile"],
}

# Reverse mapping: extension to language
EXTENSION_TO_LANGUAGE: dict[str, str] = {}
for lang, exts in SUPPORTED_LANGUAGES.items():
    for ext in exts:
        EXTENSION_TO_LANGUAGE[ext] = lang


# Default keyboard shortcuts for output control
OUTPUT_KEY_BINDINGS: dict[str, str] = {
    " ": "toggle",  # Space - expand/collapse current
    "a": "expand_all",
    "z": "collapse_all",
    "j": "down",
    "k": "up",
    "up": "up",
    "down": "down",
}


# Code block detection regex
CODE_BLOCK_PATTERN = re.compile(
    r"```(\w+)?\n(.*?)```",
    re.DOTALL,
)


# Line threshold for showing line numbers
LINE_NUMBER_THRESHOLD = 10


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ToolCall:
    """Represents a tool call with its state and output.

    Attributes:
        tool_name: Name of the tool being called.
        args: Arguments passed to the tool.
        output: Tool output or result.
        state: Current state of the tool call.
        duration: Execution duration in seconds.
        summary: Brief description of what the tool did.
        line_count: Number of lines in the output (for files).
    """

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    output: str | None = None
    state: ToolCallState = ToolCallState.PENDING
    duration: float | None = None
    summary: str | None = None
    line_count: int | None = None

    def get_display_args(self) -> str:
        """Get formatted argument string for display."""
        if not self.args:
            return ""
        if "path" in self.args:
            return f'("{self.args["path"]}")'
        if "command" in self.args:
            return f'("{self.args["command"]}")'
        if len(self.args) == 1:
            key, value = next(iter(self.args.items()))
            if isinstance(value, str) and len(value) < 50:
                return f'("{value}")'
        return f"({len(self.args)} args)"


@dataclass
class DiffHunk:
    """Represents a single hunk in a unified diff.

    Attributes:
        old_start: Starting line in old file.
        old_count: Number of lines in old file.
        new_start: Starting line in new file.
        new_count: Number of lines in new file.
        lines: Lines in the hunk with their prefixes.
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[tuple[str, str]] = field(default_factory=list)  # (prefix, line)

    @property
    def header(self) -> str:
        """Get the hunk header."""
        return f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@"


@dataclass
class FileChange:
    """Represents a file change for the file tree.

    Attributes:
        path: File path relative to project root.
        change_type: Type of change.
        lines_added: Number of lines added.
        lines_removed: Number of lines removed.
    """

    path: str
    change_type: FileChangeType | str = FileChangeType.MODIFIED
    lines_added: int = 0
    lines_removed: int = 0

    def __post_init__(self) -> None:
        """Convert string change type to enum."""
        if isinstance(self.change_type, str):
            self.change_type = FileChangeType(self.change_type)


@dataclass
class StreamChunk:
    """A chunk of streaming output.

    Attributes:
        content: The text content.
        is_code: Whether this is inside a code block.
        language: Language for code blocks.
    """

    content: str
    is_code: bool = False
    language: str | None = None


# =============================================================================
# Syntax Highlighting
# =============================================================================


def detect_language(filepath: str | None = None, content: str | None = None) -> str:
    """Detect programming language from file path or content.

    Args:
        filepath: Optional file path to detect language from.
        content: Optional content to analyze.

    Returns:
        Detected language name for syntax highlighting.
    """
    if filepath:
        # Try extension
        ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
        if ext in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext]

        # Try filename patterns
        filename = filepath.rsplit("/", 1)[-1].lower()
        if filename == "dockerfile":
            return "dockerfile"
        if filename == "makefile":
            return "makefile"
        if filename.endswith("rc"):
            return "bash"

    if content:
        # Simple heuristics
        if content.strip().startswith("#!/"):
            first_line = content.split("\n")[0]
            if "python" in first_line:
                return "python"
            if "bash" in first_line or "sh" in first_line:
                return "bash"

        # Check for common patterns
        if "def " in content and ":" in content:
            return "python"
        if "function " in content or "const " in content or "let " in content:
            return "javascript"

    return "text"


def create_syntax(
    code: str,
    language: str = "text",
    *,
    line_numbers: bool | None = None,
    start_line: int = 1,
    highlight_lines: set[int] | None = None,
    word_wrap: bool = False,
    theme: str = "monokai",
) -> Syntax:
    """Create a Rich Syntax object for code highlighting.

    Args:
        code: Source code to highlight.
        language: Programming language.
        line_numbers: Show line numbers (auto if None).
        start_line: Starting line number.
        highlight_lines: Lines to highlight.
        word_wrap: Whether to wrap long lines.
        theme: Color theme for highlighting.

    Returns:
        Configured Syntax object.
    """
    # Auto-detect line numbers
    if line_numbers is None:
        line_count = code.count("\n") + 1
        line_numbers = line_count >= LINE_NUMBER_THRESHOLD

    return Syntax(
        code,
        language,
        line_numbers=line_numbers,
        start_line=start_line,
        highlight_lines=highlight_lines,
        word_wrap=word_wrap,
        theme=theme,
    )


def extract_code_blocks(text: str) -> list[tuple[str | None, str, int, int]]:
    """Extract code blocks from markdown-style text.

    Args:
        text: Text containing code blocks.

    Returns:
        List of (language, code, start_pos, end_pos) tuples.
    """
    blocks = []
    for match in CODE_BLOCK_PATTERN.finditer(text):
        language = match.group(1)  # May be None
        code = match.group(2).strip()
        blocks.append((language, code, match.start(), match.end()))
    return blocks


# =============================================================================
# Streaming Output
# =============================================================================


class StreamingOutput:
    """Handles streaming LLM output with syntax highlighting.

    Buffers streaming text and renders with proper formatting,
    including syntax-highlighted code blocks.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        live: Live | None = None,
        show_thinking: bool = True,
    ) -> None:
        """Initialize the streaming output handler.

        Args:
            console: Rich console instance.
            live: Optional Live display context.
            show_thinking: Whether to show "thinking" animation.
        """
        self.console = console or get_console()
        self.live = live
        self.show_thinking = show_thinking

        self._buffer = ""
        self._chunks: list[StreamChunk] = []
        self._in_code_block = False
        self._current_language: str | None = None
        self._code_buffer = ""
        self._finalized = False

    @property
    def buffer(self) -> str:
        """Current buffer content."""
        return self._buffer

    @property
    def is_finalized(self) -> bool:
        """Whether the stream has been finalized."""
        return self._finalized

    def append(self, text: str) -> None:
        """Append text to the stream.

        Args:
            text: Text chunk to append.
        """
        if self._finalized:
            return

        self._buffer += text
        self._process_buffer()

    def _process_buffer(self) -> None:
        """Process buffer for code blocks and formatting."""
        # Look for code block markers
        while True:
            if not self._in_code_block:
                # Look for opening ```
                match = re.search(r"```(\w+)?\n?", self._buffer)
                if match:
                    # Emit text before code block
                    before = self._buffer[: match.start()]
                    if before:
                        self._chunks.append(StreamChunk(content=before))

                    self._in_code_block = True
                    self._current_language = match.group(1)
                    self._code_buffer = ""
                    self._buffer = self._buffer[match.end() :]
                else:
                    break
            else:
                # Look for closing ```
                close_match = re.search(r"\n?```", self._buffer)
                if close_match:
                    # Emit code block
                    self._code_buffer += self._buffer[: close_match.start()]
                    self._chunks.append(
                        StreamChunk(
                            content=self._code_buffer,
                            is_code=True,
                            language=self._current_language,
                        )
                    )

                    self._in_code_block = False
                    self._current_language = None
                    self._code_buffer = ""
                    self._buffer = self._buffer[close_match.end() :]
                else:
                    # Accumulate in code buffer
                    self._code_buffer += self._buffer
                    self._buffer = ""
                    break

    def finalize(self) -> None:
        """Finalize the stream and emit remaining content."""
        if self._finalized:
            return

        # Handle unclosed code block
        if self._in_code_block and self._code_buffer:
            self._chunks.append(
                StreamChunk(
                    content=self._code_buffer,
                    is_code=True,
                    language=self._current_language,
                )
            )

        # Handle remaining buffer
        if self._buffer:
            self._chunks.append(StreamChunk(content=self._buffer))

        self._finalized = True

    def render(self) -> Group:
        """Render the streamed content.

        Returns:
            Group of Rich renderables.
        """
        renderables: list[Any] = []

        for chunk in self._chunks:
            if chunk.is_code:
                lang = chunk.language or "text"
                syntax = create_syntax(chunk.content, lang)
                renderables.append(syntax)
            else:
                renderables.append(Text(chunk.content))

        # Add current buffer if not finalized
        if not self._finalized and self._buffer:
            renderables.append(Text(self._buffer, style="dim"))

        return Group(*renderables)

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the streaming output."""
        yield self.render()


# =============================================================================
# Tool Call Panel
# =============================================================================


class ToolCallPanel:
    """Rich renderable for displaying tool calls with collapsible output.

    Shows tool name, arguments, and expandable output with status.
    """

    def __init__(
        self,
        tool_call: ToolCall | None = None,
        *,
        tool_name: str = "",
        args: dict[str, Any] | None = None,
        output: str | None = None,
        state: ToolCallState = ToolCallState.COMPLETE,
        summary: str | None = None,
        expanded: bool = True,
        duration: float | None = None,
        line_count: int | None = None,
    ) -> None:
        """Initialize the tool call panel.

        Args:
            tool_call: ToolCall object (alternative to individual args).
            tool_name: Name of the tool.
            args: Tool arguments.
            output: Tool output.
            state: Current tool state.
            summary: Summary of what the tool did.
            expanded: Whether the output is expanded.
            duration: Execution duration in seconds.
            line_count: Number of lines affected.
        """
        if tool_call:
            self.tool_name = tool_call.tool_name
            self.args = tool_call.args
            self.output = tool_call.output
            self.state = tool_call.state
            self.summary = tool_call.summary
            self.duration = tool_call.duration
            self.line_count = tool_call.line_count
        else:
            self.tool_name = tool_name
            self.args = args or {}
            self.output = output
            self.state = state
            self.summary = summary
            self.duration = duration
            self.line_count = line_count

        self.expanded = expanded

    def _get_state_icon(self) -> tuple[str, str]:
        """Get state icon and style."""
        caps = detect_terminal_capabilities()
        use_unicode = caps.unicode_support

        icons: dict[ToolCallState, tuple[str, str, str]] = {
            ToolCallState.PENDING: ("...", "○", "dim"),
            ToolCallState.RUNNING: ("~>", "◐", "yellow"),
            ToolCallState.COMPLETE: ("ok", "●", "green"),
            ToolCallState.FAILED: ("!!", "✗", "red"),
            ToolCallState.COLLAPSED: (">", "▶", "dim"),
        }

        ascii_icon, unicode_icon, style = icons.get(self.state, icons[ToolCallState.PENDING])
        icon = unicode_icon if use_unicode else ascii_icon
        return icon, style

    def _get_expand_icon(self) -> str:
        """Get expand/collapse icon."""
        caps = detect_terminal_capabilities()
        if caps.unicode_support:
            return "▼" if self.expanded else "▶"
        return "v" if self.expanded else ">"

    def _format_args_display(self) -> str:
        """Format arguments for display."""
        if not self.args:
            return ""
        if "path" in self.args:
            return f'("{self.args["path"]}")'
        if "command" in self.args:
            cmd = self.args["command"]
            if len(cmd) > 40:
                cmd = cmd[:37] + "..."
            return f'("{cmd}")'
        return f"({len(self.args)} args)"

    def _render_header(self) -> Text:
        """Render the tool call header."""
        expand_icon = self._get_expand_icon()
        state_icon, state_style = self._get_state_icon()

        text = Text()
        text.append(f"[{expand_icon}] ", style="bold")
        text.append(self.tool_name, style=f"bold {BRAND_ACCENT}")
        text.append(self._format_args_display(), style="dim")

        # Right side info
        info_parts = []
        if self.line_count is not None:
            info_parts.append(f"[{self.line_count} lines]")
        if self.duration is not None:
            info_parts.append(f"{self.duration:.1f}s")
        if info_parts:
            text.append("  ")
            text.append(" ".join(info_parts), style="dim")

        return text

    def _render_output(self) -> list[Any]:
        """Render the tool output."""
        if not self.expanded or not self.output:
            return []

        output_parts: list[Any] = []

        # Connector line
        caps = detect_terminal_capabilities()
        pipe = "│" if caps.unicode_support else "|"

        output_parts.append(Text(f"    {pipe}", style="dim"))

        # Summary if available
        if self.summary:
            for line in self.summary.split("\n"):
                output_parts.append(Text(f"    {pipe} {line}", style="dim"))

        # Full output
        if self.output:
            # Detect if output is code
            lang = None
            if self.tool_name in ("write_file", "read_file", "create_file"):
                path = self.args.get("path", "")
                lang = detect_language(path)

            if lang and lang != "text":
                # Syntax highlight the output
                syntax = create_syntax(self.output, lang, line_numbers=True)
                output_parts.append(Text(f"    {pipe}", style="dim"))
                output_parts.append(syntax)
            else:
                for line in self.output.split("\n")[:20]:  # Limit lines
                    output_parts.append(Text(f"    {pipe} {line}", style="dim"))
                if self.output.count("\n") > 20:
                    output_parts.append(
                        Text(
                            f"    {pipe} ... ({self.output.count(chr(10)) - 20} more lines)",
                            style="dim",
                        )
                    )

        # Status line
        state_icon, state_style = self._get_state_icon()
        status_text = {
            ToolCallState.COMPLETE: "Complete",
            ToolCallState.FAILED: "Failed",
            ToolCallState.RUNNING: "Running...",
            ToolCallState.PENDING: "Pending",
            ToolCallState.COLLAPSED: "Collapsed",
        }.get(self.state, "Unknown")

        output_parts.append(Text(f"    [{state_icon}] {status_text}", style=state_style))

        return output_parts

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the tool call panel."""
        yield self._render_header()
        for part in self._render_output():
            yield part


# =============================================================================
# Tool Call Manager
# =============================================================================


class ToolCallManager:
    """Manages a collection of tool calls with navigation and toggling.

    Provides keyboard navigation and expand/collapse functionality.
    """

    def __init__(
        self,
        tool_calls: Sequence[ToolCall] | None = None,
        *,
        console: Console | None = None,
    ) -> None:
        """Initialize the tool call manager.

        Args:
            tool_calls: Initial list of tool calls.
            console: Rich console instance.
        """
        self.tool_calls = list(tool_calls) if tool_calls else []
        self.console = console or get_console()
        self._current_index = 0
        self._expanded: set[int] = set(range(len(self.tool_calls)))

    @property
    def current_index(self) -> int:
        """Current selected tool call index."""
        return self._current_index

    def add_tool_call(self, tool_call: ToolCall) -> int:
        """Add a new tool call.

        Args:
            tool_call: Tool call to add.

        Returns:
            Index of the added tool call.
        """
        index = len(self.tool_calls)
        self.tool_calls.append(tool_call)
        self._expanded.add(index)
        return index

    def toggle_current(self) -> None:
        """Toggle expand/collapse of current tool call."""
        if self._current_index in self._expanded:
            self._expanded.discard(self._current_index)
        else:
            self._expanded.add(self._current_index)

    def expand_all(self) -> None:
        """Expand all tool calls."""
        self._expanded = set(range(len(self.tool_calls)))

    def collapse_all(self) -> None:
        """Collapse all tool calls."""
        self._expanded.clear()

    def navigate(self, direction: int) -> None:
        """Navigate to next/previous tool call.

        Args:
            direction: 1 for down, -1 for up.
        """
        if not self.tool_calls:
            return
        self._current_index = max(0, min(len(self.tool_calls) - 1, self._current_index + direction))

    def handle_key(self, key: str) -> bool:
        """Handle a keypress.

        Args:
            key: Key pressed.

        Returns:
            True if key was handled.
        """
        action = OUTPUT_KEY_BINDINGS.get(key)
        if action == "toggle":
            self.toggle_current()
            return True
        elif action == "expand_all":
            self.expand_all()
            return True
        elif action == "collapse_all":
            self.collapse_all()
            return True
        elif action == "up":
            self.navigate(-1)
            return True
        elif action == "down":
            self.navigate(1)
            return True
        return False

    def render(self) -> Group:
        """Render all tool calls.

        Returns:
            Group of ToolCallPanel renderables.
        """
        panels: list[RenderableType] = []
        for i, tc in enumerate(self.tool_calls):
            panel = ToolCallPanel(
                tool_call=tc,
                expanded=i in self._expanded,
            )
            # Highlight current
            if i == self._current_index:
                panels.append(Panel(panel, border_style=BRAND_ACCENT, box=BOX_STYLES["default"]))
            else:
                panels.append(panel)
        return Group(*panels)

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the tool call manager."""
        yield self.render()


# =============================================================================
# Diff Viewer
# =============================================================================


def parse_unified_diff(diff_text: str) -> list[DiffHunk]:
    """Parse unified diff format into hunks.

    Args:
        diff_text: Unified diff text.

    Returns:
        List of DiffHunk objects.
    """
    hunks: list[DiffHunk] = []
    current_hunk: DiffHunk | None = None

    for line in diff_text.split("\n"):
        # Hunk header: @@ -start,count +start,count @@
        if line.startswith("@@"):
            # Save previous hunk
            if current_hunk:
                hunks.append(current_hunk)

            # Parse header
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                current_hunk = DiffHunk(
                    old_start=int(match.group(1)),
                    old_count=int(match.group(2) or 1),
                    new_start=int(match.group(3)),
                    new_count=int(match.group(4) or 1),
                )
        elif current_hunk is not None:
            # Diff lines
            if line.startswith("+"):
                current_hunk.lines.append(("+", line[1:]))
            elif line.startswith("-"):
                current_hunk.lines.append(("-", line[1:]))
            elif line.startswith(" "):
                current_hunk.lines.append((" ", line[1:]))
            elif line:  # Context without prefix
                current_hunk.lines.append((" ", line))

    # Save last hunk
    if current_hunk:
        hunks.append(current_hunk)

    return hunks


class DiffViewer:
    """Rich renderable for displaying unified diffs.

    Shows file modifications with syntax highlighting and line numbers.
    """

    def __init__(
        self,
        filepath: str,
        diff_text: str | None = None,
        hunks: list[DiffHunk] | None = None,
        *,
        language: str | None = None,
        context_lines: int = 3,
    ) -> None:
        """Initialize the diff viewer.

        Args:
            filepath: Path to the modified file.
            diff_text: Raw unified diff text.
            hunks: Pre-parsed diff hunks.
            language: Override language detection.
            context_lines: Number of context lines to show.
        """
        self.filepath = filepath
        self.language = language or detect_language(filepath)
        self.context_lines = context_lines

        if hunks:
            self.hunks = hunks
        elif diff_text:
            self.hunks = parse_unified_diff(diff_text)
        else:
            self.hunks = []

    def _render_hunk(self, hunk: DiffHunk) -> list[Text]:
        """Render a single diff hunk."""
        lines = []

        # Hunk header
        lines.append(Text(hunk.header, style=f"{BRAND_ACCENT} bold"))

        # Lines with coloring
        for prefix, content in hunk.lines:
            if prefix == "+":
                lines.append(Text(f"+{content}", style="green"))
            elif prefix == "-":
                lines.append(Text(f"-{content}", style="red"))
            else:
                lines.append(Text(f" {content}", style="dim"))

        return lines

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the diff viewer."""
        # File header
        header = Text()
        header.append(self.filepath, style="bold path")
        yield header
        yield Text("─" * min(60, options.max_width), style="dim")

        # Render each hunk
        for hunk in self.hunks:
            for line in self._render_hunk(hunk):
                yield line
            yield Text()  # Spacer between hunks


# =============================================================================
# File Tree Display
# =============================================================================


class FileTreeDisplay:
    """Rich renderable for displaying file tree with changes.

    Shows project structure with indicators for new, modified, and deleted files.
    """

    def __init__(
        self,
        changes: Sequence[FileChange] | None = None,
        *,
        root_label: str = "Project Structure:",
        show_unchanged: bool = False,
    ) -> None:
        """Initialize the file tree display.

        Args:
            changes: List of file changes.
            root_label: Label for the tree root.
            show_unchanged: Whether to show unchanged files.
        """
        self.changes = list(changes) if changes else []
        self.root_label = root_label
        self.show_unchanged = show_unchanged

    def _get_change_icon(self, change_type: FileChangeType) -> tuple[str, str]:
        """Get icon and style for change type."""
        caps = detect_terminal_capabilities()

        icons: dict[FileChangeType, tuple[str, str, str]] = {
            FileChangeType.NEW: ("+", "+", "green"),
            FileChangeType.MODIFIED: ("~", "~", "yellow"),
            FileChangeType.DELETED: ("-", "-", "red"),
            FileChangeType.RENAMED: (">", "→", "blue"),
        }

        ascii_icon, unicode_icon, style = icons.get(change_type, icons[FileChangeType.MODIFIED])
        icon = unicode_icon if caps.unicode_support else ascii_icon
        return icon, style

    def _build_tree(self) -> Tree:
        """Build the file tree structure."""
        tree = Tree(self.root_label, guide_style="dim")

        # Group files by directory
        dirs: dict[str, list[FileChange]] = {}
        for change in self.changes:
            parts = change.path.split("/")
            if len(parts) > 1:
                dir_path = "/".join(parts[:-1])
            else:
                dir_path = ""
            if dir_path not in dirs:
                dirs[dir_path] = []
            dirs[dir_path].append(change)

        # Sort directories
        for dir_path in sorted(dirs.keys()):
            if dir_path:
                # Add directory branch
                branch = tree.add(f"{dir_path}/", guide_style="dim")
            else:
                branch = tree

            # Add files
            for change in sorted(dirs[dir_path], key=lambda c: c.path):
                filename = change.path.rsplit("/", 1)[-1]
                change_type = change.change_type
                if isinstance(change_type, str):
                    change_type = FileChangeType(change_type)
                icon, style = self._get_change_icon(change_type)

                # Format label
                label = Text()
                label.append(f"{icon} ", style=style)
                label.append(filename, style=style)

                # Add change info
                info_parts = []
                if change_type == FileChangeType.NEW:
                    info_parts.append("(new)")
                elif change_type == FileChangeType.MODIFIED:
                    info_parts.append("(modified)")
                elif change_type == FileChangeType.DELETED:
                    info_parts.append("(deleted)")
                elif change_type == FileChangeType.RENAMED:
                    info_parts.append("(renamed)")

                if change.lines_added > 0 or change.lines_removed > 0:
                    diff_info = []
                    if change.lines_added > 0:
                        diff_info.append(f"+{change.lines_added}")
                    if change.lines_removed > 0:
                        diff_info.append(f"-{change.lines_removed}")
                    info_parts.append("/".join(diff_info))

                if info_parts:
                    label.append("  ")
                    label.append(" ".join(info_parts), style="dim")

                branch.add(label)

        return tree

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Render the file tree."""
        yield self._build_tree()


# =============================================================================
# Output Control Help
# =============================================================================


def render_output_help() -> Panel:
    """Render help panel for output control keys.

    Returns:
        Panel with output control help.
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
        ("Space", "Toggle", "Expand/collapse current tool call"),
        ("a", "Expand All", "Expand all tool calls"),
        ("z", "Collapse All", "Collapse all tool calls"),
        ("j / ↓", "Down", "Navigate to next tool call"),
        ("k / ↑", "Up", "Navigate to previous tool call"),
    ]

    for key, action, desc in commands:
        table.add_row(key, action, desc)

    return Panel(
        table,
        title="[bold]OUTPUT CONTROLS[/]",
        title_align="left",
        border_style="green",
        box=BOX_STYLES.get("default", BOX_STYLES["default"]),
        padding=(1, 2),
    )


# =============================================================================
# Live Streaming Display
# =============================================================================


class LiveStreamingDisplay:
    """Manages live updating display for streaming output.

    Combines StreamingOutput, ToolCallManager, and FileTreeDisplay
    in a single live-updating view.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_file_tree: bool = True,
        auto_scroll: bool = True,
    ) -> None:
        """Initialize the live streaming display.

        Args:
            console: Rich console instance.
            show_file_tree: Whether to show file tree updates.
            auto_scroll: Whether to auto-scroll to latest output.
        """
        self.console = console or get_console()
        self.show_file_tree = show_file_tree
        self.auto_scroll = auto_scroll

        self.streamer = StreamingOutput(console=self.console)
        self.tool_manager = ToolCallManager(console=self.console)
        self.file_changes: list[FileChange] = []

        self._live: Live | None = None

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def append_text(self, text: str) -> None:
        """Append streaming text.

        Args:
            text: Text to append.
        """
        self.streamer.append(text)
        self._update()

    def add_tool_call(self, tool_call: ToolCall) -> int:
        """Add a tool call.

        Args:
            tool_call: Tool call to add.

        Returns:
            Index of the added tool call.
        """
        index = self.tool_manager.add_tool_call(tool_call)
        self._update()
        return index

    def add_file_change(self, change: FileChange) -> None:
        """Add a file change.

        Args:
            change: File change to add.
        """
        self.file_changes.append(change)
        self._update()

    def _update(self) -> None:
        """Update the live display."""
        if self._live:
            self._live.update(self.render())

    def render(self) -> Group:
        """Render the complete display.

        Returns:
            Group of all renderables.
        """
        parts: list[Any] = []

        # Streaming output
        parts.append(self.streamer.render())

        # Tool calls
        if self.tool_manager.tool_calls:
            parts.append(Text())  # Spacer
            parts.append(self.tool_manager.render())

        # File tree
        if self.show_file_tree and self.file_changes:
            parts.append(Text())  # Spacer
            tree = FileTreeDisplay(self.file_changes)
            parts.append(tree._build_tree())

        return Group(*parts)

    def finalize(self) -> None:
        """Finalize the display."""
        self.streamer.finalize()
        self._update()

    def __enter__(self) -> LiveStreamingDisplay:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.finalize()
        self.stop()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ToolCallState",
    "FileChangeType",
    # Data classes
    "ToolCall",
    "DiffHunk",
    "FileChange",
    "StreamChunk",
    # Syntax highlighting
    "SUPPORTED_LANGUAGES",
    "EXTENSION_TO_LANGUAGE",
    "detect_language",
    "create_syntax",
    "extract_code_blocks",
    # Streaming
    "StreamingOutput",
    # Tool calls
    "ToolCallPanel",
    "ToolCallManager",
    # Diff viewing
    "parse_unified_diff",
    "DiffViewer",
    # File tree
    "FileTreeDisplay",
    # Help
    "render_output_help",
    # Live display
    "LiveStreamingDisplay",
    # Constants
    "OUTPUT_KEY_BINDINGS",
    "LINE_NUMBER_THRESHOLD",
]
