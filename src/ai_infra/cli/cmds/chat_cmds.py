"""
CLI commands for interactive chat with LLMs.

Usage:
    ai-infra chat                    # Interactive REPL (auto-detects provider)
    ai-infra chat --provider openai  # Use specific provider
    ai-infra chat --model gpt-4o     # Use specific model
    ai-infra chat -m "Hello"         # One-shot message

Sessions & Persistence:
    ai-infra chat --session my-chat  # Resume or create named session
    ai-infra chat --new              # Start fresh (don't resume last session)
    ai-infra chat sessions           # List saved sessions
    ai-infra chat session-delete <n> # Delete a session

    Within chat REPL:
    /sessions                        # List saved sessions
    /switch [name]                   # Switch session (interactive picker if no name)
    /save [name]                     # Save current session
    /load <name>                     # Load a saved session
    /new                             # Start new session (clears memory)
    /delete <name>                   # Delete a saved session
    /deleteall                       # Delete all sessions (with confirmation)

    Session names are auto-generated based on conversation topic when you exit.

MCP Tools (connect to external tools):
    ai-infra chat --mcp http://localhost:8000/mcp    # Single MCP server
    ai-infra chat --mcp server1.json --mcp s2.json   # Multiple servers

    Within chat REPL with MCP:
    /tools                           # List available tools from MCP servers
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Heading, Markdown
from rich.panel import Panel
from rich.spinner import Spinner as RichSpinner
from rich.text import Text

from ai_infra.cli.console import BRAND_ACCENT

# Rich console for formatted output
_console = Console()


# Custom Heading class that left-aligns instead of centering
class LeftAlignedHeading(Heading):
    """Heading that renders left-aligned instead of centered."""

    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"  # Override the default "center"
        if self.tag == "h1":
            yield Panel(
                text,
                box=box.HEAVY,
                style="markdown.h1.border",
            )
        else:
            if self.tag == "h2":
                yield Text("")
            yield text


# Patch Markdown to use our left-aligned heading
Markdown.elements["heading_open"] = LeftAlignedHeading

# Use Typer group for subcommands: ai-infra chat, ai-infra chat sessions, etc.
app = typer.Typer(
    help="Interactive chat with LLMs",
    invoke_without_command=True,  # Allow `ai-infra chat` to run the default command
)


# =============================================================================
# Thinking Spinner
# =============================================================================


class ThinkingSpinner:
    """Animated spinner to show while waiting for LLM response using Rich."""

    def __init__(self, message: str = "thinking"):
        self._message = message
        self._live: Live | None = None

    def start(self):
        """Start the spinner animation."""
        if self._live is not None:
            return
        spinner = RichSpinner("dots", text=f"[dim]{self._message}[/dim]", style="dim")
        self._live = Live(spinner, console=_console, refresh_per_second=10, transient=True)
        self._live.start()

    def stop(self):
        """Stop the spinner and clear the line."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# =============================================================================
# Response Rendering
# =============================================================================


def _render_response(content: str, model: str | None = None) -> None:
    """Render AI response with markdown formatting and syntax highlighting.

    Args:
        content: The response text (may contain markdown)
        model: Optional model name to show at the end
    """
    # Always render with Rich markdown for consistent formatting
    # (handles code blocks, bold, headers, lists, etc.)
    _console.print()
    md = Markdown(content, justify="left")
    _console.print(md)

    # Show model indicator at end with spacing
    if model:
        _console.print()
        _console.print(f"  [dim]↳ {model}[/dim]")


def _create_streaming_display(content: str, model: str | None = None):
    """Create a Rich renderable for streaming display.

    Returns a Group containing the markdown and optional model indicator.
    """
    md = Markdown(content, justify="left")
    if model and content:
        # Add empty line before model indicator for spacing
        return Group(md, Text(""), Text(f"  ↳ {model}", style="dim"))
    return md


async def _stream_with_markdown(content: str, model: str | None = None) -> None:
    """Stream content with live markdown rendering (simulated for pre-fetched content).

    Used for MCP responses where content is already fetched but we want streaming UX.
    Streams word-by-word with live markdown re-rendering.

    Args:
        content: The complete content to stream
        model: Optional model name to show at end
    """

    # Split by words while preserving whitespace
    tokens = re.findall(r"\S+|\s+", content)

    accumulated = ""
    with Live(
        _create_streaming_display("", model),
        console=_console,
        refresh_per_second=15,
        transient=False,
    ) as live:
        for token in tokens:
            accumulated += token
            live.update(_create_streaming_display(accumulated, model))
            # Small delay for words only (not whitespace)
            if token.strip():
                await asyncio.sleep(0.015)


def _render_tool_call(tool_name: str, result: str) -> None:
    """Render a tool call result with nice formatting.

    Args:
        tool_name: Name of the tool that was called
        result: The result from the tool
    """
    # Truncate long results for preview
    preview = result[:100] + "..." if len(result) > 100 else result
    # Remove newlines for single-line preview
    preview = preview.replace("\\n", " ").replace("\n", " ")

    _console.print(
        f"  [green]✓[/green] [{BRAND_ACCENT}]{tool_name}[/{BRAND_ACCENT}] [dim]→ {preview}[/dim]"
    )


# =============================================================================
# MCP Integration for Chat
# =============================================================================


class ChatMCPManager:
    """Manages MCP connections for chat sessions.

    Handles:
    - Discovery of MCP servers
    - Tool listing
    - Tool execution
    - Cleanup on exit
    """

    def __init__(self):
        self._client: Any = None
        self._tools: list[Any] = []
        self._discovered = False

    async def connect(self, mcp_configs: list[str]) -> bool:
        """Connect to MCP servers.

        Args:
            mcp_configs: List of MCP server URLs or JSON config file paths

        Returns:
            True if at least one server connected successfully
        """
        from ai_infra.mcp import MCPClient, McpServerConfig

        configs: list[McpServerConfig] = []

        for cfg in mcp_configs:
            if cfg.endswith(".json"):
                # Load from JSON file
                try:
                    path = Path(cfg).expanduser()
                    with open(path) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                configs.append(McpServerConfig.model_validate(item))
                        else:
                            configs.append(McpServerConfig.model_validate(data))
                except Exception as e:
                    typer.secho(f"[!] Failed to load MCP config {cfg}: {e}", fg=typer.colors.YELLOW)
            elif cfg.startswith("http://") or cfg.startswith("https://"):
                # HTTP URL
                configs.append(McpServerConfig(transport="streamable_http", url=cfg))
            else:
                typer.secho(
                    f"[!] Unknown MCP config format: {cfg} (expected URL or .json file)",
                    fg=typer.colors.YELLOW,
                )

        if not configs:
            return False

        try:
            self._client = MCPClient(configs, discover_timeout=30.0, tool_timeout=60.0)
            await self._client.discover()
            self._tools = await self._client.list_tools()
            self._discovered = True
            return True
        except Exception as e:
            typer.secho(f"[!] MCP discovery failed: {e}", fg=typer.colors.YELLOW)
            return False

    def get_tools(self) -> list[Any]:
        """Get discovered MCP tools."""
        return self._tools

    def get_tool_names(self) -> list[str]:
        """Get names of discovered tools."""
        return [getattr(t, "name", str(t)) for t in self._tools]

    async def close(self):
        """Close MCP connections."""
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None


# Global MCP manager for the chat session
_mcp_manager: ChatMCPManager | None = None


def get_mcp_manager() -> ChatMCPManager | None:
    """Get the global MCP manager if initialized."""
    return _mcp_manager


# =============================================================================
# Chat Session Storage
# =============================================================================


class ChatStorage:
    """Local file-based storage for chat sessions.

    Stores chat sessions as JSON files in ~/.ai-infra/chat_sessions/
    Each session contains:
    - messages: The conversation history
    - metadata: Provider, model, system prompt, etc.
    - timestamps: Created and updated times
    """

    def __init__(self, base_dir: Path | None = None):
        """Initialize chat storage.

        Args:
            base_dir: Base directory for storage. Defaults to ~/.ai-infra/chat_sessions/
        """
        if base_dir is None:
            base_dir = Path.home() / ".ai-infra" / "chat_sessions"
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get path to session file."""
        # Sanitize session_id to be filename-safe
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self._base_dir / f"{safe_id}.json"

    def save(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a chat session.

        Args:
            session_id: Unique identifier for the session
            messages: List of message dicts with 'role' and 'content'
            metadata: Optional metadata (provider, model, system, etc.)
        """
        path = self._session_path(session_id)
        existing = self._load_raw(session_id)

        data = {
            "session_id": session_id,
            "messages": messages,
            "metadata": metadata or {},
            "created_at": existing.get("created_at") if existing else datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_raw(self, session_id: str) -> dict[str, Any] | None:
        """Load raw session data."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with open(path) as f:
            result = json.load(f)
            return dict(result) if isinstance(result, dict) else None

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a chat session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            Session data dict or None if not found
        """
        return self._load_raw(session_id)

    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self._session_path(session_id).exists()

    def delete(self, session_id: str) -> bool:
        """Delete a chat session.

        Returns:
            True if deleted, False if not found
        """
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions with metadata.

        Returns:
            List of session info dicts sorted by updated_at (newest first)
        """
        sessions = []
        for path in self._base_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    sessions.append(
                        {
                            "session_id": data.get("session_id", path.stem),
                            "message_count": len(data.get("messages", [])),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                            "provider": data.get("metadata", {}).get("provider"),
                            "model": data.get("metadata", {}).get("model"),
                        }
                    )
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_last_session_id(self) -> str | None:
        """Get the ID of the most recently updated session."""
        sessions = self.list_sessions()
        return sessions[0]["session_id"] if sessions else None

    def save_last_session_id(self, session_id: str) -> None:
        """Save the last used session ID for auto-resume."""
        last_file = self._base_dir / ".last_session"
        with open(last_file, "w") as f:
            f.write(session_id)

    def get_auto_resume_session_id(self) -> str | None:
        """Get the session ID to auto-resume."""
        last_file = self._base_dir / ".last_session"
        if last_file.exists():
            session_id = last_file.read_text().strip()
            if self.exists(session_id):
                return session_id
        return None

    def delete_all(self, exclude_session_id: str | None = None) -> int:
        """Delete all sessions except the current one.

        Args:
            exclude_session_id: Session to keep (usually current session)

        Returns:
            Number of sessions deleted
        """
        deleted_count = 0
        for path in self._base_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    session_id = data.get("session_id", path.stem)
                    if session_id != exclude_session_id:
                        path.unlink()
                        deleted_count += 1
            except (json.JSONDecodeError, OSError):
                # Delete malformed sessions too
                path.unlink()
                deleted_count += 1
        return deleted_count

    def rename(self, old_id: str, new_id: str) -> bool:
        """Rename a session.

        Args:
            old_id: Current session ID
            new_id: New session ID

        Returns:
            True if renamed successfully
        """
        if not self.exists(old_id) or self.exists(new_id):
            return False

        old_path = self._session_path(old_id)
        new_path = self._session_path(new_id)

        try:
            # Load, update session_id, and save to new path
            with open(old_path) as f:
                data = json.load(f)
            data["session_id"] = new_id
            with open(new_path, "w") as f:
                json.dump(data, f, indent=2)
            old_path.unlink()
            return True
        except (json.JSONDecodeError, OSError):
            return False


# Global storage instance
_storage: ChatStorage | None = None


def get_storage() -> ChatStorage:
    """Get or create the chat storage instance."""
    global _storage
    if _storage is None:
        _storage = ChatStorage()
    return _storage


def _get_llm(provider: str | None, model: str | None):
    """Get LLM instance.

    Note: provider and model are not used here - they're passed to individual
    calls like .chat() or .set_model(). This factory just creates the LLM instance.
    """
    from ai_infra.llm import LLM

    return LLM()


def _get_default_provider() -> str:
    """Get the default provider that would be auto-selected."""
    from ai_infra.llm.providers.discovery import get_default_provider

    return get_default_provider() or "none"


def _extract_content(response) -> str:
    """Extract text content from LLM response (AIMessage, dict, or string)."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content)
    if isinstance(response, dict):
        return str(response.get("content", str(response)))
    return str(response)


def _build_messages_with_history(
    user_input: str,
    conversation: list[dict[str, str]],
    system: str | None = None,
) -> list[Any]:
    """Build LangChain message list with conversation history.

    Args:
        user_input: Current user message
        conversation: Previous conversation history (list of {role, content} dicts)
        system: Optional system prompt

    Returns:
        List of LangChain message objects
    """
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )

    messages: list[BaseMessage] = []

    # Add system message if provided
    if system:
        messages.append(SystemMessage(content=system))

    # Add conversation history (excluding current message if already added)
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            # Only add if not already added
            if not system:
                messages.append(SystemMessage(content=content))

    # Add current user message
    messages.append(HumanMessage(content=user_input))

    return messages


def _generate_session_id() -> str:
    """Generate a unique session ID (timestamp-based fallback)."""
    from datetime import datetime

    return f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _generate_session_name_with_llm(
    messages: list[dict[str, str]], provider: str | None = None
) -> str | None:
    """Generate a descriptive session name using LLM based on conversation content.

    Args:
        messages: The conversation history
        provider: Optional provider to use for name generation

    Returns:
        A short, descriptive name or None if generation fails
    """
    if not messages:
        return None

    # Build a summary of the conversation (first few messages)
    summary_msgs = messages[:4]  # First 4 messages should be enough context
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}" for m in summary_msgs
    )

    try:
        from ai_infra.llm import LLM

        llm = LLM()
        prompt = f"""Generate a short, descriptive name (2-5 words, no special characters except hyphens) for this chat session based on the conversation topic. Return ONLY the name, nothing else.

Conversation:
{conversation_text}

Examples of good names:
- python-debugging-help
- api-design-review
- recipe-ideas
- travel-planning-japan

Name:"""

        response = llm.chat(
            prompt,
            provider=provider,
            model="gpt-4o-mini" if provider == "openai" else None,  # Use cheap model
            temperature=0.3,
        )

        # Extract and sanitize the name
        name = _extract_content(response).strip().lower()
        # Remove any quotes, extra whitespace, punctuation except hyphens
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"\s+", "-", name)
        name = name[:50]  # Limit length

        if name and len(name) >= 3:
            return name
    except Exception:
        # Silently fail - will use fallback
        pass

    return None


def _format_time_ago(iso_str: str | None) -> str:
    """Format ISO timestamp as human-readable time ago."""
    if not iso_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now() - dt
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"
    except (ValueError, TypeError):
        return "unknown"


def _print_welcome(
    provider: str,
    model: str,
    session_id: str,
    message_count: int = 0,
    tool_count: int = 0,
):
    """Print welcome message."""
    typer.echo()
    # Use dark blue brand color for box
    _console.print("╭─────────────────────────────────────────╮", style=BRAND_ACCENT)
    _console.print("│         ai-infra Interactive Chat       │", style=BRAND_ACCENT)
    _console.print("╰─────────────────────────────────────────╯", style=BRAND_ACCENT)
    typer.echo()
    typer.echo(f"  Provider: {provider}")
    typer.echo(f"  Model:    {model}")
    typer.echo(f"  Session:  {session_id}")
    if message_count > 0:
        typer.secho(f"  Memory:   {message_count} messages restored", fg=typer.colors.GREEN)
    if tool_count > 0:
        _console.print(f"  Tools:    {tool_count} MCP tools available", style=BRAND_ACCENT)
    typer.echo()
    _console.print("  Commands:", style="dim")
    _console.print("    /help      Show all commands", style="dim")
    _console.print("    /use       Switch provider/model (e.g., /use openai gpt-4o)", style="dim")
    _console.print("    /switch    Switch session (interactive or /switch <name>)", style="dim")
    _console.print("    /sessions  List saved sessions", style="dim")
    if tool_count > 0:
        _console.print("    /tools     List available MCP tools", style="dim")
    _console.print("    /clear     Clear conversation", style="dim")
    _console.print("    /quit      Save and exit", style="dim")
    typer.echo()


def _print_help(voice_enabled: bool = False):
    """Print help message."""
    typer.echo()
    typer.secho("Conversation Commands:", bold=True)
    typer.echo("  /help              Show this help message")
    typer.echo("  /clear             Clear conversation history")
    typer.echo("  /system <prompt>   Set or update system prompt")
    typer.echo("  /history           Show conversation history")
    typer.echo()
    typer.secho("Session Commands:", bold=True)
    typer.echo("  /sessions          List all saved sessions")
    typer.echo("  /switch [name]     Switch to a session (interactive picker if no name)")
    typer.echo("  /load <name>       Load a saved session")
    typer.echo("  /new               Start a new session (current is auto-saved)")
    typer.echo("  /save [name]       Save current session")
    typer.echo("  /rename <name>     Rename current session")
    typer.echo("  /delete <name>     Delete a saved session")
    typer.echo("  /deleteall         Delete all sessions (with confirmation)")
    typer.echo()
    typer.secho("Model Commands:", bold=True)
    typer.echo("  /use <provider> [model]   Switch provider and model at once")
    typer.echo("  /model <name>             Change model (for current provider)")
    typer.echo("  /provider <name>          Change provider (resets model to default)")
    typer.echo("  /temp <value>             Set temperature (0.0-2.0)")
    typer.echo()
    if voice_enabled:
        typer.secho("Voice Commands:", bold=True)
        typer.echo("  /text              Switch to text input mode")
        typer.echo("  /voice             Switch to voice input mode")
        typer.echo("  (In voice mode: Enter=record, type text=send as text, Ctrl+C=cancel)")
        typer.echo()
    typer.secho("MCP Tools:", bold=True)
    typer.echo("  /tools             List available MCP tools")
    typer.echo("  (Tools are called automatically when the LLM needs them)")
    typer.echo()
    typer.secho("Exit Commands:", bold=True)
    typer.echo("  /quit, /exit       Save session and exit")
    typer.echo("  quit, exit, q      Quick exit (no slash needed)")
    typer.echo()
    typer.secho("Tips:", bold=True)
    typer.echo("  • Sessions auto-save on exit and auto-resume on start")
    typer.echo("  • Session names are auto-generated based on conversation topic")
    typer.echo("  • Multi-line input: end line with \\ to continue")
    typer.echo("  • Ctrl+C to cancel current generation")
    typer.echo("  • Ctrl+D to exit")
    typer.echo()


def _run_repl(
    llm,
    provider: str | None,
    model: str | None,
    system: str | None = None,
    temperature: float = 0.7,
    stream: bool = True,
    session_id: str | None = None,
    no_persist: bool = False,
    tools: list[Any] | None = None,
    voice_mode: bool = False,
):
    """Run interactive REPL with session persistence and optional MCP tools."""

    storage = get_storage()

    # Session management
    current_session_id = session_id or _generate_session_id()
    conversation: list[dict[str, str]] = []
    current_system = system
    current_temp = temperature
    current_provider = provider
    current_model = model
    current_tools = tools or []

    # Try to load existing session
    restored_count = 0
    if session_id and storage.exists(session_id):
        session_data = storage.load(session_id)
        if session_data:
            conversation = session_data.get("messages", [])
            restored_count = len(conversation)
            # Restore metadata if not overridden
            metadata = session_data.get("metadata", {})
            if not system and metadata.get("system"):
                current_system = metadata["system"]
            if not provider and metadata.get("provider"):
                current_provider = metadata["provider"]
            if not model and metadata.get("model"):
                current_model = metadata["model"]

    # Display provider/model for welcome (resolve auto to actual)
    display_provider = current_provider or _get_default_provider()
    display_model = current_model or "default"
    tool_count = len(current_tools)

    _print_welcome(display_provider, display_model, current_session_id, restored_count, tool_count)

    # Voice mode initialization
    voice_chat = None
    if voice_mode:
        try:
            from ai_infra.llm.multimodal.voice import VoiceChat

            available, missing = VoiceChat.is_available()
            if not available:
                _console.print()
                _console.print("  [red]✗[/red] [bold]Voice chat not available[/bold]")
                for item in missing:
                    _console.print(f"    [dim]• {item}[/dim]")
                _console.print()
                _console.print("  [dim]Install voice extras:[/dim] pip install ai-infra[voice]")
                _console.print()
                raise typer.Exit(1)

            voice_chat = VoiceChat()
            _console.print()
            _console.print("  [green]🎤[/green] [bold]Voice mode enabled[/bold]")
            _console.print("  [dim]• Enter        → Start recording[/dim]")
            _console.print("  [dim]• Type text    → Send as text message[/dim]")
            _console.print("  [dim]• /text        → Switch to text-only mode[/dim]")
            _console.print("  [dim]• /quit or q   → Exit[/dim]")
            _console.print()
        except ImportError:
            _console.print()
            _console.print("  [red]✗[/red] [bold]Voice dependencies not installed[/bold]")
            _console.print("  [dim]Install with:[/dim] pip install ai-infra[voice]")
            _console.print()
            raise typer.Exit(1)

    def _save_session(auto_rename: bool = False):
        """Save current session to storage.

        Args:
            auto_rename: If True and session has generic name, try to rename with LLM
        """
        nonlocal current_session_id
        if no_persist:
            return

        # Try to auto-generate a better name if session has generic name and has messages
        if auto_rename and conversation and len(conversation) >= 2:
            if current_session_id.startswith("session-") or current_session_id.startswith("chat-"):
                new_name = _generate_session_name_with_llm(conversation, current_provider)
                if new_name and not storage.exists(new_name):
                    old_id = current_session_id
                    current_session_id = new_name
                    # Delete old session file if it exists
                    storage.delete(old_id)

        metadata = {
            "provider": current_provider,
            "model": current_model,
            "system": current_system,
            "temperature": current_temp,
        }
        storage.save(current_session_id, conversation, metadata)
        storage.save_last_session_id(current_session_id)

    while True:
        try:
            # Voice mode input
            if voice_mode and voice_chat:
                _console.print()
                _console.print(
                    "[green]🎤 You:[/green] [dim]Enter=record, or type /quit, /text[/dim] ", end=""
                )
                pre_input = input().strip()

                # Check for commands before recording
                if pre_input.startswith("/"):
                    user_input = pre_input
                    # Fall through to command handling below
                elif pre_input.lower() in ("quit", "exit", "q"):
                    # Quick quit without slash
                    _save_session(auto_rename=True)
                    typer.secho(
                        f"[OK] Session saved as: {current_session_id}", fg=typer.colors.GREEN
                    )
                    typer.echo("\nGoodbye! ")
                    break
                elif pre_input:
                    # User typed something - treat as text input
                    user_input = pre_input
                    _console.print(f"[green]You:[/green] {user_input}")
                else:
                    # Empty = start recording
                    _console.print(
                        "[red]● Recording...[/red] [dim]Press Enter to stop (Ctrl+C to cancel)[/dim]"
                    )

                    try:
                        audio = voice_chat.mic.record_until_enter()
                        _console.print("[dim]Transcribing...[/dim]")
                        result = voice_chat.stt.transcribe(audio)
                        user_input = result.text.strip()

                        if not user_input:
                            _console.print("[yellow]No speech detected. Try again.[/yellow]")
                            continue

                        # Show what was transcribed
                        _console.print(f"[green]You:[/green] {user_input}")
                    except KeyboardInterrupt:
                        _console.print("\n[yellow]Recording cancelled[/yellow]")
                        continue
                    except Exception as e:
                        _console.print(f"[red]Recording/transcription error:[/red] {e}")
                        continue
            else:
                # Text mode input
                typer.secho("You: ", fg=typer.colors.GREEN, nl=False)
                user_input = input()

                # Handle empty input
                if not user_input.strip():
                    continue

                # Handle multi-line input
                while user_input.endswith("\\"):
                    user_input = user_input[:-1] + "\n"
                    continuation = input("... ")
                    user_input += continuation

                user_input = user_input.strip()

            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else None

                if cmd in ("quit", "exit", "q"):
                    _save_session(auto_rename=True)
                    typer.secho(
                        f"[OK] Session saved as: {current_session_id}", fg=typer.colors.GREEN
                    )
                    typer.echo("\nGoodbye! ")
                    break

                elif cmd == "text":
                    # Switch to text mode
                    if voice_mode:
                        voice_mode = False
                        _console.print(
                            "[green]Switched to text mode[/green] [dim](use /voice to switch back)[/dim]"
                        )
                    continue

                elif cmd == "voice":
                    # Switch to voice mode
                    if voice_chat and not voice_mode:
                        voice_mode = True
                        _console.print("[green]🎤 Switched to voice mode[/green]")
                    elif not voice_chat:
                        _console.print(
                            "[yellow]Voice not available. Start with --voice flag.[/yellow]"
                        )
                    continue

                elif cmd == "help":
                    _print_help(voice_enabled=voice_chat is not None)
                    continue

                elif cmd == "clear":
                    conversation = []
                    typer.secho("[OK] Conversation cleared", fg=typer.colors.YELLOW)
                    continue

                elif cmd == "history":
                    if not conversation:
                        typer.echo("No conversation history yet.")
                    else:
                        typer.echo()
                        for msg in conversation:
                            role = msg["role"].capitalize()
                            content = (
                                msg["content"][:100] + "..."
                                if len(msg["content"]) > 100
                                else msg["content"]
                            )
                            typer.echo(f"  [{role}] {content}")
                        typer.echo()
                    continue

                elif cmd == "system":
                    if arg:
                        current_system = arg
                        typer.secho(
                            f"[OK] System prompt set: {arg[:50]}...",
                            fg=typer.colors.YELLOW,
                        )
                    else:
                        if current_system:
                            typer.echo(f"Current system prompt: {current_system}")
                        else:
                            typer.echo("No system prompt set. Use: /system <prompt>")
                    continue

                # Session commands
                elif cmd == "sessions":
                    sessions = storage.list_sessions()
                    if not sessions:
                        typer.echo("No saved sessions.")
                    else:
                        typer.echo()
                        typer.secho("Saved Sessions:", bold=True)
                        for s in sessions:
                            active = " (current)" if s["session_id"] == current_session_id else ""
                            provider_info = s.get("provider") or "auto"
                            time_ago = _format_time_ago(s.get("updated_at"))
                            typer.echo(
                                f"  • {s['session_id']}{active} "
                                f"- {s['message_count']} msgs, {provider_info}, {time_ago}"
                            )
                        typer.echo()
                    continue

                elif cmd == "save":
                    new_id = arg.strip() if arg else current_session_id
                    current_session_id = new_id
                    _save_session()
                    typer.secho(
                        f"[OK] Session saved as: {current_session_id}",
                        fg=typer.colors.GREEN,
                    )
                    continue

                elif cmd == "load":
                    if not arg:
                        typer.secho("Usage: /load <session_name>", fg=typer.colors.RED)
                        continue
                    load_id = arg.strip()
                    if not storage.exists(load_id):
                        typer.secho(f"[X] Session not found: {load_id}", fg=typer.colors.RED)
                        typer.echo("Use /sessions to list available sessions")
                        continue
                    # Save current session before switching
                    _save_session()
                    # Load new session
                    session_data = storage.load(load_id)
                    if session_data:
                        conversation = session_data.get("messages", [])
                        current_session_id = load_id
                        metadata = session_data.get("metadata", {})
                        if metadata.get("system"):
                            current_system = metadata["system"]
                        if metadata.get("provider"):
                            current_provider = metadata["provider"]
                            llm = _get_llm(current_provider, current_model)
                        if metadata.get("model"):
                            current_model = metadata["model"]
                        if metadata.get("temperature"):
                            current_temp = metadata["temperature"]
                        typer.secho(
                            f"[OK] Loaded session: {load_id} ({len(conversation)} messages)",
                            fg=typer.colors.GREEN,
                        )
                    continue

                elif cmd == "new":
                    # Save current session
                    _save_session()
                    # Start fresh
                    current_session_id = arg.strip() if arg else _generate_session_id()
                    conversation = []
                    current_system = system  # Reset to original system prompt
                    typer.secho(f"[OK] New session: {current_session_id}", fg=typer.colors.GREEN)
                    continue

                elif cmd == "delete":
                    if not arg:
                        typer.secho("Usage: /delete <session_name>", fg=typer.colors.RED)
                        continue
                    delete_id = arg.strip()
                    if delete_id == current_session_id:
                        typer.secho(
                            "[X] Cannot delete current session. Use /new first.",
                            fg=typer.colors.RED,
                        )
                        continue
                    if storage.delete(delete_id):
                        typer.secho(f"[OK] Deleted session: {delete_id}", fg=typer.colors.GREEN)
                    else:
                        typer.secho(f"[X] Session not found: {delete_id}", fg=typer.colors.RED)
                    continue

                elif cmd == "rename":
                    if not arg:
                        typer.secho("Usage: /rename <new_name>", fg=typer.colors.RED)
                        continue
                    new_name = arg.strip()
                    if storage.exists(new_name):
                        typer.secho(
                            f"[X] Session already exists: {new_name}",
                            fg=typer.colors.RED,
                        )
                        continue
                    old_id = current_session_id
                    current_session_id = new_name
                    _save_session()
                    storage.delete(old_id)
                    typer.secho(
                        f"[OK] Renamed: {old_id} -> {new_name}",
                        fg=typer.colors.GREEN,
                    )
                    continue

                elif cmd == "switch":
                    # Switch to another session - interactive if no name provided
                    sessions = storage.list_sessions()
                    other_sessions = [s for s in sessions if s["session_id"] != current_session_id]

                    if not other_sessions:
                        typer.echo("No other sessions to switch to.")
                        typer.echo("Use /new to create a new session.")
                        continue

                    if arg:
                        # Direct switch by name
                        switch_id = arg.strip()
                        if not storage.exists(switch_id):
                            typer.secho(f"[X] Session not found: {switch_id}", fg=typer.colors.RED)
                            typer.echo("Use /sessions to list available sessions")
                            continue
                    else:
                        # Interactive picker
                        typer.echo()
                        typer.secho("Switch to session:", bold=True)
                        for i, s in enumerate(other_sessions, 1):
                            provider_info = s.get("provider") or "auto"
                            time_ago = _format_time_ago(s.get("updated_at"))
                            typer.echo(
                                f"  [{i}] {s['session_id']} "
                                f"- {s['message_count']} msgs, {provider_info}, {time_ago}"
                            )
                        typer.echo()
                        typer.echo("Enter number (or press Enter to cancel): ", nl=False)
                        choice = input().strip()

                        if not choice:
                            typer.echo("Cancelled.")
                            continue

                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(other_sessions):
                                switch_id = other_sessions[idx]["session_id"]
                            else:
                                typer.secho("[X] Invalid selection", fg=typer.colors.RED)
                                continue
                        except ValueError:
                            typer.secho("[X] Invalid number", fg=typer.colors.RED)
                            continue

                    # Save current session before switching
                    _save_session()
                    # Load new session
                    session_data = storage.load(switch_id)
                    if session_data:
                        conversation = session_data.get("messages", [])
                        current_session_id = switch_id
                        metadata = session_data.get("metadata", {})
                        if metadata.get("system"):
                            current_system = metadata["system"]
                        if metadata.get("provider"):
                            current_provider = metadata["provider"]
                            llm = _get_llm(current_provider, current_model)
                        if metadata.get("model"):
                            current_model = metadata["model"]
                        if metadata.get("temperature"):
                            current_temp = metadata["temperature"]
                        typer.secho(
                            f"[OK] Switched to: {switch_id} ({len(conversation)} messages)",
                            fg=typer.colors.GREEN,
                        )
                    continue

                elif cmd == "deleteall":
                    # Delete all sessions with confirmation
                    sessions = storage.list_sessions()
                    other_count = len(
                        [s for s in sessions if s["session_id"] != current_session_id]
                    )

                    if other_count == 0:
                        typer.echo("No other sessions to delete.")
                        continue

                    typer.secho(
                        f"Delete {other_count} session(s)? Current session will be kept.",
                        bold=True,
                    )
                    typer.echo("Type 'yes' to confirm: ", nl=False)
                    confirm = input().strip().lower()

                    if confirm == "yes":
                        deleted = storage.delete_all(exclude_session_id=current_session_id)
                        typer.secho(
                            f"[OK] Deleted {deleted} session(s)",
                            fg=typer.colors.GREEN,
                        )
                    else:
                        typer.echo("Cancelled.")
                    continue

                elif cmd == "model":
                    if arg:
                        new_model = arg.strip()
                        current_model = new_model
                        typer.secho(f"[OK] Model changed to: {new_model}", fg=typer.colors.YELLOW)
                        # Show hint about listing available models
                        provider_hint = current_provider or _get_default_provider()
                        typer.echo(
                            f"    [dim]Use 'ai-infra models {provider_hint}' to see available models[/dim]"
                        )
                    else:
                        display = current_model or "default (auto)"
                        typer.echo(f"Current model: {display}")
                        if current_provider or _get_default_provider():
                            provider_hint = current_provider or _get_default_provider()
                            typer.echo(
                                f"    [dim]Use 'ai-infra models {provider_hint}' to see options[/dim]"
                            )
                    continue

                elif cmd == "provider":
                    if arg:
                        # Parse provider name (handle accidental multi-word input)
                        new_provider = arg.split()[0]  # Take only the first word

                        # Validate provider
                        from ai_infra.llm.providers.discovery import list_providers

                        valid_providers = list_providers()
                        if new_provider not in valid_providers:
                            typer.secho(
                                f"[X] Unknown provider: {new_provider}",
                                fg=typer.colors.RED,
                            )
                            typer.echo(f"    Valid providers: {', '.join(valid_providers)}")
                            continue

                        # Reset model when provider changes (models are provider-specific)
                        old_provider = current_provider
                        current_provider = new_provider
                        current_model = None  # Reset to provider's default

                        try:
                            llm = _get_llm(current_provider, current_model)
                            typer.secho(
                                f"[OK] Provider changed to: {new_provider} (using default model)",
                                fg=typer.colors.YELLOW,
                            )
                            if old_provider and old_provider != new_provider:
                                typer.echo(
                                    "    [dim]Model reset to default. Use /model to change.[/dim]"
                                )
                        except Exception as e:
                            typer.secho(f"[X] Failed to change provider: {e}", fg=typer.colors.RED)
                            current_provider = old_provider  # Revert on failure
                    else:
                        display = current_provider or _get_default_provider() + " (auto)"
                        typer.echo(f"Current provider: {display}")
                    continue

                elif cmd == "temp":
                    if arg:
                        try:
                            current_temp = float(arg)
                            typer.secho(
                                f"[OK] Temperature set to: {current_temp}",
                                fg=typer.colors.YELLOW,
                            )
                        except ValueError:
                            typer.secho(
                                "[X] Invalid temperature. Use a number 0.0-2.0",
                                fg=typer.colors.RED,
                            )
                    else:
                        typer.echo(f"Current temperature: {current_temp}")
                    continue

                elif cmd == "use":
                    # Convenience command: /use <provider> [model]
                    # Switches both provider and model at once
                    if not arg:
                        typer.secho(
                            "Usage: /use <provider> [model]",
                            fg=typer.colors.RED,
                        )
                        typer.echo("    Example: /use openai gpt-4o")
                        typer.echo("    Example: /use anthropic claude-sonnet-4-20250514")
                        continue

                    parts = arg.split(maxsplit=1)
                    new_provider = parts[0]
                    new_model = parts[1] if len(parts) > 1 else None

                    # Validate provider
                    from ai_infra.llm.providers.discovery import list_providers

                    valid_providers = list_providers()
                    if new_provider not in valid_providers:
                        typer.secho(
                            f"[X] Unknown provider: {new_provider}",
                            fg=typer.colors.RED,
                        )
                        typer.echo(f"    Valid providers: {', '.join(valid_providers)}")
                        continue

                    current_provider = new_provider
                    current_model = new_model

                    try:
                        llm = _get_llm(current_provider, current_model)
                        if new_model:
                            typer.secho(
                                f"[OK] Now using: {new_provider}/{new_model}",
                                fg=typer.colors.YELLOW,
                            )
                        else:
                            typer.secho(
                                f"[OK] Now using: {new_provider} (default model)",
                                fg=typer.colors.YELLOW,
                            )
                    except Exception as e:
                        typer.secho(f"[X] Failed: {e}", fg=typer.colors.RED)
                    continue

                elif cmd == "tools":
                    if not current_tools:
                        typer.secho(
                            "No MCP tools connected. Use --mcp to add tools.",
                            fg=typer.colors.YELLOW,
                        )
                    else:
                        typer.echo()
                        typer.secho(f"Available MCP Tools ({len(current_tools)}):", bold=True)
                        for tool in current_tools:
                            name = getattr(tool, "name", str(tool))
                            desc = getattr(tool, "description", "")[:60]
                            if desc:
                                typer.echo(f"  • {name}: {desc}")
                            else:
                                typer.echo(f"  • {name}")
                        typer.echo()
                    continue

                else:
                    typer.secho(
                        f"Unknown command: /{cmd}. Type /help for commands.",
                        fg=typer.colors.RED,
                    )
                    continue

            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})

            # Show thinking indicator (will be replaced by response)
            spinner = ThinkingSpinner()

            try:
                # Build messages with full conversation history
                messages = _build_messages_with_history(
                    user_input,
                    conversation[:-1],  # Exclude current message (already in messages)
                    system=current_system,
                )

                # Resolve provider/model (handles None -> auto-detect)
                resolved_provider, resolved_model = llm._resolve_provider_and_model(
                    current_provider, current_model
                )

                # Get model with temperature
                chat_model: Any = llm.set_model(
                    resolved_provider,
                    resolved_model,
                    temperature=current_temp,
                )

                # Bind tools if available
                if current_tools:
                    chat_model = chat_model.bind_tools(current_tools)

                # Tool call loop - may need multiple iterations
                response_text = ""
                tts_thread = None
                tts_audio = None
                tts_error = None
                tts_spoken_len = 0  # Track how much text was sent to TTS
                max_tool_iterations = 10  # Safety limit

                async def run_with_tools():
                    nonlocal \
                        response_text, \
                        messages, \
                        tts_thread, \
                        tts_audio, \
                        tts_error, \
                        tts_spoken_len
                    from langchain_core.messages import ToolMessage

                    current_messages = list(messages)
                    has_tools = bool(current_tools)

                    for iteration in range(max_tool_iterations):
                        # Start spinner for LLM thinking
                        spinner.start()

                        try:
                            # When tools are bound, use non-streaming to get complete tool_calls
                            # Streaming tool calls are fragmented and unreliable
                            if has_tools:
                                response = await chat_model.ainvoke(current_messages)
                                tool_calls = getattr(response, "tool_calls", None) or []

                                # Stop spinner before output
                                spinner.stop()

                                if not tool_calls:
                                    # No tool calls - this is the final response
                                    content = _extract_content(response)

                                    if stream and content:
                                        # Stream with live markdown rendering
                                        _console.print(f"[{BRAND_ACCENT}]AI:[/{BRAND_ACCENT}]")
                                        await _stream_with_markdown(content, resolved_model)
                                    else:
                                        _console.print(
                                            f"[{BRAND_ACCENT}]AI:[/{BRAND_ACCENT}] ", end=""
                                        )
                                        _render_response(content, resolved_model)
                                    response_text = content
                                    return
                            else:
                                # No tools - stream with live markdown rendering
                                spinner.stop()
                                _console.print(f"[{BRAND_ACCENT}]AI:[/{BRAND_ACCENT}]")

                                if stream:
                                    # Use Rich Live for streaming with markdown
                                    text_parts = []

                                    # For voice: start TTS early on first chunk
                                    first_tts_started = False
                                    first_tts_text = ""
                                    FIRST_CHUNK_SIZE = 150  # Start TTS after this many chars

                                    with Live(
                                        _create_streaming_display("", resolved_model),
                                        console=_console,
                                        refresh_per_second=15,
                                        transient=False,
                                    ) as live:
                                        async for event in chat_model.astream(current_messages):
                                            text = getattr(event, "content", None)
                                            if text:
                                                text_parts.append(text)
                                                full_text = "".join(text_parts)
                                                live.update(
                                                    _create_streaming_display(
                                                        full_text, resolved_model
                                                    )
                                                )

                                                # Voice mode: start TTS on first chunk
                                                if (
                                                    voice_mode
                                                    and voice_chat
                                                    and not first_tts_started
                                                ):
                                                    first_tts_text += text
                                                    # Start after sentence boundary past min size
                                                    if len(first_tts_text) >= FIRST_CHUNK_SIZE:
                                                        # Find sentence end
                                                        for end in [". ", "! ", "? ", ".\n"]:
                                                            idx = first_tts_text.find(end)
                                                            if idx > 50:
                                                                import threading

                                                                chunk = first_tts_text[
                                                                    : idx + 1
                                                                ].strip()

                                                                def generate_first_tts(txt):
                                                                    nonlocal tts_audio, tts_error
                                                                    try:
                                                                        tts_audio = (
                                                                            voice_chat.tts.speak(
                                                                                txt
                                                                            )
                                                                        )
                                                                    except Exception as e:
                                                                        tts_error = e

                                                                tts_thread = threading.Thread(
                                                                    target=generate_first_tts,
                                                                    args=(chunk,),
                                                                    daemon=True,
                                                                )
                                                                tts_thread.start()
                                                                tts_spoken_len = (
                                                                    idx + 1
                                                                )  # Track spoken length
                                                                first_tts_started = True
                                                                break

                                    response_text = "".join(text_parts)

                                    # If TTS didn't start during streaming (short response), start now
                                    if (
                                        voice_mode
                                        and voice_chat
                                        and response_text
                                        and not first_tts_started
                                    ):
                                        import threading

                                        def generate_tts():
                                            nonlocal tts_audio, tts_error
                                            try:
                                                tts_audio = voice_chat.tts.speak(response_text)
                                            except Exception as e:
                                                tts_error = e

                                        tts_thread = threading.Thread(
                                            target=generate_tts, daemon=True
                                        )
                                        tts_thread.start()
                                else:
                                    response = await chat_model.ainvoke(current_messages)
                                    response_text = _extract_content(response)
                                    _render_response(response_text, resolved_model)
                                return
                        finally:
                            # Ensure spinner is stopped even on error
                            spinner.stop()

                        # Process tool calls
                        current_messages.append(response)

                        for tool_call in tool_calls:
                            # Handle both dict and object formats
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("name", "")
                                tool_args = tool_call.get("args", {})
                                tool_id = tool_call.get("id", "") or f"call_{iteration}"
                            else:
                                tool_name = getattr(tool_call, "name", "")
                                tool_args = getattr(tool_call, "args", {})
                                tool_id = getattr(tool_call, "id", "") or f"call_{iteration}"

                            # Skip empty tool calls
                            if not tool_name:
                                continue

                            # Show tool call with running indicator (brand color)
                            print()
                            _console.print(
                                f"  [dim]⧗[/dim] [{BRAND_ACCENT}]{tool_name}[/{BRAND_ACCENT}] [dim]running...[/dim]",
                                end="",
                            )

                            # Find and execute tool
                            tool_result = None
                            for tool in current_tools:
                                if getattr(tool, "name", "") == tool_name:
                                    try:
                                        # MCP tools support ainvoke
                                        if hasattr(tool, "ainvoke"):
                                            tool_result = await tool.ainvoke(tool_args)
                                        elif hasattr(tool, "invoke"):
                                            tool_result = tool.invoke(tool_args)
                                        else:
                                            tool_result = str(tool(tool_args))
                                    except Exception as e:
                                        tool_result = f"Error: {e}"
                                    break

                            if tool_result is None:
                                tool_result = f"Tool '{tool_name}' not found"

                            # Format result
                            result_str = (
                                tool_result
                                if isinstance(tool_result, str)
                                else json.dumps(tool_result, default=str)
                            )

                            # Clear "running..." and show result with new helper
                            sys.stdout.write("\r\033[K")  # Clear line
                            _render_tool_call(tool_name, result_str)

                            # Add tool result to messages
                            current_messages.append(
                                ToolMessage(content=result_str, tool_call_id=tool_id)
                            )

                        # Continue loop to get final response - show thinking again
                        print()

                asyncio.run(run_with_tools())
                typer.echo()  # Newline after response

                # Voice mode: speak the response
                if voice_mode and voice_chat and response_text:
                    try:
                        # If TTS was started during streaming, wait for it and play
                        if tts_thread is not None:
                            tts_thread.join(timeout=30)
                            if tts_error:
                                _console.print(f"[yellow]TTS error:[/yellow] {tts_error}")
                            elif tts_audio is not None:
                                _console.print("[dim]🔊 Speaking...[/dim]")

                                # Start generating remainder TTS in background BEFORE playing first chunk
                                remainder = response_text[tts_spoken_len:].strip()
                                remainder_audio = None
                                remainder_thread = None

                                if remainder:
                                    import threading

                                    def generate_remainder():
                                        nonlocal remainder_audio
                                        try:
                                            remainder_audio = voice_chat.tts.speak(remainder)
                                        except Exception:
                                            pass

                                    remainder_thread = threading.Thread(
                                        target=generate_remainder, daemon=True
                                    )
                                    remainder_thread.start()

                                # Play first chunk (remainder generating in parallel)
                                voice_chat.player.play(tts_audio, blocking=True)

                                # Play remainder (should be ready or nearly ready)
                                if remainder_thread:
                                    remainder_thread.join(timeout=30)
                                    if remainder_audio is not None:
                                        voice_chat.player.play(remainder_audio, blocking=True)
                        else:
                            # Non-streaming path: generate and play
                            _console.print("[dim]🔊 Speaking...[/dim]")
                            voice_chat.speak(response_text)
                    except Exception as e:
                        _console.print(f"[yellow]TTS error:[/yellow] {e}")

                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": response_text})

                # Auto-save after each exchange
                _save_session()

            except KeyboardInterrupt:
                typer.echo("\n[Interrupted]")
                # Remove the user message since we didn't get a response
                conversation.pop()
                continue

            except Exception as e:
                error_str = str(e)
                # Check for common API key errors and provide helpful messages
                if "api_key" in error_str.lower() and "valid string" in error_str.lower():
                    _console.print()
                    _console.print("  [red]✗[/red] [bold]API key not configured[/bold]")
                    _console.print()
                    _console.print("  [dim]Set your API key in the terminal:[/dim]")
                    _console.print("    [white]export OPENAI_API_KEY=sk-...[/white]")
                    _console.print("    [white]export ANTHROPIC_API_KEY=sk-ant-...[/white]")
                    _console.print()
                elif "authentication" in error_str.lower() or "invalid" in error_str.lower():
                    _console.print()
                    _console.print("  [red]✗[/red] [bold]Authentication failed[/bold]")
                    _console.print(f"  [dim]{error_str}[/dim]")
                    _console.print()
                else:
                    _console.print(f"\n  [red]✗[/red] Error: {e}")
                conversation.pop()
                continue

            typer.echo()  # Extra newline for readability

        except EOFError:
            _save_session(auto_rename=True)
            typer.echo("\nGoodbye! ")
            break

        except KeyboardInterrupt:
            _save_session(auto_rename=True)
            typer.echo("\nGoodbye! ")
            break


@app.callback(invoke_without_command=True)
def chat_cmd(
    ctx: typer.Context,
    message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="One-shot message (non-interactive)",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider (default: auto-detect)",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model name",
    ),
    system: str | None = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Temperature (0.0-2.0)",
    ),
    no_stream: bool = typer.Option(
        False,
        "--no-stream",
        help="Disable streaming (render complete response at once)",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON (one-shot mode only)",
    ),
    session: str | None = typer.Option(
        None,
        "--session",
        help="Session name to resume or create",
    ),
    new_session: bool = typer.Option(
        False,
        "--new",
        "-n",
        help="Start a new session (don't auto-resume)",
    ),
    no_persist: bool = typer.Option(
        False,
        "--no-persist",
        help="Disable session persistence",
    ),
    mcp: list[str] | None = typer.Option(
        None,
        "--mcp",
        help="MCP server URL or JSON config file (can be repeated)",
    ),
    voice: bool = typer.Option(
        False,
        "--voice",
        "-v",
        help="Enable voice chat mode (speak to chat, hear responses)",
    ),
):
    """
    Interactive chat with LLMs with session persistence.

    Start an interactive REPL:

        ai-infra chat

    Resume last session or create new one:

        ai-infra chat                   # Auto-resumes last session
        ai-infra chat --new             # Start fresh
        ai-infra chat --session mywork  # Resume/create named session

    Or send a one-shot message:

        ai-infra chat -m "What is the capital of France?"

    Examples:

        # Interactive with specific provider
        ai-infra chat --provider openai --model gpt-4o

        # Resume a named session
        ai-infra chat --session project-discussion

        # One-shot with system prompt
        ai-infra chat -m "Explain Python" -s "You are a teacher"

        # JSON output for scripting
        ai-infra chat -m "Hello" --json

    MCP Tools (connect to external tools):

        # Single MCP server (HTTP)
        ai-infra chat --mcp http://localhost:8000/mcp

        # Multiple MCP servers
        ai-infra chat --mcp http://server1/mcp --mcp http://server2/mcp

        # MCP config from JSON file
        ai-infra chat --mcp ~/my-mcp-config.json

    Session management:
        ai-infra chat sessions              # List saved sessions
        ai-infra chat session-delete <name> # Delete a session

    Within the REPL, use /help to see all commands including:
        /sessions   - List saved sessions
        /save       - Save current session
        /load       - Load a saved session
        /new        - Start a new session
        /tools      - List available MCP tools
    """
    # If a subcommand is being invoked, don't run the default chat
    if ctx.invoked_subcommand is not None:
        return

    # Get LLM
    try:
        llm = _get_llm(provider, model)
    except Exception as e:
        typer.secho(f"Error initializing LLM: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Initialize MCP if requested
    global _mcp_manager
    tools: list[Any] = []

    if mcp:
        import asyncio

        _mcp_manager = ChatMCPManager()
        typer.secho("Connecting to MCP servers...", fg=typer.colors.BRIGHT_BLACK)

        async def connect_mcp():
            assert _mcp_manager is not None
            return await _mcp_manager.connect(mcp)

        try:
            success = asyncio.run(connect_mcp())
            if success:
                assert _mcp_manager is not None
                tools = _mcp_manager.get_tools()
                typer.secho(
                    f"[OK] Connected: {len(tools)} tools available",
                    fg=typer.colors.GREEN,
                )
            else:
                typer.secho(
                    "[!] MCP connection failed, continuing without tools",
                    fg=typer.colors.YELLOW,
                )
        except Exception as e:
            typer.secho(f"[!] MCP error: {e}", fg=typer.colors.YELLOW)
            _mcp_manager = None

    # For display purposes only
    display_provider = provider or _get_default_provider()
    display_model = model or "default"

    # One-shot mode (no persistence)
    if message:
        try:
            # Show thinking indicator while waiting for response
            with Live(
                RichSpinner("dots", text=Text(" Thinking...", style="dim")),
                console=_console,
                transient=True,
                refresh_per_second=10,
            ):
                response = llm.chat(
                    user_msg=message,
                    system=system,
                    provider=provider,  # Pass actual value (None for auto)
                    model_name=model,  # Pass actual value (None for auto)
                    model_kwargs={"temperature": temperature},
                )

            # Extract content from response (handles AIMessage, dict, or string)
            response_text = _extract_content(response)

            if output_json:
                result = {
                    "provider": display_provider,
                    "model": display_model,
                    "message": message,
                    "response": response_text,
                }
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo(response_text)

        except Exception as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        return

    # Interactive mode - determine session ID
    session_id = session  # Explicit session name if provided

    if not session_id and not new_session and not no_persist:
        # Auto-resume: try to get last session
        storage = get_storage()
        session_id = storage.get_auto_resume_session_id()

    try:
        _run_repl(
            llm=llm,
            provider=provider,  # None means auto-detect
            model=model,  # None means use default
            system=system,
            temperature=temperature,
            stream=not no_stream,
            session_id=session_id,
            no_persist=no_persist,
            tools=tools,
            voice_mode=voice,
        )
    finally:
        # Cleanup MCP connections
        if _mcp_manager:
            import asyncio

            try:
                asyncio.run(_mcp_manager.close())
            except Exception:
                pass


@app.command("sessions")
def sessions_cmd(
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """
    List all saved chat sessions.

    Example:
        ai-infra chat sessions
        ai-infra chat sessions --json
    """
    storage = get_storage()
    sessions = storage.list_sessions()

    if output_json:
        typer.echo(json.dumps(sessions, indent=2))
    else:
        from ai_infra.cli.output import print_sessions

        print_sessions(sessions)


@app.command("session-delete")
def session_delete_cmd(
    session_name: str = typer.Argument(
        ...,
        help="Session name to delete (or 'all' to delete all sessions)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
):
    """
    Delete a chat session or all sessions.

    Examples:
        ai-infra chat session-delete chat-20251201-182558
        ai-infra chat session-delete all
        ai-infra chat session-delete all --force
    """
    storage = get_storage()

    if session_name.lower() == "all":
        sessions = storage.list_sessions()
        if not sessions:
            typer.echo("No sessions to delete.")
            return

        if not force:
            typer.echo(f"This will delete {len(sessions)} session(s):")
            for s in sessions:
                typer.echo(f"  • {s['session_id']}")
            confirm = typer.confirm("Are you sure?")
            if not confirm:
                typer.echo("Cancelled.")
                return

        deleted = 0
        for s in sessions:
            if storage.delete(s["session_id"]):
                deleted += 1
        typer.secho(f"[OK] Deleted {deleted} session(s)", fg=typer.colors.GREEN)

    else:
        if not storage.exists(session_name):
            typer.secho(f"[X] Session not found: {session_name}", fg=typer.colors.RED)
            raise typer.Exit(1)

        if not force:
            confirm = typer.confirm(f"Delete session '{session_name}'?")
            if not confirm:
                typer.echo("Cancelled.")
                return

        if storage.delete(session_name):
            typer.secho(f"[OK] Deleted: {session_name}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"[X] Failed to delete: {session_name}", fg=typer.colors.RED)
            raise typer.Exit(1)


def register(main_app: typer.Typer):
    """Register chat command group to main app."""
    # Add the chat app as a subcommand group
    # This enables: ai-infra chat, ai-infra chat sessions, ai-infra chat session-delete
    main_app.add_typer(app, name="chat", rich_help_panel="Chat")
