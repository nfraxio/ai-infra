"""Rich-powered CLI output utilities.

Provides themed, beautiful output for all ai-infra CLI commands.
Integrates with the console theme from Phase 16.6.

Example:
    ```python
    from ai_infra.cli.output import (
        print_providers,
        print_models,
        print_sessions,
        print_cli_error,
        print_cli_success,
    )

    print_providers(["openai", "anthropic"], configured=["openai"])
    print_models("openai", ["gpt-4o", "gpt-4o-mini"])
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ai_infra.cli.console import (
    BOX_STYLES,
    BRAND_ACCENT,
    get_console,
    print_error,
    print_info,
    print_success,
    print_warning,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# =============================================================================
# CLI Header / Banner
# =============================================================================


def print_banner() -> None:
    """Print the ai-infra CLI banner."""
    console = get_console()

    banner = Text()
    banner.append(
        "╭─────────────────────────────────────────────────────────╮\n", style=BRAND_ACCENT
    )
    banner.append("│", style=BRAND_ACCENT)
    banner.append("  ai-infra ", style=f"bold {BRAND_ACCENT}")
    banner.append("— Production-ready SDK for AI applications  ", style="dim")
    banner.append("│\n", style=BRAND_ACCENT)
    banner.append("╰─────────────────────────────────────────────────────────╯", style=BRAND_ACCENT)
    console.print(banner)


# =============================================================================
# Provider / Model Output
# =============================================================================


def print_providers(
    providers: Sequence[str],
    configured: Sequence[str] | None = None,
) -> None:
    """Print providers in a beautiful table.

    Args:
        providers: List of all provider names.
        configured: List of configured provider names.
    """
    console = get_console()
    configured_set = set(configured or [])

    table = Table(
        show_header=True,
        header_style="bold",
        box=BOX_STYLES["default"],
        title="[bold]AI Providers[/]",
        title_justify="left",
        padding=(0, 1),
    )
    table.add_column("Provider", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("API Key", style="dim")

    env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "bedrock": "AWS_ACCESS_KEY_ID",
    }

    for provider in sorted(providers):
        if provider in configured_set:
            status = Text("●", style="bold green")
            status.append(" configured", style="green")
        else:
            status = Text("○", style="dim")
            status.append(" not set", style="dim")

        env_var = env_vars.get(provider.lower(), f"{provider.upper()}_API_KEY")
        table.add_row(provider, status, env_var)

    console.print()
    console.print(table)
    console.print()


def print_models(
    provider: str,
    models: Sequence[str],
    *,
    show_all: bool = False,
    max_display: int = 15,
) -> None:
    """Print models for a provider in a beautiful format.

    Args:
        provider: Provider name.
        models: List of model names.
        show_all: Show all models (no truncation).
        max_display: Maximum models to show before truncating.
    """
    console = get_console()

    # Provider header
    header = Text()
    header.append(f"  {provider}", style=f"bold {BRAND_ACCENT}")
    header.append(f"  ({len(models)} models)", style="dim")

    console.print()
    console.print(header)
    console.print("  " + "─" * 50, style="dim")

    display_models = models if show_all else models[:max_display]

    # Two-column display for efficiency
    col_width = 35
    for i in range(0, len(display_models), 2):
        line = Text("    ")

        # First column
        model1 = display_models[i]
        line.append("• ", style=BRAND_ACCENT)
        line.append(model1[: col_width - 2], style="white")

        # Second column (if exists)
        if i + 1 < len(display_models):
            padding = col_width - len(model1) - 2
            if padding > 0:
                line.append(" " * padding)
            model2 = display_models[i + 1]
            line.append("• ", style=BRAND_ACCENT)
            line.append(model2[: col_width - 2], style="white")

        console.print(line)

    # Truncation message
    if not show_all and len(models) > max_display:
        remaining = len(models) - max_display
        console.print(f"    [dim]... and {remaining} more (use --all to see all)[/]")

    console.print()


def print_all_models(
    provider_models: Mapping[str, Sequence[str]],
    *,
    max_per_provider: int = 10,
) -> None:
    """Print models for all providers.

    Args:
        provider_models: Dict mapping provider names to model lists.
        max_per_provider: Max models to show per provider.
    """
    console = get_console()

    console.print()
    console.print("[bold]Available Models by Provider[/]")
    console.print("─" * 60, style="dim")

    for provider, models in sorted(provider_models.items()):
        if not models:
            # Empty provider
            header = Text()
            header.append(f"\n  {provider}", style="dim")
            header.append("  (no models available)", style="dim italic")
            console.print(header)
            continue

        # Provider header
        header = Text()
        header.append(f"\n  {provider}", style=f"bold {BRAND_ACCENT}")
        header.append(f"  ({len(models)} models)", style="dim")
        console.print(header)

        # Models list
        display_models = models[:max_per_provider]
        for model in display_models:
            console.print(f"    [{BRAND_ACCENT}]•[/] {model}")

        if len(models) > max_per_provider:
            remaining = len(models) - max_per_provider
            console.print(f"    [dim]... and {remaining} more[/]")

    console.print()


# =============================================================================
# Session Output
# =============================================================================


def print_sessions(
    sessions: Sequence[dict[str, Any]],
) -> None:
    """Print chat sessions in a beautiful table.

    Args:
        sessions: List of session info dicts with session_id, message_count, etc.
    """
    console = get_console()

    if not sessions:
        print_info("No saved sessions found.")
        console.print("  [dim]Start a chat with: ai-infra chat[/]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        box=BOX_STYLES["default"],
        title="[bold]Chat Sessions[/]",
        title_justify="left",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Session", style="bold")
    table.add_column("Messages", justify="right")
    table.add_column("Model", style=BRAND_ACCENT)
    table.add_column("Updated", style="dim")

    for i, session in enumerate(sessions, 1):
        session_id = session.get("session_id", "unknown")
        msg_count = str(session.get("message_count", 0))
        model = session.get("model") or session.get("provider") or "—"

        # Format timestamp
        updated = session.get("updated_at", "")
        if updated:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(updated)
                updated = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                pass

        table.add_row(str(i), session_id, msg_count, model, updated)

    console.print()
    console.print(table)
    console.print()
    console.print("  [dim]Resume a session: ai-infra chat --session <name>[/]")
    console.print()


# =============================================================================
# Error / Success Output
# =============================================================================


def print_cli_error(
    message: str,
    *,
    hint: str | None = None,
    exit_code: int = 1,
) -> None:
    """Print a CLI error with optional hint.

    Args:
        message: Error message.
        hint: Optional hint for resolution.
        exit_code: Exit code (not used, for signature compatibility).
    """
    print_error(message, hint=hint)


def print_cli_warning(message: str) -> None:
    """Print a CLI warning."""
    print_warning(message)


def print_cli_success(message: str, *, details: str | None = None) -> None:
    """Print a CLI success message."""
    print_success(message, details=details)


def print_cli_info(message: str) -> None:
    """Print a CLI info message."""
    print_info(message)


# =============================================================================
# Help Panel
# =============================================================================


def print_quick_start() -> None:
    """Print quick start guide."""
    console = get_console()

    content = Text()
    content.append("QUICK START\n\n", style="bold")

    commands = [
        ("ai-infra chat", "Start interactive chat"),
        ('ai-infra chat -m "Hello"', "One-shot message"),
        ("ai-infra providers", "List AI providers"),
        ("ai-infra models --all", "List all models"),
    ]

    for cmd, desc in commands:
        content.append("  $ ", style="dim")
        content.append(cmd, style=f"bold {BRAND_ACCENT}")
        content.append(f"  {desc}\n", style="dim")

    panel = Panel(
        content,
        title="[bold]ai-infra[/]",
        title_align="left",
        border_style=BRAND_ACCENT,
        box=BOX_STYLES["default"],
        padding=(1, 2),
    )
    console.print(panel)


# =============================================================================
# Image / TTS / STT Output
# =============================================================================


def print_image_providers(
    providers: Sequence[str],
    configured: Sequence[str] | None = None,
) -> None:
    """Print image generation providers."""
    console = get_console()
    configured_set = set(configured or [])

    console.print()
    console.print("[bold]Image Generation Providers[/]")
    console.print("─" * 40, style="dim")

    for provider in sorted(providers):
        if provider in configured_set:
            status = "[bold green]●[/] "
        else:
            status = "[dim]○[/] "
        console.print(f"  {status}{provider}")

    console.print()


def print_tts_providers(
    providers: Sequence[str],
    configured: Sequence[str] | None = None,
) -> None:
    """Print text-to-speech providers."""
    console = get_console()
    configured_set = set(configured or [])

    console.print()
    console.print("[bold]Text-to-Speech Providers[/]")
    console.print("─" * 40, style="dim")

    for provider in sorted(providers):
        if provider in configured_set:
            status = "[bold green]●[/] "
        else:
            status = "[dim]○[/] "
        console.print(f"  {status}{provider}")

    console.print()


def print_voices(
    provider: str,
    voices: Sequence[dict[str, Any]],
) -> None:
    """Print available voices for a TTS provider."""
    console = get_console()

    console.print()
    console.print(f"[bold]{provider} Voices[/]")
    console.print("─" * 50, style="dim")

    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Voice ID", style=BRAND_ACCENT)
    table.add_column("Name", style="white")
    table.add_column("Language", style="dim")

    for voice in voices[:20]:
        voice_id = voice.get("id", voice.get("voice_id", ""))
        name = voice.get("name", "")
        lang = voice.get("language", voice.get("lang", ""))
        table.add_row(voice_id, name, lang)

    console.print(table)

    if len(voices) > 20:
        console.print(f"  [dim]... and {len(voices) - 20} more[/]")

    console.print()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Banner
    "print_banner",
    # Providers / Models
    "print_providers",
    "print_models",
    "print_all_models",
    # Sessions
    "print_sessions",
    # Errors / Success
    "print_cli_error",
    "print_cli_warning",
    "print_cli_success",
    "print_cli_info",
    # Help
    "print_quick_start",
    # Image / TTS
    "print_image_providers",
    "print_tts_providers",
    "print_voices",
]
