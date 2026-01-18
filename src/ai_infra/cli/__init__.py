from __future__ import annotations

import sys
import time

import typer
from rich.console import Console
from rich.live import Live
from rich.text import Text
from svc_infra.cli.foundation.typer_bootstrap import pre_cli

from ai_infra.cli.cmds import (
    benchmark_app,
    register_chat,
    register_discovery,
    register_executor,
    register_imagegen,
    register_mcp,
    register_multimodal,
    register_setup,
    register_stdio_publisher,
)
from ai_infra.cli.credentials import load_credentials_into_env
from ai_infra.llm.defaults import DEFAULT_MODELS
from ai_infra.llm.providers.discovery import (
    get_default_provider,
    list_configured_providers,
)

# Version - should match pyproject.toml
__version__ = "1.5.1"

console = Console()

# ASCII art logo - stylized "ai-infra"
_LOGO_LINES = [
    "    ▄▀▀▀▄ ▀       ▀ █▀▀▄  █▀▀▀ █▀▀▄  ▄▀▀▀▄",
    "    █▀▀▀█ █  ▄▄▄  █ █  █  █▀▀  █▄▄▀  █▀▀▀█",
    "    █   █ █       █ █  █  █    █  █  █   █",
    "    ▀   ▀ ▀       ▀ ▀  ▀  ▀    ▀  ▀  ▀   ▀",
]

# Gradient colors for the logo (nfrax blue spectrum)
_LOGO_GRADIENT = ["#7ba7d4", "#5b8bd4", "#3b82f6", "#1e3a5f"]

# Animation frames for loading dots
_SPINNER_FRAMES = ["◇", "◈", "◆", "◈"]
_GRADIENT_COLORS = ["#5b8bd4", "#3b82f6", "#1e3a5f", "#3b82f6"]


def _animate_banner():
    """Display animated ai-infra banner with ASCII art logo."""
    # Skip animation if not a TTY (e.g., piped output)
    if not sys.stdout.isatty():
        _show_banner_static()
        return

    console.print()

    # Phase 1: Build up the logo line by line with color gradient
    for idx, line in enumerate(_LOGO_LINES):
        color = _LOGO_GRADIENT[idx % len(_LOGO_GRADIENT)]
        # Reveal effect - print characters progressively
        with Live(console=console, refresh_per_second=60, transient=True) as live:
            for i in range(0, len(line) + 1, 3):  # Step by 3 for speed
                text = Text(line[:i], style=f"bold {color}")
                live.update(text)
                time.sleep(0.008)
        console.print(Text(line, style=f"bold {color}"))

    # Small pause after logo
    time.sleep(0.15)

    # Phase 2: Version badge slides in
    version_line = f"                          v{__version__}"
    with Live(console=console, refresh_per_second=30, transient=True) as live:
        for i in range(len(version_line) + 1):
            text = Text(version_line[:i], style="dim")
            live.update(text)
            time.sleep(0.015)
    console.print(Text(version_line, style="dim"))

    console.print()

    # Phase 3: Tagline with fade-in
    tagline = "   Production-ready SDK for AI applications"
    with Live(console=console, refresh_per_second=30, transient=True) as live:
        for i in range(len(tagline) + 1):
            text = Text(tagline[:i], style="dim italic")
            live.update(text)
            time.sleep(0.012)
    console.print(Text(tagline, style="dim italic"))
    console.print()


def _show_banner_static():
    """Display static banner (no animation)."""
    console.print()
    # Print logo with gradient colors
    for idx, line in enumerate(_LOGO_LINES):
        color = _LOGO_GRADIENT[idx % len(_LOGO_GRADIENT)]
        console.print(Text(line, style=f"bold {color}"))
    console.print(Text(f"                          v{__version__}", style="dim"))
    console.print()
    console.print(Text("   Production-ready SDK for AI applications", style="dim italic"))
    console.print()


def _show_banner():
    """Display the ai-infra banner (with animation if TTY)."""
    _animate_banner()


def _show_provider_status() -> bool:
    """Display configured provider status. Returns True if any configured."""
    # Auto-load credentials from config file
    load_credentials_into_env()

    configured = list_configured_providers()

    if not configured:
        console.print("  [yellow]⚠[/yellow] [dim]No providers configured[/dim]")
        console.print()
        console.print("    [dim]Run the setup wizard to configure your API keys:[/dim]")
        console.print("    [white]ai-infra setup[/white]")
        console.print()
        console.print("    [dim]Or set environment variables directly:[/dim]")
        console.print("    [dim]export OPENAI_API_KEY=sk-...[/dim]")
        console.print()
        return False

    # Show configured providers with their default models
    default_provider = get_default_provider()
    is_tty = sys.stdout.isatty()

    for provider in configured:
        model = DEFAULT_MODELS.get(provider, "default")

        # Animate checkmark if TTY
        if is_tty:
            # Brief loading indicator
            with Live(console=console, refresh_per_second=20, transient=True) as live:
                live.update(Text(f"  ○ {provider}", style="dim"))
                time.sleep(0.05)

        if provider == default_provider:
            console.print(
                f"  [green]✓[/green] [bold]{provider}[/bold] [dim]({model})[/dim] "
                f"[#3b82f6]← default[/#3b82f6]"
            )
        else:
            console.print(f"  [green]✓[/green] [white]{provider}[/white] [dim]({model})[/dim]")

    console.print()
    return True
    return True


def _show_help():
    """Display beautiful help output."""
    _show_banner()

    # Show provider status
    has_providers = _show_provider_status()

    # Quick start section
    console.print(Text("  Quick Start", style="bold #3b82f6"))
    console.print()

    if has_providers:
        console.print(
            "    [dim]$[/dim] [white]ai-infra chat[/white]              [dim]Interactive chat[/dim]"
        )
        console.print(
            '    [dim]$[/dim] [white]ai-infra chat -m "Hello"[/white]   [dim]One-shot message[/dim]'
        )
        console.print(
            "    [dim]$[/dim] [white]ai-infra executor run[/white]"
            "       [dim]Run autonomous tasks[/dim]"
        )
    else:
        console.print(
            "    [dim]$[/dim] [white]ai-infra providers[/white]"
            "          [dim]Check provider status[/dim]"
        )

    console.print()

    # Commands section
    console.print(Text("  Commands", style="bold #3b82f6"))
    console.print()

    commands = [
        ("setup", "Configure API keys (interactive wizard)"),
        ("chat", "Interactive chat with LLMs (supports --mcp for tools)"),
        ("chat --voice", "Voice chat mode (speak to chat)"),
        ("executor", "Autonomous task execution (roadmaps, coding)"),
        ("mcp", "MCP server debugging and testing"),
        ("providers", "List available AI providers"),
        ("models", "List models for a provider"),
        ("image-*", "Image generation (providers, models)"),
        ("tts-*", "Text-to-speech (providers, voices, models)"),
        ("stt-*", "Speech-to-text (providers, models)"),
        ("voice-*", "Voice chat (status, devices)"),
    ]

    for cmd, desc in commands:
        console.print(f"    [bold #1e3a5f]{cmd:16}[/bold #1e3a5f] [dim]{desc}[/dim]")

    console.print()
    console.print("    [dim]Run[/dim] [white]ai-infra --help[/white] [dim]for all commands[/dim]")
    console.print()

    # Footer
    console.print(Text("  https://github.com/Aliikhatami94/ai-infra", style="dim underline"))
    console.print()


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        _show_banner()
        raise typer.Exit()


_TYPER_HELP = """Production-ready SDK for building AI applications.

**Quick start:**

* `ai-infra chat` — Start interactive chat
* `ai-infra chat -m "Hello"` — One-shot message
* `ai-infra providers` — List available providers

More info: https://github.com/Aliikhatami94/ai-infra
"""

app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    add_completion=False,
    help=_TYPER_HELP,
    rich_markup_mode="markdown",
)

# Register callback AFTER pre_cli to ensure our logic runs
# pre_cli adds env loading, we wrap it with our display logic
pre_cli(app)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """ai-infra — Production-ready SDK for building AI applications."""
    # Always load credentials from config file into environment
    # This ensures all commands have access to stored API keys
    load_credentials_into_env()

    # If no command given, show our custom help
    if ctx.invoked_subcommand is None:
        _show_help()
        raise typer.Exit()


register_chat(app)
register_executor(app)
register_setup(app)
register_stdio_publisher(app)
register_discovery(app)
register_imagegen(app)
register_multimodal(app)
register_mcp(app)
app.add_typer(benchmark_app, name="benchmark", hidden=True)


def main():
    app()


if __name__ == "__main__":
    main()
