"""Setup commands for ai-infra.

This module provides interactive setup for configuring API keys and
provider credentials with a user-friendly experience.

Commands:
    ai-infra setup          Interactive setup wizard
    ai-infra setup status   Show current configuration status
    ai-infra setup clear    Remove saved credentials
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ai_infra.cli.console import BRAND_ACCENT
from ai_infra.cli.credentials import (
    CredentialStore,
    load_credentials_into_env,
    validate_api_key,
)
from ai_infra.providers import ProviderCapability, ProviderRegistry

if TYPE_CHECKING:
    pass

console = Console()

# Provider display configuration
PROVIDER_INFO = {
    "openai": {
        "name": "OpenAI",
        "placeholder": "sk-...",
        "url": "https://platform.openai.com/api-keys",
        "models": "GPT-4o, o1, DALL-E 3, Whisper",
    },
    "anthropic": {
        "name": "Anthropic",
        "placeholder": "sk-ant-...",
        "url": "https://console.anthropic.com/settings/keys",
        "models": "Claude 3.5 Sonnet, Claude 3 Opus",
    },
    "google_genai": {
        "name": "Google AI",
        "placeholder": "AIza...",
        "url": "https://aistudio.google.com/apikey",
        "models": "Gemini 2.0, Gemini 1.5 Pro",
    },
    "xai": {
        "name": "xAI",
        "placeholder": "xai-...",
        "url": "https://console.x.ai/",
        "models": "Grok-2, Grok-beta",
    },
}


def _mask_key(key: str) -> str:
    """Mask API key for display, showing only first and last 4 chars."""
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _show_welcome() -> None:
    """Show welcome message for setup wizard."""
    console.print()
    console.print(
        Panel(
            f"[bold {BRAND_ACCENT}]ai-infra Setup Wizard[/bold {BRAND_ACCENT}]\n\n"
            "[dim]Configure your AI provider API keys for use with ai-infra.\n"
            "Keys are stored securely in ~/.config/ai-infra/credentials[/dim]",
            border_style=BRAND_ACCENT,
            padding=(1, 2),
        )
    )
    console.print()


def _show_provider_selection() -> list[str]:
    """Show provider selection and return list of selected providers."""
    console.print("[bold]Available Providers[/bold]")
    console.print()

    providers = ProviderRegistry.list_for_capability(ProviderCapability.CHAT)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Num", style="dim", width=4)
    table.add_column("Provider", style="bold", width=12)
    table.add_column("Models", style="dim")
    table.add_column("Status", width=12)

    for i, provider in enumerate(providers, 1):
        info = PROVIDER_INFO.get(provider, {"name": provider, "models": ""})
        is_configured = ProviderRegistry.is_configured(provider)
        status = "[green]✓ configured[/green]" if is_configured else "[dim]not set[/dim]"
        table.add_row(
            f"{i}.",
            info["name"],
            info.get("models", ""),
            status,
        )

    console.print(table)
    console.print()

    # Ask which providers to configure
    console.print("[dim]Enter provider numbers to configure (e.g., 1,2) or 'all' for all:[/dim]")
    selection = Prompt.ask("Configure", default="all")

    if selection.lower() == "all":
        return providers

    try:
        indices = [int(x.strip()) for x in selection.split(",")]
        return [providers[i - 1] for i in indices if 1 <= i <= len(providers)]
    except (ValueError, IndexError):
        console.print("[yellow]Invalid selection, configuring all providers[/yellow]")
        return providers


def _configure_provider(provider: str, store: CredentialStore) -> bool:
    """Configure a single provider interactively. Returns True if configured."""
    info = PROVIDER_INFO.get(provider, {"name": provider, "placeholder": "...", "url": ""})

    console.print()
    console.print(f"[bold {BRAND_ACCENT}]━━━ {info['name']} ━━━[/bold {BRAND_ACCENT}]")

    # Show existing key status
    existing_key = store.get(provider) or os.environ.get(
        ProviderRegistry.get(provider).env_var if ProviderRegistry.get(provider) else ""
    )

    if existing_key:
        console.print(f"[dim]Current key:[/dim] [green]{_mask_key(existing_key)}[/green]")
        if not Confirm.ask("Replace existing key?", default=False):
            return True

    # Show where to get the key
    if info.get("url"):
        console.print(f"[dim]Get your API key at:[/dim] [underline]{info['url']}[/underline]")

    # Prompt for key with validation
    while True:
        # Use getpass-style input for security
        console.print(f"[dim]Enter API key ({info['placeholder']}):[/dim]")

        # Read key securely (hide input)
        if sys.stdin.isatty():
            import getpass

            try:
                api_key = getpass.getpass(prompt="API Key: ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Skipped[/yellow]")
                return False
        else:
            api_key = Prompt.ask("API Key", password=True)

        if not api_key:
            console.print("[yellow]Skipped[/yellow]")
            return False

        # Validate key format
        console.print("[dim]Validating...[/dim]", end=" ")

        is_valid, error = validate_api_key(provider, api_key)

        if is_valid:
            console.print("[green]✓ Valid[/green]")
            store.set(provider, api_key)
            console.print(f"[green]✓ {info['name']} configured successfully[/green]")
            return True
        else:
            console.print(f"[red]✗ {error}[/red]")
            if not Confirm.ask("Try again?", default=True):
                return False


def _show_summary(store: CredentialStore) -> None:
    """Show configuration summary."""
    console.print()
    console.print("[bold]Configuration Summary[/bold]")
    console.print()

    providers = ProviderRegistry.list_for_capability(ProviderCapability.CHAT)

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Provider", width=12)
    table.add_column("Status", width=15)
    table.add_column("Source", style="dim")

    for provider in providers:
        info = PROVIDER_INFO.get(provider, {"name": provider})

        # Check both credential store and environment
        stored_key = store.get(provider)
        config = ProviderRegistry.get(provider)
        env_key = os.environ.get(config.env_var) if config else None

        if stored_key:
            table.add_row(
                info["name"],
                "[green]✓ configured[/green]",
                "~/.config/ai-infra/credentials",
            )
        elif env_key:
            table.add_row(
                info["name"],
                "[green]✓ configured[/green]",
                f"${config.env_var}",
            )
        else:
            table.add_row(
                info["name"],
                "[dim]not configured[/dim]",
                "",
            )

    console.print(table)
    console.print()


# =============================================================================
# CLI Commands
# =============================================================================

setup_app = typer.Typer(
    name="setup",
    help="Configure AI provider API keys",
    no_args_is_help=False,
    invoke_without_command=True,
)


@setup_app.callback(invoke_without_command=True)
def setup_main(ctx: typer.Context) -> None:
    """Interactive setup wizard for configuring AI providers.

    Run this command to securely configure your API keys for various AI
    providers. Keys are stored in ~/.config/ai-infra/credentials and loaded
    automatically when using ai-infra.

    Examples:
        ai-infra setup              # Interactive wizard
        ai-infra setup status       # Show current status
        ai-infra setup clear        # Remove saved credentials
    """
    if ctx.invoked_subcommand is not None:
        return

    _show_welcome()

    store = CredentialStore()
    providers = _show_provider_selection()

    configured_count = 0
    for provider in providers:
        if _configure_provider(provider, store):
            configured_count += 1

    if configured_count > 0:
        store.save()
        console.print()
        console.print(
            f"[green]✓ Saved {configured_count} credential(s) to "
            f"~/.config/ai-infra/credentials[/green]"
        )

    _show_summary(store)

    console.print("[dim]Run [white]ai-infra chat[/white] to start chatting![/dim]")
    console.print()


@setup_app.command("status")
def setup_status() -> None:
    """Show current API key configuration status."""
    console.print()
    console.print(f"[bold {BRAND_ACCENT}]Provider Status[/bold {BRAND_ACCENT}]")
    console.print()

    # Load credentials
    store = CredentialStore()
    load_credentials_into_env()

    providers = ProviderRegistry.list_for_capability(ProviderCapability.CHAT)

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Provider", width=12)
    table.add_column("API Key", width=20)
    table.add_column("Source", style="dim", width=30)

    for provider in providers:
        info = PROVIDER_INFO.get(provider, {"name": provider})
        config = ProviderRegistry.get(provider)

        stored_key = store.get(provider)
        env_var = config.env_var if config else ""
        env_key = os.environ.get(env_var)

        if stored_key:
            table.add_row(
                info["name"],
                f"[green]{_mask_key(stored_key)}[/green]",
                "~/.config/ai-infra/credentials",
            )
        elif env_key:
            table.add_row(
                info["name"],
                f"[green]{_mask_key(env_key)}[/green]",
                f"${env_var}",
            )
        else:
            table.add_row(
                info["name"],
                "[dim]not set[/dim]",
                f"[dim]Set ${env_var} or run setup[/dim]",
            )

    console.print(table)
    console.print()


@setup_app.command("clear")
def setup_clear(
    provider: str | None = typer.Argument(
        None,
        help="Provider to clear (e.g., 'openai'). If not specified, clears all.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Remove saved credentials.

    Examples:
        ai-infra setup clear           # Clear all saved credentials
        ai-infra setup clear openai    # Clear only OpenAI credentials
    """
    store = CredentialStore()

    if provider:
        # Clear single provider
        if provider not in store._credentials:
            console.print(f"[yellow]No saved credentials for {provider}[/yellow]")
            return

        if not force and not Confirm.ask(f"Remove saved credentials for {provider}?"):
            console.print("[dim]Cancelled[/dim]")
            return

        store.delete(provider)
        store.save()
        console.print(f"[green]✓ Removed credentials for {provider}[/green]")
    else:
        # Clear all
        if not store._credentials:
            console.print("[dim]No saved credentials to clear[/dim]")
            return

        if not force and not Confirm.ask("Remove all saved credentials?"):
            console.print("[dim]Cancelled[/dim]")
            return

        store.clear()
        console.print("[green]✓ Removed all saved credentials[/green]")


def register_setup(app: typer.Typer) -> None:
    """Register setup commands with the main app."""
    app.add_typer(setup_app, name="setup")
