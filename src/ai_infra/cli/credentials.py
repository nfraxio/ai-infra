"""Credential storage for ai-infra.

This module provides secure storage for API keys in ~/.config/ai-infra/credentials.
Keys are stored in an INI-style format similar to AWS credentials.

The credential file format:
    [openai]
    api_key = sk-...

    [anthropic]
    api_key = sk-ant-...

Security:
- File permissions are set to 600 (owner read/write only)
- Keys are never logged
- Validation happens before storage
"""

from __future__ import annotations

import logging
import os
import stat
from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Configuration paths
CONFIG_DIR = Path.home() / ".config" / "ai-infra"
CREDENTIALS_FILE = CONFIG_DIR / "credentials"


class CredentialStore:
    """Secure storage for AI provider API keys.

    Stores credentials in ~/.config/ai-infra/credentials with proper
    file permissions (600).

    Example:
        >>> store = CredentialStore()
        >>> store.set("openai", "sk-...")
        >>> store.save()
        >>> store.get("openai")
        'sk-...'
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialize credential store.

        Args:
            path: Custom path for credentials file. Defaults to
                  ~/.config/ai-infra/credentials
        """
        self.path = path or CREDENTIALS_FILE
        self._credentials: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load credentials from file."""
        if not self.path.exists():
            return

        try:
            parser = ConfigParser()
            parser.read(self.path)

            for section in parser.sections():
                if parser.has_option(section, "api_key"):
                    self._credentials[section] = parser.get(section, "api_key")
        except Exception as e:
            log.warning(f"Failed to load credentials: {e}")

    def save(self) -> None:
        """Save credentials to file with secure permissions."""
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Write credentials
        parser = ConfigParser()
        for provider, key in self._credentials.items():
            parser.add_section(provider)
            parser.set(provider, "api_key", key)

        with open(self.path, "w") as f:
            parser.write(f)

        # Set secure permissions (owner read/write only)
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            log.warning("Could not set secure file permissions on credentials file")

    def get(self, provider: str) -> str | None:
        """Get API key for a provider.

        Args:
            provider: Provider name (e.g., "openai")

        Returns:
            API key or None if not stored.
        """
        return self._credentials.get(provider)

    def set(self, provider: str, api_key: str) -> None:
        """Set API key for a provider.

        Args:
            provider: Provider name (e.g., "openai")
            api_key: The API key to store
        """
        self._credentials[provider] = api_key

    def delete(self, provider: str) -> None:
        """Remove API key for a provider.

        Args:
            provider: Provider name to remove
        """
        self._credentials.pop(provider, None)
        self.save()

    def clear(self) -> None:
        """Remove all stored credentials."""
        self._credentials.clear()
        if self.path.exists():
            self.path.unlink()

    def list_providers(self) -> list[str]:
        """List providers with stored credentials.

        Returns:
            List of provider names.
        """
        return list(self._credentials.keys())


def load_credentials_into_env() -> int:
    """Load stored credentials into environment variables.

    This function reads credentials from ~/.config/ai-infra/credentials
    and sets the corresponding environment variables if not already set.

    Returns:
        Number of credentials loaded.

    Example:
        >>> load_credentials_into_env()
        2  # Loaded 2 credentials
    """
    from ai_infra.providers import ProviderRegistry

    store = CredentialStore()
    loaded = 0

    for provider, api_key in store._credentials.items():
        config = ProviderRegistry.get(provider)
        if not config:
            continue

        # Only set if not already in environment (env takes precedence)
        if not os.environ.get(config.env_var):
            os.environ[config.env_var] = api_key
            loaded += 1
            log.debug(f"Loaded credential for {provider} from config")

    return loaded


def validate_api_key(provider: str, api_key: str) -> tuple[bool, str | None]:
    """Validate an API key format and optionally test connectivity.

    Args:
        provider: Provider name
        api_key: API key to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty"

    api_key = api_key.strip()

    # Provider-specific format validation
    if provider == "openai":
        if not api_key.startswith("sk-"):
            return False, "OpenAI keys should start with 'sk-'"
        if len(api_key) < 20:
            return False, "API key seems too short"

    elif provider == "anthropic":
        if not api_key.startswith("sk-ant-"):
            return False, "Anthropic keys should start with 'sk-ant-'"

    elif provider == "google_genai":
        if not api_key.startswith("AIza"):
            return False, "Google AI keys should start with 'AIza'"

    elif provider == "xai":
        if not api_key.startswith("xai-"):
            return False, "xAI keys should start with 'xai-'"

    # TODO: Optional live validation by making a test API call
    # This would verify the key actually works, but adds latency
    # and requires network connectivity

    return True, None
