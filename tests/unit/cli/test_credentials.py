"""Tests for credential storage."""

from __future__ import annotations

from pathlib import Path

from ai_infra.cli.credentials import CredentialStore, validate_api_key


class TestCredentialStore:
    """Tests for CredentialStore class."""

    def test_set_and_get(self, tmp_path: Path) -> None:
        """Test setting and getting credentials."""
        creds_file = tmp_path / "credentials"
        store = CredentialStore(path=creds_file)

        store.set("openai", "sk-test-key")
        assert store.get("openai") == "sk-test-key"
        assert store.get("anthropic") is None

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading credentials from file."""
        creds_file = tmp_path / "credentials"

        # Save credentials
        store1 = CredentialStore(path=creds_file)
        store1.set("openai", "sk-test-key")
        store1.set("anthropic", "sk-ant-test")
        store1.save()

        # Load in new instance
        store2 = CredentialStore(path=creds_file)
        assert store2.get("openai") == "sk-test-key"
        assert store2.get("anthropic") == "sk-ant-test"

    def test_delete(self, tmp_path: Path) -> None:
        """Test deleting a credential."""
        creds_file = tmp_path / "credentials"
        store = CredentialStore(path=creds_file)

        store.set("openai", "sk-test-key")
        store.save()
        store.delete("openai")

        assert store.get("openai") is None

    def test_clear(self, tmp_path: Path) -> None:
        """Test clearing all credentials."""
        creds_file = tmp_path / "credentials"
        store = CredentialStore(path=creds_file)

        store.set("openai", "sk-test-key")
        store.set("anthropic", "sk-ant-test")
        store.save()
        store.clear()

        assert store.list_providers() == []
        assert not creds_file.exists()

    def test_list_providers(self, tmp_path: Path) -> None:
        """Test listing providers with credentials."""
        creds_file = tmp_path / "credentials"
        store = CredentialStore(path=creds_file)

        store.set("openai", "sk-test-key")
        store.set("anthropic", "sk-ant-test")

        providers = store.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers


class TestValidateApiKey:
    """Tests for API key validation."""

    def test_empty_key(self) -> None:
        """Test empty key validation."""
        is_valid, error = validate_api_key("openai", "")
        assert not is_valid
        assert "empty" in error.lower()

    def test_openai_valid_prefix(self) -> None:
        """Test OpenAI key with valid prefix."""
        is_valid, _ = validate_api_key("openai", "sk-" + "x" * 50)
        assert is_valid

    def test_openai_invalid_prefix(self) -> None:
        """Test OpenAI key with invalid prefix."""
        is_valid, error = validate_api_key("openai", "invalid-key")
        assert not is_valid
        assert "sk-" in error

    def test_anthropic_valid_prefix(self) -> None:
        """Test Anthropic key with valid prefix."""
        is_valid, _ = validate_api_key("anthropic", "sk-ant-" + "x" * 50)
        assert is_valid

    def test_anthropic_invalid_prefix(self) -> None:
        """Test Anthropic key with invalid prefix."""
        is_valid, error = validate_api_key("anthropic", "sk-wrong")
        assert not is_valid
        assert "sk-ant-" in error

    def test_google_valid_prefix(self) -> None:
        """Test Google key with valid prefix."""
        is_valid, _ = validate_api_key("google_genai", "AIza" + "x" * 35)
        assert is_valid

    def test_google_invalid_prefix(self) -> None:
        """Test Google key with invalid prefix."""
        is_valid, error = validate_api_key("google_genai", "invalid")
        assert not is_valid
        assert "AIza" in error

    def test_xai_valid_prefix(self) -> None:
        """Test xAI key with valid prefix."""
        is_valid, _ = validate_api_key("xai", "xai-" + "x" * 50)
        assert is_valid

    def test_xai_invalid_prefix(self) -> None:
        """Test xAI key with invalid prefix."""
        is_valid, error = validate_api_key("xai", "invalid")
        assert not is_valid
        assert "xai-" in error

    def test_unknown_provider(self) -> None:
        """Test unknown provider validation (should pass basic checks)."""
        is_valid, _ = validate_api_key("unknown", "some-key")
        assert is_valid
