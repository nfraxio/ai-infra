"""Tests for ScaffolderAgent.

This module tests the ScaffolderAgent functionality:
- ScaffolderAgent instantiation and registration
- Project type detection
- Initialization check (any language/framework)
- File analysis from command history
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.executor.agents.scaffolder import (
    DETECT_PROJECT_TYPE_PROMPT,
    PROJECT_MARKERS,
    SETUP_PROJECT_PROMPT,
    ScaffolderAgent,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def ensure_scaffolder_registered():
    """Ensure ScaffolderAgent is properly registered before tests."""
    SubAgentRegistry._agents[SubAgentType.SCAFFOLDER] = ScaffolderAgent
    yield


# =============================================================================
# SubAgentType Tests
# =============================================================================


class TestSubAgentTypeEnum:
    """Tests for SCAFFOLDER enum value in SubAgentType."""

    def test_scaffolder_enum_exists(self) -> None:
        """Test SCAFFOLDER is defined in SubAgentType."""
        assert hasattr(SubAgentType, "SCAFFOLDER")
        assert SubAgentType.SCAFFOLDER.value == "scaffolder"

    def test_scaffolder_is_string_enum(self) -> None:
        """Test SCAFFOLDER inherits from str."""
        assert isinstance(SubAgentType.SCAFFOLDER, str)
        assert SubAgentType.SCAFFOLDER == "scaffolder"


# =============================================================================
# ScaffolderAgent Class Attributes Tests
# =============================================================================


class TestScaffolderAgentAttributes:
    """Tests for ScaffolderAgent class attributes."""

    def test_name_property(self) -> None:
        """Test ScaffolderAgent has correct name."""
        agent = ScaffolderAgent()
        assert agent.name == "ScaffolderAgent"

    def test_model_property(self) -> None:
        """Test ScaffolderAgent has correct default model."""
        agent = ScaffolderAgent()
        assert agent.model == "claude-sonnet-4-20250514"

    def test_custom_model(self) -> None:
        """Test ScaffolderAgent accepts custom model."""
        agent = ScaffolderAgent(model="gpt-4o")
        assert agent.model == "gpt-4o"

    def test_detection_model_default(self) -> None:
        """Test ScaffolderAgent has fast detection model."""
        agent = ScaffolderAgent()
        assert agent._detection_model == "gpt-4o-mini"

    def test_custom_detection_model(self) -> None:
        """Test ScaffolderAgent accepts custom detection model."""
        agent = ScaffolderAgent(detection_model="gpt-4o")
        assert agent._detection_model == "gpt-4o"


# =============================================================================
# Registry Tests
# =============================================================================


class TestScaffolderAgentRegistry:
    """Tests for ScaffolderAgent registration in SubAgentRegistry."""

    def test_agent_registered(self) -> None:
        """Test ScaffolderAgent is registered in registry."""
        assert SubAgentType.SCAFFOLDER in SubAgentRegistry._agents

    def test_get_agent_by_type(self) -> None:
        """Test can retrieve ScaffolderAgent by type."""
        agent = SubAgentRegistry.get(SubAgentType.SCAFFOLDER, cached=False)
        assert isinstance(agent, ScaffolderAgent)


# =============================================================================
# Project Markers Tests
# =============================================================================


class TestProjectMarkers:
    """Tests for PROJECT_MARKERS covering multiple languages/frameworks."""

    def test_nodejs_markers(self) -> None:
        """Test Node.js project markers are defined."""
        assert "package.json" in PROJECT_MARKERS

    def test_python_markers(self) -> None:
        """Test Python project markers are defined."""
        assert "pyproject.toml" in PROJECT_MARKERS
        assert "setup.py" in PROJECT_MARKERS
        assert "requirements.txt" in PROJECT_MARKERS

    def test_rust_markers(self) -> None:
        """Test Rust project markers are defined."""
        assert "Cargo.toml" in PROJECT_MARKERS

    def test_go_markers(self) -> None:
        """Test Go project markers are defined."""
        assert "go.mod" in PROJECT_MARKERS

    def test_ruby_markers(self) -> None:
        """Test Ruby project markers are defined."""
        assert "Gemfile" in PROJECT_MARKERS

    def test_java_markers(self) -> None:
        """Test Java/Kotlin project markers are defined."""
        assert "pom.xml" in PROJECT_MARKERS
        assert "build.gradle" in PROJECT_MARKERS

    def test_elixir_markers(self) -> None:
        """Test Elixir project markers are defined."""
        assert "mix.exs" in PROJECT_MARKERS


# =============================================================================
# Initialization Check Tests
# =============================================================================


class TestInitializationCheck:
    """Tests for _check_initialized method."""

    def test_empty_directory_not_initialized(self) -> None:
        """Test empty directory is detected as not initialized."""
        agent = ScaffolderAgent()
        with TemporaryDirectory() as tmpdir:
            is_init, markers = agent._check_initialized(Path(tmpdir))
            assert is_init is False
            assert markers == []

    def test_nodejs_project_initialized(self) -> None:
        """Test Node.js project is detected as initialized."""
        agent = ScaffolderAgent()
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text('{"name": "test"}')
            is_init, markers = agent._check_initialized(Path(tmpdir))
            assert is_init is True
            assert "package.json" in markers

    def test_python_project_initialized(self) -> None:
        """Test Python project is detected as initialized."""
        agent = ScaffolderAgent()
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("[project]\nname = 'test'")
            is_init, markers = agent._check_initialized(Path(tmpdir))
            assert is_init is True
            assert "pyproject.toml" in markers

    def test_rust_project_initialized(self) -> None:
        """Test Rust project is detected as initialized."""
        agent = ScaffolderAgent()
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Cargo.toml").write_text('[package]\nname = "test"')
            is_init, markers = agent._check_initialized(Path(tmpdir))
            assert is_init is True
            assert "Cargo.toml" in markers

    def test_go_project_initialized(self) -> None:
        """Test Go project is detected as initialized."""
        agent = ScaffolderAgent()
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "go.mod").write_text("module test")
            is_init, markers = agent._check_initialized(Path(tmpdir))
            assert is_init is True
            assert "go.mod" in markers

    def test_multiple_markers_found(self) -> None:
        """Test multiple markers can be detected."""
        agent = ScaffolderAgent()
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text("{}")
            (Path(tmpdir) / "requirements.txt").write_text("")
            is_init, markers = agent._check_initialized(Path(tmpdir))
            assert is_init is True
            assert len(markers) == 2

    def test_nonexistent_directory(self) -> None:
        """Test nonexistent directory returns not initialized."""
        agent = ScaffolderAgent()
        is_init, markers = agent._check_initialized(Path("/nonexistent/path"))
        assert is_init is False
        assert markers == []


# =============================================================================
# Prompt Tests
# =============================================================================


class TestPrompts:
    """Tests for prompt templates."""

    def test_detect_prompt_is_short(self) -> None:
        """Test detection prompt is concise."""
        # Should be under 500 chars (excluding variable placeholders)
        base_prompt = DETECT_PROJECT_TYPE_PROMPT.replace("{roadmap_content}", "")
        assert len(base_prompt) < 500

    def test_setup_prompt_is_generic(self) -> None:
        """Test setup prompt works for any framework."""
        # Should mention forbidden commands
        assert "FORBIDDEN" in SETUP_PROJECT_PROMPT or "DO NOT" in SETUP_PROJECT_PROMPT
        # Should contain generic guidance
        assert "cat >" in SETUP_PROJECT_PROMPT
        assert "mkdir -p" in SETUP_PROJECT_PROMPT

    def test_setup_prompt_has_critical_rules(self) -> None:
        """Test setup prompt includes critical rules."""
        assert "FORBIDDEN COMMANDS" in SETUP_PROJECT_PROMPT
        assert "create-next-app" in SETUP_PROJECT_PROMPT
        assert "subdirectory" in SETUP_PROJECT_PROMPT.lower()


# =============================================================================
# File Analysis Tests
# =============================================================================


class TestFileAnalysis:
    """Tests for file analysis from command history."""

    def test_analyze_file_changes_cat(self) -> None:
        """Test analyzing files from cat commands."""
        agent = ScaffolderAgent()

        class MockEntry:
            def __init__(self, command: str):
                self.command = command

        history = [
            MockEntry("cat > package.json << 'EOF'"),
            MockEntry("cat > tsconfig.json << 'EOF'"),
            MockEntry("mkdir -p src/app"),
        ]

        files_created, files_modified = agent._analyze_file_changes(history)
        assert "package.json" in files_created
        assert "tsconfig.json" in files_created
        assert len(files_created) == 2

    def test_analyze_file_changes_touch(self) -> None:
        """Test analyzing files from touch commands."""
        agent = ScaffolderAgent()

        class MockEntry:
            def __init__(self, command: str):
                self.command = command

        history = [
            MockEntry("touch README.md"),
            MockEntry("touch src/__init__.py"),
        ]

        files_created, files_modified = agent._analyze_file_changes(history)
        assert "README.md" in files_created
        assert "src/__init__.py" in files_created

    def test_empty_history(self) -> None:
        """Test analyzing empty command history."""
        agent = ScaffolderAgent()
        files_created, files_modified = agent._analyze_file_changes([])
        assert files_created == []
        assert files_modified == []
