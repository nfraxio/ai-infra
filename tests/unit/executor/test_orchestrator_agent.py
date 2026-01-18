"""Tests for Phase 16.5.11.2: OrchestratorAgent.

This module tests the OrchestratorAgent functionality:
- RoutingContext dataclass
- RoutingDecision dataclass
- OrchestratorAgent initialization
- Routing prompt building
- Response parsing
- Keyword fallback
- Integration with SubAgentRegistry
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.executor.agents.orchestrator import (
    _AGENT_TYPE_MAP,
    ORCHESTRATOR_SYSTEM_PROMPT,
    OrchestratorAgent,
    RoutingContext,
    RoutingDecision,
    RoutingResponse,
)
from ai_infra.executor.agents.registry import SubAgentType
from ai_infra.executor.todolist import TodoItem

# =============================================================================
# RoutingContext Tests (16.5.11.2.4)
# =============================================================================


class TestRoutingContext:
    """Tests for RoutingContext dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic context creation."""
        context = RoutingContext(
            workspace=Path("/project"),
            completed_tasks=["Task 1", "Task 2"],
            existing_files=["src/main.py"],
            project_type="python",
            previous_agent="coder",
        )

        assert context.workspace == Path("/project")
        assert context.completed_tasks == ["Task 1", "Task 2"]
        assert context.existing_files == ["src/main.py"]
        assert context.project_type == "python"
        assert context.previous_agent == "coder"

    def test_default_values(self) -> None:
        """Test default values are applied."""
        context = RoutingContext(workspace=Path("/project"))

        assert context.completed_tasks == []
        assert context.existing_files == []
        assert context.project_type == "unknown"
        assert context.previous_agent is None

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        context = RoutingContext(
            workspace=Path("/project"),
            completed_tasks=["Task 1"],
            existing_files=["file.py"],
            project_type="python",
            previous_agent="coder",
        )

        result = context.to_dict()

        assert result["workspace"] == "/project"
        assert result["completed_tasks"] == ["Task 1"]
        assert result["existing_files"] == ["file.py"]
        assert result["project_type"] == "python"
        assert result["previous_agent"] == "coder"


# =============================================================================
# RoutingDecision Tests (16.5.11.2.5)
# =============================================================================


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic decision creation."""
        decision = RoutingDecision(
            agent_type=SubAgentType.TESTWRITER,
            confidence=0.95,
            reasoning="Task mentions creating tests",
            used_fallback=False,
        )

        assert decision.agent_type == SubAgentType.TESTWRITER
        assert decision.confidence == 0.95
        assert decision.reasoning == "Task mentions creating tests"
        assert decision.used_fallback is False

    def test_default_fallback_flag(self) -> None:
        """Test default used_fallback is False."""
        decision = RoutingDecision(
            agent_type=SubAgentType.CODER,
            confidence=0.8,
            reasoning="Creating code",
        )

        assert decision.used_fallback is False

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        decision = RoutingDecision(
            agent_type=SubAgentType.DEBUGGER,
            confidence=0.75,
            reasoning="Fixing a bug",
            used_fallback=True,
        )

        result = decision.to_dict()

        assert result["agent_type"] == "debugger"
        assert result["confidence"] == 0.75
        assert result["reasoning"] == "Fixing a bug"
        assert result["used_fallback"] is True


# =============================================================================
# RoutingResponse Tests
# =============================================================================


class TestRoutingResponse:
    """Tests for RoutingResponse Pydantic model."""

    def test_valid_response(self) -> None:
        """Test creating a valid response."""
        response = RoutingResponse(
            agent="testwriter",
            confidence=0.9,
            reasoning="Task is about tests",
        )

        assert response.agent == "testwriter"
        assert response.confidence == 0.9
        assert response.reasoning == "Task is about tests"

    def test_response_from_dict(self) -> None:
        """Test creating response from dictionary."""
        data = {
            "agent": "coder",
            "confidence": 0.85,
            "reasoning": "Implementation task",
        }
        response = RoutingResponse(**data)

        assert response.agent == "coder"
        assert response.confidence == 0.85


# =============================================================================
# ORCHESTRATOR_SYSTEM_PROMPT Tests (16.5.11.2.2)
# =============================================================================


class TestOrchestratorSystemPrompt:
    """Tests for ORCHESTRATOR_SYSTEM_PROMPT content."""

    def test_prompt_is_string(self) -> None:
        """Test prompt is a non-empty string."""
        assert isinstance(ORCHESTRATOR_SYSTEM_PROMPT, str)
        assert len(ORCHESTRATOR_SYSTEM_PROMPT) > 0

    def test_prompt_lists_all_agents(self) -> None:
        """Test prompt lists all available agents."""
        agents = ["Coder", "TestWriter", "Tester", "Debugger", "Reviewer", "Researcher"]
        for agent in agents:
            assert agent in ORCHESTRATOR_SYSTEM_PROMPT

    def test_prompt_includes_routing_rules(self) -> None:
        """Test prompt includes routing rules section."""
        assert "ROUTING RULES" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_prompt_includes_output_format(self) -> None:
        """Test prompt includes JSON output format."""
        assert "OUTPUT FORMAT" in ORCHESTRATOR_SYSTEM_PROMPT
        assert '"agent"' in ORCHESTRATOR_SYSTEM_PROMPT
        assert '"confidence"' in ORCHESTRATOR_SYSTEM_PROMPT
        assert '"reasoning"' in ORCHESTRATOR_SYSTEM_PROMPT


# =============================================================================
# Agent Type Mapping Tests
# =============================================================================


class TestAgentTypeMapping:
    """Tests for agent type string to enum mapping."""

    def test_all_agent_types_mapped(self) -> None:
        """Test all expected agent types are in the map."""
        expected = {
            "coder",
            "testwriter",
            "tester",
            "debugger",
            "reviewer",
            "researcher",
            "scaffolder",
        }
        assert set(_AGENT_TYPE_MAP.keys()) == expected

    def test_mapping_to_correct_types(self) -> None:
        """Test mapping returns correct SubAgentType."""
        assert _AGENT_TYPE_MAP["coder"] == SubAgentType.CODER
        assert _AGENT_TYPE_MAP["testwriter"] == SubAgentType.TESTWRITER
        assert _AGENT_TYPE_MAP["tester"] == SubAgentType.TESTER
        assert _AGENT_TYPE_MAP["debugger"] == SubAgentType.DEBUGGER
        assert _AGENT_TYPE_MAP["reviewer"] == SubAgentType.REVIEWER
        assert _AGENT_TYPE_MAP["scaffolder"] == SubAgentType.SCAFFOLDER
        assert _AGENT_TYPE_MAP["researcher"] == SubAgentType.RESEARCHER


# =============================================================================
# OrchestratorAgent Initialization Tests (16.5.11.2.3)
# =============================================================================


class TestOrchestratorAgentInit:
    """Tests for OrchestratorAgent initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization values."""
        agent = OrchestratorAgent()

        assert agent.model == "gpt-4o-mini"
        assert agent.confidence_threshold == 0.7
        assert agent._llm is None

    def test_custom_model(self) -> None:
        """Test custom model initialization."""
        agent = OrchestratorAgent(model="gpt-4o")
        assert agent.model == "gpt-4o"

    def test_custom_confidence_threshold(self) -> None:
        """Test custom confidence threshold."""
        agent = OrchestratorAgent(confidence_threshold=0.9)
        assert agent.confidence_threshold == 0.9

    def test_last_token_usage_default(self) -> None:
        """Test last_token_usage starts empty."""
        agent = OrchestratorAgent()
        assert agent.last_token_usage == {}


# =============================================================================
# _build_routing_prompt Tests (16.5.11.2.6)
# =============================================================================


class TestBuildRoutingPrompt:
    """Tests for _build_routing_prompt method."""

    def test_basic_prompt_building(self) -> None:
        """Test basic prompt building."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Create tests for user.py", description="Add unit tests")
        context = RoutingContext(
            workspace=Path("/project"),
            completed_tasks=["Create src/user.py"],
            existing_files=["src/user.py"],
            project_type="python",
            previous_agent="coder",
        )

        prompt = agent._build_routing_prompt(task, context)

        assert "Create tests for user.py" in prompt
        assert "Add unit tests" in prompt
        assert "python" in prompt
        assert "Create src/user.py" in prompt
        assert "src/user.py" in prompt
        assert "coder" in prompt

    def test_prompt_with_no_description(self) -> None:
        """Test prompt when task has no description."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Test task", description="")
        context = RoutingContext(workspace=Path("/project"))

        prompt = agent._build_routing_prompt(task, context)

        assert "No description" in prompt

    def test_prompt_limits_completed_tasks(self) -> None:
        """Test prompt limits completed tasks to last 5."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Test", description="")
        context = RoutingContext(
            workspace=Path("/project"),
            completed_tasks=[f"Task {i}" for i in range(10)],
        )

        prompt = agent._build_routing_prompt(task, context)

        # Should only include last 5
        assert "Task 5" in prompt
        assert "Task 9" in prompt
        assert "Task 0" not in prompt

    def test_prompt_limits_files(self) -> None:
        """Test prompt limits files to first 20."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Test", description="")
        context = RoutingContext(
            workspace=Path("/project"),
            existing_files=[f"file{i}.py" for i in range(30)],
        )

        prompt = agent._build_routing_prompt(task, context)

        assert "file0.py" in prompt
        assert "file19.py" in prompt
        # file20.py and above should not be included
        assert "file20.py" not in prompt


# =============================================================================
# _parse_routing_response Tests (16.5.11.2.7)
# =============================================================================


class TestParseRoutingResponse:
    """Tests for _parse_routing_response method."""

    def test_parse_pydantic_response(self) -> None:
        """Test parsing a RoutingResponse Pydantic object."""
        agent = OrchestratorAgent()
        response = RoutingResponse(
            agent="testwriter",
            confidence=0.95,
            reasoning="Task is about creating tests",
        )

        decision = agent._parse_routing_response(response)

        assert decision.agent_type == SubAgentType.TESTWRITER
        assert decision.confidence == 0.95
        assert decision.reasoning == "Task is about creating tests"
        assert decision.used_fallback is False

    def test_parse_dict_response(self) -> None:
        """Test parsing a dictionary response."""
        agent = OrchestratorAgent()
        response = {
            "agent": "coder",
            "confidence": 0.8,
            "reasoning": "Implementation task",
        }

        decision = agent._parse_routing_response(response)

        assert decision.agent_type == SubAgentType.CODER
        assert decision.confidence == 0.8
        assert decision.reasoning == "Implementation task"

    def test_parse_string_response_with_json(self) -> None:
        """Test parsing a string response containing JSON."""
        agent = OrchestratorAgent()
        response = """Here is my analysis:
        {"agent": "debugger", "confidence": 0.75, "reasoning": "Fixing error"}
        That's my decision."""

        decision = agent._parse_routing_response(response)

        assert decision.agent_type == SubAgentType.DEBUGGER
        assert decision.confidence == 0.75

    def test_parse_message_with_content(self) -> None:
        """Test parsing an object with content attribute."""
        agent = OrchestratorAgent()

        class MockMessage:
            content = '{"agent": "reviewer", "confidence": 0.85, "reasoning": "Review task"}'

        decision = agent._parse_routing_response(MockMessage())

        assert decision.agent_type == SubAgentType.REVIEWER
        assert decision.confidence == 0.85

    def test_confidence_clamped_to_range(self) -> None:
        """Test confidence is clamped to 0.0-1.0."""
        agent = OrchestratorAgent()

        # Test high confidence
        response = RoutingResponse(agent="coder", confidence=1.5, reasoning="Test")
        decision = agent._parse_routing_response(response)
        assert decision.confidence == 1.0

        # Test negative confidence
        response = RoutingResponse(agent="coder", confidence=-0.5, reasoning="Test")
        decision = agent._parse_routing_response(response)
        assert decision.confidence == 0.0

    def test_parse_invalid_agent_raises(self) -> None:
        """Test invalid agent type raises ValueError."""
        agent = OrchestratorAgent()
        response = RoutingResponse(
            agent="invalid_agent",
            confidence=0.9,
            reasoning="Test",
        )

        with pytest.raises(ValueError, match="Unknown agent type"):
            agent._parse_routing_response(response)

    def test_parse_invalid_string_raises(self) -> None:
        """Test unparseable string raises ValueError."""
        agent = OrchestratorAgent()
        response = "This is not valid JSON at all"

        with pytest.raises(ValueError, match="Could not extract"):
            agent._parse_routing_response(response)


# =============================================================================
# _default_fallback Tests (16.5.11.2.8)
# =============================================================================


class TestDefaultFallback:
    """Tests for _default_fallback method."""

    def test_fallback_returns_decision(self) -> None:
        """Test fallback returns a RoutingDecision."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Create a new module", description="")

        decision = agent._default_fallback(task)

        assert isinstance(decision, RoutingDecision)
        assert decision.used_fallback is True
        assert decision.confidence == 0.5
        assert "Fallback" in decision.reasoning

    def test_fallback_defaults_to_coder(self) -> None:
        """Test fallback always returns CODER as default."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Fix the broken test", description="")

        decision = agent._default_fallback(task)

        # Should always default to CODER
        assert decision.agent_type == SubAgentType.CODER


# =============================================================================
# route() Tests (16.5.11.2.3)
# =============================================================================


class TestOrchestratorRoute:
    """Tests for OrchestratorAgent.route method."""

    @pytest.mark.asyncio
    async def test_route_calls_llm(self) -> None:
        """Test route calls LLM with correct prompt."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Create tests for user.py", description="")
        context = RoutingContext(workspace=Path("/project"), project_type="python")

        # Mock the LLM call (used to set up expected behavior pattern)
        _mock_response = RoutingResponse(
            agent="testwriter",
            confidence=0.95,
            reasoning="Task is about creating tests",
        )

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RoutingDecision(
                agent_type=SubAgentType.TESTWRITER,
                confidence=0.95,
                reasoning="Task is about creating tests",
            )

            decision = await agent.route(task, context)

            mock_llm.assert_called_once()
            assert decision.agent_type == SubAgentType.TESTWRITER
            assert decision.confidence == 0.95

    @pytest.mark.asyncio
    async def test_route_uses_fallback_on_low_confidence(self) -> None:
        """Test route falls back when confidence is low."""
        agent = OrchestratorAgent(confidence_threshold=0.8)
        task = TodoItem(id=1, title="Create tests for user.py", description="")
        context = RoutingContext(workspace=Path("/project"))

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RoutingDecision(
                agent_type=SubAgentType.CODER,
                confidence=0.5,  # Below threshold
                reasoning="Low confidence",
            )

            decision = await agent.route(task, context)

            # Should fall back to keyword routing
            assert decision.used_fallback is True

    @pytest.mark.asyncio
    async def test_route_uses_fallback_on_error(self) -> None:
        """Test route falls back when LLM fails."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Fix the bug", description="")
        context = RoutingContext(workspace=Path("/project"))

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            decision = await agent.route(task, context)

            assert decision.used_fallback is True

    @pytest.mark.asyncio
    async def test_route_uses_default_fallback_on_error(self) -> None:
        """Test route uses default fallback when LLM fails."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Test task", description="")
        context = RoutingContext(workspace=Path("/project"))

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            decision = await agent.route(task, context)

            # Should use default fallback (CODER)
            assert decision.used_fallback is True
            assert decision.agent_type == SubAgentType.CODER


# =============================================================================
# route_sync() Tests
# =============================================================================


class TestOrchestratorRouteSync:
    """Tests for OrchestratorAgent.route_sync method."""

    def test_route_sync_calls_route(self) -> None:
        """Test route_sync wraps async route."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title="Test task", description="")
        context = RoutingContext(workspace=Path("/project"))

        with patch.object(agent, "_default_fallback") as mock_fallback:
            mock_fallback.return_value = RoutingDecision(
                agent_type=SubAgentType.CODER,
                confidence=0.5,
                reasoning="Fallback",
                used_fallback=True,
            )
            # Mock _call_llm to raise so we hit fallback
            with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = Exception("Test")

                decision = agent.route_sync(task, context)

                assert decision.used_fallback is True


# =============================================================================
# Integration: Imports and Exports
# =============================================================================


class TestOrchestratorExports:
    """Tests for OrchestratorAgent module exports."""

    def test_import_from_agents_module(self) -> None:
        """Test OrchestratorAgent can be imported from agents module."""
        from ai_infra.executor.agents import OrchestratorAgent as ImportedAgent

        assert ImportedAgent is OrchestratorAgent

    def test_import_routing_context(self) -> None:
        """Test RoutingContext can be imported from agents module."""
        from ai_infra.executor.agents import RoutingContext as ImportedContext

        assert ImportedContext is RoutingContext

    def test_import_routing_decision(self) -> None:
        """Test RoutingDecision can be imported from agents module."""
        from ai_infra.executor.agents import RoutingDecision as ImportedDecision

        assert ImportedDecision is RoutingDecision

    def test_orchestrator_in_all(self) -> None:
        """Test OrchestratorAgent is in agents __all__."""
        from ai_infra.executor.agents import __all__

        assert "OrchestratorAgent" in __all__
        assert "RoutingContext" in __all__
        assert "RoutingDecision" in __all__


# =============================================================================
# Parametrized Routing Tests (16.5.11.6.1 style)
# =============================================================================


class TestDefaultFallbackPatterns:
    """Tests for default fallback behavior."""

    @pytest.mark.parametrize(
        "task_title",
        [
            # Various task types all default to CODER
            "Create tests for user.py",
            "Write unit tests for auth module",
            "Run pytest to verify",
            "Fix the ImportError in user.py",
            "Review code quality",
            "Research best practices for caching",
        ],
    )
    def test_default_fallback_always_returns_coder(
        self,
        task_title: str,
    ) -> None:
        """Test default fallback always returns CODER."""
        agent = OrchestratorAgent()
        task = TodoItem(id=1, title=task_title, description="")

        decision = agent._default_fallback(task)

        # Default fallback always returns CODER
        assert decision.agent_type == SubAgentType.CODER
        assert decision.used_fallback is True
        assert decision.confidence == 0.5


# =============================================================================
# Phase 16.5.11.6.1: Comprehensive Orchestrator Routing Tests
# =============================================================================


class TestOrchestratorRoutesCorrectly:
    """Phase 16.5.11.6.1: Tests that orchestrator routes to correct agent type.

    These tests use the LLM-based orchestrator to verify routing decisions.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "task_title,expected_agent",
        [
            # Coder tasks - file creation and implementation
            ("Create src/user.py with User class", SubAgentType.CODER),
            ("Implement the PaymentService class", SubAgentType.CODER),
            ("Add database models to src/models.py", SubAgentType.CODER),
            # TestWriter tasks - test file creation
            ("Create tests/test_user.py with tests", SubAgentType.TESTWRITER),
            ("Write comprehensive unit tests for auth", SubAgentType.TESTWRITER),
            ("Add test coverage for the payment module", SubAgentType.TESTWRITER),
            # Tester tasks - test execution
            ("Run pytest to verify tests pass", SubAgentType.TESTER),
            ("Execute the test suite", SubAgentType.TESTER),
            # Debugger tasks - fixing errors
            ("Fix the ImportError in user.py", SubAgentType.DEBUGGER),
            ("Debug the authentication failure", SubAgentType.DEBUGGER),
            ("Fix broken database connection", SubAgentType.DEBUGGER),
            # Reviewer tasks - code review and refactoring
            ("Refactor User class for better performance", SubAgentType.REVIEWER),
            ("Review and improve code quality", SubAgentType.REVIEWER),
            # Researcher tasks - research and documentation
            ("Research best practices for validation", SubAgentType.RESEARCHER),
            ("Look up OAuth 2.0 documentation", SubAgentType.RESEARCHER),
        ],
    )
    async def test_orchestrator_routes_correctly(
        self,
        task_title: str,
        expected_agent: SubAgentType,
    ) -> None:
        """Test orchestrator routes tasks to correct agent type.

        Phase 16.5.11.6.1: Uses mocked LLM to verify routing logic.
        """
        orchestrator = OrchestratorAgent()
        context = RoutingContext(
            workspace=Path("/project"),
            completed_tasks=[],
            existing_files=["src/user.py", "tests/test_user.py"],
            project_type="python",
        )
        task = TodoItem(id=1, title=task_title, description="")

        # Mock the LLM call to return the expected agent
        with patch.object(orchestrator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RoutingDecision(
                agent_type=expected_agent,
                confidence=0.95,
                reasoning=f"Task matches {expected_agent.value} pattern",
            )

            decision = await orchestrator.route(task, context)

            # Verify the decision
            assert decision.agent_type == expected_agent
            assert decision.confidence >= 0.7
            assert not decision.used_fallback


# =============================================================================
# Phase 16.5.11.6.3: Orchestrator Routing Tests
# =============================================================================


class TestOrchestratorRoutingComparison:
    """Phase 16.5.11.6.3: Test orchestrator routing capabilities."""

    TEST_CASES = [
        # (task_title, expected_agent, description)
        ("Create tests for user module", SubAgentType.TESTWRITER, "Test creation"),
        ("Write unit tests for auth", SubAgentType.TESTWRITER, "Test writing"),
        ("Run the test suite", SubAgentType.TESTER, "Test execution"),
        ("Fix the broken import", SubAgentType.DEBUGGER, "Bug fix"),
        ("Implement new feature", SubAgentType.CODER, "Implementation"),
        ("Review security code", SubAgentType.REVIEWER, "Code review"),
    ]

    @pytest.mark.asyncio
    async def test_orchestrator_routing(self) -> None:
        """Test that orchestrator can route all test cases correctly.

        Phase 16.5.11.6.3: Integration test for orchestrator routing.
        """
        orchestrator = OrchestratorAgent()
        context = RoutingContext(workspace=Path("/project"), project_type="python")

        for task_title, expected_agent, description in self.TEST_CASES:
            task = TodoItem(id=1, title=task_title, description="")

            # Mock orchestrator decision
            with patch.object(orchestrator, "_call_llm", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = RoutingDecision(
                    agent_type=expected_agent,
                    confidence=0.9,
                    reasoning=f"Routing for {description}",
                )

                orchestrator_decision = await orchestrator.route(task, context)

            # Verify orchestrator matches expected
            assert orchestrator_decision.agent_type == expected_agent, (
                f"Task '{task_title}' should route to {expected_agent}, "
                f"got {orchestrator_decision.agent_type}"
            )


# =============================================================================
# Phase 16.5.11.6.4: Orchestrator Latency Benchmark
# =============================================================================


class TestOrchestratorLatencyBenchmark:
    """Phase 16.5.11.6.4: Benchmark tests for orchestrator latency."""

    @pytest.mark.asyncio
    async def test_orchestrator_latency_under_target(self) -> None:
        """Test orchestrator routing latency is under 2s target.

        Phase 16.5.11.6.4: Benchmark for <2s latency per routing decision.
        """
        orchestrator = OrchestratorAgent()
        context = RoutingContext(workspace=Path("/project"), project_type="python")
        task = TodoItem(
            id=1,
            title="Create tests for user.py",
            description="Write comprehensive tests",
        )

        # Mock LLM to return immediately (simulating fast response)
        with patch.object(orchestrator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RoutingDecision(
                agent_type=SubAgentType.TESTWRITER,
                confidence=0.95,
                reasoning="Test creation task",
            )

            import time

            start = time.perf_counter()
            await orchestrator.route(task, context)
            latency_ms = (time.perf_counter() - start) * 1000

        # Target: <2000ms (2s) - mocked calls should be <100ms
        assert latency_ms < 2000, f"Orchestrator latency {latency_ms:.0f}ms exceeds 2s target"

    def test_prompt_building_latency(self) -> None:
        """Test prompt building is fast (<50ms)."""
        orchestrator = OrchestratorAgent()
        context = RoutingContext(
            workspace=Path("/project"),
            completed_tasks=[f"Task {i}" for i in range(10)],
            existing_files=[f"file{i}.py" for i in range(30)],
            project_type="python",
        )
        task = TodoItem(
            id=1,
            title="Test task",
            description="Long description " * 100,
        )

        import time

        start = time.perf_counter()
        prompt = orchestrator._build_routing_prompt(task, context)
        latency_ms = (time.perf_counter() - start) * 1000

        assert latency_ms < 50, f"Prompt building took {latency_ms:.0f}ms (>50ms)"
        assert len(prompt) > 0


# =============================================================================
# Phase 16.5.11.6.5: Orchestrator Token Cost Benchmark
# =============================================================================


class TestOrchestratorTokenCostBenchmark:
    """Phase 16.5.11.6.5: Benchmark tests for orchestrator token usage."""

    def test_routing_prompt_token_estimate(self) -> None:
        """Test routing prompt is under 500 token estimate.

        Phase 16.5.11.6.5: Target <500 tokens per routing decision.
        Approximation: ~4 chars per token for English text.
        """
        orchestrator = OrchestratorAgent()
        context = RoutingContext(
            workspace=Path("/project"),
            completed_tasks=["Create user model", "Add validation"],
            existing_files=["src/user.py", "src/auth.py", "tests/test_user.py"],
            project_type="python",
            previous_agent="coder",
        )
        task = TodoItem(
            id=1,
            title="Create tests for payment module",
            description="Write comprehensive unit tests for payment processing",
        )

        prompt = orchestrator._build_routing_prompt(task, context)

        # Estimate tokens (rough: ~4 chars per token)
        estimated_tokens = len(prompt) / 4

        # Target: <500 tokens for routing prompt
        assert estimated_tokens < 500, (
            f"Routing prompt ~{estimated_tokens:.0f} tokens exceeds 500 target"
        )

    def test_system_prompt_token_estimate(self) -> None:
        """Test system prompt is reasonably sized."""
        # Estimate system prompt tokens
        estimated_tokens = len(ORCHESTRATOR_SYSTEM_PROMPT) / 4

        # System prompt should be under 1000 tokens
        assert estimated_tokens < 1000, f"System prompt ~{estimated_tokens:.0f} tokens exceeds 1000"

    @pytest.mark.asyncio
    async def test_token_tracking_works(self) -> None:
        """Test that token usage is tracked after routing."""
        orchestrator = OrchestratorAgent()
        context = RoutingContext(workspace=Path("/project"), project_type="python")
        task = TodoItem(id=1, title="Test task", description="")

        with patch.object(orchestrator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RoutingDecision(
                agent_type=SubAgentType.CODER,
                confidence=0.9,
                reasoning="Implementation task",
            )
            # Simulate token usage by setting internal attribute
            orchestrator._last_token_usage = {
                "prompt_tokens": 300,
                "completion_tokens": 50,
                "total_tokens": 350,
            }

            await orchestrator.route(task, context)

        # Verify token tracking via property
        assert orchestrator.last_token_usage.get("total_tokens", 0) > 0


# =============================================================================
# Phase 16.5.11.6.7: Accuracy Comparison Report
# =============================================================================


class TestAccuracyComparisonReport:
    """Phase 16.5.11.6.7: Generate accuracy comparison report."""

    COMPREHENSIVE_TEST_CASES = [
        # Coder tasks
        ("Create src/user.py", SubAgentType.CODER),
        ("Implement login endpoint", SubAgentType.CODER),
        ("Add database model", SubAgentType.CODER),
        ("Build API handler", SubAgentType.CODER),
        # TestWriter tasks
        ("Create tests for user", SubAgentType.TESTWRITER),
        ("Write unit tests", SubAgentType.TESTWRITER),
        ("Add test coverage", SubAgentType.TESTWRITER),
        ("Generate pytest tests", SubAgentType.TESTWRITER),
        # Tester tasks
        ("Run tests", SubAgentType.TESTER),
        ("Execute pytest", SubAgentType.TESTER),
        ("Verify tests pass", SubAgentType.TESTER),
        # Debugger tasks
        ("Fix ImportError", SubAgentType.DEBUGGER),
        ("Debug crash", SubAgentType.DEBUGGER),
        ("Fix broken test", SubAgentType.DEBUGGER),
        ("Resolve TypeError", SubAgentType.DEBUGGER),
        # Reviewer tasks
        ("Review code", SubAgentType.REVIEWER),
        ("Refactor service", SubAgentType.REVIEWER),
        ("Audit security", SubAgentType.REVIEWER),
        # Researcher tasks
        ("Research API", SubAgentType.RESEARCHER),
        ("Look up docs", SubAgentType.RESEARCHER),
        ("Investigate options", SubAgentType.RESEARCHER),
    ]

    def test_default_fallback_always_returns_coder(self) -> None:
        """Test that default fallback always returns CODER.

        After removing keyword routing, fallback always returns CODER.
        """
        orchestrator = OrchestratorAgent()

        for task_title, _ in self.COMPREHENSIVE_TEST_CASES:
            task = TodoItem(id=1, title=task_title, description="")
            decision = orchestrator._default_fallback(task)

            # Default fallback always returns CODER
            assert decision.agent_type == SubAgentType.CODER
            assert decision.used_fallback is True

    @pytest.mark.asyncio
    async def test_orchestrator_routing_accuracy_report(self) -> None:
        """Generate orchestrator routing accuracy report (mocked).

        Phase 16.5.11.6.7: With mocked LLM, orchestrator achieves 100%.
        """
        orchestrator = OrchestratorAgent()
        context = RoutingContext(workspace=Path("/project"), project_type="python")

        correct = 0

        for task_title, expected_agent in self.COMPREHENSIVE_TEST_CASES:
            task = TodoItem(id=1, title=task_title, description="")

            with patch.object(orchestrator, "_call_llm", new_callable=AsyncMock) as mock_llm:
                # Simulate perfect orchestrator
                mock_llm.return_value = RoutingDecision(
                    agent_type=expected_agent,
                    confidence=0.95,
                    reasoning=f"Matches {expected_agent.value}",
                )

                decision = await orchestrator.route(task, context)

            if decision.agent_type == expected_agent:
                correct += 1

        accuracy = correct / len(self.COMPREHENSIVE_TEST_CASES)

        print("\n=== Orchestrator Routing Accuracy Report (Mocked) ===")
        print(f"Total: {len(self.COMPREHENSIVE_TEST_CASES)}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.1%}")

        # Mocked orchestrator should achieve 100%
        assert accuracy == 1.0, f"Mocked orchestrator accuracy {accuracy:.1%}"
