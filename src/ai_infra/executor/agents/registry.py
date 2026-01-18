"""Subagent registry for specialized agent selection.

Phase 3.3.1 of EXECUTOR_1.md: Provides a registry for specialized subagents
that can be selected based on task type.

The registry pattern allows:
- Registration of specialized agents via decorator
- Selection of agents by type
- Extensibility for new agent types

Task routing is handled by OrchestratorAgent (see orchestrator.py).

Example:
    ```python
    from ai_infra.executor.agents import SubAgentRegistry, SubAgentType

    # Get a specific agent type
    coder = SubAgentRegistry.get(SubAgentType.CODER)

    # Use OrchestratorAgent for task routing
    from ai_infra.executor.agents.orchestrator import OrchestratorAgent
    orchestrator = OrchestratorAgent()
    decision = await orchestrator.route(task, context)
    ```
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ai_infra.executor.agents.base import SubAgent

__all__ = [
    "SubAgentRegistry",
    "SubAgentType",
]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="SubAgent")


class SubAgentType(str, Enum):
    """Types of specialized subagents.

    Each type represents a different role in the development workflow:
    - SCAFFOLDER: Sets up new project structure based on tech stack
    - CODER: Writes and edits code
    - REVIEWER: Reviews code for bugs and style
    - TESTER: Discovers and runs tests
    - DEBUGGER: Analyzes and fixes failures
    - RESEARCHER: Searches for information
    - TESTWRITER: Creates comprehensive test suites
    """

    SCAFFOLDER = "scaffolder"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DEBUGGER = "debugger"
    RESEARCHER = "researcher"
    TESTWRITER = "testwriter"


class SubAgentRegistry:
    """Registry of specialized subagents.

    Provides registration and selection of specialized agents by type.
    Agents can be registered via decorator or direct registration.

    Example:
        ```python
        @SubAgentRegistry.register(SubAgentType.CODER)
        class CoderAgent(SubAgent):
            ...

        # Get agent by type
        agent = SubAgentRegistry.get(SubAgentType.CODER)
        ```
    """

    _agents: dict[SubAgentType, type[SubAgent]] = {}
    _instances: dict[SubAgentType, SubAgent] = {}

    @classmethod
    def register(
        cls,
        agent_type: SubAgentType,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a subagent class.

        Args:
            agent_type: The type of agent being registered.

        Returns:
            Decorator function that registers the agent class.

        Example:
            ```python
            @SubAgentRegistry.register(SubAgentType.CODER)
            class CoderAgent(SubAgent):
                ...
            ```
        """

        def decorator(agent_class: type[T]) -> type[T]:
            cls._agents[agent_type] = agent_class
            logger.debug(f"Registered subagent: {agent_type.value} -> {agent_class.__name__}")
            return agent_class

        return decorator

    @classmethod
    def register_class(
        cls,
        agent_type: SubAgentType,
        agent_class: type[SubAgent],
    ) -> None:
        """Directly register a subagent class.

        Args:
            agent_type: The type of agent being registered.
            agent_class: The agent class to register.
        """
        cls._agents[agent_type] = agent_class
        logger.debug(f"Registered subagent: {agent_type.value} -> {agent_class.__name__}")

    @classmethod
    def get(
        cls,
        agent_type: SubAgentType,
        *,
        model: str | None = None,
        cached: bool = True,
    ) -> SubAgent:
        """Get a subagent instance by type.

        Args:
            agent_type: The type of agent to get.
            model: Optional model name override. If provided, creates new instance.
            cached: If True, return cached instance. If False, create new.
                Note: Caching is disabled when model is provided.

        Returns:
            SubAgent instance.

        Raises:
            ValueError: If no agent is registered for the type.
        """
        if agent_type not in cls._agents:
            raise ValueError(
                f"No subagent registered for type: {agent_type.value}. "
                f"Available types: {[t.value for t in cls._agents.keys()]}"
            )

        # When model is specified, always create new instance (no caching)
        if model is not None:
            logger.debug(f"Creating {agent_type.value} agent with custom model: {model}")
            return cls._agents[agent_type](model=model)

        if cached and agent_type in cls._instances:
            return cls._instances[agent_type]

        instance = cls._agents[agent_type]()
        if cached:
            cls._instances[agent_type] = instance

        return instance

    @classmethod
    def available_types(cls) -> list[SubAgentType]:
        """Get list of registered agent types.

        Returns:
            List of SubAgentType values that have registered agents.
        """
        return list(cls._agents.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents (for testing)."""
        cls._agents.clear()
        cls._instances.clear()

    @classmethod
    def reset_instances(cls) -> None:
        """Clear cached instances but keep registrations."""
        cls._instances.clear()
