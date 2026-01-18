"""Orchestrator agent for intelligent task routing.

Phase 16.5.11.2 of EXECUTOR_5.md: Replaces keyword-based routing with
LLM-based semantic understanding for better task classification.

The OrchestratorAgent analyzes task semantics and context to route
tasks to the most appropriate specialist agent:
- Coder: Implements application code
- TestWriter: Creates test files
- Tester: Runs existing tests
- Debugger: Fixes bugs and errors
- Reviewer: Reviews and refactors code
- Researcher: Gathers information

Example:
    ```python
    from ai_infra.executor.agents.orchestrator import (
        OrchestratorAgent,
        RoutingContext,
    )

    orchestrator = OrchestratorAgent()
    context = RoutingContext(
        workspace=Path("/project"),
        completed_tasks=["Create src/user.py"],
        existing_files=["src/user.py"],
        project_type="python",
    )
    task = TodoItem(id=1, title="Create tests for user.py")
    decision = await orchestrator.route(task, context)
    print(f"Route to: {decision.agent_type.value}")  # testwriter
    ```
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ai_infra.executor.agents.registry import SubAgentType
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

__all__ = [
    "OrchestratorAgent",
    "RoutingContext",
    "RoutingDecision",
    "ORCHESTRATOR_SYSTEM_PROMPT",
]

logger = get_logger("executor.agents.orchestrator")


# =============================================================================
# Orchestrator System Prompt (16.5.11.2.2)
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are a task routing orchestrator for a software development team.

Your ONLY job is to analyze tasks and decide which specialist agent should handle them.

AVAILABLE AGENTS:

0. **Scaffolder** - Sets up and configures projects
   - Analyzes roadmap to determine tech stack
   - Creates project structure (package.json, tsconfig.json, pyproject.toml)
   - Initializes directories and configuration files
   - Adds missing config files to existing projects
   - Sets up build tools, linters, formatters
   - Used for NEW projects OR adding missing infrastructure to existing ones
   - CRITICAL: Routes to this for project setup/config tasks

1. **Coder** - Implements application code
   - Creates modules, classes, functions
   - Implements features and business logic
   - Edits existing source files
   - Does NOT create project config or tests

2. **TestWriter** - Creates test files
   - Writes unit tests, integration tests
   - Creates test fixtures and mocks
   - Ensures code coverage
   - Does NOT run tests

3. **Tester** - Runs existing tests
   - Executes pytest, jest, etc.
   - Interprets test results
   - Identifies flaky tests
   - Does NOT write tests

4. **Debugger** - Fixes bugs and errors
   - Analyzes error messages
   - Identifies root causes
   - Implements fixes
   - Used when something is BROKEN

5. **Reviewer** - Reviews and refactors
   - Code review feedback
   - Performance optimization
   - Style improvements
   - Used for EXISTING code improvement

6. **Researcher** - Gathers information
   - Documentation lookup
   - API research
   - Best practices research
   - Used when information is MISSING

ROUTING RULES:

1. Project setup/config tasks → **Scaffolder**:
   - "Set up project", "Initialize", "Bootstrap", "Create project structure"
   - "Add/configure tsconfig", "Add eslint", "Configure prettier"
   - "Set up Tailwind", "Configure Next.js", "Add dependencies"
   - "Create package.json", "Create pyproject.toml"
   - Missing config files (even if project exists)

2. "Create tests", "Write tests", "Add tests" → **TestWriter**
3. "Run tests", "Execute tests", "Verify tests pass" → **Tester**
4. "Fix", "Debug", "Resolve error", "Bug" → **Debugger**
5. "Review", "Refactor", "Optimize", "Improve" → **Reviewer**
6. "Research", "Find out", "Look up", "How to" → **Researcher**
7. Creating application code, components, modules → **Coder**

CONTEXT CONSIDERATIONS:
- If no existing files and task is about creating project → Scaffolder
- If task mentions config/setup even with existing project → Scaffolder
- If task mentions framework/tech stack setup (Next.js, Django, etc.) → Scaffolder
- If task is about adding config files (tsconfig, eslint, tailwind.config) → Scaffolder
- If previous tasks created source files and current task is about tests → TestWriter
- If tests exist and task mentions running/verifying → Tester
- If task mentions an error or failure → Debugger
- If task is about creating app components/pages/routes → Coder

OUTPUT FORMAT:
Respond with ONLY a JSON object:
{{
  "agent": "scaffolder|coder|testwriter|tester|debugger|reviewer|researcher",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this agent"
}}
"""


# =============================================================================
# Data Classes (16.5.11.2.4, 16.5.11.2.5)
# =============================================================================


@dataclass
class RoutingContext:
    """Context for intelligent task routing.

    Provides the orchestrator with information about the current project
    state to make informed routing decisions.

    Attributes:
        workspace: Path to the workspace root.
        completed_tasks: Titles of tasks already completed in this session.
        existing_files: Files that exist in the workspace.
        project_type: Detected project type (python, node, rust, etc.).
        previous_agent: The agent type used for the previous task.
    """

    workspace: Path
    completed_tasks: list[str] = field(default_factory=list)
    existing_files: list[str] = field(default_factory=list)
    project_type: str = "unknown"
    previous_agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workspace": str(self.workspace),
            "completed_tasks": self.completed_tasks,
            "existing_files": self.existing_files,
            "project_type": self.project_type,
            "previous_agent": self.previous_agent,
        }


@dataclass
class RoutingDecision:
    """Result of orchestrator routing decision.

    Attributes:
        agent_type: The SubAgentType selected for this task.
        confidence: Confidence score from 0.0 to 1.0.
        reasoning: Brief explanation of why this agent was chosen.
        used_fallback: True if keyword fallback was used instead of LLM.
    """

    agent_type: SubAgentType
    confidence: float
    reasoning: str
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_type": self.agent_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "used_fallback": self.used_fallback,
        }


# =============================================================================
# Pydantic Schema for Structured Output
# =============================================================================


class RoutingResponse(BaseModel):
    """Structured response from the orchestrator LLM."""

    agent: str
    confidence: float
    reasoning: str


# =============================================================================
# Agent Type Mapping
# =============================================================================

_AGENT_TYPE_MAP: dict[str, SubAgentType] = {
    "scaffolder": SubAgentType.SCAFFOLDER,
    "coder": SubAgentType.CODER,
    "testwriter": SubAgentType.TESTWRITER,
    "tester": SubAgentType.TESTER,
    "debugger": SubAgentType.DEBUGGER,
    "reviewer": SubAgentType.REVIEWER,
    "researcher": SubAgentType.RESEARCHER,
}


# =============================================================================
# OrchestratorAgent Class (16.5.11.2.3)
# =============================================================================


class OrchestratorAgent:
    """Routes tasks to specialized agents using LLM reasoning.

    The OrchestratorAgent uses a fast, cheap LLM (gpt-4o-mini by default)
    to analyze task semantics and context, then routes to the appropriate
    specialist agent. Falls back to CODER if the LLM call fails.

    Example:
        ```python
        orchestrator = OrchestratorAgent()
        context = RoutingContext(workspace=Path("/project"), ...)
        task = TodoItem(id=1, title="Create tests for user.py")
        decision = await orchestrator.route(task, context)
        ```
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize the orchestrator agent.

        Args:
            model: Model to use for routing decisions. Default is gpt-4o-mini
                   for speed and cost efficiency.
            confidence_threshold: Minimum confidence required to use LLM
                                  routing decision. Below this, defaults to CODER.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._llm = None
        self._last_token_usage: dict[str, int] = {}

    @property
    def last_token_usage(self) -> dict[str, int]:
        """Get token usage from the last routing call."""
        return self._last_token_usage

    async def route(
        self,
        task: TodoItem,
        context: RoutingContext,
    ) -> RoutingDecision:
        """Route a task to the appropriate specialist agent.

        Analyzes the task and context using LLM reasoning, then returns
        a routing decision. Falls back to CODER if LLM fails.

        Args:
            task: The task to route.
            context: Routing context with project information.

        Returns:
            RoutingDecision with agent type, confidence, and reasoning.
        """
        logger.info(
            f"Routing task: '{task.title[:50]}...' (project_type={context.project_type}, completed={len(context.completed_tasks)} tasks)"
            if len(task.title) > 50
            else f"Routing task: '{task.title}' (project_type={context.project_type}, completed={len(context.completed_tasks)} tasks)"
        )

        try:
            # Build the routing prompt
            prompt = self._build_routing_prompt(task, context)

            # Call LLM for routing decision
            decision = await self._call_llm(prompt)

            # Log the decision
            reasoning_preview = (
                decision.reasoning[:50] + "..."
                if len(decision.reasoning) > 50
                else decision.reasoning
            )
            logger.info(
                f"Orchestrator decision: {decision.agent_type.value} "
                f"(confidence={decision.confidence:.2f}, reason='{reasoning_preview}')"
            )

            # Check confidence threshold - default to CODER if low confidence
            if decision.confidence < self.confidence_threshold:
                logger.warning(
                    f"Low confidence ({decision.confidence:.2f} < {self.confidence_threshold:.2f}), defaulting to CODER"
                )
                return self._default_fallback(task)

            return decision

        except Exception as e:
            logger.warning(f"Orchestrator LLM failed: {e}, defaulting to CODER")
            return self._default_fallback(task)

    def _build_routing_prompt(
        self,
        task: TodoItem,
        context: RoutingContext,
    ) -> str:
        """Build the routing prompt with task and context.

        Args:
            task: The task to route.
            context: Routing context.

        Returns:
            Formatted prompt string.
        """
        # Limit lists to avoid token bloat
        completed = context.completed_tasks[-5:] if context.completed_tasks else []
        files = context.existing_files[:20] if context.existing_files else []

        return f"""TASK TO ROUTE:
Title: {task.title}
Description: {task.description or "No description"}

CONTEXT:
Project type: {context.project_type}
Completed tasks: {", ".join(completed) or "None"}
Existing files: {", ".join(files) or "None"}
Previous agent: {context.previous_agent or "None"}

Which agent should handle this task?"""

    async def _call_llm(self, prompt: str) -> RoutingDecision:
        """Call the LLM for routing decision.

        Args:
            prompt: The routing prompt.

        Returns:
            RoutingDecision from LLM response.
        """
        # Lazy import to avoid circular dependencies
        from ai_infra.llm import LLM

        if self._llm is None:
            self._llm = LLM()

        # Use structured output for reliable JSON parsing
        response = await self._llm.achat(
            user_msg=prompt,
            model_name=self.model,
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            output_schema=RoutingResponse,
            output_method="prompt",
            temperature=0.0,  # Deterministic routing
        )

        # Track token usage if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self._last_token_usage = {
                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                "output_tokens": response.usage_metadata.get("output_tokens", 0),
                "total_tokens": response.usage_metadata.get("total_tokens", 0),
            }

        # Parse response - handle both structured and raw string responses
        return self._parse_routing_response(response)

    def _parse_routing_response(self, response: Any) -> RoutingDecision:
        """Parse the LLM response into a RoutingDecision.

        Handles both structured Pydantic responses and raw string/dict responses.

        Args:
            response: LLM response (RoutingResponse, dict, or string).

        Returns:
            RoutingDecision from parsed response.

        Raises:
            ValueError: If response cannot be parsed or agent type is invalid.
        """
        # Handle structured Pydantic response
        if isinstance(response, RoutingResponse):
            agent_str = response.agent.lower().strip()
            if agent_str not in _AGENT_TYPE_MAP:
                raise ValueError(f"Unknown agent type: {agent_str}")
            return RoutingDecision(
                agent_type=_AGENT_TYPE_MAP[agent_str],
                confidence=max(0.0, min(1.0, response.confidence)),
                reasoning=response.reasoning,
                used_fallback=False,
            )

        # Handle dict response
        if isinstance(response, dict):
            agent_str = response.get("agent", "").lower().strip()
            if agent_str not in _AGENT_TYPE_MAP:
                raise ValueError(f"Unknown agent type: {agent_str}")
            return RoutingDecision(
                agent_type=_AGENT_TYPE_MAP[agent_str],
                confidence=max(0.0, min(1.0, float(response.get("confidence", 0.5)))),
                reasoning=response.get("reasoning", "No reasoning provided"),
                used_fallback=False,
            )

        # Handle raw string response - try to extract JSON
        if isinstance(response, str):
            return self._parse_json_from_string(response)

        # Handle message-like response with content attribute
        if hasattr(response, "content"):
            content = str(response.content)
            return self._parse_json_from_string(content)

        raise ValueError(f"Cannot parse response type: {type(response)}")

    def _parse_json_from_string(self, text: str) -> RoutingDecision:
        """Extract and parse JSON from a string response.

        Args:
            text: String that may contain JSON.

        Returns:
            RoutingDecision from extracted JSON.

        Raises:
            ValueError: If no valid JSON found or agent type is invalid.
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*"agent"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                agent_str = data.get("agent", "").lower().strip()
                if agent_str in _AGENT_TYPE_MAP:
                    return RoutingDecision(
                        agent_type=_AGENT_TYPE_MAP[agent_str],
                        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                        reasoning=data.get("reasoning", "No reasoning provided"),
                        used_fallback=False,
                    )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        raise ValueError(f"Could not extract valid routing JSON from: {text[:100]}...")

    def _default_fallback(self, task: TodoItem) -> RoutingDecision:
        """Fall back to CODER as the default agent.

        Used when LLM routing fails or has low confidence.
        CODER is the safest default as it can handle most tasks.

        Args:
            task: The task to route.

        Returns:
            RoutingDecision defaulting to CODER.
        """
        logger.info(f"Using CODER fallback for task: {task.title[:50]}")

        return RoutingDecision(
            agent_type=SubAgentType.CODER,
            confidence=0.5,
            reasoning="Fallback: defaulting to CODER",
            used_fallback=True,
        )

    def route_sync(
        self,
        task: TodoItem,
        context: RoutingContext,
    ) -> RoutingDecision:
        """Synchronous version of route().

        Convenience method for non-async contexts.

        Args:
            task: The task to route.
            context: Routing context with project information.

        Returns:
            RoutingDecision with agent type, confidence, and reasoning.
        """
        import asyncio

        return asyncio.run(self.route(task, context))
