"""Executor agents module.

Provides specialized agents for the executor:

Phase 3.1:
- VerificationAgent: Autonomous verification using shell tools

Phase 3.3:
- SubAgent: Base class for specialized subagents
- SubAgentRegistry: Registry for subagent selection
- CoderAgent: Writes and edits code
- ReviewerAgent: Reviews code changes
- TesterAgent: Runs tests and reports results
- DebuggerAgent: Analyzes and fixes failures
- ResearcherAgent: Searches for information
- spawn_for_task: Spawns appropriate subagent for a task

Phase 7.4:
- SubAgentConfig: Configuration for subagent models
- SubAgentModelConfig: Model settings per agent type

Phase 16.5.11:
- TestWriterAgent: Creates comprehensive test suites
- OrchestratorAgent: Routes tasks to specialists via LLM reasoning
- RoutingContext: Context for intelligent task routing
- RoutingDecision: Result of orchestrator routing

Phase 16.5.12:
- SubagentContextBuilder: Builds rich context for subagent execution
- SubagentContext: Rich context data for subagents
- SubagentOutputValidator: Validates subagent output quality
- ValidationResult: Result of output validation

Phase 16.5.13:
- OrchestratorMetrics: Aggregate metrics for routing decisions
- MetricsCollector: Collects and tracks routing metrics
- RoutingRecord: Individual routing decision record
- check_routing_mismatch: Detects potential misrouting
"""

from __future__ import annotations

# Phase 3.3: Subagent infrastructure
from ai_infra.executor.agents.base import (
    ExecutionContext,
    SubAgent,
    SubAgentResult,
)
from ai_infra.executor.agents.coder import CoderAgent

# Phase 7.4: Subagent configuration
from ai_infra.executor.agents.config import SubAgentConfig, SubAgentModelConfig

# Phase 16.5.12: Context builder and validator
from ai_infra.executor.agents.context_builder import (
    CodePatterns,
    SubagentContext,
    SubagentContextBuilder,
)
from ai_infra.executor.agents.debugger import DebuggerAgent

# Phase 16.5.13: Orchestrator metrics and observability
from ai_infra.executor.agents.metrics import (
    MetricsCollector,
    OrchestratorMetrics,
    RoutingOutcome,
    RoutingRecord,
    RoutingTimer,
    check_routing_mismatch,
    format_metrics_summary,
)
from ai_infra.executor.agents.orchestrator import (
    OrchestratorAgent,
    RoutingContext,
    RoutingDecision,
)
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.executor.agents.researcher import ResearcherAgent
from ai_infra.executor.agents.reviewer import ReviewerAgent

# Phase 16.5.14: Scaffolder agent for project setup
from ai_infra.executor.agents.scaffolder import ScaffolderAgent
from ai_infra.executor.agents.spawner import (
    compute_quality_score,
    spawn_for_task,
    spawn_for_task_from_state,
    spawn_for_task_with_validation,
    validate_expected_files,
)
from ai_infra.executor.agents.tester import TesterAgent

# Phase 16.5.11: TestWriter and Orchestrator agents
from ai_infra.executor.agents.testwriter import TestWriterAgent
from ai_infra.executor.agents.validator import (
    SubagentOutputValidator,
    ValidationResult,
    needs_retry,
    validate_subagent_output,
)

# Phase 3.1: Verification agent
from ai_infra.executor.agents.verify_agent import (
    VerificationAgent,
    VerificationFailure,
    VerificationResult,
)

__all__ = [
    # Phase 3.1: Verification
    "VerificationAgent",
    "VerificationFailure",
    "VerificationResult",
    # Phase 3.3: Subagent base
    "ExecutionContext",
    "SubAgent",
    "SubAgentResult",
    # Phase 3.3: Registry
    "SubAgentRegistry",
    "SubAgentType",
    # Phase 3.3: Specialized agents
    "CoderAgent",
    "DebuggerAgent",
    "ResearcherAgent",
    "ReviewerAgent",
    "ScaffolderAgent",
    "TesterAgent",
    # Phase 16.5.11: TestWriter and Orchestrator
    "TestWriterAgent",
    "OrchestratorAgent",
    "RoutingContext",
    "RoutingDecision",
    # Phase 3.3: Spawning
    "compute_quality_score",
    "spawn_for_task",
    "spawn_for_task_from_state",
    "spawn_for_task_with_validation",
    "validate_expected_files",
    # Phase 7.4: Configuration
    "SubAgentConfig",
    "SubAgentModelConfig",
    # Phase 16.5.12: Context builder and validator
    "CodePatterns",
    "SubagentContext",
    "SubagentContextBuilder",
    "SubagentOutputValidator",
    "ValidationResult",
    "needs_retry",
    "validate_subagent_output",
    # Phase 16.5.13: Metrics and observability
    "MetricsCollector",
    "OrchestratorMetrics",
    "RoutingOutcome",
    "RoutingRecord",
    "RoutingTimer",
    "check_routing_mismatch",
    "format_metrics_summary",
]
