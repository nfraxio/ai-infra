"""Scaffolder subagent for project initialization.

The ScaffolderAgent is a thin orchestrator that:
1. Detects project type from roadmap content
2. Checks if project is already initialized
3. Delegates actual setup to Coder agent with proper context

This design works for ANY language/framework because:
- Project type detection is done by LLM (knows all languages)
- Initialization check is generic (looks for common project markers)
- Actual setup is delegated to Coder with context about non-empty dirs
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import MetricsCallbacks
from ai_infra.executor.agents.base import (
    ExecutionContext,
    SubAgent,
    SubAgentResult,
)
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.llm import LLM
from ai_infra.llm.agent import Agent
from ai_infra.llm.shell.session import SessionConfig, ShellSession
from ai_infra.llm.shell.tool import create_shell_tool, set_current_session
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

__all__ = ["ScaffolderAgent"]

logger = get_logger("executor.agents.scaffolder")


# =============================================================================
# Project Markers - Generic indicators that a project is initialized
# =============================================================================

PROJECT_MARKERS = {
    # Node.js / JavaScript / TypeScript
    "package.json",
    # Python
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    # Rust
    "Cargo.toml",
    # Go
    "go.mod",
    # Ruby
    "Gemfile",
    # PHP
    "composer.json",
    # Java / Kotlin
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    # .NET
    "*.csproj",
    "*.fsproj",
    # Elixir
    "mix.exs",
    # Haskell
    "*.cabal",
    "stack.yaml",
}


# =============================================================================
# Prompts - Keep them SHORT and GENERIC
# =============================================================================

DETECT_PROJECT_TYPE_PROMPT = """Analyze this roadmap and identify the project's tech stack.

Roadmap content:
{roadmap_content}

Reply with a single line naming the tech stack. Examples of valid responses:
Next.js 14 with TypeScript and Tailwind CSS
Python FastAPI with SQLAlchemy
Rust CLI application
Go web server with Chi router

Your response (just the tech stack, no other text):"""


SETUP_PROJECT_PROMPT = """You are setting up a {project_type} project in {workspace}.

## Task
{task_title}: {task_description}

## Current State
{state_description}

## CRITICAL: How to Set Up This Project

You MUST create files manually because this directory already contains files.

### Step 1: Create package.json
```bash
cat > package.json << 'EOF'
{{
  "name": "project-name",
  "version": "0.1.0",
  "private": true,
  "scripts": {{ ... }},
  "dependencies": {{ ... }},
  "devDependencies": {{ ... }}
}}
EOF
```

### Step 2: Create config files (tsconfig.json, next.config.mjs, etc.)
Use `cat > filename << 'EOF'` for each file.

### Step 3: Create source directories and files
```bash
mkdir -p src/app
cat > src/app/page.tsx << 'EOF'
// page content
EOF
```

### Step 4: Install dependencies
```bash
npm install  # or pnpm install
```

### Step 5: Verify it builds
```bash
npm run build  # or pnpm build
```

## FORBIDDEN COMMANDS (will fail, do not use)
- `npx create-next-app` - fails on non-empty directories
- `pnpm create next-app` - fails on non-empty directories
- `mkdir web && cd web` - DO NOT create subdirectories
- Any command that creates a new subdirectory for the project

Create the {project_type} project files directly in {workspace} using manual file creation."""


# =============================================================================
# ScaffolderAgent
# =============================================================================


@SubAgentRegistry.register(SubAgentType.SCAFFOLDER)
class ScaffolderAgent(SubAgent):
    """Thin orchestrator for project initialization.

    Works for any language/framework by:
    1. Using LLM to detect project type from roadmap
    2. Checking if project is already initialized (generic markers)
    3. Delegating setup to Coder agent with non-empty-dir context
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        detection_model: str = "gpt-4o-mini",  # Fast model for type detection
        max_shell_iterations: int = 30,
    ):
        """Initialize the scaffolder agent.

        Args:
            model: LLM model for project setup.
            detection_model: Fast model for project type detection.
            max_shell_iterations: Maximum shell iterations.
        """
        super().__init__(model=model)
        self._name = "ScaffolderAgent"
        self._detection_model = detection_model
        self._max_shell_iterations = max_shell_iterations

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    @property
    def model(self) -> str:
        """Get model name."""
        return self._model

    def _get_tools(self) -> list[Any]:
        """Get tools for this agent."""
        return []

    async def execute(
        self,
        task: TodoItem,
        context: ExecutionContext,
    ) -> SubAgentResult:
        """Execute scaffolding task.

        Args:
            task: The scaffolding task to execute.
            context: Execution context with workspace info.

        Returns:
            SubAgentResult with scaffolding outcome.
        """
        start_time = time.perf_counter()
        logger.info(f"ScaffolderAgent executing task: {task.title}")

        workspace = context.workspace
        roadmap_content = self._get_roadmap_content(context)

        # Step 1: Detect project type from roadmap
        project_type = await self._detect_project_type(roadmap_content)
        logger.info(f"Detected project type: {project_type}")

        # Step 2: Check if already initialized
        is_initialized, markers_found = self._check_initialized(workspace)

        if is_initialized:
            logger.info(f"Project already initialized (found: {markers_found})")
            state_description = (
                f"Project is ALREADY INITIALIZED.\n"
                f"Found markers: {', '.join(markers_found)}\n"
                f"Only add missing pieces, do NOT recreate existing files."
            )
        else:
            state_description = (
                "Project is NOT initialized yet.\n"
                "Create the full project structure from scratch.\n"
                "Note: Directory may have non-project files (README, .git, etc.)"
            )

        # Step 3: Execute setup with proper context
        result = await self._setup_project(
            task=task,
            context=context,
            project_type=project_type,
            state_description=state_description,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"ScaffolderAgent completed in {duration_ms:.0f}ms: "
            f"{len(result.files_created)} created, {len(result.files_modified)} modified"
        )

        return SubAgentResult(
            success=result.success,
            output=result.output,
            error=result.error,
            files_created=result.files_created,
            files_modified=result.files_modified,
            metrics={
                "duration_ms": duration_ms,
                "agent_type": SubAgentType.SCAFFOLDER.value,
                "project_type": project_type,
                "was_initialized": is_initialized,
                **result.metrics,
            },
        )

    async def _detect_project_type(self, roadmap_content: str) -> str:
        """Use LLM to detect project type from roadmap.

        Args:
            roadmap_content: Content of the roadmap file.

        Returns:
            Project type description (e.g., "Next.js 14 with TypeScript").
        """
        try:
            llm = LLM()
            prompt = DETECT_PROJECT_TYPE_PROMPT.format(
                roadmap_content=roadmap_content[:3000]  # Limit content
            )
            response = await llm.achat(prompt, model_name=self._detection_model)
            # Response is an AIMessage - extract content
            content = response.content if hasattr(response, "content") else str(response)
            # Clean up the response - remove markdown formatting, quotes, etc.
            project_type = content.strip()
            # Remove leading dash/bullet
            if project_type.startswith("-") or project_type.startswith("*"):
                project_type = project_type[1:].strip()
            # Remove surrounding quotes
            project_type = project_type.strip('"').strip("'")
            return project_type or "Unknown project type"
        except Exception as e:
            logger.warning(f"Failed to detect project type: {e}")
            return "Unknown project type"

    def _check_initialized(self, workspace: Path) -> tuple[bool, list[str]]:
        """Check if project is already initialized.

        Args:
            workspace: Path to workspace directory.

        Returns:
            Tuple of (is_initialized, list of found markers).
        """
        if not workspace.exists():
            return False, []

        found_markers: list[str] = []

        for marker in PROJECT_MARKERS:
            if "*" in marker:
                # Handle glob patterns
                pattern = marker
                if list(workspace.glob(pattern)):
                    found_markers.append(marker)
            else:
                if (workspace / marker).exists():
                    found_markers.append(marker)

        return len(found_markers) > 0, found_markers

    async def _setup_project(
        self,
        task: TodoItem,
        context: ExecutionContext,
        project_type: str,
        state_description: str,
    ) -> SubAgentResult:
        """Execute project setup using shell commands.

        Args:
            task: The task to execute.
            context: Execution context.
            project_type: Detected project type.
            state_description: Description of current project state.

        Returns:
            SubAgentResult with setup outcome.
        """
        prompt = SETUP_PROJECT_PROMPT.format(
            project_type=project_type,
            task_title=task.title,
            task_description=task.description or "Set up project structure",
            workspace=context.workspace,
            state_description=state_description,
        )

        session_config = SessionConfig(workspace_root=context.workspace)
        session = ShellSession(session_config)

        try:
            await session.start()
            set_current_session(session)

            shell_tool = create_shell_tool(
                session=session,
                default_timeout=120.0,
            )

            metrics_cb = MetricsCallbacks()

            agent = Agent(
                tools=[shell_tool],
                model_name=self._model,
                system=prompt,
                callbacks=metrics_cb,
            )

            result = await agent.arun(
                f"Set up the project for: {task.title}\n\n"
                f"Project type: {project_type}\n"
                f"Create all necessary config files and install dependencies."
            )

            files_created, files_modified = self._analyze_file_changes(session.command_history)

            llm_metrics = metrics_cb.get_metrics() if hasattr(metrics_cb, "get_metrics") else {}

            return SubAgentResult(
                success=True,
                output=str(result) if result else "",
                files_created=files_created,
                files_modified=files_modified,
                metrics={
                    "total_tokens": llm_metrics.get("total_tokens", 0),
                    "commands_run": len(session.command_history),
                },
            )

        except Exception as e:
            logger.exception(f"Scaffolder setup failed: {e}")
            return SubAgentResult(
                success=False,
                error=str(e),
                metrics={},
            )

        finally:
            set_current_session(None)
            await session.close()

    def _analyze_file_changes(self, command_history: list[Any]) -> tuple[list[str], list[str]]:
        """Analyze command history to detect file changes.

        Args:
            command_history: List of shell command history entries.

        Returns:
            Tuple of (files_created, files_modified).
        """
        files_created: set[str] = set()
        files_modified: set[str] = set()

        for entry in command_history:
            cmd = entry.command if hasattr(entry, "command") else str(entry)

            # Detect file creation patterns
            if "cat >" in cmd or "cat>" in cmd:
                parts = cmd.split(">")
                if len(parts) >= 2:
                    filename = parts[1].split()[0].strip()
                    if filename and filename != "EOF" and "<<" not in filename:
                        files_created.add(filename)
            elif cmd.startswith("touch "):
                parts = cmd.split()
                if len(parts) >= 2:
                    files_created.add(parts[1])

        return list(files_created), list(files_modified)

    def _get_roadmap_content(self, context: ExecutionContext) -> str:
        """Get roadmap content from file (preferred) or context.

        For project type detection, we need the FULL roadmap including
        the Tech Stack section, not just the summary.

        Args:
            context: Execution context.

        Returns:
            Roadmap content as string.
        """
        # First try to read the actual roadmap file - this has the full content
        workspace = context.workspace
        roadmap_paths = [
            workspace / "ROADMAP.md",
            workspace / "ROADMAP-2.md",
            workspace / "roadmap.md",
        ]

        for roadmap_path in roadmap_paths:
            if roadmap_path.exists():
                try:
                    content = roadmap_path.read_text()
                    return content[:3000]  # Limit for LLM
                except Exception as e:
                    logger.warning(f"Failed to read {roadmap_path}: {e}")

        # Fall back to summary only if no roadmap file found
        if context.summary and len(context.summary) > 100:
            return context.summary

        return "(No roadmap content available)"
