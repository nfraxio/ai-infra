"""
CLI commands for the Executor module.

Usage:
    ai-infra executor run --roadmap ./ROADMAP.md --max-tasks 5
    ai-infra executor status --roadmap ./ROADMAP.md
    ai-infra executor resume --roadmap ./ROADMAP.md --approve
    ai-infra executor rollback --roadmap ./ROADMAP.md
    ai-infra executor reset --roadmap ./ROADMAP.md
    ai-infra executor run --dry-run --roadmap ./ROADMAP.md

The executor reads tasks from a ROADMAP.md file and executes them
autonomously using an AI agent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_infra import Agent
from ai_infra.cli.console import BRAND_ACCENT
from ai_infra.executor import (
    ExecutionStatus,
    Executor,
    ExecutorCallbacks,
    ExecutorConfig,
    ReviewInfo,
    RunStatus,
    RunSummary,
    TaskStatus,
    VerifyMode,
)
from ai_infra.executor.adaptive import AdaptiveMode
from ai_infra.llm.workspace import Workspace

if TYPE_CHECKING:
    from ai_infra.executor.hitl import HITLAction, HITLResponse

app = typer.Typer(help="Autonomous task execution from ROADMAP.md")
console = Console()


# =============================================================================
# Constants
# =============================================================================

# Status icons for task display
STATUS_ICONS = {
    TaskStatus.COMPLETED: "[green]✓[/green]",
    TaskStatus.FAILED: "[red]✗[/red]",
    TaskStatus.IN_PROGRESS: "[yellow]→[/yellow]",
    TaskStatus.PENDING: "[dim]○[/dim]",
    TaskStatus.SKIPPED: "[dim]⊘[/dim]",
}

# Status colors for run status
RUN_STATUS_COLORS = {
    RunStatus.COMPLETED: "green",
    RunStatus.PAUSED: "yellow",
    RunStatus.FAILED: "red",
    RunStatus.STOPPED: "yellow",
    RunStatus.NO_TASKS: "dim",
}


# =============================================================================
# Output Formatting
# =============================================================================


def _format_duration(ms: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _format_tokens(tokens: int) -> str:
    """Format token count with cost estimate."""
    # Rough estimate: $0.003 per 1K tokens (average of input/output)
    cost = (tokens / 1000) * 0.003
    return f"{tokens:,} (≈${cost:.2f})"


def _render_node_metrics(node_metrics: dict[str, Any] | None) -> Panel | None:
    """Render per-node metrics as a Rich panel.

    Phase 2.4.3: Display per-node cost breakdown.

    Args:
        node_metrics: Dict of node_name -> NodeMetrics.to_dict()

    Returns:
        Rich Panel with node breakdown, or None if no metrics.
    """
    if not node_metrics:
        return None

    from ai_infra.executor.metrics import aggregate_node_metrics

    aggregated = aggregate_node_metrics(node_metrics)

    if not aggregated.node_metrics:
        return None

    # Build table
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Node", style=BRAND_ACCENT)
    table.add_column("Tokens", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Calls", justify="right")

    # Sort by total tokens descending
    sorted_nodes = sorted(
        aggregated.node_metrics.items(),
        key=lambda x: x[1].total_tokens,
        reverse=True,
    )

    # Find highest for highlighting
    highest_tokens = max((m.total_tokens for _, m in sorted_nodes), default=0)

    for name, metrics in sorted_nodes:
        pct = aggregated.get_node_percentage(name)
        tokens_str = f"{metrics.total_tokens:,}"

        # Highlight highest consumer
        if metrics.total_tokens == highest_tokens and metrics.total_tokens > 0:
            name = f"[bold yellow]{name}[/bold yellow]"
            tokens_str = f"[bold yellow]{tokens_str}[/bold yellow]"

        table.add_row(
            name,
            tokens_str,
            f"{pct:.1f}%",
            _format_duration(metrics.duration_ms),
            str(metrics.llm_calls) if metrics.llm_calls > 0 else "-",
        )

    # Add totals row
    table.add_row("", "", "", "", "")
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{aggregated.total_tokens:,}[/bold]",
        "[bold]100%[/bold]",
        f"[bold]{_format_duration(aggregated.total_duration_ms)}[/bold]",
        f"[bold]{aggregated.total_llm_calls}[/bold]"
        if aggregated.total_llm_calls > 0
        else "[bold]-[/bold]",
    )

    return Panel(
        table,
        title="Per-Node Cost Breakdown",
        subtitle="[dim]Highest consumer highlighted[/dim]",
        border_style="blue",
    )


def _render_routing_history(
    routing_history: list[dict[str, Any]] | None,
    orchestrator_tokens: int = 0,
) -> Panel | None:
    """Render orchestrator routing history as a Rich panel.

    Phase 16.5.11.3: Display routing decisions made by orchestrator.

    Args:
        routing_history: List of routing decision dicts.
        orchestrator_tokens: Total tokens used by orchestrator.

    Returns:
        Rich Panel with routing breakdown, or None if no history.
    """
    if not routing_history:
        return None

    # Build table
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Task", style=BRAND_ACCENT, no_wrap=False, max_width=40)
    table.add_column("Agent", style="green")
    table.add_column("Conf.", justify="right")
    table.add_column("Method", style="dim")

    # Agent type counts for summary
    agent_counts: dict[str, int] = {}

    for decision in routing_history:
        task_title = decision.get("task_title", "Unknown")
        agent_type = decision.get("agent_type", "unknown")
        confidence = decision.get("confidence", 0.0)
        used_fallback = decision.get("used_fallback", False)

        # Truncate long titles
        if len(task_title) > 37:
            task_title = task_title[:37] + "..."

        # Color confidence based on value
        if confidence >= 0.8:
            conf_str = f"[green]{confidence:.0%}[/green]"
        elif confidence >= 0.5:
            conf_str = f"[yellow]{confidence:.0%}[/yellow]"
        else:
            conf_str = f"[red]{confidence:.0%}[/red]"

        method = "keyword" if used_fallback else "LLM"

        table.add_row(task_title, agent_type, conf_str, method)

        # Count agents
        agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1

    # Summary line
    agent_summary = ", ".join(f"{k}={v}" for k, v in sorted(agent_counts.items()))
    subtitle = f"[dim]{agent_summary}[/dim]"
    if orchestrator_tokens > 0:
        subtitle += f" | [dim]orchestrator tokens: {orchestrator_tokens:,}[/dim]"

    return Panel(
        table,
        title="Orchestrator Routing Decisions",
        subtitle=subtitle,
        border_style=BRAND_ACCENT,
    )


# =============================================================================
# Rich HITL Interaction (Phase 4.1.6)
# =============================================================================


def prompt_hitl_action(
    proposal_description: str, task_id: str = "", task_title: str = ""
) -> HITLAction:
    """Rich CLI prompt for HITL actions.

    Phase 4.1.6: Provides a user-friendly interface for all HITL action types.

    Args:
        proposal_description: Description of what the agent proposes to do.
        task_id: Optional task ID for context.
        task_title: Optional task title for context.

    Returns:
        HITLAction based on user's choice.

    Example:
        ```python
        action = prompt_hitl_action(
            proposal_description="Add logging to auth module",
            task_id="2.1.3",
            task_title="Add structured logging",
        )

        if action.type == HITLActionType.APPROVE:
            # Continue execution
            pass
        elif action.type == HITLActionType.EDIT:
            # Process edit request
            pass
        ```
    """
    from ai_infra.executor.hitl import HITLAction, HITLActionType

    # Display proposal
    console.print()
    console.print("=" * 60)
    if task_id:
        console.print(f"[bold {BRAND_ACCENT}]Task {task_id}:[/bold {BRAND_ACCENT}] {task_title}")
    console.print(f"[bold]PROPOSAL:[/bold] {proposal_description}")
    console.print("=" * 60)

    # Display action menu
    console.print()
    console.print("[bold]Actions:[/bold]")
    console.print("  [green][a]pprove[/green]  - Accept and continue")
    console.print("  [red][r]eject[/red]   - Stop or try differently")
    console.print("  [yellow][e]dit[/yellow]     - Modify the proposal")
    console.print("  [blue][s]uggest[/blue]  - Give a hint")
    console.print("  [magenta][x]plain[/magenta]  - Ask for reasoning")
    console.print("  [dim][b]ack[/dim]     - Rollback to checkpoint")
    console.print("  [dim][k]ip[/dim]      - Skip this task")
    console.print()

    while True:
        choice = console.input("[bold]Action [a/r/e/s/x/b/k]:[/bold] ").lower().strip()

        if choice in ("a", "approve"):
            return HITLAction(type=HITLActionType.APPROVE)

        elif choice in ("r", "reject"):
            return HITLAction(type=HITLActionType.REJECT)

        elif choice in ("e", "edit"):
            edit_text = console.input("[yellow]How should the proposal change?[/yellow] ")
            if edit_text.strip():
                return HITLAction(type=HITLActionType.EDIT, content=edit_text.strip())
            console.print("[dim]Empty edit, try again[/dim]")
            continue

        elif choice in ("s", "suggest"):
            suggestion = console.input("[blue]Your suggestion:[/blue] ")
            if suggestion.strip():
                return HITLAction(type=HITLActionType.SUGGEST, content=suggestion.strip())
            console.print("[dim]Empty suggestion, try again[/dim]")
            continue

        elif choice in ("x", "explain"):
            question = console.input("[magenta]What do you want explained?[/magenta] ")
            return HITLAction(
                type=HITLActionType.EXPLAIN,
                content=question.strip() if question.strip() else None,
            )

        elif choice in ("b", "back", "rollback"):
            target = console.input("[dim]Rollback target (press Enter for last checkpoint):[/dim] ")
            return HITLAction(
                type=HITLActionType.ROLLBACK,
                target=target.strip() if target.strip() else None,
            )

        elif choice in ("k", "skip"):
            return HITLAction(type=HITLActionType.SKIP)

        else:
            console.print("[red]Invalid choice. Please enter a/r/e/s/x/b/k[/red]")


def display_hitl_response(response: HITLResponse) -> None:
    """Display a HITL response to the user.

    Args:
        response: The response from a HITL action handler.
    """

    console.print()

    if response.understood:
        if response.explanation:
            console.print(
                Panel(
                    response.explanation,
                    title="[bold magenta]Explanation[/bold magenta]",
                    border_style="magenta",
                )
            )

        if response.revised_plan:
            console.print(
                Panel(
                    response.revised_plan,
                    title="[bold yellow]Revised Plan[/bold yellow]",
                    border_style="yellow",
                )
            )

        if response.next_step:
            console.print(f"[bold]Next:[/bold] {response.next_step}")
    else:
        console.print(f"[red]Error:[/red] {response.error or 'Action not understood'}")
        if response.next_step:
            console.print(f"[dim]{response.next_step}[/dim]")


# =============================================================================
# Run Summary Rendering
# =============================================================================


def _render_run_summary(summary: RunSummary) -> Panel:
    """Render a run summary as a Rich panel."""
    status_color = RUN_STATUS_COLORS.get(summary.status, "white")

    # Build summary lines
    lines = []
    lines.append(
        f"[bold]Status:[/bold]     [{status_color}]{summary.status.value}[/{status_color}]"
    )

    # Task counts
    total = summary.total_tasks
    completed = summary.tasks_completed
    failed = summary.tasks_failed
    remaining = summary.tasks_remaining
    skipped = summary.tasks_skipped

    tasks_line = f"[bold]Tasks:[/bold]      {completed}/{total} completed"
    if remaining > 0:
        tasks_line += f" ({remaining} remaining"
        if skipped > 0:
            tasks_line += f", {skipped} skipped"
        tasks_line += ")"
    elif failed > 0:
        tasks_line += f" ({failed} failed)"
    lines.append(tasks_line)

    # Duration
    lines.append(f"[bold]Duration:[/bold]   {_format_duration(summary.duration_ms)}")

    # Tokens
    if summary.total_tokens > 0:
        lines.append(f"[bold]Tokens:[/bold]     {_format_tokens(summary.total_tokens)}")

    # Files modified
    total_files = sum(len(r.files_modified) + len(r.files_created) for r in summary.results)
    if total_files > 0:
        files_modified = sum(len(r.files_modified) for r in summary.results)
        files_created = sum(len(r.files_created) for r in summary.results)
        files_line = f"[bold]Files:[/bold]      {files_modified} modified"
        if files_created > 0:
            files_line += f", {files_created} created"
        lines.append(files_line)

    # Paused reason
    if summary.paused and summary.pause_reason:
        lines.append(f"[bold]Paused:[/bold]     {summary.pause_reason}")

    content = "\n".join(lines)

    return Panel(
        content,
        title="Executor Run Summary",
        border_style=status_color,
    )


def _render_results_table(summary: RunSummary) -> Table:
    """Render execution results as a Rich table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("", width=3)  # Status icon
    table.add_column("Task ID", style=BRAND_ACCENT)
    table.add_column("Title", no_wrap=False)
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right")

    for result in summary.results:
        if result.status == ExecutionStatus.SUCCESS:
            icon = "[green]✓[/green]"
            duration = _format_duration(result.duration_ms) if result.duration_ms > 0 else "-"
            tokens = f"{sum(result.token_usage.values()):,}" if result.token_usage else "-"
        elif result.status == ExecutionStatus.FAILED:
            icon = "[red]✗[/red]"
            duration = "-"
            tokens = "(failed)"
        elif result.status == ExecutionStatus.SKIPPED:
            icon = "[dim]⊘[/dim]"
            duration = "-"
            tokens = "(skipped)"
        else:
            icon = "[yellow]?[/yellow]"
            duration = "-"
            tokens = "-"

        # Truncate title if too long (max 50 chars)
        title = result.title[:47] + "..." if len(result.title) > 50 else result.title
        table.add_row(icon, result.task_id, title, duration, tokens)

    return table


def _render_status_table(executor: Executor) -> Table:
    """Render task status as a Rich table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("", width=3)  # Status icon
    table.add_column("Task ID", style=BRAND_ACCENT)
    table.add_column("Title", no_wrap=False)
    table.add_column("Status")

    for task in executor.roadmap.all_tasks():
        status = executor.state.get_status(task.id)
        icon = STATUS_ICONS.get(status, "?")
        status_text = status.value

        if status == TaskStatus.COMPLETED:
            status_style = "green"
        elif status == TaskStatus.FAILED:
            status_style = "red"
        elif status == TaskStatus.IN_PROGRESS:
            status_style = "yellow"
        else:
            status_style = "dim"

        table.add_row(icon, task.id, task.title, f"[{status_style}]{status_text}[/{status_style}]")

    return table


def _render_review_info(review: ReviewInfo) -> Panel:
    """Render review info as a Rich panel."""
    lines = []
    lines.append(f"[bold]Tasks Executed:[/bold] {len(review.task_ids)}")

    if review.task_ids:
        lines.append(f"  {', '.join(review.task_ids)}")

    lines.append(f"\n[bold]Files Modified:[/bold] {len(review.files_modified)}")
    for f in review.files_modified[:10]:  # Limit to first 10
        lines.append(f"  • {f}")
    if len(review.files_modified) > 10:
        lines.append(f"  ... and {len(review.files_modified) - 10} more")

    if review.files_created:
        lines.append(f"\n[bold]Files Created:[/bold] {len(review.files_created)}")
        for f in review.files_created[:5]:
            lines.append(f"  • {f}")

    if review.files_deleted:
        lines.append(f"\n[bold red]Files Deleted:[/bold red] {len(review.files_deleted)}")
        for f in review.files_deleted:
            lines.append(f"  • [red]{f}[/red]")

    if review.has_destructive:
        lines.append("\n[bold red]WARNING: Destructive operations detected[/bold red]")

    if review.commits:
        lines.append(f"\n[bold]Git Commits:[/bold] {len(review.commits)}")
        for commit in review.commits[:5]:
            lines.append(f"  • {commit.short_sha}: {commit.message[:50]}")

    content = "\n".join(lines)

    border_color = "red" if review.has_destructive else "yellow"
    return Panel(
        content,
        title="Changes for Review",
        border_style=border_color,
    )


# =============================================================================
# executor run - Run the executor
# =============================================================================


def parse_mcp_arg(mcp_arg: str) -> Any:
    """Parse MCP argument into server config (Phase 15.4).

    Supports two formats:
    - URL format: `http://localhost:8000/mcp` or `https://api.example.com/mcp`
      Uses streamable_http transport.
    - stdio format: `stdio:command arg1 arg2 ...`
      Uses stdio transport with command and args.

    Args:
        mcp_arg: MCP server specification string.

    Returns:
        McpServerConfig instance.

    Raises:
        ValueError: If the format is invalid.

    Examples:
        >>> parse_mcp_arg("http://localhost:8000/mcp")
        McpServerConfig(transport="streamable_http", url="http://localhost:8000/mcp")

        >>> parse_mcp_arg("stdio:npx -y @anthropic/mcp-server-filesystem /tmp")
        McpServerConfig(transport="stdio", command="npx", args=["-y", "..."])
    """
    from ai_infra.mcp import McpServerConfig

    mcp_arg = mcp_arg.strip()

    if mcp_arg.startswith("stdio:"):
        # Parse stdio format: "stdio:command arg1 arg2 ..."
        command_part = mcp_arg[6:].strip()
        if not command_part:
            raise ValueError("stdio: format requires a command (e.g., 'stdio:npx -y server')")

        parts = command_part.split()
        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        return McpServerConfig(
            transport="stdio",
            command=command,
            args=args,
        )
    elif mcp_arg.startswith("http://") or mcp_arg.startswith("https://"):
        # URL format for HTTP transport
        return McpServerConfig(
            transport="streamable_http",
            url=mcp_arg,
        )
    else:
        raise ValueError(
            f"Invalid MCP server format: {mcp_arg}. "
            "Expected URL (http://...) or stdio format (stdio:command args)"
        )


@app.command("run")
def run_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    max_tasks: Annotated[
        int,
        typer.Option(
            "--max-tasks",
            "-n",
            help="Maximum tasks to execute (0 = unlimited)",
        ),
    ] = 0,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model to use for execution",
        ),
    ] = "claude-sonnet-4-20250514",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be done without executing",
        ),
    ] = False,
    pause_destructive: Annotated[
        bool,
        typer.Option(
            "--pause-destructive/--no-pause-destructive",
            help="Pause and confirm before destructive operations (rm -rf, DROP TABLE, etc.)",
        ),
    ] = True,
    enable_planning: Annotated[
        bool,
        typer.Option(
            "--enable-planning/--no-planning",
            help="Enable pre-execution planning to identify files, dependencies, and risks (Phase 2.4.2)",
        ),
    ] = False,
    skip_verification: Annotated[
        bool,
        typer.Option(
            "--skip-verification",
            help="Skip task verification after execution",
        ),
    ] = False,
    stop_on_failure: Annotated[
        bool,
        typer.Option(
            "--stop-on-failure/--continue-on-failure",
            help="Stop execution on first failure",
        ),
    ] = True,
    # Phase 4.2: Collaboration mode options
    collaboration_mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Collaboration mode: autonomous, supervised, pair, or guided",
        ),
    ] = "supervised",
    approve_all: Annotated[
        bool,
        typer.Option(
            "--approve-all",
            help="Approve all actions automatically (sets mode to autonomous)",
        ),
    ] = False,
    step_mode: Annotated[
        bool,
        typer.Option(
            "--step",
            help="Pause after every step for review (sets mode to pair)",
        ),
    ] = False,
    require_approval: Annotated[
        int,
        typer.Option(
            "--require-approval",
            help="Pause for human approval after N tasks (0 = disabled)",
        ),
    ] = 0,
    checkpoint: Annotated[
        int,
        typer.Option(
            "--checkpoint",
            help="Create git checkpoint every N tasks (0 = disabled)",
        ),
    ] = 1,
    sync_roadmap: Annotated[
        bool,
        typer.Option(
            "--sync/--no-sync",
            help="Sync completed tasks to ROADMAP.md checkboxes after each task",
        ),
    ] = True,
    retry_failed: Annotated[
        int,
        typer.Option(
            "--retry-failed",
            help="Number of retry attempts for failed tasks (1 = no retry)",
        ),
    ] = 1,
    adaptive_mode: Annotated[
        str,
        typer.Option(
            "--adaptive-mode",
            help="Adaptive planning mode: no-adapt, suggest, or auto-fix",
        ),
    ] = "auto-fix",
    verify_mode: Annotated[
        str,
        typer.Option(
            "--verify-mode",
            help="Verification mode: auto (detect runner), agent (agent verifies), skip, pytest",
        ),
    ] = "auto",
    # Phase 5.8.5: Memory options
    no_run_memory: Annotated[
        bool,
        typer.Option(
            "--no-run-memory",
            help="Disable run memory (task-to-task context within a run)",
        ),
    ] = False,
    no_project_memory: Annotated[
        bool,
        typer.Option(
            "--no-project-memory",
            help="Disable project memory (cross-run persistence)",
        ),
    ] = False,
    memory_budget: Annotated[
        int,
        typer.Option(
            "--memory-budget",
            help="Token budget for memory context in prompts",
        ),
    ] = 6000,
    extract_with_llm: Annotated[
        bool,
        typer.Option(
            "--extract-with-llm",
            help="Use LLM for outcome extraction (more accurate, slower)",
        ),
    ] = False,
    clear_project_memory: Annotated[
        bool,
        typer.Option(
            "--clear-project-memory",
            help="Clear project memory before run (fresh start)",
        ),
    ] = False,
    normalize_with_llm: Annotated[
        bool,
        typer.Option(
            "--normalize-with-llm",
            help="Use LLM to normalize non-checkbox ROADMAP formats (emojis, prose, etc.)",
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON",
        ),
    ] = False,
    visualize: Annotated[
        bool,
        typer.Option(
            "--visualize",
            help="Generate and display Mermaid diagram of the executor graph",
        ),
    ] = False,
    interrupt_before: Annotated[
        list[str] | None,
        typer.Option(
            "--interrupt-before",
            help="Nodes to pause before (graph mode only). Valid: execute_task, verify_task, checkpoint",
        ),
    ] = None,
    interrupt_after: Annotated[
        list[str] | None,
        typer.Option(
            "--interrupt-after",
            help="Nodes to pause after (graph mode only). Valid: execute_task, verify_task, checkpoint",
        ),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            help="Maximum graph transitions before abort (default: 100, LangGraph default is 25)",
        ),
    ] = 100,
    # Phase 2.4: Shell tool CLI options
    enable_shell: Annotated[
        bool,
        typer.Option(
            "--enable-shell/--no-shell",
            help="Enable shell tool for command execution (npm, make, pytest, etc.)",
        ),
    ] = True,
    shell_timeout: Annotated[
        float,
        typer.Option(
            "--shell-timeout",
            help="Default timeout in seconds for shell commands",
        ),
    ] = 120.0,
    shell_allowed_commands: Annotated[
        str | None,
        typer.Option(
            "--shell-allowed-commands",
            help="Comma-separated allowlist of shell commands (e.g., 'pytest,npm,make'). Empty = all allowed.",
        ),
    ] = None,
    shell_snapshots: Annotated[
        bool,
        typer.Option(
            "--shell-snapshots/--no-shell-snapshots",
            help="Capture pre/post shell snapshots on task boundaries",
        ),
    ] = False,
    # Phase 2.2: Shell resource limits
    shell_memory_limit: Annotated[
        int,
        typer.Option(
            "--shell-memory-limit",
            help="Maximum memory in MB for shell processes (0 = unlimited)",
        ),
    ] = 512,
    shell_cpu_limit: Annotated[
        int,
        typer.Option(
            "--shell-cpu-limit",
            help="Maximum CPU time in seconds for shell processes (0 = unlimited)",
        ),
    ] = 60,
    shell_file_limit: Annotated[
        int,
        typer.Option(
            "--shell-file-limit",
            help="Maximum file size in MB for shell processes (0 = unlimited)",
        ),
    ] = 100,
    # Phase 11.1: Strict resource limits preset (EXECUTOR_4.md)
    strict_limits: Annotated[
        bool,
        typer.Option(
            "--strict-limits/--no-strict-limits",
            help="Use strict resource limits for shell execution (Phase 11.1: 256MB, 30s CPU, 10MB files)",
        ),
    ] = False,
    # Phase 2.3: Docker isolation
    docker_isolation: Annotated[
        bool,
        typer.Option(
            "--docker-isolation/--no-docker-isolation",
            help="Execute shell commands in isolated Docker containers",
        ),
    ] = False,
    docker_image: Annotated[
        str,
        typer.Option(
            "--docker-image",
            help="Docker image for isolated execution (e.g., 'python:3.11-slim', 'node:18-slim')",
        ),
    ] = "python:3.11-slim",
    docker_allow_network: Annotated[
        bool,
        typer.Option(
            "--allow-network/--no-network",
            help="Allow network access in Docker containers (for npm install, pip install, etc.)",
        ),
    ] = False,
    # Phase 3.1: Generate roadmap from prompt
    from_prompt: Annotated[
        str | None,
        typer.Option(
            "--from-prompt",
            "-p",
            help="Generate roadmap from natural language prompt before executing",
        ),
    ] = None,
    generation_style: Annotated[
        str,
        typer.Option(
            "--generation-style",
            help="Style for roadmap generation: minimal, standard, or detailed (used with --from-prompt)",
        ),
    ] = "standard",
    # Phase 3.3: Autonomous verification CLI options
    enable_autonomous_verify: Annotated[
        bool,
        typer.Option(
            "--enable-autonomous-verify/--no-autonomous-verify",
            help="Enable autonomous verification agent to discover and run tests (Phase 3.3)",
        ),
    ] = False,
    verify_timeout: Annotated[
        float,
        typer.Option(
            "--verify-timeout",
            help="Timeout in seconds for autonomous verification (default: 300, max 5 minutes)",
        ),
    ] = 300.0,
    # Phase 7.1: Subagent routing CLI options (EXECUTOR_3.md)
    use_subagents: Annotated[
        bool,
        typer.Option(
            "--use-subagents/--no-subagents",
            help="Route tasks to specialized subagents (Coder, Tester, Debugger) based on task type",
        ),
    ] = False,
    # Phase 7.4.3: Subagent model configuration (EXECUTOR_3.md)
    subagent_model: Annotated[
        list[str] | None,
        typer.Option(
            "--subagent-model",
            help="Override model for subagent type. Format: type=model (e.g., coder=gpt-4o). Can repeat.",
        ),
    ] = None,
    # Orchestrator routing CLI options
    orchestrator_model: Annotated[
        str,
        typer.Option(
            "--orchestrator-model",
            help="Model to use for orchestrator routing (default: gpt-4o-mini)",
        ),
    ] = "gpt-4o-mini",
    orchestrator_threshold: Annotated[
        float,
        typer.Option(
            "--orchestrator-threshold",
            help="Confidence threshold for orchestrator routing (default: 0.7)",
        ),
    ] = 0.7,
    # Phase 8.1.2: Skills learning CLI options (EXECUTOR_3.md)
    enable_learning: Annotated[
        bool,
        typer.Option(
            "--enable-learning/--no-learning",
            help="Enable skills learning from task execution to improve future performance",
        ),
    ] = True,
    # Phase 15.4: MCP server integration CLI options (EXECUTOR_5.md)
    mcp: Annotated[
        list[str] | None,
        typer.Option(
            "--mcp",
            help=(
                "MCP server to connect to. Can be specified multiple times. "
                "Formats: URL (http://localhost:8000/mcp) or "
                "stdio (stdio:npx -y @anthropic/mcp-server-filesystem /tmp)"
            ),
        ),
    ] = None,
    mcp_timeout: Annotated[
        float,
        typer.Option(
            "--mcp-timeout",
            help="Timeout in seconds for MCP server discovery (default: 30.0)",
        ),
    ] = 30.0,
):
    """Run the executor on a ROADMAP.md file.

    Can also generate a roadmap from a prompt before executing:
        ai-infra executor run --from-prompt "Add user authentication"
    """
    # Phase 3.1: Generate roadmap from prompt if --from-prompt is provided
    if from_prompt:
        from ai_infra.executor.roadmap_generator import (
            GenerationStyle,
            RoadmapGenerator,
        )

        # Validate generation style
        valid_styles = {
            "minimal": GenerationStyle.MINIMAL,
            "standard": GenerationStyle.STANDARD,
            "detailed": GenerationStyle.DETAILED,
        }
        if generation_style.lower() not in valid_styles:
            console.print(f"[red]Invalid generation style: {generation_style}[/red]")
            console.print("Valid options: minimal, standard, detailed")
            raise typer.Exit(1)

        workspace_path = roadmap.parent.resolve()
        if not output_json:
            console.print(f"[blue]Generating roadmap from prompt: {from_prompt}[/blue]")

        # Create agent for generation
        gen_agent = Agent(model=model)
        generator = RoadmapGenerator(gen_agent)

        try:
            generated = asyncio.run(
                generator.generate_and_save(
                    prompt=from_prompt,
                    workspace=workspace_path,
                    output=roadmap.name,
                    style=generation_style.lower(),
                    validate=True,
                )
            )
            if not output_json:
                console.print(
                    f"[green]Generated roadmap: {generated.task_count} tasks, "
                    f"complexity: {generated.complexity}[/green]"
                )
                if generated.validation_issues:
                    for issue in generated.validation_issues:
                        console.print(f"  [yellow]Warning: {issue.message}[/yellow]")
                console.print()
        except ValueError as e:
            console.print(f"[red]Error generating roadmap: {e}[/red]")
            raise typer.Exit(1)

    # Validate roadmap exists
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    # Parse adaptive mode
    mode_map = {
        "no-adapt": AdaptiveMode.NO_ADAPT,
        "suggest": AdaptiveMode.SUGGEST,
        "auto-fix": AdaptiveMode.AUTO_FIX,
    }
    parsed_mode = mode_map.get(adaptive_mode.lower())
    if parsed_mode is None:
        console.print(f"[red]Invalid adaptive mode: {adaptive_mode}[/red]")
        console.print("Valid options: no-adapt, suggest, auto-fix")
        raise typer.Exit(1)

    # Parse verify mode (Phase 5.9.2)
    verify_mode_map = {
        "auto": VerifyMode.AUTO,
        "agent": VerifyMode.AGENT,
        "skip": VerifyMode.SKIP,
        "pytest": VerifyMode.PYTEST,
    }
    parsed_verify_mode = verify_mode_map.get(verify_mode.lower())
    if parsed_verify_mode is None:
        console.print(f"[red]Invalid verify mode: {verify_mode}[/red]")
        console.print("Valid options: auto, agent, skip, pytest")
        raise typer.Exit(1)

    # Phase 4.2: Parse collaboration mode
    from ai_infra.executor.collaboration import CollaborationConfig, CollaborationMode

    # Override mode with shortcut flags
    effective_collab_mode = collaboration_mode
    if approve_all:
        effective_collab_mode = "autonomous"
    if step_mode:
        effective_collab_mode = "pair"

    collab_mode_map = {
        "autonomous": CollaborationMode.AUTONOMOUS,
        "supervised": CollaborationMode.SUPERVISED,
        "pair": CollaborationMode.PAIR,
        "guided": CollaborationMode.GUIDED,
    }
    parsed_collab_mode = collab_mode_map.get(effective_collab_mode.lower())
    if parsed_collab_mode is None:
        console.print(f"[red]Invalid collaboration mode: {effective_collab_mode}[/red]")
        console.print("Valid options: autonomous, supervised, pair, guided")
        raise typer.Exit(1)

    # Create collaboration config (will be used for mode-aware pausing)
    # Integration with Executor is pending Phase 4.3
    _ = CollaborationConfig.for_mode(parsed_collab_mode)

    # Phase 5.8.5: Clear project memory if requested
    if clear_project_memory:
        memory_path = roadmap.parent / ".executor" / "project-memory.json"
        if memory_path.exists():
            memory_path.unlink()
            if not output_json:
                console.print("[yellow]Cleared project memory[/yellow]")
        elif not output_json:
            console.print("[dim]No project memory to clear[/dim]")

    # Create config
    # Phase 11.1: Apply strict limits preset if enabled
    effective_shell_memory = shell_memory_limit
    effective_shell_cpu = shell_cpu_limit
    effective_shell_file = shell_file_limit
    if strict_limits:
        # ResourceLimits.strict() values
        effective_shell_memory = 256
        effective_shell_cpu = 30
        effective_shell_file = 10

    # Phase 15.4: Parse MCP server arguments
    mcp_servers = []
    if mcp:
        for mcp_arg in mcp:
            try:
                mcp_config = parse_mcp_arg(mcp_arg)
                mcp_servers.append(mcp_config)
            except ValueError as e:
                console.print(f"[red]Invalid MCP server: {e}[/red]")
                raise typer.Exit(1)

    # Store config for future use (currently passed as individual params to ExecutorGraph)
    _config = ExecutorConfig(
        model=model,
        max_tasks=max_tasks,
        dry_run=dry_run,
        skip_verification=skip_verification,
        stop_on_failure=stop_on_failure,
        require_human_approval_after=require_approval,
        pause_before_destructive=pause_destructive,
        checkpoint_every=checkpoint,
        sync_roadmap=sync_roadmap,
        retry_failed=retry_failed,
        adaptive_mode=parsed_mode,
        verify_mode=parsed_verify_mode,
        # Phase 5.8.5: Memory configuration
        enable_run_memory=not no_run_memory,
        enable_project_memory=not no_project_memory,
        memory_token_budget=memory_budget,
        extract_outcomes_with_llm=extract_with_llm,
        # Phase 5.13: LLM normalization
        normalize_with_llm=normalize_with_llm,
        # Phase 2.2: Shell resource limits (Phase 11.1: with strict override)
        shell_memory_limit_mb=effective_shell_memory,
        shell_cpu_limit_seconds=effective_shell_cpu,
        shell_file_limit_mb=effective_shell_file,
        # Phase 16.4: Shell snapshots
        enable_shell_snapshots=shell_snapshots,
        # Phase 2.3: Docker isolation
        docker_isolation=docker_isolation,
        docker_image=docker_image,
        docker_allow_network=docker_allow_network,
        # Phase 15.4: MCP server integration
        mcp_servers=mcp_servers,
        mcp_discover_timeout=mcp_timeout,
    )

    # Show what we're doing
    if not output_json:
        if dry_run:
            console.print("[yellow]Dry run mode - no changes will be made[/yellow]\n")
        console.print(f"Running executor on [{BRAND_ACCENT}]{roadmap}[/{BRAND_ACCENT}]")
        if max_tasks > 0:
            console.print(f"  Max tasks: {max_tasks}")
        console.print(f"  Model: {model}")
        if retry_failed > 1:
            console.print(f"  Retry attempts: {retry_failed}")
            console.print(f"  Adaptive mode: {adaptive_mode}")
        if normalize_with_llm:
            console.print(f"  [{BRAND_ACCENT}]LLM normalization: enabled[/{BRAND_ACCENT}]")
        # Phase 2.3: Show Docker mode
        if docker_isolation:
            network_mode = "bridge" if docker_allow_network else "none"
            console.print(
                f"  [{BRAND_ACCENT}]Docker isolation: {docker_image} (network: {network_mode})[/{BRAND_ACCENT}]"
            )
        # Phase 16.4: Show shell snapshots
        if shell_snapshots:
            console.print(f"  [{BRAND_ACCENT}]Shell snapshots: enabled[/{BRAND_ACCENT}]")
        # Phase 11.1: Show strict limits mode
        if strict_limits:
            console.print(
                "  [yellow]Strict resource limits: enabled (256MB, 30s CPU, 10MB files)[/yellow]"
            )
        # Show interrupt points
        if interrupt_before:
            console.print(
                f"  [{BRAND_ACCENT}]Interrupt before: {', '.join(interrupt_before)}[/{BRAND_ACCENT}]"
            )
        if interrupt_after:
            console.print(
                f"  [{BRAND_ACCENT}]Interrupt after: {', '.join(interrupt_after)}[/{BRAND_ACCENT}]"
            )
        # Phase 4.2: Show collaboration mode
        collab_mode_colors = {
            "autonomous": "green",
            "supervised": "yellow",
            "pair": BRAND_ACCENT,
            "guided": "magenta",
        }
        collab_color = collab_mode_colors.get(effective_collab_mode.lower(), "white")
        console.print(f"  [{collab_color}]Collaboration: {effective_collab_mode}[/{collab_color}]")
        # Phase 7.1: Show subagent routing status
        if use_subagents:
            console.print(f"  [{BRAND_ACCENT}]Subagent routing: enabled[/{BRAND_ACCENT}]")
            console.print(
                f"    [dim]Orchestrator: {orchestrator_model} (threshold={orchestrator_threshold})[/dim]"
            )
            # Phase 7.4.3: Show model overrides
            if subagent_model:
                for override in subagent_model:
                    console.print(f"    [dim]Model override: {override}[/dim]")
        # Phase 15.4: Show MCP servers
        if mcp_servers:
            console.print(
                f"  [{BRAND_ACCENT}]MCP servers: {len(mcp_servers)} configured[/{BRAND_ACCENT}]"
            )
            for mcp_cfg in mcp_servers:
                transport = mcp_cfg.transport
                if transport == "stdio":
                    console.print(f"    [dim]stdio: {mcp_cfg.command}[/dim]")
                else:
                    console.print(f"    [dim]{transport}: {mcp_cfg.url}[/dim]")
        console.print()

    # Phase 7.4.3 + 16.5.3: Parse subagent model config
    # Priority: --subagent-model > --model > defaults
    subagent_config = None
    if use_subagents:
        from ai_infra.executor.agents.config import SubAgentConfig

        overrides: dict[str, str] = {}
        if subagent_model:
            for override in subagent_model:
                if "=" not in override:
                    console.print(
                        f"[yellow]Warning: Invalid subagent model format: {override}[/yellow]"
                    )
                    console.print("  Expected format: type=model (e.g., coder=gpt-4o)")
                    continue
                agent_type, model_name = override.split("=", 1)
                agent_type = agent_type.strip().lower()
                model_name = model_name.strip()
                if not model_name:
                    console.print(f"[yellow]Warning: Empty model for {agent_type}[/yellow]")
                    continue
                overrides[agent_type] = model_name

        # Phase 16.5.3: Create config with base_model inheritance from --model
        if overrides:
            subagent_config = SubAgentConfig.with_overrides(overrides, base_model=model)
            if not output_json:
                console.print(
                    f"[blue]Subagent config: {len(overrides)} override(s), "
                    f"base model: {model}[/blue]"
                )
        else:
            # No explicit overrides - use base_model for all subagents
            subagent_config = SubAgentConfig(base_model=model)
            if not output_json:
                console.print(f"[blue]Subagent config: inheriting model {model}[/blue]")

    # Phase 1.8.1: Handle --visualize (just show diagram and exit)
    if visualize:
        from ai_infra.executor.graph import ExecutorGraph

        try:
            graph_executor = ExecutorGraph(
                agent=None,
                roadmap_path=str(roadmap),
            )
            mermaid = graph_executor.get_mermaid()
            console.print("[bold]Executor Graph Diagram (Mermaid)[/bold]\n")
            console.print("```mermaid")
            console.print(mermaid)
            console.print("```")
            console.print("\n[dim]Copy the above diagram to a Mermaid-compatible viewer.[/dim]")
            return
        except Exception as e:
            console.print(f"[red]Error generating graph diagram: {e}[/red]")
            raise typer.Exit(1)

    # Create callbacks for observability (token tracking, metrics)
    callbacks = ExecutorCallbacks()

    # Create agent for task execution (unless dry-run, but still needed for LLM normalization)
    agent = None
    needs_agent = not dry_run or normalize_with_llm
    if needs_agent:
        try:
            # Use workspace in sandboxed mode to confine agent to project directory
            # "sandboxed" prevents filesystem access outside roadmap.parent
            from typing import Literal, cast

            workspace_mode = cast(
                Literal["virtual", "sandboxed", "full"], "sandboxed" if not dry_run else "virtual"
            )
            workspace = Workspace(roadmap.parent, mode=workspace_mode)
            agent = Agent(
                deep=True,
                model_name=model,
                workspace=workspace,
                callbacks=callbacks,  # Track LLM tokens
            )
        except Exception as e:
            console.print(f"[red]Error creating agent: {e}[/red]")
            raise typer.Exit(1)

    # Run using ExecutorGraph (graph-based executor is now the only mode)
    async def _run():
        from ai_infra.executor.checkpoint import Checkpointer
        from ai_infra.executor.graph import ExecutorGraph
        from ai_infra.executor.todolist import TodoListManager

        # Initialize checkpointer if git repo exists
        checkpointer = None
        try:
            checkpointer = Checkpointer(roadmap.parent)
            if not checkpointer.is_repo:
                checkpointer = None
        except Exception:
            checkpointer = None

        # Initialize todo manager for ROADMAP sync and todos.json persistence
        todo_manager = TodoListManager(roadmap)

        graph_executor = ExecutorGraph(
            agent=agent,
            roadmap_path=str(roadmap),
            checkpointer=checkpointer,
            todo_manager=todo_manager,
            callbacks=callbacks,  # Phase 2.2.1: Pass callbacks for token tracking
            use_llm_normalization=normalize_with_llm,
            sync_roadmap=sync_roadmap,
            max_tasks=max_tasks if max_tasks > 0 else None,
            max_retries=retry_failed,  # Phase 2.2.2: Configurable retry count
            dry_run=dry_run,  # Phase 2.3.2: Dry run mode
            pause_destructive=pause_destructive,  # Phase 2.3.3: Pause destructive
            enable_planning=enable_planning,  # Phase 2.4.2: Pre-execution planning
            adaptive_mode=parsed_mode,  # Phase 2.3.1: Adaptive replanning mode
            recursion_limit=max_iterations,  # Phase 1.6: Max graph transitions
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            # Phase 2.4: Shell tool options
            enable_shell=enable_shell,
            shell_timeout=shell_timeout,
            shell_allowed_commands=(
                tuple(cmd.strip() for cmd in shell_allowed_commands.split(","))
                if shell_allowed_commands
                else None
            ),
            # Phase 16.4: Shell snapshots
            enable_shell_snapshots=shell_snapshots,
            # Phase 3.3: Autonomous verification options
            enable_autonomous_verify=enable_autonomous_verify,
            verify_timeout=verify_timeout,
            # Phase 7.1: Subagent routing (EXECUTOR_3.md)
            use_subagents=use_subagents,
            # Phase 7.4: Subagent model configuration (EXECUTOR_3.md)
            subagent_config=subagent_config,
            # Orchestrator routing
            orchestrator_model=orchestrator_model,
            orchestrator_confidence_threshold=orchestrator_threshold,
            # Phase 8.1: Skills learning (EXECUTOR_3.md)
            enable_learning=enable_learning,
            # Phase 15.4: MCP server integration (EXECUTOR_5.md)
            mcp_servers=mcp_servers,
            mcp_discover_timeout=mcp_timeout,
        )
        result = await graph_executor.arun()

        # Convert graph result to RunSummary for consistent output
        completed = result.get("tasks_completed_count", 0)
        failed = len(result.get("failed_todos", []))
        total = len(result.get("todos", []))

        # Phase 2.4.3: Capture node metrics for display
        node_metrics_data = result.get("node_metrics")

        # Phase 16.5.11.3: Capture routing history for display
        routing_history_data = result.get("orchestrator_routing_history", [])
        orchestrator_tokens_data = result.get("orchestrator_tokens_total", 0)

        summary = RunSummary(
            status=(
                RunStatus.COMPLETED
                if failed == 0 and not result.get("interrupt_requested")
                else RunStatus.PAUSED
                if result.get("interrupt_requested")
                else RunStatus.FAILED
            ),
            total_tasks=total,
            tasks_completed=completed,
            tasks_failed=failed,
            tasks_remaining=total - completed - failed,
            tasks_skipped=0,
            duration_ms=float(result.get("duration_ms") or 0),  # type: ignore[arg-type]
            total_tokens=int(result.get("tokens_used") or 0),  # type: ignore[call-overload]
            results=[],
            paused=result.get("interrupt_requested", False),
            pause_reason="HITL interrupt" if result.get("interrupt_requested") else "",
        )
        return (summary, node_metrics_data, routing_history_data, orchestrator_tokens_data)

    try:
        run_result = asyncio.run(_run())
        summary, node_metrics, routing_history, orchestrator_tokens = run_result
    except KeyboardInterrupt:
        console.print("\n[yellow]Execution interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]Execution error: {e}[/red]")
        raise typer.Exit(1)

    # Output results
    if output_json:
        import json

        output_data = summary.to_dict()
        # Phase 2.4.3: Include node_metrics in JSON output
        if node_metrics:
            output_data["node_metrics"] = node_metrics
        # Phase 16.5.11.3: Include routing history in JSON output
        if routing_history:
            output_data["orchestrator_routing_history"] = routing_history
            output_data["orchestrator_tokens_total"] = orchestrator_tokens
        console.print(json.dumps(output_data, indent=2))
    else:
        # Render summary panel
        console.print(_render_run_summary(summary))

        # Phase 2.4.3: Render per-node cost breakdown
        if node_metrics:
            node_panel = _render_node_metrics(node_metrics)
            if node_panel:
                console.print()
                console.print(node_panel)

        # Phase 16.5.11.3: Render routing history
        if routing_history:
            routing_panel = _render_routing_history(routing_history, orchestrator_tokens)
            if routing_panel:
                console.print()
                console.print(routing_panel)

        # Render results table if we have results
        if summary.results:
            console.print()
            console.print(_render_results_table(summary))

        # Show next steps if paused
        if summary.paused:
            console.print()
            console.print("[yellow]Execution paused. Review changes and run:[/yellow]")
            console.print(
                f"  [{BRAND_ACCENT}]ai-infra executor resume --roadmap {roadmap} --approve[/{BRAND_ACCENT}]"
            )
            console.print("Or to reject and rollback:")
            console.print(
                f"  [{BRAND_ACCENT}]ai-infra executor resume --roadmap {roadmap} --reject --rollback[/{BRAND_ACCENT}]"
            )

    # Exit with appropriate code
    if summary.status == RunStatus.FAILED:
        raise typer.Exit(1)


# =============================================================================
# executor status - Show current status
# =============================================================================


@app.command("status")
def status_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON",
        ),
    ] = False,
):
    """Show current executor status for a ROADMAP.md file."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    try:
        executor = Executor(roadmap=roadmap)
    except Exception as e:
        console.print(f"[red]Error loading executor: {e}[/red]")
        raise typer.Exit(1)

    # Get state summary
    state_summary = executor.state.get_summary()

    if output_json:
        import json

        data = {
            "roadmap": str(roadmap),
            "run_id": executor.state.run_id,
            "total_tasks": executor.roadmap.total_tasks,
            "completed": state_summary.completed,
            "failed": state_summary.failed,
            "in_progress": state_summary.in_progress,
            "pending": state_summary.pending,
            "progress": state_summary.completed / executor.roadmap.total_tasks
            if executor.roadmap.total_tasks > 0
            else 0,
        }
        console.print(json.dumps(data, indent=2))
    else:
        # Summary panel
        total = executor.roadmap.total_tasks
        completed = state_summary.completed
        progress = completed / total if total > 0 else 0

        lines = [
            f"[bold]ROADMAP:[/bold]     {roadmap}",
            f"[bold]Run ID:[/bold]      {executor.state.run_id}",
            f"[bold]Progress:[/bold]    {completed}/{total} ({progress:.0%})",
            f"  • Completed:   [green]{state_summary.completed}[/green]",
            f"  • Failed:      [red]{state_summary.failed}[/red]",
            f"  • In Progress: [yellow]{state_summary.in_progress}[/yellow]",
            f"  • Pending:     [dim]{state_summary.pending}[/dim]",
        ]

        console.print(Panel("\n".join(lines), title="Executor Status", border_style="blue"))

        # Task table
        console.print()
        console.print(_render_status_table(executor))


# =============================================================================
# executor resume - Resume after pause
# =============================================================================


@app.command("resume")
def resume_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    approve: Annotated[
        bool,
        typer.Option(
            "--approve",
            help="Approve pending changes and continue",
        ),
    ] = False,
    reject: Annotated[
        bool,
        typer.Option(
            "--reject",
            help="Reject pending changes",
        ),
    ] = False,
    rollback: Annotated[
        bool,
        typer.Option(
            "--rollback",
            help="Rollback git changes (only with --reject)",
        ),
    ] = False,
):
    """Resume executor after a pause."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    # Validate options
    if approve and reject:
        console.print("[red]Error: Cannot use both --approve and --reject[/red]")
        raise typer.Exit(1)

    if not approve and not reject:
        console.print("[red]Error: Must specify either --approve or --reject[/red]")
        raise typer.Exit(1)

    if rollback and not reject:
        console.print("[yellow]Warning: --rollback only has effect with --reject[/yellow]")

    try:
        executor = Executor(roadmap=roadmap)
    except Exception as e:
        console.print(f"[red]Error loading executor: {e}[/red]")
        raise typer.Exit(1)

    # Show what's being reviewed
    review = executor.get_changes_for_review()
    if review.task_ids:
        console.print(_render_review_info(review))
        console.print()

    # Resume with approval or rejection
    if approve:
        executor.resume(approved=True)
        console.print("[green]Changes approved. Ready to continue.[/green]")
        console.print(
            f"Run [{BRAND_ACCENT}]ai-infra executor run --roadmap {roadmap}[/{BRAND_ACCENT}] to continue."
        )
    else:
        result = executor.resume(approved=False, rollback=rollback)
        if rollback and result:
            if result.success:
                console.print(
                    f"[green]Rolled back {result.commits_reverted} commit(s) "
                    f"to {result.target_sha}[/green]"
                )
            else:
                console.print(f"[red]Rollback failed: {result.error}[/red]")
        console.print("[yellow]Changes rejected. In-progress tasks reset.[/yellow]")


# =============================================================================
# executor rollback - Rollback last task
# =============================================================================


@app.command("rollback")
def rollback_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    task_id: Annotated[
        str | None,
        typer.Option(
            "--task",
            "-t",
            help="Task ID to rollback to (default: last completed task)",
        ),
    ] = None,
    hard: Annotated[
        bool,
        typer.Option(
            "--hard",
            help="Hard reset (discard all changes)",
        ),
    ] = False,
):
    """Rollback to the state before a task was executed."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    try:
        executor = Executor(roadmap=roadmap)
    except Exception as e:
        console.print(f"[red]Error loading executor: {e}[/red]")
        raise typer.Exit(1)

    checkpointer = executor.checkpointer
    if checkpointer is None:
        console.print("[red]Error: Checkpointing is not enabled or not in a git repository[/red]")
        raise typer.Exit(1)

    # Get task to rollback
    if task_id is None:
        # Get last completed task
        completed_tasks = [
            tid
            for tid in executor.state._tasks
            if executor.state.get_status(tid) == TaskStatus.COMPLETED
        ]
        if not completed_tasks:
            console.print("[red]Error: No completed tasks to rollback[/red]")
            raise typer.Exit(1)
        task_id = completed_tasks[-1]

    console.print(f"Rolling back task [{BRAND_ACCENT}]{task_id}[/{BRAND_ACCENT}]...")

    result = checkpointer.rollback(task_id, hard=hard)

    if result.success:
        console.print(
            f"[green]Successfully rolled back {result.commits_reverted} commit(s)[/green]"
        )
        console.print(f"  Target: {result.target_sha}")
        console.print(f"  Message: {result.message}")

        # Reset state for the rolled back task
        executor.state.reset_task(task_id)
        executor.state.save()
        console.print(f"  Task [{BRAND_ACCENT}]{task_id}[/{BRAND_ACCENT}] reset to pending")
    else:
        console.print(f"[red]Rollback failed: {result.error}[/red]")
        raise typer.Exit(1)


# =============================================================================
# executor reset - Reset executor state
# =============================================================================


@app.command("reset")
def reset_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt",
        ),
    ] = False,
):
    """Reset executor state completely (re-parse ROADMAP)."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(
            "This will reset all executor state. Tasks will be re-read from ROADMAP.md. Continue?"
        )
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    try:
        executor = Executor(roadmap=roadmap)
        executor.reset()
        console.print("[green]Executor state reset successfully.[/green]")
        console.print(f"  Tasks: {executor.roadmap.total_tasks}")
        console.print(f"  Run ID: {executor.state.run_id}")
    except Exception as e:
        console.print(f"[red]Error resetting executor: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# executor sync-roadmap - Sync from todos.json to ROADMAP (Phase 5.13.5)
# =============================================================================


@app.command("sync-roadmap")
def sync_roadmap_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be updated without making changes",
        ),
    ] = False,
):
    """Sync completed todos from .executor/todos.json back to ROADMAP.md.

    This command reads the normalized todos from .executor/todos.json
    (created during executor run with LLM normalization) and updates
    the checkboxes in the original ROADMAP.md file.

    Use this after execution to update the ROADMAP with completion status.

    Example:
        ai-infra executor sync-roadmap --roadmap ./ROADMAP.md
        ai-infra executor sync-roadmap --dry-run
    """
    from ai_infra.executor.todolist import NormalizedTodoFile, TodoListManager

    todos_json = roadmap.parent / ".executor" / "todos.json"

    if not todos_json.exists():
        console.print(
            f"[red]Error: No todos.json found at {todos_json}[/red]\n"
            "[dim]Run 'executor run' first to create normalized todos.[/dim]"
        )
        raise typer.Exit(1)

    try:
        # Load and display what will be synced
        todo_file = NormalizedTodoFile.load(todos_json)
        completed = [t for t in todo_file.todos if t.status == "completed"]
        pending = [t for t in todo_file.todos if t.status == "pending"]
        skipped = [t for t in todo_file.todos if t.status == "skipped"]

        # Display summary
        table = Table(title="Todos Status Summary")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("[green]Completed[/green]", str(len(completed)))
        table.add_row("[yellow]Pending[/yellow]", str(len(pending)))
        table.add_row("[dim]Skipped[/dim]", str(len(skipped)))
        console.print(table)

        if not completed:
            console.print("\n[dim]No completed todos to sync.[/dim]")
            return

        if dry_run:
            console.print("\n[bold]Completed todos to sync:[/bold]")
            for todo in completed:
                console.print(f"  [green]✓[/green] {todo.title}")
            console.print("\n[dim]Dry run - no changes made.[/dim]")
            return

        # Perform the sync
        updated = TodoListManager.sync_json_to_roadmap(roadmap)

        if updated > 0:
            console.print(
                f"\n[green]Successfully updated {updated} checkbox(es) in ROADMAP.md[/green]"
            )
        else:
            console.print(
                "\n[yellow]Warning: No checkboxes were updated.[/yellow]\n"
                "[dim]The completed todos may not have matching checkboxes in ROADMAP.[/dim]"
            )

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error syncing roadmap: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# executor sync - Sync state to ROADMAP
# =============================================================================


@app.command("sync")
def sync_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
):
    """Sync completed tasks back to ROADMAP.md checkboxes."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    try:
        executor = Executor(roadmap=roadmap)
        updated = executor.sync_roadmap()

        if updated > 0:
            console.print(f"[green]Updated {updated} checkbox(es) in ROADMAP.md[/green]")
        else:
            console.print("[dim]No checkboxes to update.[/dim]")
    except Exception as e:
        console.print(f"[red]Error syncing: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# executor review - Show changes for review
# =============================================================================


@app.command("review")
def review_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON",
        ),
    ] = False,
):
    """Show changes pending human review."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    try:
        executor = Executor(roadmap=roadmap)
    except Exception as e:
        console.print(f"[red]Error loading executor: {e}[/red]")
        raise typer.Exit(1)

    review = executor.get_changes_for_review()

    if output_json:
        import json

        console.print(json.dumps(review.to_dict(), indent=2))
    else:
        if not review.task_ids:
            console.print("[dim]No changes pending review.[/dim]")
        else:
            console.print(_render_review_info(review))


# =============================================================================
# executor memory - Show project memory (Phase 5.8.5)
# =============================================================================


@app.command("memory")
def memory_cmd(
    project: Annotated[
        Path,
        typer.Argument(
            help="Project root directory (containing .executor/)",
        ),
    ] = Path("."),
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: summary, json, files, history",
        ),
    ] = "summary",
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON (shortcut for --format json)",
        ),
    ] = False,
):
    """Show project memory for a project.

    Project memory tracks:
    - Key files created/modified during execution
    - Run history (completed tasks, failures)
    - Learned patterns from previous runs

    Examples:
        ai-infra executor memory .
        ai-infra executor memory --format files
        ai-infra executor memory --format history
        ai-infra executor memory --json
    """
    from ai_infra.executor.project_memory import ProjectMemory

    # Resolve project path
    project_path = project.resolve()
    if not project_path.exists():
        console.print(f"[red]Error: Project directory not found: {project}[/red]")
        raise typer.Exit(1)

    # Load project memory
    try:
        memory = ProjectMemory.load(project_path)
    except Exception as e:
        console.print(f"[red]Error loading project memory: {e}[/red]")
        raise typer.Exit(1)

    # Override format if --json flag is used
    if output_json:
        format = "json"

    # Output based on format
    if format == "json":
        import json

        console.print(json.dumps(memory._to_dict(), indent=2))

    elif format == "files":
        if not memory.key_files:
            console.print("[dim]No files tracked in project memory.[/dim]")
        else:
            console.print(f"[bold]Key Files ({len(memory.key_files)}):[/bold]\n")
            for path, info in memory.key_files.items():
                purpose = info.purpose or "(no description)"
                created_by = (
                    f" [dim](created by {info.created_by_task})[/dim]"
                    if info.created_by_task
                    else ""
                )
                console.print(f"  [{BRAND_ACCENT}]{path}[/{BRAND_ACCENT}]: {purpose}{created_by}")

    elif format == "history":
        if not memory.run_history:
            console.print("[dim]No run history in project memory.[/dim]")
        else:
            console.print(f"[bold]Run History ({len(memory.run_history)} runs):[/bold]\n")
            # Show most recent runs first
            for run in reversed(memory.run_history[-10:]):
                date = run.timestamp[:10] if run.timestamp else "unknown"
                status_color = "green" if run.tasks_failed == 0 else "red"
                console.print(
                    f"  [{status_color}]{date}[/{status_color}] "
                    f"[bold]{run.run_id[:8]}...[/bold] - "
                    f"{run.tasks_completed} completed, {run.tasks_failed} failed"
                )
                if run.lessons_learned:
                    for lesson in run.lessons_learned[:2]:
                        console.print(
                            f"    └─ [dim]{lesson[:60]}...[/dim]"
                            if len(lesson) > 60
                            else f"    └─ [dim]{lesson}[/dim]"
                        )

    else:  # summary (default)
        console.print("[bold]Project Memory Summary[/bold]\n")

        # Project type
        project_type = memory.project_type or "unknown"
        console.print(f"  [bold]Project Type:[/bold] {project_type}")

        # Files tracked
        console.print(f"  [bold]Files Tracked:[/bold] {len(memory.key_files)}")

        # Run history
        console.print(f"  [bold]Runs Recorded:[/bold] {len(memory.run_history)}")

        # Last run info
        if memory.run_history:
            last = memory.run_history[-1]
            date = last.timestamp[:10] if last.timestamp else "unknown"
            status = "success" if last.tasks_failed == 0 else "had failures"
            console.print(
                f"  [bold]Last Run:[/bold] {date} - "
                f"{last.tasks_completed} completed, {last.tasks_failed} failed ({status})"
            )

        # Lessons learned count
        total_lessons = sum(len(r.lessons_learned) for r in memory.run_history)
        if total_lessons > 0:
            console.print(f"  [bold]Lessons Learned:[/bold] {total_lessons}")

        # Memory file location
        memory_file = project_path / ".executor" / "project-memory.json"
        if memory_file.exists():
            size_kb = memory_file.stat().st_size / 1024
            console.print(f"\n  [dim]Memory file: {memory_file} ({size_kb:.1f} KB)[/dim]")


@app.command("memory-clear")
def memory_clear_cmd(
    project: Annotated[
        Path,
        typer.Argument(
            help="Project root directory (containing .executor/)",
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt",
        ),
    ] = False,
):
    """Clear project memory for a fresh start.

    This removes:
    - Key files map
    - Run history
    - Learned patterns

    Use this when you want the executor to start fresh without
    any context from previous runs.

    Examples:
        ai-infra executor memory-clear .
        ai-infra executor memory-clear . --force
    """
    project_path = project.resolve()
    memory_path = project_path / ".executor" / "project-memory.json"

    if not memory_path.exists():
        console.print("[dim]No project memory to clear.[/dim]")
        return

    # Confirm unless --force
    if not force:
        confirm = typer.confirm(
            f"Clear project memory at {project_path}? This cannot be undone.",
            default=False,
        )
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # Remove the file
    try:
        memory_path.unlink()
        console.print("[green]Project memory cleared.[/green]")
    except Exception as e:
        console.print(f"[red]Error clearing memory: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Generate Command (Phase 3.1)
# =============================================================================


@app.command("generate")
def generate_cmd(
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            "-p",
            help="Natural language description of the work to do",
        ),
    ],
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            help="Project directory to analyze",
        ),
    ] = Path("."),
    style: Annotated[
        str,
        typer.Option(
            "--style",
            "-s",
            help="Generation style: minimal (3-5 tasks), standard (5-10), detailed (10+)",
        ),
    ] = "standard",
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (relative to workspace or absolute)",
        ),
    ] = Path("ROADMAP.md"),
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model to use for generation",
        ),
    ] = "claude-sonnet-4-20250514",
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            "-x",
            help="Execute the roadmap immediately after generating",
        ),
    ] = False,
    no_validate: Annotated[
        bool,
        typer.Option(
            "--no-validate",
            help="Skip validation of generated roadmap",
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output metadata as JSON",
        ),
    ] = False,
):
    """Generate ROADMAP.md from natural language prompt.

    Analyzes your project structure and generates a structured roadmap
    that can be executed by the executor.

    Examples:
        ai-infra executor generate -p "Add JWT authentication"
        ai-infra executor generate -p "Add user registration" --style detailed
        ai-infra executor generate -p "Add tests for auth module" --execute
    """
    from ai_infra.executor.roadmap_generator import (
        GenerationStyle,
        RoadmapGenerator,
    )

    # Validate style
    valid_styles = {
        "minimal": GenerationStyle.MINIMAL,
        "standard": GenerationStyle.STANDARD,
        "detailed": GenerationStyle.DETAILED,
    }
    if style.lower() not in valid_styles:
        console.print(f"[red]Invalid style: {style}[/red]")
        console.print("Valid options: minimal, standard, detailed")
        raise typer.Exit(1)

    workspace_path = workspace.resolve()
    if not workspace_path.exists():
        console.print(f"[red]Workspace does not exist: {workspace_path}[/red]")
        raise typer.Exit(1)

    # Create agent and generator
    agent = Agent(model=model)
    generator = RoadmapGenerator(agent)

    # Generate roadmap
    console.print(f"[blue]Analyzing project: {workspace_path}[/blue]")
    console.print(f"[blue]Generating roadmap for: {prompt}[/blue]")

    try:
        roadmap = asyncio.run(
            generator.generate_and_save(
                prompt=prompt,
                workspace=workspace_path,
                output=output,
                style=style.lower(),
                validate=not no_validate,
            )
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Output results
    if output_json:
        import json

        console.print(json.dumps(roadmap.to_dict(), indent=2))
    else:
        output_path = workspace_path / output if not output.is_absolute() else output
        console.print()
        console.print(
            Panel(
                f"[bold green]Generated:[/bold green] {output_path}\n"
                f"[bold]Title:[/bold] {roadmap.title}\n"
                f"[bold]Tasks:[/bold] {roadmap.task_count}\n"
                f"[bold]Estimated Time:[/bold] {roadmap.estimated_time}\n"
                f"[bold]Complexity:[/bold] {roadmap.complexity}\n"
                f"[bold]Confidence:[/bold] {roadmap.confidence:.0%}",
                title="Roadmap Generated",
                border_style="green" if roadmap.is_valid else "yellow",
            )
        )

        # Show validation warnings
        if roadmap.validation_issues:
            console.print()
            console.print("[yellow]Validation Issues:[/yellow]")
            for issue in roadmap.validation_issues:
                severity_color = {
                    "high": "red",
                    "medium": "yellow",
                    "low": "dim",
                }.get(issue.severity, "white")
                console.print(
                    f"  [{severity_color}]{issue.severity.upper()}[/{severity_color}] "
                    f"{issue.message}"
                )

    # Execute if requested
    if execute:
        console.print()
        console.print("[blue]Executing generated roadmap...[/blue]")
        # Import run command logic here
        from ai_infra.executor import Executor, ExecutorConfig

        executor = Executor(
            roadmap=workspace_path / output if not output.is_absolute() else output,
            config=ExecutorConfig(),
            agent=agent,
        )
        summary = asyncio.run(executor.run())
        console.print(_render_run_summary(summary))


# =============================================================================
# Skills CLI Commands (Phase 8.4)
# =============================================================================


@app.command("skills-list")
def skills_list_cmd(
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Filter skills by programming language",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of skills to display",
        ),
    ] = 10,
):
    """List learned skills from successful task executions.

    Skills are automatically extracted from successful tasks and stored
    for future reference. Use this command to view what the executor
    has learned.

    Examples:
        ai-infra executor skills-list
        ai-infra executor skills-list --language python
        ai-infra executor skills-list --limit 20
    """
    from ai_infra.executor.skills.database import SkillsDatabase

    db = SkillsDatabase()
    skills = db.skills

    if not skills:
        console.print("[dim]No skills learned yet.[/dim]")
        console.print(
            "[dim]Skills are automatically extracted from successful task executions.[/dim]"
        )
        return

    # Filter by language if specified
    if language:
        skills = [s for s in skills if language.lower() in [lang.lower() for lang in s.languages]]
        if not skills:
            console.print(f"[dim]No skills found for language: {language}[/dim]")
            return

    # Apply limit
    skills = skills[:limit]

    # Display skills in a table
    from rich.table import Table

    table = Table(title=f"Learned Skills ({len(db)} total)")
    table.add_column("ID", style="dim", width=12)
    table.add_column("Title", style="bold")
    table.add_column("Type", style=BRAND_ACCENT)
    table.add_column("Languages", style="green")
    table.add_column("Confidence", style="yellow", justify="right")
    table.add_column("Uses", style="magenta", justify="right")

    for skill in skills:
        table.add_row(
            skill.id[:12],
            skill.title[:40] + ("..." if len(skill.title) > 40 else ""),
            skill.type.value if hasattr(skill.type, "value") else str(skill.type),
            ", ".join(skill.languages[:3]) or "-",
            f"{skill.confidence:.0%}",
            str(skill.success_count + skill.failure_count),
        )

    console.print(table)

    # Show stats summary
    stats = db.get_stats()
    console.print()
    console.print(f"[dim]Average confidence: {stats.get('avg_confidence', 0):.0%}[/dim]")
    console.print(f"[dim]Total uses: {stats.get('total_uses', 0)}[/dim]")


@app.command("skills-show")
def skills_show_cmd(
    skill_id: Annotated[
        str,
        typer.Argument(
            help="Skill ID (full or partial match)",
        ),
    ],
):
    """Show full details of a specific learned skill.

    Displays all information about a skill including its pattern,
    rationale, keywords, and usage statistics.

    Examples:
        ai-infra executor skills-show abc123
        ai-infra executor skills-show 7f2a1b3c
    """
    from ai_infra.executor.skills.database import SkillsDatabase

    db = SkillsDatabase()

    # Try exact match first
    skill = db.get(skill_id)

    # If not found, try partial match
    if skill is None:
        for s in db.skills:
            if s.id.startswith(skill_id) or skill_id in s.id:
                skill = s
                break

    if skill is None:
        console.print(f"[red]Skill not found: {skill_id}[/red]")
        console.print("[dim]Use 'skills-list' to see available skills.[/dim]")
        raise typer.Exit(1)

    # Display full skill details
    console.print(
        Panel(
            f"[bold]{skill.title}[/bold]\n\n"
            f"[dim]ID:[/dim] {skill.id}\n"
            f"[dim]Type:[/dim] {skill.type.value if hasattr(skill.type, 'value') else skill.type}\n"
            f"[dim]Languages:[/dim] {', '.join(skill.languages) or 'None'}\n"
            f"[dim]Frameworks:[/dim] {', '.join(skill.frameworks) or 'None'}\n"
            f"[dim]Keywords:[/dim] {', '.join(skill.task_keywords[:10]) or 'None'}\n"
            f"[dim]Confidence:[/dim] {skill.confidence:.0%}\n"
            f"[dim]Success/Failure:[/dim] {skill.success_count}/{skill.failure_count}\n"
            f"[dim]Created:[/dim] {skill.created_at}\n"
            f"[dim]Updated:[/dim] {skill.updated_at}",
            title="Skill Details",
            border_style="blue",
        )
    )

    # Show description
    if skill.description:
        console.print()
        console.print("[bold]Description:[/bold]")
        console.print(skill.description)

    # Show pattern
    if skill.pattern:
        console.print()
        console.print("[bold]Pattern:[/bold]")
        console.print(Panel(skill.pattern, border_style="dim"))

    # Show rationale
    if skill.rationale:
        console.print()
        console.print("[bold]Rationale:[/bold]")
        console.print(skill.rationale)


@app.command("skills-clear")
def skills_clear_cmd(
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm clearing all skills (required)",
        ),
    ] = False,
):
    """Clear all learned skills from the database.

    This permanently removes all skills that have been automatically
    extracted from successful task executions. Use with caution.

    Examples:
        ai-infra executor skills-clear --confirm
    """
    from ai_infra.executor.skills.database import SkillsDatabase

    if not confirm:
        console.print("[red]This will permanently delete all learned skills.[/red]")
        console.print("[dim]Use --confirm to proceed.[/dim]")
        raise typer.Exit(1)

    db = SkillsDatabase()
    count = len(db)

    if count == 0:
        console.print("[dim]No skills to clear.[/dim]")
        return

    db.clear()
    console.print(f"[green]Cleared {count} skill(s) from the database.[/green]")


# =============================================================================
# Audit Command (Phase 11.3.3)
# =============================================================================


@app.command("audit")
def audit_cmd(
    last: Annotated[
        int,
        typer.Option(
            "--last",
            "-n",
            help="Show last N audit entries",
        ),
    ] = 20,
    event_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Filter by event type (executed, failed, timeout, violation, suspicious)",
        ),
    ] = None,
    show_suspicious: Annotated[
        bool,
        typer.Option(
            "--suspicious",
            "-s",
            help="Show only suspicious pattern detections",
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output in JSON format",
        ),
    ] = False,
):
    """View shell command audit log.

    Shows recent shell command executions, security violations,
    and suspicious pattern detections from the audit log.

    Examples:
        ai-infra executor audit                    # Show last 20 entries
        ai-infra executor audit --last 50         # Show last 50 entries
        ai-infra executor audit --type violation  # Show security violations
        ai-infra executor audit --suspicious      # Show suspicious patterns only
        ai-infra executor audit --json            # Output as JSON
    """

    from ai_infra.llm.shell.audit import (
        AuditEventType,
    )

    # Map short names to event types
    type_map = {
        "executed": AuditEventType.COMMAND_EXECUTED,
        "failed": AuditEventType.COMMAND_FAILED,
        "timeout": AuditEventType.COMMAND_TIMEOUT,
        "violation": AuditEventType.SECURITY_VIOLATION,
        "suspicious": AuditEventType.SUSPICIOUS_PATTERN,
        "redacted": AuditEventType.SECRET_REDACTED,
        "session_started": AuditEventType.SESSION_STARTED,
        "session_ended": AuditEventType.SESSION_ENDED,
    }

    # Handle --suspicious flag (filter_type reserved for future use)
    _filter_type = None
    if show_suspicious:
        _filter_type = AuditEventType.SUSPICIOUS_PATTERN
    elif event_type:
        if event_type.lower() not in type_map:
            console.print(f"[red]Unknown event type: {event_type}[/red]")
            console.print(f"[dim]Valid types: {', '.join(type_map.keys())}[/dim]")
            raise typer.Exit(1)
        _filter_type = type_map[event_type.lower()]

    # Get audit logger and explain that logs are in Python logging
    console.print("[dim]Audit logs are captured via Python logging.[/dim]")
    console.print("[dim]Configure logging handlers to view audit events.[/dim]")
    console.print()

    # Show configured patterns
    from ai_infra.llm.shell.audit import SUSPICIOUS_PATTERNS

    if show_suspicious or event_type == "suspicious":
        console.print(f"[bold {BRAND_ACCENT}]Suspicious Pattern Detection[/bold {BRAND_ACCENT}]")
        console.print()
        console.print("[bold]Monitored Patterns:[/bold]")
        for pattern, description in SUSPICIOUS_PATTERNS:
            console.print(f"  [yellow]{description}[/yellow]")
            console.print(f"    [dim]{pattern}[/dim]")
        console.print()
        console.print(f"[dim]Total patterns monitored: {len(SUSPICIOUS_PATTERNS)}[/dim]")
    else:
        console.print(f"[bold {BRAND_ACCENT}]Shell Audit Configuration[/bold {BRAND_ACCENT}]")
        console.print()
        console.print("[bold]Event Types Logged:[/bold]")
        for name, etype in type_map.items():
            console.print(f"  [green]{name}[/green]: {etype.value}")
        console.print()
        console.print("[bold]To enable audit logging:[/bold]")
        console.print("  1. Configure Python logging to capture 'ai_infra.shell.audit'")
        console.print("  2. Use ShellMiddleware with enable_audit=True (default)")
        console.print()
        console.print("[bold]Example logging config:[/bold]")
        console.print("  [dim]import logging")
        console.print("  logging.basicConfig(level=logging.INFO)")
        console.print("  logging.getLogger('ai_infra.shell.audit').setLevel(logging.INFO)[/dim]")


# =============================================================================
# executor mcp-status - MCP server health status (Phase 15.5)
# =============================================================================


@app.command("mcp-status")
def mcp_status_cmd(
    mcp: Annotated[
        list[str] | None,
        typer.Option(
            "--mcp",
            help=(
                "MCP server to check. Can be specified multiple times. "
                "Formats: URL (http://localhost:8000/mcp) or "
                "stdio (stdio:npx -y @anthropic/mcp-server-filesystem /tmp)"
            ),
        ),
    ] = None,
    mcp_timeout: Annotated[
        float,
        typer.Option(
            "--mcp-timeout",
            help="Timeout in seconds for MCP server discovery (default: 30.0)",
        ),
    ] = 30.0,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON",
        ),
    ] = False,
):
    """Check health status of MCP servers (Phase 15.5).

    Shows connection status, discovered tools, and health check results
    for configured MCP servers.

    Examples:
        ai-infra executor mcp-status --mcp http://localhost:8000/mcp
        ai-infra executor mcp-status --mcp "stdio:npx -y my-server" --json
    """
    import json as json_module

    if not mcp:
        console.print("[yellow]No MCP servers specified. Use --mcp to specify servers.[/yellow]")
        console.print()
        console.print("[dim]Example:[/dim]")
        console.print("  ai-infra executor mcp-status --mcp http://localhost:8000/mcp")
        console.print('  ai-infra executor mcp-status --mcp "stdio:npx -y my-server"')
        raise typer.Exit(0)

    # Parse MCP server arguments
    mcp_servers = []
    for mcp_arg in mcp:
        try:
            mcp_config = parse_mcp_arg(mcp_arg)
            mcp_servers.append(mcp_config)
        except ValueError as e:
            console.print(f"[red]Invalid MCP server: {e}[/red]")
            raise typer.Exit(1)

    if not output_json:
        console.print(f"[bold {BRAND_ACCENT}]MCP Server Status[/bold {BRAND_ACCENT}]")
        console.print(f"Checking {len(mcp_servers)} server(s)...\n")

    # Create MCP manager and check status
    from ai_infra.executor.mcp_integration import ExecutorMCPManager

    async def _check_status():
        manager = ExecutorMCPManager(
            configs=mcp_servers,
            discover_timeout=mcp_timeout,
        )

        try:
            # Try to discover servers
            discovery_result = await manager.discover(strict=False)

            # Get health check
            health_result = await manager.health_check(auto_reconnect=False)

            # Get full summary
            summary = manager.get_health_summary()

            return {
                "discovery": discovery_result.to_dict(),
                "health": health_result.to_dict(),
                "summary": summary,
            }
        except Exception as e:
            return {
                "error": str(e),
                "discovery": None,
                "health": None,
                "summary": None,
            }
        finally:
            await manager.close()

    result = asyncio.run(_check_status())

    if output_json:
        console.print(json_module.dumps(result, indent=2, default=str))
        return

    # Display results
    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    summary = result.get("summary", {})
    discovery = result.get("discovery", {})
    _health = result.get("health", {})  # Reserved for future health details display

    # Overall status
    overall = summary.get("overall_status", "unknown")
    status_color = "green" if overall == "healthy" else "red"
    console.print(f"[bold]Overall Status:[/bold] [{status_color}]{overall}[/{status_color}]")
    console.print(
        f"[bold]Servers:[/bold] {summary.get('healthy_count', 0)}/{summary.get('total_count', 0)} healthy"
    )
    console.print(f"[bold]Tools Discovered:[/bold] {discovery.get('tool_count', 0)}")
    console.print()

    # Per-server details
    servers = summary.get("servers", [])
    if servers:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Server", style=BRAND_ACCENT)
        table.add_column("Transport", style="dim")
        table.add_column("Status")
        table.add_column("Tools", justify="right")
        table.add_column("Failures", justify="right")

        for server in servers:
            status = server.get("health_status", "unknown")
            status_style = "green" if status == "healthy" else "red"
            failures = server.get("consecutive_failures", 0)
            failure_style = "yellow" if failures > 0 else "dim"

            table.add_row(
                server.get("name", "unknown"),
                server.get("transport", "unknown"),
                f"[{status_style}]{status}[/{status_style}]",
                str(server.get("tool_count", 0)),
                f"[{failure_style}]{failures}[/{failure_style}]",
            )

        console.print(table)

        # Show errors if any
        for server in servers:
            if server.get("error"):
                console.print(f"\n[red]Error for {server['name']}:[/red] {server['error']}")
    else:
        console.print("[yellow]No servers discovered[/yellow]")

    # Show discovery errors
    errors = discovery.get("errors", [])
    if errors:
        console.print("\n[bold red]Discovery Errors:[/bold red]")
        for err in errors:
            console.print(f"  [red]{err.get('error', 'Unknown error')}[/red]")


# =============================================================================
# Registration
# =============================================================================


def register(parent: typer.Typer):
    """Register executor commands with the parent CLI app."""
    parent.add_typer(
        app,
        name="executor",
        help="Autonomous task execution from ROADMAP.md",
        rich_help_panel="Automation",
    )
