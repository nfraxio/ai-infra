"""
CLI commands for debugging and testing MCP servers.

Usage:
    ai-infra mcp test http://localhost:8000/mcp        # Test connection
    ai-infra mcp tools http://localhost:8000/mcp       # List tools
    ai-infra mcp prompts http://localhost:8000/mcp     # List prompts
    ai-infra mcp resources http://localhost:8000/mcp   # List resources
    ai-infra mcp call http://localhost:8000/mcp tool_name '{"arg": "value"}'
    ai-infra mcp prompt http://localhost:8000/mcp prompt_name '{"skill": "math"}'
    ai-infra mcp info http://localhost:8000/mcp        # Server metadata

For stdio transport:
    ai-infra mcp tools --transport stdio --command npx --args "-y @modelcontextprotocol/server-filesystem /tmp"
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from ai_infra.cli.console import BRAND_ACCENT
from ai_infra.mcp import (
    MCPClient,
    MCPResource,
    McpServerConfig,
    PromptInfo,
    ResourceInfo,
)

app = typer.Typer(help="MCP server debugging and testing commands")
console = Console()

# Type alias for transport options
TransportType = Literal["stdio", "streamable_http", "sse"]

# Valid transport values
_VALID_TRANSPORTS = {"stdio", "streamable_http", "sse"}


def _create_config(
    url: str | None = None,
    transport: str = "streamable_http",
    command: str | None = None,
    args: str | None = None,
    env: str | None = None,
) -> McpServerConfig:
    """Create MCP server config from CLI options."""
    # Validate transport
    if transport not in _VALID_TRANSPORTS:
        console.print(
            f"[red]Error: Invalid transport '{transport}'. Must be one of: {', '.join(_VALID_TRANSPORTS)}[/red]"
        )
        raise typer.Exit(1)

    # Cast to Literal type after validation
    validated_transport: TransportType = transport  # type: ignore[assignment]

    if validated_transport == "stdio":
        if not command:
            console.print("[red]Error: --command required for stdio transport[/red]")
            raise typer.Exit(1)

        args_list = args.split() if args else []
        env_dict = json.loads(env) if env else None

        return McpServerConfig(
            transport="stdio",
            command=command,
            args=args_list,
            env=env_dict,
        )
    else:
        if not url:
            console.print("[red]Error: URL required for HTTP transport[/red]")
            raise typer.Exit(1)

        return McpServerConfig(
            transport=validated_transport,
            url=url,
        )


async def _run_with_client(
    config: McpServerConfig,
    timeout: float,
    operation: str,
    func,
):
    """Run an async operation with MCPClient."""
    client = MCPClient([config], discover_timeout=timeout)
    try:
        await client.discover()
        return await func(client)
    except Exception as e:
        console.print(f"[red][X] {operation} failed: {e}[/red]")
        raise typer.Exit(1)
    finally:
        await client.close()


# =============================================================================
# mcp test - Test connection to MCP server
# =============================================================================


@app.command("test")
def test_cmd(
    url: str | None = typer.Argument(None, help="MCP server URL"),
    transport: str = typer.Option(
        "streamable_http",
        "--transport",
        "-t",
        help="Transport type: streamable_http, stdio",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command for stdio transport",
    ),
    args: str | None = typer.Option(
        None,
        "--args",
        "-a",
        help="Arguments for stdio command (space-separated)",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        help="Connection timeout in seconds",
    ),
):
    """Test connection to an MCP server."""
    config = _create_config(url, transport, command, args)

    async def _test(client: MCPClient):
        # Get tools count (list_tools returns a List)
        tools = await client.list_tools(server="cli-server")
        tools_count = len(tools)

        # Try to get prompts (list_prompts returns a Dict)
        try:
            prompts = await client.list_prompts(server_name="cli-server")
            prompts_count = len(prompts.get("cli-server", []))
        except Exception:
            prompts_count = 0

        # Try to get resources (list_resources returns a Dict)
        try:
            resources = await client.list_resources(server_name="cli-server")
            resources_count = len(resources.get("cli-server", []))
        except Exception:
            resources_count = 0

        # Success panel
        target = url or f"{command} {args or ''}"
        console.print(
            Panel(
                f"[green][OK] Connected successfully[/green]\n\n"
                f"Server: {target}\n"
                f"Transport: {transport}\n"
                f"Tools: {tools_count}\n"
                f"Prompts: {prompts_count}\n"
                f"Resources: {resources_count}",
                title="MCP Server Test",
                border_style="green",
            )
        )

    asyncio.run(_run_with_client(config, timeout, "Connection test", _test))


# =============================================================================
# mcp tools - List tools from MCP server
# =============================================================================


@app.command("tools")
def tools_cmd(
    url: str | None = typer.Argument(None, help="MCP server URL"),
    transport: str = typer.Option(
        "streamable_http",
        "--transport",
        "-t",
        help="Transport type: streamable_http, stdio",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command for stdio transport",
    ),
    args: str | None = typer.Option(
        None,
        "--args",
        "-a",
        help="Arguments for stdio command (space-separated)",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        help="Connection timeout in seconds",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show full tool schemas",
    ),
):
    """List tools available from an MCP server."""
    config = _create_config(url, transport, command, args)

    async def _list_tools(client: MCPClient):
        # list_tools returns a List when called with server=
        tools = await client.list_tools(server="cli-server")

        if output_json:
            # Convert to serializable format
            tools_data = []
            for tool in tools:
                tool_dict = {
                    "name": tool.name,
                    "description": tool.description,
                }
                if verbose and hasattr(tool, "inputSchema"):
                    tool_dict["inputSchema"] = tool.inputSchema
                tools_data.append(tool_dict)
            console.print(json.dumps(tools_data, indent=2))
        else:
            if not tools:
                console.print("[yellow]No tools found[/yellow]")
                return

            table = Table(title=f"MCP Tools ({len(tools)})")
            table.add_column("Name", style=BRAND_ACCENT, no_wrap=True)
            table.add_column("Description", style="white")

            if verbose:
                table.add_column("Parameters", style="dim")

            for tool in tools:
                desc = tool.description or ""
                if len(desc) > 60 and not verbose:
                    desc = desc[:57] + "..."

                if verbose:
                    # Extract parameter names from schema
                    params = ""
                    if hasattr(tool, "inputSchema") and tool.inputSchema:
                        schema = tool.inputSchema
                        if "properties" in schema:
                            param_names = list(schema["properties"].keys())
                            required = schema.get("required", [])
                            params = ", ".join(f"{p}*" if p in required else p for p in param_names)
                    table.add_row(tool.name, desc, params)
                else:
                    table.add_row(tool.name, desc)

            console.print(table)

    asyncio.run(_run_with_client(config, timeout, "List tools", _list_tools))


# =============================================================================
# mcp prompts - List prompts from MCP server
# =============================================================================


@app.command("prompts")
def prompts_cmd(
    url: str | None = typer.Argument(None, help="MCP server URL"),
    transport: str = typer.Option(
        "streamable_http",
        "--transport",
        "-t",
        help="Transport type: streamable_http, stdio",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command for stdio transport",
    ),
    args: str | None = typer.Option(
        None,
        "--args",
        "-a",
        help="Arguments for stdio command (space-separated)",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        help="Connection timeout in seconds",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List prompts available from an MCP server."""
    config = _create_config(url, transport, command, args)

    async def _list_prompts(client: MCPClient):
        prompts_map = await client.list_prompts(server_name="cli-server")
        prompts: list[PromptInfo] = prompts_map.get("cli-server", [])

        if output_json:
            prompts_data = [
                {
                    "name": p.name,
                    "description": p.description,
                    "arguments": [
                        {
                            "name": a["name"],
                            "description": a.get("description"),
                            "required": a.get("required", False),
                        }
                        for a in (p.arguments or [])
                    ],
                }
                for p in prompts
            ]
            console.print(json.dumps(prompts_data, indent=2))
        else:
            if not prompts:
                console.print("[yellow]No prompts found[/yellow]")
                return

            table = Table(title=f"MCP Prompts ({len(prompts)})")
            table.add_column("Name", style=BRAND_ACCENT, no_wrap=True)
            table.add_column("Description", style="white")
            table.add_column("Arguments", style="dim")

            for prompt in prompts:
                desc = prompt.description or ""
                if len(desc) > 50:
                    desc = desc[:47] + "..."

                args_str = ""
                if prompt.arguments:
                    args_str = ", ".join(
                        f"{a['name']}*" if a.get("required") else a["name"]
                        for a in prompt.arguments
                    )

                table.add_row(prompt.name, desc, args_str)

            console.print(table)

    asyncio.run(_run_with_client(config, timeout, "List prompts", _list_prompts))


# =============================================================================
# mcp resources - List resources from MCP server
# =============================================================================


@app.command("resources")
def resources_cmd(
    url: str | None = typer.Argument(None, help="MCP server URL"),
    transport: str = typer.Option(
        "streamable_http",
        "--transport",
        "-t",
        help="Transport type: streamable_http, stdio",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command for stdio transport",
    ),
    args: str | None = typer.Option(
        None,
        "--args",
        "-a",
        help="Arguments for stdio command (space-separated)",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        help="Connection timeout in seconds",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List resources available from an MCP server."""
    config = _create_config(url, transport, command, args)

    async def _list_resources(client: MCPClient):
        resources_map = await client.list_resources(server_name="cli-server")
        resources: list[ResourceInfo] = resources_map.get("cli-server", [])

        if output_json:
            resources_data = [
                {
                    "uri": r.uri,
                    "name": r.name,
                    "description": r.description,
                    "mime_type": r.mime_type,
                }
                for r in resources
            ]
            console.print(json.dumps(resources_data, indent=2))
        else:
            if not resources:
                console.print("[yellow]No resources found[/yellow]")
                return

            table = Table(title=f"MCP Resources ({len(resources)})")
            table.add_column("Name", style=BRAND_ACCENT, no_wrap=True)
            table.add_column("URI", style="blue")
            table.add_column("MIME Type", style="dim")

            for resource in resources:
                table.add_row(
                    resource.name,
                    resource.uri[:50] + "..." if len(resource.uri) > 50 else resource.uri,
                    resource.mime_type or "-",
                )

            console.print(table)

    asyncio.run(_run_with_client(config, timeout, "List resources", _list_resources))


# =============================================================================
# mcp call - Call a tool on MCP server
# =============================================================================


@app.command("call")
def call_cmd(
    url: str | None = typer.Argument(None, help="MCP server URL"),
    tool_name: str = typer.Argument(..., help="Tool name to call"),
    tool_args: str | None = typer.Argument(None, help="Tool arguments as JSON"),
    transport: str = typer.Option(
        "streamable_http",
        "--transport",
        "-t",
        help="Transport type: streamable_http, stdio",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command for stdio transport",
    ),
    args: str | None = typer.Option(
        None,
        "--args",
        "-a",
        help="Arguments for stdio command (space-separated)",
    ),
    timeout: float = typer.Option(
        30.0,
        "--timeout",
        help="Call timeout in seconds",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Call a tool on an MCP server."""
    config = _create_config(url, transport, command, args)

    # Parse tool arguments
    try:
        parsed_args = json.loads(tool_args) if tool_args else {}
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON arguments: {e}[/red]")
        raise typer.Exit(1)

    async def _call_tool(client: MCPClient):
        result = await client.call_tool("cli-server", tool_name, parsed_args)

        if output_json:
            # Convert result to JSON-serializable format
            if hasattr(result, "content"):
                content = []
                for item in result.content:
                    if hasattr(item, "text"):
                        content.append({"type": "text", "text": item.text})
                    elif hasattr(item, "data"):
                        content.append({"type": "data", "data": str(item.data)})
                    else:
                        content.append({"type": "unknown", "value": str(item)})
                console.print(json.dumps({"content": content}, indent=2))
            else:
                console.print(json.dumps({"result": str(result)}, indent=2))
        else:
            console.print(
                Panel(
                    f"[green][OK] Tool '{tool_name}' called successfully[/green]",
                    title="MCP Tool Call",
                    border_style="green",
                )
            )

            # Display result
            if hasattr(result, "content"):
                for item in result.content:
                    if hasattr(item, "text"):
                        console.print(item.text)
                    else:
                        console.print(str(item))
            else:
                console.print(str(result))

    asyncio.run(_run_with_client(config, timeout, f"Call tool '{tool_name}'", _call_tool))


# =============================================================================
# mcp prompt - Get a prompt from MCP server
# =============================================================================


@app.command("prompt")
def prompt_cmd(
    url: str | None = typer.Argument(None, help="MCP server URL"),
    prompt_name: str = typer.Argument(..., help="Prompt name to get"),
    prompt_args: str | None = typer.Argument(None, help="Prompt arguments as JSON"),
    transport: str = typer.Option(
        "streamable_http",
        "--transport",
        "-t",
        help="Transport type: streamable_http, stdio",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command for stdio transport",
    ),
    args: str | None = typer.Option(
        None,
        "--args",
        "-a",
        help="Arguments for stdio command (space-separated)",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        help="Request timeout in seconds",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Get a prompt from an MCP server with arguments."""
    config = _create_config(url, transport, command, args)

    # Parse prompt arguments
    try:
        parsed_args = json.loads(prompt_args) if prompt_args else {}
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON arguments: {e}[/red]")
        raise typer.Exit(1)

    async def _get_prompt(client: MCPClient):
        messages = await client.get_prompt("cli-server", prompt_name, arguments=parsed_args)

        if output_json:
            messages_data = [{"role": m.type, "content": m.content} for m in messages]
            console.print(json.dumps(messages_data, indent=2))
        else:
            console.print(
                Panel(
                    f"[green][OK] Prompt '{prompt_name}' loaded[/green]\nMessages: {len(messages)}",
                    title="MCP Prompt",
                    border_style="green",
                )
            )

            for msg in messages:
                role_color = "blue" if msg.type == "human" else "green"
                console.print(f"\n[{role_color}]{msg.type.upper()}:[/{role_color}]")
                console.print(msg.content)

    asyncio.run(_run_with_client(config, timeout, f"Get prompt '{prompt_name}'", _get_prompt))


# =============================================================================
# mcp resource - Get a resource from MCP server
# =============================================================================


@app.command("resource")
def resource_cmd(
    url: str | None = typer.Argument(None, help="MCP server URL"),
    resource_uri: str = typer.Argument(..., help="Resource URI to fetch"),
    transport: str = typer.Option(
        "streamable_http",
        "--transport",
        "-t",
        help="Transport type: streamable_http, stdio",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command for stdio transport",
    ),
    args: str | None = typer.Option(
        None,
        "--args",
        "-a",
        help="Arguments for stdio command (space-separated)",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        help="Request timeout in seconds",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
    output_file: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write resource content to file",
    ),
):
    """Fetch a resource from an MCP server."""
    config = _create_config(url, transport, command, args)

    async def _get_resource(client: MCPClient):
        resources: list[MCPResource] = await client.get_resources("cli-server", uris=resource_uri)

        if not resources:
            console.print(f"[red]Resource not found: {resource_uri}[/red]")
            raise typer.Exit(1)

        resource = resources[0]

        if output_file:
            # Write to file
            mode = "w" if resource.is_text else "wb"
            data = resource.data
            with open(output_file, mode) as f:
                f.write(data)
            console.print(f"[green][OK] Written to {output_file}[/green]")
        elif output_json:
            resource_data = {
                "uri": resource.uri,
                "mime_type": resource.mime_type,
                "is_text": resource.is_text,
                "content": (
                    resource.as_text()
                    if resource.is_text
                    else f"<binary: {len(resource.as_bytes())} bytes>"
                ),
            }
            console.print(json.dumps(resource_data, indent=2))
        else:
            console.print(
                Panel(
                    f"[green][OK] Resource fetched[/green]\n"
                    f"URI: {resource.uri}\n"
                    f"MIME: {resource.mime_type or 'unknown'}\n"
                    f"Type: {'text' if resource.is_text else 'binary'}",
                    title="MCP Resource",
                    border_style="green",
                )
            )

            if resource.is_text:
                content = resource.as_text()
                if len(content) > 1000:
                    console.print(content[:1000] + "\n... (truncated)")
                else:
                    console.print(content)
            else:
                console.print(f"[dim]Binary content: {len(resource.as_bytes())} bytes[/dim]")

    asyncio.run(_run_with_client(config, timeout, f"Get resource '{resource_uri}'", _get_resource))


# =============================================================================
# mcp info - Get server metadata
# =============================================================================


@app.command("info")
def info_cmd(
    url: str | None = typer.Argument(None, help="MCP server URL"),
    transport: str = typer.Option(
        "streamable_http",
        "--transport",
        "-t",
        help="Transport type: streamable_http, stdio",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command for stdio transport",
    ),
    args: str | None = typer.Option(
        None,
        "--args",
        "-a",
        help="Arguments for stdio command (space-separated)",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        help="Connection timeout in seconds",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Get MCP server information and capabilities."""
    config = _create_config(url, transport, command, args)

    async def _get_info(client: MCPClient):
        # Get server info
        info: dict[str, Any] = {
            "name": "cli-server",
            "transport": transport,
            "url": url if transport != "stdio" else None,
            "command": command if transport == "stdio" else None,
        }

        # Get counts (list_tools returns a List)
        tools = await client.list_tools(server="cli-server")
        info["tools_count"] = len(tools)

        try:
            prompts = await client.list_prompts(server_name="cli-server")
            info["prompts_count"] = len(prompts.get("cli-server", []))
        except Exception:
            info["prompts_count"] = 0

        try:
            resources = await client.list_resources(server_name="cli-server")
            info["resources_count"] = len(resources.get("cli-server", []))
        except Exception:
            info["resources_count"] = 0

        if output_json:
            console.print(json.dumps(info, indent=2))
        else:
            # Build tree view
            tree = Tree("[bold]MCP Server Info[/bold]")

            conn = tree.add(f"[{BRAND_ACCENT}]Connection[/{BRAND_ACCENT}]")
            conn.add(f"Transport: {transport}")
            if url:
                conn.add(f"URL: {url}")
            if command:
                conn.add(f"Command: {command}")
                if args:
                    conn.add(f"Args: {args}")

            caps = tree.add(f"[{BRAND_ACCENT}]Capabilities[/{BRAND_ACCENT}]")
            caps.add(f"Tools: {info['tools_count']}")
            caps.add(f"Prompts: {info['prompts_count']}")
            caps.add(f"Resources: {info['resources_count']}")

            console.print(tree)

    asyncio.run(_run_with_client(config, timeout, "Get server info", _get_info))


# =============================================================================
# Registration
# =============================================================================


def register(parent: typer.Typer):
    """Register MCP commands with the parent CLI app."""
    # Hidden from main help - advanced debugging tools
    parent.add_typer(app, name="mcp", help="MCP server debugging and testing", hidden=True)
