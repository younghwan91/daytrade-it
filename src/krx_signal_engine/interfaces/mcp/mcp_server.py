"""
MCP Server Entry Point for GPT Quant v2

This module provides the MCP server implementation that exposes trading tools
to AI agents following the Model Context Protocol specification.

Usage:
    # Run as a module
    python -m krx_signal_engine.interfaces.mcp.mcp_server

    # Or import and run programmatically
    from krx_signal_engine.interfaces.mcp.mcp_server import MCPServer
    server = MCPServer()
    await server.run()
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any

from krx_signal_engine.interfaces.mcp.tools import (
    MCPToolContext,
    get_mcp_tools,
    get_tool_by_name,
)
from krx_signal_engine.shared.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# MCP Protocol Constants and Types
# =============================================================================


class MCPErrorCode(IntEnum):
    """MCP JSON-RPC error codes following the specification."""

    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific errors (-32000 to -32099)
    TOOL_NOT_FOUND = -32001
    TOOL_EXECUTION_ERROR = -32002
    AUTHENTICATION_ERROR = -32003
    RESOURCE_NOT_FOUND = -32004
    RATE_LIMIT_EXCEEDED = -32005
    INVALID_TOOL_INPUT = -32006


MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "gpt-quant-v2"
SERVER_VERSION = "1.0.0"


@dataclass
class MCPError(Exception):
    """MCP protocol error with code and data."""

    code: MCPErrorCode
    message: str
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-RPC error object."""
        error = {
            "code": int(self.code),
            "message": self.message,
        }
        if self.data:
            error["data"] = self.data
        return error


@dataclass
class MCPRequest:
    """Parsed MCP request."""

    jsonrpc: str
    method: str
    id: str | int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """MCP response structure."""

    jsonrpc: str = "2.0"
    id: str | int | None = None
    result: Any = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-RPC response object."""
        response: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response


# =============================================================================
# MCP Server Implementation
# =============================================================================


class MCPServer:
    """MCP Server implementing the Model Context Protocol.

    This server exposes trading tools to AI agents via JSON-RPC over stdio.
    It handles tool discovery, invocation, and error handling according to
    the MCP specification.

    Example:
        server = MCPServer()
        await server.run()  # Runs the server on stdio
    """

    def __init__(
        self,
        context: MCPToolContext | None = None,
        name: str = SERVER_NAME,
        version: str = SERVER_VERSION,
    ) -> None:
        """Initialize the MCP server.

        Args:
            context: Tool execution context with service dependencies
            name: Server name for capability negotiation
            version: Server version string
        """
        self.name = name
        self.version = version
        self.context = context or MCPToolContext.get_instance()
        self._initialized = False
        self._request_id = 0

        # Register method handlers
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping,
            "shutdown": self._handle_shutdown,
        }

        logger.info(
            "MCP Server initialized",
            server_name=self.name,
            server_version=self.version,
        )

    async def run(self) -> None:
        """Run the MCP server on stdin/stdout.

        This method reads JSON-RPC requests from stdin and writes
        responses to stdout, following the MCP protocol.
        """
        logger.info("Starting MCP server on stdio")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, reader, asyncio.get_event_loop()
        )

        try:
            while True:
                try:
                    line = await reader.readline()
                    if not line:
                        break

                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    response = await self.handle_request(line_str)
                    if response:
                        response_str = json.dumps(response.to_dict()) + "\n"
                        writer.write(response_str.encode("utf-8"))
                        await writer.drain()

                except asyncio.CancelledError:
                    logger.info("Server shutdown requested")
                    break
                except Exception as e:
                    logger.error("Error processing request", error=str(e))
                    error_response = MCPResponse(
                        error=MCPError(
                            code=MCPErrorCode.INTERNAL_ERROR,
                            message=str(e),
                        ).to_dict()
                    )
                    writer.write((json.dumps(error_response.to_dict()) + "\n").encode("utf-8"))
                    await writer.drain()

        finally:
            writer.close()
            logger.info("MCP server stopped")

    async def handle_request(self, request_str: str) -> MCPResponse | None:
        """Handle a single MCP request.

        Args:
            request_str: JSON-RPC request string

        Returns:
            MCPResponse or None for notifications
        """
        try:
            request = self._parse_request(request_str)
        except MCPError as e:
            return MCPResponse(error=e.to_dict())

        handler = self._handlers.get(request.method)
        if not handler:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCode.METHOD_NOT_FOUND,
                    message=f"Method not found: {request.method}",
                ).to_dict(),
            )

        try:
            result = await handler(request)

            # Notifications (no id) don't get responses
            if request.id is None:
                return None

            return MCPResponse(id=request.id, result=result)

        except MCPError as e:
            return MCPResponse(id=request.id, error=e.to_dict())
        except Exception as e:
            logger.error(
                "Error handling request",
                method=request.method,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCode.INTERNAL_ERROR,
                    message=str(e),
                    data={"traceback": traceback.format_exc()},
                ).to_dict(),
            )

    def _parse_request(self, request_str: str) -> MCPRequest:
        """Parse a JSON-RPC request string.

        Args:
            request_str: Raw JSON string

        Returns:
            Parsed MCPRequest

        Raises:
            MCPError: If parsing fails
        """
        try:
            data = json.loads(request_str)
        except json.JSONDecodeError as e:
            raise MCPError(
                code=MCPErrorCode.PARSE_ERROR,
                message=f"Invalid JSON: {e}",
            ) from e

        if not isinstance(data, dict):
            raise MCPError(
                code=MCPErrorCode.INVALID_REQUEST,
                message="Request must be an object",
            )

        if data.get("jsonrpc") != "2.0":
            raise MCPError(
                code=MCPErrorCode.INVALID_REQUEST,
                message="Invalid JSON-RPC version",
            )

        method = data.get("method")
        if not isinstance(method, str):
            raise MCPError(
                code=MCPErrorCode.INVALID_REQUEST,
                message="Method must be a string",
            )

        return MCPRequest(
            jsonrpc=data["jsonrpc"],
            method=method,
            id=data.get("id"),
            params=data.get("params", {}),
        )

    # =========================================================================
    # Protocol Handlers
    # =========================================================================

    async def _handle_initialize(self, request: MCPRequest) -> dict[str, Any]:
        """Handle the initialize request.

        Returns server capabilities and protocol version.
        """
        client_info = request.params.get("clientInfo", {})
        protocol_version = request.params.get("protocolVersion", MCP_PROTOCOL_VERSION)

        logger.info(
            "Client initializing",
            client_name=client_info.get("name"),
            client_version=client_info.get("version"),
            protocol_version=protocol_version,
        )

        self._initialized = True

        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
                "resources": {
                    "subscribe": False,
                    "listChanged": False,
                },
                "prompts": {
                    "listChanged": False,
                },
                "logging": {},
            },
        }

    async def _handle_initialized(self, request: MCPRequest) -> dict[str, Any]:
        """Handle the initialized notification."""
        logger.info("Client initialized successfully")
        return {}

    async def _handle_tools_list(self, request: MCPRequest) -> dict[str, Any]:
        """Handle tools/list request.

        Returns list of available tools with their schemas.
        """
        tools = get_mcp_tools()

        # Convert to MCP tool format
        mcp_tools = []
        for tool in tools:
            # Build JSON Schema for parameters
            properties = {}
            required = []

            for param in tool.get("parameters", []):
                param_schema: dict[str, Any] = {
                    "description": param.get("description", ""),
                }

                # Map Python types to JSON Schema types
                param_type = param.get("type", "string")
                if param_type in ("str", "string"):
                    param_schema["type"] = "string"
                elif param_type in ("int", "integer"):
                    param_schema["type"] = "integer"
                elif param_type in ("float", "number"):
                    param_schema["type"] = "number"
                elif param_type in ("bool", "boolean"):
                    param_schema["type"] = "boolean"
                elif param_type in ("list", "array"):
                    param_schema["type"] = "array"
                elif param_type in ("dict", "object"):
                    param_schema["type"] = "object"
                else:
                    param_schema["type"] = "string"

                if "default" in param:
                    param_schema["default"] = param["default"]

                properties[param["name"]] = param_schema

                if param.get("required", False):
                    required.append(param["name"])

            mcp_tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )

        return {"tools": mcp_tools}

    async def _handle_tools_call(self, request: MCPRequest) -> dict[str, Any]:
        """Handle tools/call request.

        Invokes the specified tool with provided arguments.
        """
        tool_name = request.params.get("name")
        arguments = request.params.get("arguments", {})

        if not tool_name:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Tool name is required",
            )

        logger.info(
            "Tool call requested",
            tool=tool_name,
            arguments=list(arguments.keys()),
        )

        # Get the tool function
        tool_func = get_tool_by_name(tool_name)
        if not tool_func:
            raise MCPError(
                code=MCPErrorCode.TOOL_NOT_FOUND,
                message=f"Tool not found: {tool_name}",
                data={"available_tools": [t["name"] for t in get_mcp_tools()]},
            )

        try:
            # Execute the tool
            result = await tool_func(**arguments)

            # Serialize the result
            if hasattr(result, "model_dump"):
                # Pydantic model
                serialized = result.model_dump(mode="json")
            elif hasattr(result, "dict"):
                # Legacy Pydantic
                serialized = result.dict()
            elif isinstance(result, list):
                serialized = [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in result
                ]
            else:
                serialized = result

            logger.info(
                "Tool call completed",
                tool=tool_name,
                success=True,
            )

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(serialized, indent=2, default=str),
                    }
                ],
                "isError": False,
            }

        except ValueError as e:
            logger.warning(
                "Tool call validation error",
                tool=tool_name,
                error=str(e),
            )
            raise MCPError(
                code=MCPErrorCode.INVALID_TOOL_INPUT,
                message=str(e),
            ) from e

        except Exception as e:
            logger.error(
                "Tool execution error",
                tool=tool_name,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            raise MCPError(
                code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Tool execution failed: {e}",
                data={"tool": tool_name, "error_type": type(e).__name__},
            ) from e

    async def _handle_ping(self, request: MCPRequest) -> dict[str, Any]:
        """Handle ping request for health checks."""
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    async def _handle_shutdown(self, request: MCPRequest) -> dict[str, Any]:
        """Handle shutdown request."""
        logger.info("Shutdown requested")
        return {"status": "shutting_down"}


# =============================================================================
# Server Factory and Entry Point
# =============================================================================


def create_server(
    tiingo_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> MCPServer:
    """Create an MCP server with configured services.

    Args:
        tiingo_api_key: Optional Tiingo API key
        openai_api_key: Optional OpenAI API key

    Returns:
        Configured MCPServer instance
    """
    # Create context with available services
    context = MCPToolContext()

    # Initialize Tiingo client if API key provided
    if tiingo_api_key:
        try:
            from krx_signal_engine.infrastructure.external.tiingo_client import TiingoClient

            context.tiingo_client = TiingoClient(api_key=tiingo_api_key)
        except Exception as e:
            logger.warning("Failed to initialize Tiingo client", error=str(e))

    # Initialize OpenAI client if API key provided
    if openai_api_key:
        try:
            from krx_signal_engine.infrastructure.external.openai_client import (
                OpenAIClient,
                OpenAIClientConfig,
            )

            config = OpenAIClientConfig(api_key=openai_api_key)
            context.openai_client = OpenAIClient(config=config)
        except Exception as e:
            logger.warning("Failed to initialize OpenAI client", error=str(e))

    MCPToolContext.set_instance(context)

    return MCPServer(context=context)


async def main() -> None:
    """Main entry point for running the MCP server."""
    import os

    # Get API keys from environment
    tiingo_key = os.environ.get("TIINGO_API_KEY") or os.environ.get("GPTQ_TIINGO_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GPTQ_OPENAI_API_KEY")

    server = create_server(
        tiingo_api_key=tiingo_key,
        openai_api_key=openai_key,
    )

    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
