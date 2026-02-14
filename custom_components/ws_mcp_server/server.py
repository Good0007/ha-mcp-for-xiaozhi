"""The Model Context Protocol Server implementation.

The Model Context Protocol python sdk defines a Server API that provides the
MCP message handling logic and error handling. The server implementation provided
here is independent of the lower level transport protocol.

See https://modelcontextprotocol.io/docs/concepts/architecture#implementation-example
"""

from collections.abc import Callable, Sequence
import json
import logging
from typing import Any

from mcp import types
from mcp.server import Server
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from .const import STATELESS_LLM_API

_LOGGER = logging.getLogger(__name__)


def _list_entities_tool(hass: HomeAssistant, area_filter: str | None = None) -> list[dict]:
    """List all available entities with their details.
    
    Args:
        hass: Home Assistant instance
        area_filter: Optional area name to filter entities
        
    Returns:
        List of entity details including name, entity_id, area, and state
    """
    from homeassistant.helpers import area_registry, device_registry, entity_registry
    
    ent_reg = entity_registry.async_get(hass)
    dev_reg = device_registry.async_get(hass)
    area_reg = area_registry.async_get(hass)
    
    entities_list = []
    for state in hass.states.async_all():
        ent_id = state.entity_id
        friendly_name = state.attributes.get("friendly_name", "")
        
        # Get area information
        entry = ent_reg.async_get(ent_id)
        area_name = "Unknown"
        if entry:
            if entry.area_id:
                area_entry = area_reg.async_get_area(entry.area_id)
                if area_entry:
                    area_name = area_entry.name
            elif entry.device_id:
                device = dev_reg.async_get(entry.device_id)
                if device and device.area_id:
                    area_entry = area_reg.async_get_area(device.area_id)
                    if area_entry:
                        area_name = area_entry.name
        
        if not area_filter or area_filter in area_name:
            entities_list.append({
                "name": friendly_name,
                "entity_id": ent_id,
                "area": area_name,
                "state": state.state
            })
    
    return entities_list


def _get_entity_state_tool(hass: HomeAssistant, entity_id_or_name: str) -> dict:
    """Get detailed state and attributes for a specific entity.
    
    Args:
        hass: Home Assistant instance
        entity_id_or_name: Either entity_id (e.g., 'light.kitchen') or friendly name
        
    Returns:
        Dictionary with entity details including state and all attributes
        
    Raises:
        ValueError: If entity is not found
    """
    # First try to match by entity_id
    state = hass.states.get(entity_id_or_name)
    
    # If not found, try to search by friendly_name or entity name
    if not state:
        for s in hass.states.async_all():
            if (s.attributes.get("friendly_name") == entity_id_or_name or 
                s.entity_id.split('.')[-1] == entity_id_or_name):
                state = s
                break
    
    if not state:
        raise ValueError(
            f"Entity '{entity_id_or_name}' not found. "
            "Please use ListEntities to find the correct entity_id."
        )
    
    return {
        "entity_id": state.entity_id,
        "name": state.attributes.get("friendly_name"),
        "state": state.state,
        "attributes": dict(state.attributes),
        "last_changed": state.last_changed.isoformat(),
    }


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> types.Tool:
    """Format tool specification."""
    input_schema = convert(tool.parameters, custom_serializer=custom_serializer)
    return types.Tool(
        name=tool.name,
        description=tool.description or "",
        inputSchema={
            "type": "object",
            "properties": input_schema["properties"],
        },
    )


async def create_server(
    hass: HomeAssistant, llm_api_id: str | list[str], llm_context: llm.LLMContext
) -> Server:
    """Create a new Model Context Protocol Server.

    A Model Context Protocol Server object is associated with a single session.
    The MCP SDK handles the details of the protocol.
    """
    _LOGGER.info("Creating MCP Server for LLM API: %s", llm_api_id)
    
    # Backwards compatibility with old MCP Server config
    if llm_api_id == STATELESS_LLM_API:
        _LOGGER.debug("Converting STATELESS_LLM_API to LLM_API_ASSIST")
        llm_api_id = llm.LLM_API_ASSIST

    server = Server("home-assistant")

    async def get_api_instance() -> llm.APIInstance:
        """Get the LLM API selected."""
        _LOGGER.debug("Getting API instance for llm_api_id=%s", llm_api_id)
        api_instance = await llm.async_get_api(hass, llm_api_id, llm_context)
        _LOGGER.debug("API instance: %s with %d tools", api_instance.api.name, len(api_instance.tools))
        return api_instance

    @server.list_prompts()  # type: ignore[no-untyped-call, misc]
    async def handle_list_prompts() -> list[types.Prompt]:
        llm_api = await get_api_instance()
        return [
            types.Prompt(
                name=llm_api.api.name,
                description=f"Default prompt for Home Assistant {llm_api.api.name} API",
            )
        ]

    @server.get_prompt()  # type: ignore[no-untyped-call, misc]
    async def handle_get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> types.GetPromptResult:
        llm_api = await get_api_instance()
        
        if name != llm_api.api.name:
            _LOGGER.warning("Unknown prompt: %s (expected: %s)", name, llm_api.api.name)
            raise ValueError(f"Unknown prompt: {name}")

        _LOGGER.info(
            "[MCP] get_prompt: %s",
            json.dumps({"name": name, "prompt_length": len(llm_api.api_prompt)}, ensure_ascii=False)
        )
        
        return types.GetPromptResult(
            description=f"Default prompt for Home Assistant {llm_api.api.name} API",
            messages=[
                types.PromptMessage(
                    role="assistant",
                    content=types.TextContent(
                        type="text",
                        text=llm_api.api_prompt,
                    ),
                )
            ],
        )

    @server.list_tools()  # type: ignore[no-untyped-call, misc]
    async def list_tools() -> list[types.Tool]:
        """List available tools."""
        llm_api = await get_api_instance()
        
        formatted_tools = [_format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools]
        
        # Add custom entity query tools
        formatted_tools.extend([
            types.Tool(
                name="ListEntities",
                description="List all available entities in Home Assistant with their IDs, names, areas, and current states. Use this when you need to discover entity_id values before controlling devices.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "area": {
                            "type": "string",
                            "description": "Optional: Filter entities by area name (e.g., 'Living Room', 'Kitchen')",
                        }
                    },
                },
            ),
            types.Tool(
                name="GetEntityState",
                description="Retrieve detailed state information and all attributes for a specific entity. Accepts either entity_id (e.g., 'light.kitchen') or friendly name (e.g., 'Kitchen Light'). Returns comprehensive entity data including current state, attributes, and last update time.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity_id_or_name": {
                            "type": "string",
                            "description": "Entity ID (like 'light.kitchen') or friendly name (like 'Kitchen Light')",
                        }
                    },
                    "required": ["entity_id_or_name"],
                },
            )
        ])
        
        tool_names = [tool.name for tool in formatted_tools]
        _LOGGER.info(
            "[MCP] list_tools: %s",
            json.dumps({"count": len(formatted_tools), "tools": tool_names}, ensure_ascii=False)
        )
        return formatted_tools

    @server.call_tool()  # type: ignore[no-untyped-call, misc]
    async def call_tool(name: str, arguments: dict) -> Sequence[types.TextContent]:
        """Handle calling tools."""
        _LOGGER.info(
            "[MCP] call_tool: %s",
            json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=False)
        )
        
        # Handle custom tools
        try:
            if name == "ListEntities":
                area_filter = arguments.get("area")
                entities_list = _list_entities_tool(hass, area_filter)
                result_text = json.dumps(entities_list, ensure_ascii=False)
                _LOGGER.info(
                    "[MCP] call_tool_result: %s",
                    json.dumps({"tool": name, "result_count": len(entities_list)}, ensure_ascii=False)
                )
                return [types.TextContent(type="text", text=result_text)]
            
            if name == "GetEntityState":
                entity_id_or_name = arguments.get("entity_id_or_name")
                if not entity_id_or_name:
                    raise ValueError("Missing required parameter: entity_id_or_name")
                
                detail = _get_entity_state_tool(hass, entity_id_or_name)
                result_text = json.dumps(detail, ensure_ascii=False)
                _LOGGER.info(
                    "[MCP] call_tool_result: %s",
                    json.dumps({"tool": name, "entity_id": detail["entity_id"], "state": detail["state"]}, ensure_ascii=False)
                )
                return [types.TextContent(type="text", text=result_text)]
        except Exception as e:
            error_msg = f"Error: {e}"
            _LOGGER.error(
                "[MCP] call_tool_error: %s",
                json.dumps({"tool": name, "error": str(e)}, ensure_ascii=False)
            )
            return [types.TextContent(type="text", text=error_msg)]
        
        # Handle LLM API tools
        llm_api = await get_api_instance()
        tool_input = llm.ToolInput(tool_name=name, tool_args=arguments)

        try:
            tool_response = await llm_api.async_call_tool(tool_input)
            result_text = json.dumps(tool_response, ensure_ascii=False)
            result_preview = result_text[:200] + "..." if len(result_text) > 200 else result_text
            _LOGGER.info(
                "[MCP] call_tool_result: %s",
                json.dumps({"tool": name, "success": True, "result_preview": result_preview}, ensure_ascii=False)
            )
        except (HomeAssistantError, vol.Invalid) as e:
            _LOGGER.error(
                "[MCP] call_tool_error: %s",
                json.dumps({"tool": name, "error": str(e)}, ensure_ascii=False)
            )
            raise HomeAssistantError(f"Error calling tool: {e}") from e
            
        return [
            types.TextContent(
                type="text",
                text=json.dumps(tool_response, ensure_ascii=False),
            )
        ]

    return server

