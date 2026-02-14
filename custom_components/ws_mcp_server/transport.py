import logging
import anyio
import asyncio
import aiohttp
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import types
from mcp.shared.message import SessionMessage

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm

from .const import DOMAIN
from .server import create_server
from .session import Session
from .types import WsMCPServerConfigEntry

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant,  entry: WsMCPServerConfigEntry) -> None:
    """Set up MCP Server from a config entry and run the connection loop."""
    await _connect_loop(hass, entry)

def async_get_config_entry(hass: HomeAssistant) -> WsMCPServerConfigEntry:
    """Get the first enabled MCP server config entry."""
    config_entries: list[WsMCPServerConfigEntry] = (
        hass.config_entries.async_loaded_entries(DOMAIN)
    )
    if not config_entries:
        raise RuntimeError("Model Context Protocol server is not configured")
    if len(config_entries) > 1:
        raise RuntimeError("Found multiple Model Context Protocol configurations")
    return config_entries[0]


async def _connect_loop(hass: HomeAssistant, entry: WsMCPServerConfigEntry) -> None:
    """Reconnect on failure loop."""
    session_manager = entry.runtime_data
    
    while not session_manager.is_stopping():
        try:
            _LOGGER.info("Starting WebSocket connection")
            should_reconnect = await _connect_to_client(hass, entry)
            if not should_reconnect or session_manager.is_stopping():
                _LOGGER.info("Graceful disconnect, stopping reconnection loop")
                break
        except Exception as e:
            if session_manager.is_stopping():
                _LOGGER.info("Session manager stopping, exiting connection loop")
                break
            _LOGGER.warning("WebSocket disconnected or failed: %s", e)
        
        if not session_manager.is_stopping():
            _LOGGER.info("Reconnecting in 20 seconds")
            await asyncio.sleep(20)
    
    _LOGGER.info("Connection loop terminated")


async def _ws_reader(
    ws: aiohttp.ClientWebSocketResponse,
    read_stream_writer: MemoryObjectSendStream,
    tg: anyio.abc.TaskGroup,
) -> None:
    """读取 WebSocket 消息并转发到 MCP server。"""
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    json_data = msg.json()
                    message = types.JSONRPCMessage.model_validate(json_data)
                    session_message = SessionMessage(message=message)
                    _LOGGER.debug("Received message: %s", message)
                    await read_stream_writer.send(session_message)
                except Exception as err:
                    _LOGGER.error("Invalid message from client: %s", err)
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                _LOGGER.info("WebSocket closed by client: %s", msg.extra)
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                _LOGGER.error("WebSocket error: %s", msg.data)
                break
    finally:
        _LOGGER.debug("WebSocket reader ended, cancelling task group")
        tg.cancel_scope.cancel()


async def _ws_writer(
    ws: aiohttp.ClientWebSocketResponse,
    write_stream_reader: MemoryObjectReceiveStream,
) -> None:
    """从 MCP server 读取消息并发送到 WebSocket。"""
    try:
        async for session_message in write_stream_reader:
            message = session_message.message
            _LOGGER.debug("Sending message: %s", message)
            await ws.send_str(message.model_dump_json(by_alias=True, exclude_none=True))
    except Exception as e:
        _LOGGER.debug("Write stream ended: %s", e)
    finally:
        _LOGGER.debug("Closing WebSocket connection")
        await ws.close()


async def _heartbeat(ws: aiohttp.ClientWebSocketResponse) -> None:
    """保持 WebSocket 连接活跃。"""
    try:
        while True:
            await asyncio.sleep(50)
            _LOGGER.debug("Sending heartbeat ping")
            await ws.ping()
    except Exception as e:
        _LOGGER.debug("Heartbeat failed: %s", e)


async def _connect_to_client(hass: HomeAssistant, entry: WsMCPServerConfigEntry) -> bool:
    """Connect to external WebSocket endpoint as MCP server.
    
    Returns:
        True if should reconnect on failure, False for graceful disconnect.
    """
    session_manager = entry.runtime_data
    endpoint = entry.data.get("client_endpoint")
    if not endpoint:
        _LOGGER.error("No client endpoint configured in config entry")
        return False

    _LOGGER.info("Connecting to MCP client at: %s", endpoint)
    
    context = llm.LLMContext(
        platform=DOMAIN,
        context={},
        language="*",
        assistant=conversation.DOMAIN,
        device_id=None,
    )
    llm_api_id = entry.data[CONF_LLM_HASS_API]
    server = await create_server(hass, llm_api_id, context)
    options = await hass.async_add_executor_job(server.create_initialization_options)

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)
    
    try:
        async with session_manager.create(Session(read_stream_writer)) as session_id:
            _LOGGER.debug("Created session: %s", session_id)
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as client_session:
                async with client_session.ws_connect(endpoint) as ws:
                    _LOGGER.info("WebSocket connected successfully")
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(_ws_reader, ws, read_stream_writer, tg)
                        tg.start_soon(_ws_writer, ws, write_stream_reader)
                        tg.start_soon(_heartbeat, ws)
                        await server.run(read_stream, write_stream, options)
    except Exception as e:
        _LOGGER.exception("Failed to connect to client WebSocket at %s: %s", endpoint, e)
        return True
    
    return True
