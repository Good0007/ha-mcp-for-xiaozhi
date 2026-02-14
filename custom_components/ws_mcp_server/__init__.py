
"""The Model Context Protocol Server integration."""

from __future__ import annotations
import asyncio

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType
import logging
from . import transport
from .const import DOMAIN
from .session import SessionManager
from .types import WsMCPServerConfigEntry
_LOGGER = logging.getLogger(__name__)

__all__ = [
    "CONFIG_SCHEMA",
    "DOMAIN",
    "async_setup",
    "async_setup_entry",
    "async_unload_entry",
]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Model Context Protocol component."""
    #websocket_transport.async_register(hass)
    return True



async def async_setup_entry(hass: HomeAssistant, entry: WsMCPServerConfigEntry) -> bool:
    """Set up Model Context Protocol Server from a config entry."""
    async def _system_started(event):
        session_manager = SessionManager()
        entry.runtime_data = session_manager
        
        # Start the connection loop and store the task
        connect_task = hass.async_create_task(transport.async_setup_entry(hass, entry))
        
        # 存储到 hass.data 以便于访问
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
            "session_manager": session_manager,
            "config": entry.data,
            "connect_task": connect_task
        }
        
        try:
            hass.states.async_set(f"{DOMAIN}.{entry.entry_id}_status", "connecting")
            # Note: We don't await the task here as it runs continuously
            hass.states.async_set(f"{DOMAIN}.{entry.entry_id}_status", "connected")
        except Exception as ex:
            _LOGGER.error("MCP connect failed: %s", ex)
            hass.states.async_set(f"{DOMAIN}.{entry.entry_id}_status", "error")
            # Cleanup
            if DOMAIN in hass.data and entry.entry_id in hass.data[DOMAIN]:
                hass.data[DOMAIN].pop(entry.entry_id)
            raise
    
    if hass.is_running:
        await _system_started(None)
    else:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _system_started)
    return True
    


async def async_unload_entry(hass: HomeAssistant, entry: WsMCPServerConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading MCP Server entry: %s", entry.entry_id)
    
    if DOMAIN in hass.data and entry.entry_id in hass.data[DOMAIN]:
        data = hass.data[DOMAIN][entry.entry_id]
        session_manager = data["session_manager"]
        connect_task = data.get("connect_task")
        
        # Signal the session manager to stop
        session_manager.close()
        
        # Cancel the connection task if it exists
        if connect_task and not connect_task.done():
            _LOGGER.debug("Cancelling connection task")
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                _LOGGER.debug("Connection task cancelled successfully")
        
        # Remove from hass.data
        hass.data[DOMAIN].pop(entry.entry_id)
        
        # Update status
        hass.states.async_set(f"{DOMAIN}.{entry.entry_id}_status", "disconnected")
    
    _LOGGER.info("MCP Server entry unloaded: %s", entry.entry_id)
    return True
