"""tool_mappings — Modular tool-call format mapping system for MLXr.

Typical usage in server.py::

    from tool_mappings import get_registry
    from tool_mappings.converter import MappedToolCallParser

    # At request time:
    mapping = get_registry().resolve(cur.name) if tools_active else None
    tool_parser = MappedToolCallParser(enabled=tools_active, mapping=mapping)

    # Drop-in for ToolCallParser — feed/flush/try_extract_raw_json work identically.
"""
from __future__ import annotations

from typing import Optional

from .registry import MappingRegistry
from .converter import MappedToolCallParser
from .schema import MappingDefinition

__all__ = ["get_registry", "MappingRegistry", "MappedToolCallParser", "MappingDefinition"]

_registry: Optional[MappingRegistry] = None


def get_registry() -> MappingRegistry:
    """Return the global MappingRegistry singleton, initialising it on first call."""
    global _registry
    if _registry is None:
        _registry = MappingRegistry()
    return _registry
