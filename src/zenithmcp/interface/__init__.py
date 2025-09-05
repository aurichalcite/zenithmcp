"""MCP Interface Layer."""

from zenithmcp.interface.server import app

__all__ = ["app", "server"]

# Re-export server module for convenience
from zenithmcp.interface import server
