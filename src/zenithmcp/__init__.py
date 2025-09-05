"""
ZenithMCP - High-performance MCP server for code context retrieval.

An open-source implementation of the Model Context Protocol that provides
real-time, relevant context from private codebases to LLM-powered coding agents.
"""

__version__ = "0.1.0"
__author__ = "ZenithMCP Contributors"
__license__ = "Apache-2.0"

from zenithmcp.interface import server
from zenithmcp.rag_core import retrieval

__all__ = ["__version__", "retrieval", "server"]
