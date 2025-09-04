"""MCP Interface Layer - FastAPI server implementation."""

from fastapi import FastAPI

app = FastAPI(title="ZenithMCP", version="0.1.0")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "ZenithMCP Server", "version": "0.1.0"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
