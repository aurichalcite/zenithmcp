.. ZenithMCP documentation master file

Welcome to ZenithMCP's Documentation
=====================================

**ZenithMCP** is a high-performance server implementation of the Model Context Protocol (MCP) that provides real-time, relevant context from private codebases to LLM-powered coding agents.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   architecture
   usage
   api

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   contributing
   development
   testing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Key Features
------------

* **Model Context Protocol (MCP) Standard**: Built as a server implementation of the open MCP standard
* **Two-Stage RAG Pipeline**: Sophisticated retrieval architecture for high precision and low latency
* **Structure-Aware Code Parsing**: Uses tree-sitter for semantically coherent code chunking
* **GraphCodeBERT Embeddings**: Understands both text and code structure for superior retrieval
* **High Performance**: Built on modern async Python stack with FastAPI and Qdrant
* **Modular Architecture**: Four distinct services for scalability and maintainability

Quick Example
-------------

.. code-block:: python

   from zenithmcp import server
   from zenithmcp.rag_core import CandidateRetriever

   # Initialize the retriever
   retriever = CandidateRetriever()

   # Search for relevant code
   results = await retriever.search("calculate sum of numbers")

   # Start the MCP server
   import uvicorn
   uvicorn.run(server.app, host="0.0.0.0", port=8000)

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
