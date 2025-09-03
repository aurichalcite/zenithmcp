Of course. Here is the detailed `ARCHITECTURE.md` document based on the provided sources.

***

# ZenithMCP Architecture

## 1. Executive Summary

This document outlines the architectural blueprint for ZenithMCP, an open-source, high-performance server for the Model Context Protocol (MCP). The server is designed to solve the "context gap" faced by Large Language Models (LLMs) in software development by providing them with real-time, relevant context from private or recently developed codebases.

The core of this architecture is a specialized, **two-stage Retrieval-Augmented Generation (RAG) pipeline** optimized for the semantic and structural complexities of source code. By indexing private repositories, the system empowers LLM-powered coding agents to reason about and generate code for projects not included in their training data, significantly reducing hallucinations and improving overall utility.

The system is built on a modular, four-pillar design to promote separation of concerns, scalability, and maintainability. This document details each of these pillars, the retrieval strategy, the data processing pipeline, and the mandated technology stack.

## 2. The Model Context Protocol (MCP)

ZenithMCP is a server implementation of the Model Context Protocol (MCP), an open-source standard introduced by Anthropic to standardize communication between AI systems and external services. Its rapid adoption by providers like OpenAI and Google DeepMind positions it as a foundational protocol for AI interoperability.

The protocol uses a client-server architecture with three main components:
*   **MCP Host:** The user-facing LLM application (e.g., an AI-powered IDE).
*   **MCP Client:** A component within the Host that manages communication.
*   **MCP Server:** The external service (ZenithMCP) that exposes data and capabilities.

An MCP server offers its capabilities through three standardized feature types: **Resources** (read-only data), **Tools** (executable functions), and **Prompts** (reusable templates). Communication is handled via JSON-RPC 2.0 over either `stdio` for local servers or HTTP with Server-Sent Events (SSE) for remote ones.

## 3. High-Level System Architecture

The ZenithMCP system is architected as four distinct, interacting services. This modular design is a non-negotiable principle of the project. The physical codebase is mandated to reflect this logical structure, with each service corresponding to a specific sub-package (e.g., `src/zenithmcp/interface/`).

![A diagram showing the four pillars of the architecture: MCP Interface Layer, Multi-Level Caching Service, RAG Core Engine, and Data Ingestion & Indexing Pipeline. Arrows indicate the flow of requests from the Interface Layer through the Caching Service to the RAG Core, and the Ingestion Pipeline feeding data into the RAG Core's knowledge base.]
*(Information from outside the provided sources: This is a conceptual representation of the architecture described in the sources for clarity.)*

### 3.1. The Four System Pillars

1.  **MCP Interface Layer**: This is the public-facing API gateway and the sole entry point for all external requests.
    *   **Responsibilities**: It handles all incoming JSON-RPC 2.0 requests, terminates TLS, validates authentication, parses MCP requests, and routes them to the appropriate internal services. It serves as the strict implementation of the MCP contract.
    *   **Technology**: Built with **FastAPI** and the core **mcp SDK** on a Streamable HTTP transport. The implementation will be managed in `src/zenithmcp/interface/`.

2.  **RAG Core Engine**: This is the intelligent heart of the server, responsible for finding and ranking relevant code context.
    *   **Responsibilities**: It orchestrates the two-stage retrieval process, interacting with the vector database and the re-ranking model to produce a concise, highly relevant set of context snippets for the LLM.
    *   **Location**: Implemented within the `src/zenithmcp/rag_core/` package.

3.  **Multi-Level Caching Service**: A critical component for optimizing performance and cost by avoiding redundant computation.
    *   **Responsibilities**: Sits between the Interface Layer and the RAG Core, intercepting requests and serving cached results for semantically similar queries, pre-computed embeddings, and final LLM responses.
    *   **Technology**: Implemented with a combination of **Redis** for key-value caching and a dedicated vector store for semantic query caching. It will be located in `src/zenithmcp/caching/`.

4.  **Data Ingestion & Indexing Pipeline**: An offline, asynchronous system responsible for populating the knowledge base that the RAG Core Engine uses.
    *   **Responsibilities**: It discovers new or updated code, parses it into semantically meaningful chunks, generates vector embeddings, and indexes them in the vector database.
    *   **Location**: Implemented within the `src/zenithmcp/ingestion/` package.

## 4. The Two-Stage RAG Core Strategy

Standard single-stage RAG systems force an unacceptable trade-off between speed (recall) and accuracy (precision). To overcome this, ZenithMCP mandates a **two-stage retrieval process** that delivers both high accuracy and low latency, a pattern proven effective in large-scale information retrieval systems.

### Stage 1: Candidate Retrieval (High Recall)

*   **Goal**: To maximize **recall** by quickly identifying a broad set of potentially relevant documents from the entire corpus. The objective is to cast a wide net to ensure the correct context is not missed, even at the cost of including some irrelevant results.
*   **Mechanism**: A user query is converted into a vector using a code-optimized embedding model (a bi-encoder). This vector is used to perform an **Approximate Nearest Neighbor (ANN) search** against the vector database, efficiently returning a large set of candidate code chunks (e.g., top 50-100).

### Stage 2: Re-ranking (High Precision)

*   **Goal**: To maximize **precision** by filtering the broad candidate set down to only the most relevant items for the LLM. This provides a concise, noise-free context, which improves the quality of the final generated response.
*   **Mechanism**: The candidates from Stage 1 are passed to a more powerful but computationally expensive **cross-encoder model (reranker)**. The reranker evaluates each `(query, document)` pair directly, performing a full transformer inference step to produce a highly accurate relevance score. Only the top-scoring candidates (e.g., 5-10) are selected for the final context.

This two-stage design is a foundational requirement, as it is the only way to deliver both the high accuracy needed for complex code tasks and the low latency expected from a real-time assistant.

## 5. Data Ingestion and Processing

The foundation of the RAG system is the quality of its indexed data. The offline ingestion pipeline transforms raw source code into a searchable knowledge base through several automated stages.

1.  **Source Discovery**: The pipeline is triggered by code changes via Git hooks, scheduled polling, or a local filesystem watcher.
2.  **File Filtering**: Loads only relevant source files based on a configurable list of extensions, ignoring binaries, build artifacts, and lock files.
3.  **Semantic Chunking**: Instead of naive text splitting (which is explicitly forbidden), code is parsed into an **Abstract Syntax Tree (AST)**. This allows the system to split code along logical boundaries (functions, classes, methods), preserving semantic integrity. This structure-aware method is critical for effective code retrieval.
4.  **Embedding Generation**: Each coherent code chunk is passed through a specialized code embedding model to generate a dense vector embedding that captures its meaning.
5.  **Vector Storage**: The vector and rich metadata (file path, line numbers, commit hash, etc.) are stored in the vector database.

## 6. Mandated Technology Stack

The following technology stack is mandated for the project based on performance, feature set, and alignment with architectural goals.

*   **Server Framework**: **FastMCP 2.0**
    *   *Rationale*: A high-level framework that abstracts away MCP complexity, accelerating development with simple decorators and providing a rich ecosystem of tools.
*   **Code Chunking**: **tree-sitter** (via the `astchunk` library)
    *   *Rationale*: A powerful, error-tolerant parser that supports a wide array of languages, making it ideal for creating the structurally-aware chunks required for high-quality retrieval.
*   **Embedding Model**: **GraphCodeBERT**
    *   *Rationale*: This model understands not just text but also code structure, specifically the data flow graph (DFG). This creates a powerful synergy with the AST-based chunks, maximizing retrieval accuracy.
*   **Vector Database**: **Qdrant**
    *   *Rationale*: An open-source, high-performance database written in Rust. Its key advantage is its advanced metadata filtering system, which is crucial for narrowing the search space in a multi-repository environment.

### Vector Database Schema

Each object stored in Qdrant will consist of a dense vector and a rich metadata payload to enable filtering and provide traceable context.

```json
{
  "chunk_content": "def calculate_bmi(weight_kg: float, height_m: float) -> float:\n    \"\"\"Calculate BMI given weight in kg and height in meters\"\"\"\n    return weight_kg / (height_m ** 2)",
  "file_path": "/src/health_metrics/calculator.py",
  "repository": "healthcare-analytics-api",
  "start_line": 25,
  "end_line": 28,
  "symbol_name": "calculate_bmi",
  "symbol_type": "function",
  "commit_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
  "language": "python"
}
```

## 7. Performance and Optimization

To ensure a responsive user experience and manage costs, a multi-layered caching strategy is essential.

*   **Semantic Query Cache**: The highest-impact layer. Before running the full RAG pipeline, an incoming query's embedding is checked against a cache of previous queries. If a semantically similar query is found, the stored final response is returned immediately.
*   **Embedding Cache**: A Redis key-value store caches vector embeddings of code chunks, preventing re-computation for unchanged code during ingestion.
*   **Reranker/LLM Response Cache**: A key-value store caches the final output of the most expensive stages. The cache key is a hash of the query and the retrieved context chunks.

## 8. Development Roadmap

The project will be developed in four distinct phases to deliver value incrementally.

1.  **Phase 1: Foundational RAG Pipeline**: Implement the full data ingestion pipeline and a single-stage (Stage 1 only) retrieval mechanism to validate the core data flow.
2.  **Phase 2: Precision Enhancement via Re-ranking**: Introduce the second-stage reranker (cross-encoder) and the multi-level caching service to transition to the full two-stage architecture and improve performance.
3.  **Phase 3: Context Expansion with Documentation Parsing**: Enhance the knowledge base by adding a parallel pipeline to ingest and index natural language documentation (e.g., Markdown files) alongside source code.
4.  **Phase 4: Surgical Precision with Targeted Snippet Retrieval**: Evolve the system to return precise, line-level code snippets by performing a focused search *within* the top-ranked semantic chunk.
