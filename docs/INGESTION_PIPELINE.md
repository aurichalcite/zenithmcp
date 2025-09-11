# ZenithMCP Data Ingestion Pipeline

The ZenithMCP Data Ingestion Pipeline is a high-performance, production-grade system that transforms source code repositories into a searchable knowledge base. It implements a complete RAG (Retrieval-Augmented Generation) pipeline with AST-based semantic chunking, GraphCodeBERT embeddings, and Qdrant vector storage.

## 🏗️ Architecture

The pipeline consists of four main stages:

1. **File Discovery** - Identifies new or modified files using Git or filesystem scanning
2. **Code Chunking** - Parses source code into semantic chunks using AST analysis
3. **Embedding Generation** - Creates vector embeddings using GraphCodeBERT
4. **Vector Indexing** - Stores chunks in Qdrant vector database for retrieval

## 🚀 Features

- **Idempotent Processing** - Tracks processing state to avoid duplicate work
- **Configuration-Driven** - All parameters configurable via YAML
- **Type-Safe** - Uses Pydantic models for data contracts
- **Multi-Language Support** - Supports 25+ programming languages
- **Incremental Updates** - Only processes changed files
- **Robust Error Handling** - Graceful fallbacks and comprehensive logging
- **CLI Interface** - Easy-to-use command-line tools
- **Comprehensive Testing** - Full test suite with mocking

## 📋 Requirements

### System Requirements
- Python 3.11+
- 4GB+ RAM (8GB recommended)
- GPU support (optional, for faster embedding generation)

### External Services
- **Qdrant Vector Database** - For storing and retrieving embeddings
- **Git Repository** - For change detection (optional)

### Python Dependencies
- `astchunk` - AST-based code chunking
- `transformers` - GraphCodeBERT model
- `qdrant-client` - Vector database client
- `GitPython` - Git repository interaction
- `typer` - CLI framework
- `pydantic` - Data validation
- `torch` - PyTorch for embeddings

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   uv add astchunk transformers qdrant-client GitPython typer
   ```

2. **Set up Qdrant** (using Docker):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Configure Pipeline**:
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your settings
   ```

## ⚙️ Configuration

The pipeline is configured via `config.yaml`. Key sections:

### Qdrant Configuration
```yaml
qdrant:
  host: localhost
  port: 6333
  collection_name: zenithmcp_code_chunks
  vector_size: 768
  distance: Cosine
```

### Embedding Configuration
```yaml
embedding:
  model_name: microsoft/graphcodebert-base
  batch_size: 32
  max_length: 512
  device: auto  # auto, cpu, cuda, mps
```

### Chunking Configuration
```yaml
chunking:
  file_extensions: [".py", ".js", ".ts", ".java", ...]
  exclude_patterns: ["*/node_modules/*", "*/__pycache__/*", ...]
  min_chunk_size: 50
  max_chunk_size: 500
  overlap_lines: 10
```

## 🎯 Usage

### Command Line Interface

**Run Complete Pipeline**:
```bash
python -m zenithmcp.ingestion.pipeline run /path/to/repository
```

**Options**:
- `--config PATH` - Custom configuration file
- `--dry-run` - Run without indexing to database
- `--verbose` - Enable debug logging
- `--force` - Process all files regardless of changes

**Health Check**:
```bash
python -m zenithmcp.ingestion.pipeline health
```

**Configuration Info**:
```bash
python -m zenithmcp.ingestion.pipeline info
```

### Programmatic Usage

```python
from zenithmcp.ingestion import (
    SourceFileDiscoverer,
    CodeChunker,
    EmbeddingGenerator,
    VectorIndexer,
    load_and_set_config
)

# Load configuration
config = load_and_set_config("config.yaml")

# Initialize components
discoverer = SourceFileDiscoverer(config)
chunker = CodeChunker(config)
embedder = EmbeddingGenerator(config)
indexer = VectorIndexer(config)

# Run pipeline
files = discoverer.run("/path/to/repo")
chunks = chunker.run(files, "/path/to/repo", "repo_name", "commit_hash")
embedded_chunks = embedder.run(chunks)
result = indexer.run(embedded_chunks)
```

## 📊 Data Models

### CodeChunk
The primary data model representing a semantic code unit:

```python
class CodeChunk(BaseModel):
    id: str                           # Unique deterministic ID
    content: str                      # Code content
    file_path: str                    # Relative file path
    repository: str                   # Repository name
    start_line: int                   # Starting line number
    end_line: int                     # Ending line number
    commit_hash: str                  # Git commit hash
    language: str                     # Programming language
    symbol_name: Optional[str]        # Function/class name
    symbol_type: Optional[str]        # Symbol type (function, class, etc.)
    embedding: Optional[List[float]]  # Vector embedding
```

### Processing State
Tracks pipeline execution state for idempotency:

```python
class ProcessingState(BaseModel):
    repository: str                   # Repository path
    last_commit_hash: str            # Last processed commit
    last_processed_at: str           # ISO timestamp
    total_files_processed: int       # File count
    total_chunks_created: int        # Chunk count
    failed_files: List[str]          # Failed file paths
```

## 🔧 Components

### 1. SourceFileDiscoverer
**Purpose**: Identify files that need processing

**Features**:
- Git-based change detection
- Filesystem scanning fallback
- File extension filtering
- Exclusion pattern matching
- Size limit enforcement

**Usage**:
```python
discoverer = SourceFileDiscoverer(config)
files = discoverer.run("/path/to/repo")
```

### 2. CodeChunker
**Purpose**: Parse source code into semantic chunks

**Features**:
- AST-based chunking via `astchunk`
- Language-specific configuration
- Fallback line-based chunking
- Symbol extraction (functions, classes)
- Configurable chunk sizes

**Usage**:
```python
chunker = CodeChunker(config)
chunks = chunker.run(files, repo_path, repo_name, commit_hash)
```

### 3. EmbeddingGenerator
**Purpose**: Generate vector embeddings for code chunks

**Features**:
- GraphCodeBERT model
- Batch processing
- GPU acceleration
- Mean pooling
- Embedding validation

**Usage**:
```python
embedder = EmbeddingGenerator(config)
embedded_chunks = embedder.run(chunks)
```

### 4. VectorIndexer
**Purpose**: Store chunks in Qdrant vector database

**Features**:
- Batch upserts
- Automatic collection creation
- Retry logic
- Health checking
- Idempotent operations

**Usage**:
```python
indexer = VectorIndexer(config)
result = indexer.run(embedded_chunks)
```

## 🧪 Testing

Run the complete test suite:
```bash
pytest tests/ingestion/ -v
```

Run specific test modules:
```bash
pytest tests/ingestion/test_models.py -v
pytest tests/ingestion/test_chunking.py -v
pytest tests/ingestion/test_embedding.py -v
```

Run with coverage:
```bash
pytest tests/ingestion/ --cov=src/zenithmcp/ingestion
```

## 🎮 Demo

Run the demo script to see the pipeline in action:
```bash
python demo_pipeline.py
```

This demonstrates:
- Data model creation and validation
- File discovery functionality
- Code chunking with AST parsing
- Error handling and fallbacks

## 🔍 Monitoring

### Logging
The pipeline provides comprehensive logging at multiple levels:
- `DEBUG` - Detailed execution information
- `INFO` - Progress updates and summaries
- `WARNING` - Non-fatal issues and fallbacks
- `ERROR` - Processing failures

### Metrics
Key metrics tracked during execution:
- Files discovered and processed
- Chunks generated and embedded
- Processing times per stage
- Success/failure rates
- Memory and GPU usage

### Health Checks
Monitor pipeline health:
```bash
python -m zenithmcp.ingestion.pipeline health
```

Checks:
- Qdrant connectivity and collection status
- GraphCodeBERT model loading
- Configuration validation
- Resource availability

## 🚨 Troubleshooting

### Common Issues

**1. Qdrant Connection Failed**
```
Solution: Ensure Qdrant is running on configured host:port
Check: docker ps | grep qdrant
```

**2. CUDA Out of Memory**
```
Solution: Reduce batch_size in config or use CPU
Config: embedding.device = "cpu"
```

**3. AST Parsing Failed**
```
Solution: Pipeline automatically falls back to line-based chunking
Check: Review logs for unsupported languages
```

**4. Git Repository Not Found**
```
Solution: Disable Git discovery or initialize repository
Config: discovery.git.enabled = false
```

### Performance Optimization

**1. GPU Acceleration**
- Enable CUDA/MPS for embedding generation
- Increase batch size for better GPU utilization

**2. Memory Management**
- Process files in smaller batches
- Set memory limits in configuration

**3. Parallel Processing**
- Increase worker count for I/O operations
- Use async processing for independent tasks

## 📈 Scalability

The pipeline is designed for production use with:

- **Horizontal Scaling** - Multiple pipeline instances
- **Incremental Processing** - Only changed files
- **Batch Optimization** - Configurable batch sizes
- **Resource Management** - Memory and GPU limits
- **State Persistence** - Resume interrupted processing

## 🔮 Future Enhancements

- **Distributed Processing** - Multi-node execution
- **Advanced Chunking** - Semantic similarity-based chunking
- **Model Fine-tuning** - Domain-specific embeddings
- **Real-time Updates** - File system watching
- **Multi-modal Support** - Documentation and comments
- **Query Optimization** - Embedding compression and quantization

## 📚 References

- [GraphCodeBERT Paper](https://arxiv.org/abs/2009.08366)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [AST Chunking Library](https://github.com/codelion/astchunk)
- [Tree-sitter Parsers](https://tree-sitter.github.io/tree-sitter/)

---

For more information, see the [ZenithMCP Documentation](../README.md).
