"""Test configuration and fixtures for ingestion tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from zenithmcp.ingestion.config import Config
from zenithmcp.ingestion.models import CodeChunk


@pytest.fixture
def sample_config() -> Config:
    """Create a sample configuration for testing."""
    return Config(
        qdrant={
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_collection",
            "vector_size": 768,
            "distance": "Cosine",
            "timeout": 30.0,
            "prefer_grpc": False,
        },
        embedding={
            "model_name": "microsoft/graphcodebert-base",
            "batch_size": 2,  # Small batch for testing
            "max_length": 512,
            "device": "cpu",  # Force CPU for testing
            "cache_dir": ".cache/huggingface",
        },
        chunking={
            "file_extensions": [".py", ".js", ".ts"],
            "exclude_patterns": ["*/node_modules/*", "*/__pycache__/*"],
            "min_chunk_size": 5,
            "max_chunk_size": 100,
            "overlap_lines": 2,
            "languages": {
                "python": {
                    "chunk_types": ["function", "class"],
                    "min_lines": 3,
                },
                "javascript": {
                    "chunk_types": ["function", "class"],
                    "min_lines": 3,
                },
            },
        },
        discovery={
            "state_file": ".test_state",
            "git": {
                "enabled": True,
                "max_commits": 100,
                "branches": [],
            },
            "filesystem": {
                "enabled": False,
                "poll_interval": 5.0,
                "full_scan_on_start": True,
            },
        },
        indexing={
            "batch_size": 10,
            "create_collection": True,
            "recreate_on_schema_change": False,
            "parallel_workers": 1,
            "max_retries": 2,
            "retry_delay": 0.1,
        },
        logging={
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None,
            "max_file_size": "10MB",
            "backup_count": 5,
        },
        performance={
            "max_memory_mb": 1024,
            "max_workers": 1,
            "use_gpu": False,
            "batch_timeout": 30.0,
        },
        pipeline={
            "incremental": True,
            "validate_embeddings": True,
            "skip_failed_files": True,
            "max_file_size_mb": 5,
            "process_binary_files": False,
        },
    )


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing."""
    return '''"""Sample Python module for testing."""

import os
from typing import List


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result


def fibonacci(n: int) -> List[int]:
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])

    return sequence


def main():
    """Main function."""
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    print(fibonacci(10))


if __name__ == "__main__":
    main()
'''


@pytest.fixture
def sample_javascript_code() -> str:
    """Sample JavaScript code for testing."""
    return """/**
 * Sample JavaScript module for testing.
 */

class Calculator {
    constructor() {
        this.history = [];
    }

    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }

    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }
}

function fibonacci(n) {
    if (n <= 0) return [];
    if (n === 1) return [0];
    if (n === 2) return [0, 1];

    const sequence = [0, 1];
    for (let i = 2; i < n; i++) {
        sequence.push(sequence[i-1] + sequence[i-2]);
    }

    return sequence;
}

function main() {
    const calc = new Calculator();
    console.log(calc.add(5, 3));
    console.log(calc.multiply(4, 7));
    console.log(fibonacci(10));
}

if (require.main === module) {
    main();
}

module.exports = { Calculator, fibonacci };
"""


@pytest.fixture
def sample_code_chunk() -> CodeChunk:
    """Create a sample code chunk for testing."""
    return CodeChunk(
        content="def hello_world():\n    print('Hello, World!')\n    return True",
        file_path="test/hello.py",
        repository="test_repo",
        start_line=1,
        end_line=3,
        commit_hash="abc123",
        language="python",
        symbol_name="hello_world",
        symbol_type="function",
    )


@pytest.fixture
def sample_code_chunks() -> list[CodeChunk]:
    """Create multiple sample code chunks for testing."""
    chunks = []

    # Python function chunk
    chunks.append(
        CodeChunk(
            content="def add(a, b):\n    return a + b",
            file_path="math/operations.py",
            repository="test_repo",
            start_line=1,
            end_line=2,
            commit_hash="abc123",
            language="python",
            symbol_name="add",
            symbol_type="function",
        )
    )

    # Python class chunk
    chunks.append(
        CodeChunk(
            content="class Calculator:\n    def __init__(self):\n        self.value = 0",
            file_path="math/calculator.py",
            repository="test_repo",
            start_line=5,
            end_line=7,
            commit_hash="abc123",
            language="python",
            symbol_name="Calculator",
            symbol_type="class",
        )
    )

    # JavaScript function chunk
    chunks.append(
        CodeChunk(
            content="function multiply(a, b) {\n    return a * b;\n}",
            file_path="utils/math.js",
            repository="test_repo",
            start_line=10,
            end_line=12,
            commit_hash="abc123",
            language="javascript",
            symbol_name="multiply",
            symbol_type="function",
        )
    )

    return chunks


@pytest.fixture
def fake_git_repo(fs) -> Path:
    """Create a fake Git repository for testing."""
    repo_path = Path("/test/repo")
    fs.create_dir(repo_path)
    fs.create_dir(repo_path / ".git")

    # Create some sample files
    fs.create_file(repo_path / "main.py", contents="print('Hello, World!')")
    fs.create_file(
        repo_path / "utils.js", contents="function add(a, b) { return a + b; }"
    )
    fs.create_file(repo_path / "README.md", contents="# Test Repository")

    # Create Git config
    fs.create_file(
        repo_path / ".git" / "config", contents="[core]\nrepositoryformatversion = 0"
    )
    fs.create_file(repo_path / ".git" / "HEAD", contents="ref: refs/heads/main")

    return repo_path


@pytest.fixture
def temp_config_file() -> Generator[Path, None, None]:
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
qdrant:
  host: localhost
  port: 6333
  collection_name: test_collection
  vector_size: 768

embedding:
  model_name: microsoft/graphcodebert-base
  batch_size: 2
  device: cpu

chunking:
  file_extensions: [".py", ".js"]
  min_chunk_size: 5
  max_chunk_size: 100

discovery:
  state_file: .test_state

indexing:
  batch_size: 10
  create_collection: true

logging:
  level: DEBUG

performance:
  use_gpu: false

pipeline:
  max_file_size_mb: 5
"""
        f.write(config_content)
        f.flush()

        yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def mock_embedding_vector() -> list[float]:
    """Create a mock embedding vector for testing."""
    return [0.1] * 768  # GraphCodeBERT dimension


@pytest.fixture
def sample_processing_state() -> dict:
    """Create sample processing state data."""
    return {
        "repository": "/test/repo",
        "last_commit_hash": "abc123def456",
        "last_processed_at": "2025-01-01T12:00:00",
        "total_files_processed": 5,
        "total_chunks_created": 15,
        "failed_files": ["broken.py"],
    }
