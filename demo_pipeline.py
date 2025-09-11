#!/usr/bin/env python3
"""
Demo script for ZenithMCP Data Ingestion Pipeline.

This script demonstrates the complete pipeline functionality without requiring
external services like Qdrant or actual Git repositories.
"""

import tempfile
from pathlib import Path

from zenithmcp.ingestion.chunking import CodeChunker
from zenithmcp.ingestion.config import Config
from zenithmcp.ingestion.discovery import SourceFileDiscoverer
from zenithmcp.ingestion.models import CodeChunk


def create_demo_config() -> Config:
    """Create a demo configuration for testing."""
    return Config(
        qdrant={
            "host": "localhost",
            "port": 6333,
            "collection_name": "demo_collection",
            "vector_size": 768,
        },
        embedding={
            "model_name": "microsoft/graphcodebert-base",
            "batch_size": 2,
            "device": "cpu",
        },
        chunking={
            "file_extensions": [".py", ".js"],
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
            "state_file": ".demo_state",
            "git": {"enabled": False},  # Disable Git for demo
            "filesystem": {"enabled": True},
        },
        pipeline={
            "max_file_size_mb": 5,
        },
    )


def create_demo_files(repo_path: Path) -> None:
    """Create demo source files for testing."""
    # Python file
    python_code = '''"""Demo Python module."""

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


def fibonacci(n: int) -> list:
    """Generate Fibonacci sequence."""
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


if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    print(fibonacci(10))
'''

    # JavaScript file
    js_code = """/**
 * Demo JavaScript module.
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

module.exports = { Calculator, fibonacci };
"""

    # Create files
    (repo_path / "calculator.py").write_text(python_code)
    (repo_path / "calculator.js").write_text(js_code)
    (repo_path / "README.md").write_text(
        "# Demo Repository\n\nThis is a demo repository."
    )


def demo_file_discovery():
    """Demonstrate file discovery functionality."""
    print("🔍 Demo: File Discovery")
    print("=" * 50)

    config = create_demo_config()

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        create_demo_files(repo_path)

        discoverer = SourceFileDiscoverer(config)
        files = discoverer._discover_filesystem_changes(repo_path)

        print(f"Repository path: {repo_path}")
        print(f"Found {len(files)} files to process:")
        for file_path in files:
            print(f"  - {file_path}")

        print()


def demo_code_chunking():
    """Demonstrate code chunking functionality."""
    print("🔧 Demo: Code Chunking")
    print("=" * 50)

    config = create_demo_config()

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        create_demo_files(repo_path)

        chunker = CodeChunker(config)

        # Chunk Python file
        python_result = chunker.chunk_file(
            "calculator.py", repo_path, "demo_repo", "abc123"
        )

        print("Python file chunking result:")
        print(f"  Success: {python_result.success}")
        print(f"  Language: {python_result.language}")
        print(f"  Chunks: {len(python_result.chunks)}")

        for i, chunk in enumerate(python_result.chunks):
            print(
                f"    Chunk {i + 1}: {chunk.symbol_type} '{chunk.symbol_name}' "
                f"(lines {chunk.start_line}-{chunk.end_line})"
            )

        # Chunk JavaScript file
        js_result = chunker.chunk_file(
            "calculator.js", repo_path, "demo_repo", "abc123"
        )

        print("\nJavaScript file chunking result:")
        print(f"  Success: {js_result.success}")
        print(f"  Language: {js_result.language}")
        print(f"  Chunks: {len(js_result.chunks)}")

        for i, chunk in enumerate(js_result.chunks):
            print(
                f"    Chunk {i + 1}: {chunk.symbol_type} '{chunk.symbol_name}' "
                f"(lines {chunk.start_line}-{chunk.end_line})"
            )

        print()


def demo_data_models():
    """Demonstrate data model functionality."""
    print("📊 Demo: Data Models")
    print("=" * 50)

    # Create a sample code chunk
    chunk = CodeChunk(
        content="def hello_world():\n    print('Hello, World!')\n    return True",
        file_path="demo/hello.py",
        repository="demo_repo",
        start_line=1,
        end_line=3,
        commit_hash="abc123",
        language="python",
        symbol_name="hello_world",
        symbol_type="function",
    )

    print("Created CodeChunk:")
    print(f"  ID: {chunk.id[:16]}...")
    print(f"  File: {chunk.file_path}")
    print(f"  Language: {chunk.language}")
    print(f"  Symbol: {chunk.symbol_type} '{chunk.symbol_name}'")
    print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
    print(f"  Content preview: {chunk.content[:50]!r}...")

    # Test metadata extraction
    metadata = chunk.get_metadata()
    print("\nMetadata:")
    print(f"  Line count: {metadata['line_count']}")
    print(f"  Character count: {metadata['char_count']}")

    print()


def main():
    """Run all demos."""
    print("🚀 ZenithMCP Data Ingestion Pipeline Demo")
    print("=" * 60)
    print()

    try:
        demo_data_models()
        demo_file_discovery()
        demo_code_chunking()

        print("✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Set up Qdrant vector database")
        print("2. Configure GraphCodeBERT model")
        print(
            "3. Run the full pipeline with: python -m zenithmcp.ingestion.pipeline run /path/to/repo"
        )

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
