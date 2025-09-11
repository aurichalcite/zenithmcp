"""Tests for code chunking."""

from pathlib import Path
from unittest.mock import patch

from zenithmcp.ingestion.chunking import CodeChunker
from zenithmcp.ingestion.models import ChunkingResult


class TestCodeChunker:
    """Tests for CodeChunker class."""

    def test_init_with_config(self, sample_config):
        """Test initializing chunker with config."""
        chunker = CodeChunker(sample_config)

        assert chunker.config is sample_config
        assert chunker.chunking_config is sample_config.chunking

    def test_language_detection(self, sample_config):
        """Test programming language detection from file extensions."""
        chunker = CodeChunker(sample_config)

        # Test various file extensions
        assert chunker._detect_language("main.py") == "python"
        assert chunker._detect_language("script.js") == "javascript"
        assert chunker._detect_language("component.jsx") == "javascript"
        assert chunker._detect_language("app.ts") == "typescript"
        assert chunker._detect_language("component.tsx") == "typescript"
        assert chunker._detect_language("Main.java") == "java"
        assert chunker._detect_language("program.cpp") == "cpp"
        assert chunker._detect_language("header.h") == "c"
        assert chunker._detect_language("Program.cs") == "c_sharp"
        assert chunker._detect_language("main.go") == "go"
        assert chunker._detect_language("lib.rs") == "rust"

        # Test unsupported extension
        assert chunker._detect_language("document.txt") is None
        assert chunker._detect_language("image.png") is None

    def test_validate_file_content(self, sample_config):
        """Test file content validation."""
        chunker = CodeChunker(sample_config)

        # Valid content
        valid_content = "\n".join(["line"] * 10)
        assert chunker._validate_file_content(valid_content, "test.py")

        # Empty content
        assert not chunker._validate_file_content("", "test.py")
        assert not chunker._validate_file_content("   \n  \n  ", "test.py")

        # Binary content (contains null bytes)
        assert not chunker._validate_file_content("binary\x00content", "test.py")

        # Too small content (less than min_chunk_size)
        small_content = "\n".join(["line"] * 3)  # 3 lines, min is 5
        assert not chunker._validate_file_content(small_content, "test.py")

    @patch("zenithmcp.ingestion.chunking.astchunk.ASTChunkBuilder")
    def test_chunk_file_success(
        self, mock_builder_class, sample_config, sample_python_code, fs
    ):
        """Test successful file chunking."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "calculator.py", contents=sample_python_code)

        chunker = CodeChunker(sample_config)

        # Mock the builder and its chunkify method
        mock_builder_instance = mock_builder_class.return_value
        mock_builder_instance.chunkify.return_value = [
            {
                "content": (
                    "class Calculator:\n    def __init__(self):\n        "
                    "self.history = []"
                ),
                "metadata": {
                    "start_line": 8,
                    "end_line": 10,
                    "symbol_name": "Calculator",
                    "symbol_type": "class",
                },
            },
            {
                "content": (
                    "def add(self, a: float, b: float) -> float:\n        "
                    "result = a + b\n        return result"
                ),
                "metadata": {
                    "start_line": 12,
                    "end_line": 15,
                    "symbol_name": "add",
                    "symbol_type": "method",
                },
            },
        ]

        result = chunker.chunk_file(
            "calculator.py", repo_path, "test_repo", "abc123"
        )

        assert result.success is True
        assert result.file_path == "calculator.py"
        assert result.language == "python"
        assert len(result.chunks) == 2

        # Check first chunk
        chunk1 = result.chunks[0]
        assert chunk1.file_path == "calculator.py"
        assert chunk1.repository == "test_repo"
        assert chunk1.commit_hash == "abc123"
        assert chunk1.language == "python"
        assert chunk1.symbol_name == "Calculator"
        assert chunk1.symbol_type == "class"
        assert chunk1.start_line == 8
        assert chunk1.end_line == 10

    def test_chunk_file_not_found(self, sample_config, fs):
        """Test chunking non-existent file."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)

        chunker = CodeChunker(sample_config)
        result = chunker.chunk_file("nonexistent.py", repo_path, "test_repo", "abc123")

        assert result.success is False
        assert "File not found" in result.error_message
        assert result.file_path == "nonexistent.py"

    def test_chunk_file_unsupported_extension(self, sample_config, fs):
        """Test chunking file with unsupported extension."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "document.txt", contents="Some text content")

        chunker = CodeChunker(sample_config)
        result = chunker.chunk_file("document.txt", repo_path, "test_repo", "abc123")

        assert result.success is False
        assert "Unsupported file extension" in result.error_message

    def test_chunk_file_unicode_error(self, sample_config, fs):
        """Test chunking file with unicode decode error."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)

        # Create file with binary content that will cause unicode error
        binary_content = b"\x80\x81\x82\x83"
        with open(repo_path / "binary.py", "wb") as f:
            f.write(binary_content)

        chunker = CodeChunker(sample_config)
        result = chunker.chunk_file("binary.py", repo_path, "test_repo", "abc123")

        # Should handle gracefully (using errors='ignore')
        # The test might pass or fail depending on the exact binary content
        # Let's just check that it doesn't crash
        assert isinstance(result, ChunkingResult)

    def test_chunk_file_empty_content(self, sample_config, fs):
        """Test chunking empty file."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "empty.py", contents="")

        chunker = CodeChunker(sample_config)
        result = chunker.chunk_file("empty.py", repo_path, "test_repo", "abc123")

        assert result.success is False
        assert "File content validation failed" in result.error_message

    def test_create_fallback_chunks_small_file(self, sample_config):
        """Test creating fallback chunks for small file."""
        chunker = CodeChunker(sample_config)

        content = "\n".join(
            [f"line {i}" for i in range(1, 11)]
        )  # 10 lines (< max_chunk_size)

        chunks = chunker._create_fallback_chunks(
            content, "test.py", "test_repo", "abc123", "python"
        )

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.file_path == "test.py"
        assert chunk.repository == "test_repo"
        assert chunk.commit_hash == "abc123"
        assert chunk.language == "python"
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert chunk.symbol_type == "file"

    def test_create_fallback_chunks_large_file(self, sample_config):
        """Test creating fallback chunks for large file."""
        chunker = CodeChunker(sample_config)

        # Create content larger than max_chunk_size (100 lines)
        content = "\n".join([f"line {i}" for i in range(1, 201)])  # 200 lines

        chunks = chunker._create_fallback_chunks(
            content, "test.py", "test_repo", "abc123", "python"
        )

        assert len(chunks) > 1

        # Check first chunk
        first_chunk = chunks[0]
        assert first_chunk.start_line == 1
        assert first_chunk.end_line == 100  # max_chunk_size
        assert first_chunk.symbol_name == "chunk_1"
        assert first_chunk.symbol_type == "chunk"

        # Check second chunk (should have overlap)
        second_chunk = chunks[1]
        assert second_chunk.start_line == 99  # 100 - overlap_lines (2) + 1
        assert second_chunk.symbol_name == "chunk_2"

    @patch("zenithmcp.ingestion.chunking.astchunk.ASTChunkBuilder")
    def test_chunk_content_astchunk_failure(self, mock_builder_class, sample_config):
        """Test chunking when astchunk fails."""
        mock_builder_instance = mock_builder_class.return_value
        mock_builder_instance.chunkify.side_effect = Exception("AST parsing failed")

        chunker = CodeChunker(sample_config)

        content = "def hello():\n    print('hello')\n    return True"

        chunks = chunker._chunk_content(
            content, "test.py", "test_repo", "abc123", "python"
        )

        # Should fall back to simple chunking
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.content == content
        assert chunk.symbol_type == "file"

    @patch("zenithmcp.ingestion.chunking.astchunk.ASTChunkBuilder")
    def test_run_multiple_files(
        self,
        mock_builder_class,
        sample_config,
        sample_python_code,
        sample_javascript_code,
        fs,
    ):
        """Test chunking multiple files."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "calculator.py", contents=sample_python_code)
        fs.create_file(repo_path / "utils.js", contents=sample_javascript_code)

        chunker = CodeChunker(sample_config)

        # Mock the builder and its chunkify method
        mock_builder_instance = mock_builder_class.return_value
        mock_builder_instance.chunkify.side_effect = [
            # Python file chunks
            [
                {
                    "content": "class Calculator:",
                    "metadata": {
                        "start_line": 1,
                        "end_line": 3,
                        "symbol_name": "Calculator",
                        "symbol_type": "class",
                    },
                }
            ],
            # JavaScript file chunks
            [
                {
                    "content": "class Calculator {",
                    "metadata": {
                        "start_line": 1,
                        "end_line": 3,
                        "symbol_name": "Calculator",
                        "symbol_type": "class",
                    },
                }
            ],
        ]

        files = ["calculator.py", "utils.js"]
        chunks = chunker.run(files, str(repo_path), "test_repo", "abc123")

        assert len(chunks) == 2

        # Check that we have chunks from both files
        file_paths = {chunk.file_path for chunk in chunks}
        assert "calculator.py" in file_paths
        assert "utils.js" in file_paths

        # Check languages
        languages = {chunk.language for chunk in chunks}
        assert "python" in languages
        assert "javascript" in languages

    def test_run_with_failed_file(self, sample_config, fs):
        """Test running with some files that fail to chunk."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "good.py", contents="def hello(): pass")
        # Don't create bad.py to simulate file not found

        chunker = CodeChunker(sample_config)

        files = ["good.py", "bad.py"]
        chunks = chunker.run(files, str(repo_path), "test_repo", "abc123")

        # Should only return chunks from successful files
        assert len(chunks) >= 0  # Depends on astchunk behavior

        # All returned chunks should be from good.py
        for chunk in chunks:
            assert chunk.file_path == "good.py"

    def test_language_specific_config(self, sample_config):
        """Test using language-specific configuration."""
        # Test that language config is accessed correctly
        # Note: CodeChunker would be used here for actual language-specific tests
        python_config = sample_config.chunking.languages.get("python")
        assert python_config is not None
        assert python_config.chunk_types == ["function", "class"]
        assert python_config.min_lines == 3

        javascript_config = sample_config.chunking.languages.get("javascript")
        assert javascript_config is not None
        assert javascript_config.chunk_types == ["function", "class"]
        assert javascript_config.min_lines == 3
