"""Tests for ingestion data models."""

import pytest
from pydantic import ValidationError

from zenithmcp.ingestion.models import (
    ChunkingResult,
    CodeChunk,
    EmbeddingResult,
    IndexingResult,
    ProcessingState,
)


class TestCodeChunk:
    """Tests for CodeChunk model."""

    def test_create_valid_chunk(self, sample_code_chunk):
        """Test creating a valid code chunk."""
        chunk = sample_code_chunk
        assert (
            chunk.content
            == "def hello_world():\n    print('Hello, World!')\n    return True"
        )
        assert chunk.file_path == "test/hello.py"
        assert chunk.repository == "test_repo"
        assert chunk.start_line == 1
        assert chunk.end_line == 3
        assert chunk.commit_hash == "abc123"
        assert chunk.language == "python"
        assert chunk.symbol_name == "hello_world"
        assert chunk.symbol_type == "function"
        assert chunk.embedding is None

    def test_id_generation(self):
        """Test automatic ID generation."""
        chunk = CodeChunk(
            content="print('test')",
            file_path="test.py",
            repository="repo",
            start_line=1,
            end_line=1,
            commit_hash="abc",
            language="python",
        )
        assert chunk.id is not None
        assert len(chunk.id) == 64  # SHA256 hex length

    def test_deterministic_id(self):
        """Test that ID generation is deterministic."""
        chunk1 = CodeChunk(
            content="print('test')",
            file_path="test.py",
            repository="repo",
            start_line=1,
            end_line=1,
            commit_hash="abc",
            language="python",
        )
        chunk2 = CodeChunk(
            content="print('test')",
            file_path="test.py",
            repository="repo",
            start_line=1,
            end_line=1,
            commit_hash="abc",
            language="python",
        )
        assert chunk1.id == chunk2.id

    def test_custom_id(self):
        """Test using custom ID."""
        custom_id = "custom_id_123"
        chunk = CodeChunk(
            id=custom_id,
            content="print('test')",
            file_path="test.py",
            repository="repo",
            start_line=1,
            end_line=1,
            commit_hash="abc",
            language="python",
        )
        assert chunk.id == custom_id

    def test_end_line_validation(self):
        """Test that end_line must be >= start_line."""
        with pytest.raises(ValidationError):
            CodeChunk(
                content="test",
                file_path="test.py",
                repository="repo",
                start_line=5,
                end_line=3,  # Invalid: less than start_line
                commit_hash="abc",
                language="python",
            )

    def test_language_normalization(self):
        """Test that language is normalized to lowercase."""
        chunk = CodeChunk(
            content="test",
            file_path="test.py",
            repository="repo",
            start_line=1,
            end_line=1,
            commit_hash="abc",
            language="PYTHON",
        )
        assert chunk.language == "python"

    def test_file_path_normalization(self):
        """Test file path normalization."""
        chunk = CodeChunk(
            content="test",
            file_path="/src\\test.py",
            repository="repo",
            start_line=1,
            end_line=1,
            commit_hash="abc",
            language="python",
        )
        assert chunk.file_path == "src/test.py"

    def test_empty_content_validation(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValidationError):
            CodeChunk(
                content="   ",  # Only whitespace
                file_path="test.py",
                repository="repo",
                start_line=1,
                end_line=1,
                commit_hash="abc",
                language="python",
            )

    def test_embedding_validation(self):
        """Test embedding validation."""
        # Valid embedding
        chunk = CodeChunk(
            content="test",
            file_path="test.py",
            repository="repo",
            start_line=1,
            end_line=1,
            commit_hash="abc",
            language="python",
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.embedding == [0.1, 0.2, 0.3]

        # Empty embedding should fail
        with pytest.raises(ValidationError):
            CodeChunk(
                content="test",
                file_path="test.py",
                repository="repo",
                start_line=1,
                end_line=1,
                commit_hash="abc",
                language="python",
                embedding=[],
            )

        # Non-numeric embedding should fail
        with pytest.raises(ValidationError):
            CodeChunk(
                content="test",
                file_path="test.py",
                repository="repo",
                start_line=1,
                end_line=1,
                commit_hash="abc",
                language="python",
                embedding=["not", "numeric"],
            )

    def test_to_qdrant_point(self, sample_code_chunk, mock_embedding_vector):
        """Test conversion to Qdrant point format."""
        chunk = sample_code_chunk
        chunk.embedding = mock_embedding_vector

        point = chunk.to_qdrant_point()

        assert point["id"] == chunk.id
        assert point["vector"] == mock_embedding_vector
        assert point["payload"]["content"] == chunk.content
        assert point["payload"]["file_path"] == chunk.file_path
        assert point["payload"]["repository"] == chunk.repository
        assert point["payload"]["start_line"] == chunk.start_line
        assert point["payload"]["end_line"] == chunk.end_line
        assert point["payload"]["commit_hash"] == chunk.commit_hash
        assert point["payload"]["language"] == chunk.language
        assert point["payload"]["symbol_name"] == chunk.symbol_name
        assert point["payload"]["symbol_type"] == chunk.symbol_type
        assert point["payload"]["line_count"] == 3
        assert point["payload"]["char_count"] == len(chunk.content)

    def test_to_qdrant_point_without_embedding(self, sample_code_chunk):
        """Test that to_qdrant_point fails without embedding."""
        chunk = sample_code_chunk

        with pytest.raises(
            ValueError, match="Cannot create Qdrant point without embedding"
        ):
            chunk.to_qdrant_point()

    def test_from_qdrant_point(self, mock_embedding_vector):
        """Test creation from Qdrant point."""
        point = {
            "id": "test_id",
            "vector": mock_embedding_vector,
            "payload": {
                "content": "test content",
                "file_path": "test.py",
                "repository": "test_repo",
                "start_line": 1,
                "end_line": 5,
                "commit_hash": "abc123",
                "language": "python",
                "symbol_name": "test_func",
                "symbol_type": "function",
            },
        }

        chunk = CodeChunk.from_qdrant_point(point)

        assert chunk.id == "test_id"
        assert chunk.embedding == mock_embedding_vector
        assert chunk.content == "test content"
        assert chunk.file_path == "test.py"
        assert chunk.repository == "test_repo"
        assert chunk.start_line == 1
        assert chunk.end_line == 5
        assert chunk.commit_hash == "abc123"
        assert chunk.language == "python"
        assert chunk.symbol_name == "test_func"
        assert chunk.symbol_type == "function"

    def test_get_metadata(self, sample_code_chunk):
        """Test metadata extraction."""
        chunk = sample_code_chunk
        metadata = chunk.get_metadata()

        assert metadata["id"] == chunk.id
        assert metadata["file_path"] == chunk.file_path
        assert metadata["repository"] == chunk.repository
        assert metadata["start_line"] == chunk.start_line
        assert metadata["end_line"] == chunk.end_line
        assert metadata["commit_hash"] == chunk.commit_hash
        assert metadata["language"] == chunk.language
        assert metadata["symbol_name"] == chunk.symbol_name
        assert metadata["symbol_type"] == chunk.symbol_type
        assert metadata["line_count"] == 3
        assert metadata["char_count"] == len(chunk.content)

    def test_string_representations(self, sample_code_chunk):
        """Test string representations."""
        chunk = sample_code_chunk

        str_repr = str(chunk)
        assert "CodeChunk[python]" in str_repr
        assert "(function: hello_world)" in str_repr
        assert "test/hello.py:1-3" in str_repr

        repr_str = repr(chunk)
        assert "CodeChunk(" in repr_str
        assert f"id='{chunk.id[:8]}...'" in repr_str
        assert "file_path='test/hello.py'" in repr_str
        assert "has_embedding=False" in repr_str


class TestProcessingState:
    """Tests for ProcessingState model."""

    def test_create_processing_state(self, sample_processing_state):
        """Test creating processing state."""
        state = ProcessingState(**sample_processing_state)

        assert state.repository == "/test/repo"
        assert state.last_commit_hash == "abc123def456"
        assert state.last_processed_at == "2025-01-01T12:00:00"
        assert state.total_files_processed == 5
        assert state.total_chunks_created == 15
        assert state.failed_files == ["broken.py"]

    def test_to_dict(self, sample_processing_state):
        """Test conversion to dictionary."""
        state = ProcessingState(**sample_processing_state)
        state_dict = state.to_dict()

        assert state_dict == sample_processing_state

    def test_from_dict(self, sample_processing_state):
        """Test creation from dictionary."""
        state = ProcessingState.from_dict(sample_processing_state)

        assert state.repository == sample_processing_state["repository"]
        assert state.last_commit_hash == sample_processing_state["last_commit_hash"]


class TestChunkingResult:
    """Tests for ChunkingResult model."""

    def test_successful_chunking_result(self, sample_code_chunks):
        """Test successful chunking result."""
        result = ChunkingResult(
            file_path="test.py",
            chunks=sample_code_chunks,
            success=True,
            processing_time=1.5,
            language="python",
        )

        assert result.file_path == "test.py"
        assert result.chunks == sample_code_chunks
        assert result.success is True
        assert result.error_message is None
        assert result.processing_time == 1.5
        assert result.language == "python"
        assert result.chunk_count == len(sample_code_chunks)

    def test_failed_chunking_result(self):
        """Test failed chunking result."""
        result = ChunkingResult(
            file_path="broken.py",
            success=False,
            error_message="Syntax error",
            processing_time=0.1,
        )

        assert result.file_path == "broken.py"
        assert result.chunks == []
        assert result.success is False
        assert result.error_message == "Syntax error"
        assert result.processing_time == 0.1
        assert result.chunk_count == 0


class TestEmbeddingResult:
    """Tests for EmbeddingResult model."""

    def test_successful_embedding_result(self):
        """Test successful embedding result."""
        chunk_ids = ["id1", "id2", "id3"]
        result = EmbeddingResult(
            chunk_ids=chunk_ids,
            success=True,
            processing_time=2.5,
        )

        assert result.chunk_ids == chunk_ids
        assert result.success is True
        assert result.error_message is None
        assert result.processing_time == 2.5
        assert result.chunk_count == 3

    def test_failed_embedding_result(self):
        """Test failed embedding result."""
        result = EmbeddingResult(
            success=False,
            error_message="Model loading failed",
            processing_time=0.5,
        )

        assert result.chunk_ids == []
        assert result.success is False
        assert result.error_message == "Model loading failed"
        assert result.processing_time == 0.5
        assert result.chunk_count == 0


class TestIndexingResult:
    """Tests for IndexingResult model."""

    def test_successful_indexing_result(self):
        """Test successful indexing result."""
        chunk_ids = ["id1", "id2", "id3"]
        result = IndexingResult(
            chunk_ids=chunk_ids,
            success=True,
            processing_time=1.0,
        )

        assert result.chunk_ids == chunk_ids
        assert result.success is True
        assert result.error_message is None
        assert result.processing_time == 1.0
        assert result.chunk_count == 3

    def test_failed_indexing_result(self):
        """Test failed indexing result."""
        result = IndexingResult(
            success=False,
            error_message="Database connection failed",
            processing_time=0.2,
        )

        assert result.chunk_ids == []
        assert result.success is False
        assert result.error_message == "Database connection failed"
        assert result.processing_time == 0.2
        assert result.chunk_count == 0
