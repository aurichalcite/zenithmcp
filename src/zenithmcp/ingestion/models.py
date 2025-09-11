"""Data models for the ingestion pipeline."""

import hashlib

from pydantic import BaseModel, Field, field_validator, model_validator


class CodeChunk(BaseModel):
    """Data model for a single, semantically coherent chunk of code."""

    id: str = Field(..., description="Unique and deterministic ID for the chunk")
    content: str = Field(..., description="The actual code content")
    file_path: str = Field(..., description="Relative path to the source file")
    repository: str = Field(..., description="Repository name or identifier")
    start_line: int = Field(..., description="Starting line number in the file", ge=1)
    end_line: int = Field(..., description="Ending line number in the file", ge=1)
    commit_hash: str = Field(
        ..., description="Git commit hash when this chunk was processed"
    )
    language: str = Field(..., description="Programming language of the code")
    symbol_name: str | None = Field(
        None, description="Name of the function/class/symbol"
    )
    symbol_type: str | None = Field(
        None, description="Type of symbol (function, class, method, etc.)"
    )
    embedding: list[float] | None = Field(
        None, description="Vector embedding of the code chunk"
    )

    @model_validator(mode="before")
    @classmethod
    def generate_id_if_missing(cls, values):
        """Generate deterministic ID if not provided."""
        if isinstance(values, dict):
            if "id" not in values or values["id"] is None:
                # Generate deterministic ID from content and metadata
                content = values.get("content", "")
                file_path = values.get("file_path", "")
                repository = values.get("repository", "")
                start_line = values.get("start_line", 0)
                symbol_name = values.get("symbol_name", "")

                # Create a unique string to hash
                unique_string = (
                    f"{repository}:{file_path}:{start_line}:{symbol_name}:{content}"
                )

                # Generate SHA256 hash
                values["id"] = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
        return values

    @model_validator(mode="after")
    def validate_end_line(self):
        """Validate that end_line >= start_line."""
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self

    @field_validator("language")
    @classmethod
    def normalize_language(cls, v: str) -> str:
        """Normalize language name to lowercase."""
        return v.lower().strip()

    @field_validator("file_path")
    @classmethod
    def normalize_file_path(cls, v: str) -> str:
        """Normalize file path (remove leading slashes, use forward slashes)."""
        # Remove leading slash and normalize separators
        path = v.lstrip("/").replace("\\", "/")
        return path

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate that content is not empty."""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: list[float] | None) -> list[float] | None:
        """Validate embedding dimensions if provided."""
        if v is not None:
            if not v:
                raise ValueError("Embedding cannot be empty list")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Embedding must contain only numeric values")
        return v

    def to_qdrant_point(self) -> dict:
        """
        Convert to Qdrant point format for indexing.

        Returns
        -------
        dict
            Dictionary in Qdrant point format with id, vector, and payload.

        Raises
        ------
        ValueError
            If embedding is not set.
        """
        if self.embedding is None:
            raise ValueError("Cannot create Qdrant point without embedding")

        return {
            "id": self.id,
            "vector": self.embedding,
            "payload": {
                "content": self.content,
                "file_path": self.file_path,
                "repository": self.repository,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "commit_hash": self.commit_hash,
                "language": self.language,
                "symbol_name": self.symbol_name,
                "symbol_type": self.symbol_type,
                "line_count": self.end_line - self.start_line + 1,
                "char_count": len(self.content),
            },
        }

    @classmethod
    def from_qdrant_point(cls, point: dict) -> "CodeChunk":
        """
        Create CodeChunk from Qdrant point.

        Parameters
        ----------
        point : dict
            Qdrant point with id, vector, and payload.

        Returns
        -------
        CodeChunk
            CodeChunk instance created from the point.
        """
        payload = point.get("payload", {})

        return cls(
            id=point["id"],
            content=payload["content"],
            file_path=payload["file_path"],
            repository=payload["repository"],
            start_line=payload["start_line"],
            end_line=payload["end_line"],
            commit_hash=payload["commit_hash"],
            language=payload["language"],
            symbol_name=payload.get("symbol_name"),
            symbol_type=payload.get("symbol_type"),
            embedding=point.get("vector"),
        )

    def get_metadata(self) -> dict:
        """
        Get metadata dictionary for the chunk.

        Returns
        -------
        dict
            Metadata dictionary with all non-content fields.
        """
        return {
            "id": self.id,
            "file_path": self.file_path,
            "repository": self.repository,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "commit_hash": self.commit_hash,
            "language": self.language,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "line_count": self.end_line - self.start_line + 1,
            "char_count": len(self.content),
        }

    def __str__(self) -> str:
        """String representation of the chunk."""
        symbol_info = (
            f" ({self.symbol_type}: {self.symbol_name})" if self.symbol_name else ""
        )
        return f"CodeChunk[{self.language}]{symbol_info} {self.file_path}:{self.start_line}-{self.end_line}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"CodeChunk(id='{self.id[:8]}...', "
            f"file_path='{self.file_path}', "
            f"lines={self.start_line}-{self.end_line}, "
            f"language='{self.language}', "
            f"symbol='{self.symbol_name}', "
            f"has_embedding={self.embedding is not None})"
        )


class ProcessingState(BaseModel):
    """Model for tracking processing state of repositories."""

    repository: str = Field(..., description="Repository name or path")
    last_commit_hash: str = Field(..., description="Last processed commit hash")
    last_processed_at: str = Field(..., description="ISO timestamp of last processing")
    total_files_processed: int = Field(0, description="Total number of files processed")
    total_chunks_created: int = Field(0, description="Total number of chunks created")
    failed_files: list[str] = Field(
        default_factory=list, description="List of files that failed processing"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return self.dict()

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingState":
        """Create from dictionary."""
        return cls(**data)


class ChunkingResult(BaseModel):
    """Result of chunking a single file."""

    file_path: str = Field(..., description="Path to the processed file")
    chunks: list[CodeChunk] = Field(
        default_factory=list, description="Generated chunks"
    )
    success: bool = Field(..., description="Whether chunking was successful")
    error_message: str | None = Field(
        None, description="Error message if chunking failed"
    )
    processing_time: float = Field(
        0.0, description="Time taken to process the file in seconds"
    )
    language: str | None = Field(None, description="Detected language of the file")

    @property
    def chunk_count(self) -> int:
        """Number of chunks generated."""
        return len(self.chunks)

    def __str__(self) -> str:
        """String representation."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"ChunkingResult[{status}] {self.file_path}: {self.chunk_count} chunks"


class EmbeddingResult(BaseModel):
    """Result of embedding generation for chunks."""

    chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of processed chunks"
    )
    success: bool = Field(
        ..., description="Whether embedding generation was successful"
    )
    error_message: str | None = Field(
        None, description="Error message if embedding failed"
    )
    processing_time: float = Field(
        0.0, description="Time taken to generate embeddings in seconds"
    )

    @property
    def chunk_count(self) -> int:
        """Number of chunks processed."""
        return len(self.chunk_ids)

    def __str__(self) -> str:
        """String representation."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"EmbeddingResult[{status}]: {self.chunk_count} chunks"


class IndexingResult(BaseModel):
    """Result of indexing chunks to vector database."""

    chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of indexed chunks"
    )
    success: bool = Field(..., description="Whether indexing was successful")
    error_message: str | None = Field(
        None, description="Error message if indexing failed"
    )
    processing_time: float = Field(
        0.0, description="Time taken to index chunks in seconds"
    )

    @property
    def chunk_count(self) -> int:
        """Number of chunks indexed."""
        return len(self.chunk_ids)

    def __str__(self) -> str:
        """String representation."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"IndexingResult[{status}]: {self.chunk_count} chunks"
