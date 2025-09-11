"""Code chunking using tree-sitter and astchunk."""

import logging
import time
from pathlib import Path

import astchunk

from zenithmcp.ingestion.config import Config, get_config
from zenithmcp.ingestion.models import ChunkingResult, CodeChunk

logger = logging.getLogger(__name__)


class CodeChunker:
    """Chunks source code files into semantic units using AST parsing."""

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize the code chunker.

        Parameters
        ----------
        config : Optional[Config]
            Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.chunking_config = self.config.chunking

        # Language mapping for astchunk
        self.language_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "c_sharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".clj": "clojure",
            ".hs": "haskell",
            ".ml": "ocaml",
            ".fs": "fsharp",
            ".elm": "elm",
            ".dart": "dart",
            ".lua": "lua",
            ".r": "r",
            ".jl": "julia",
            ".m": "objective_c",
            ".pl": "perl",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".fish": "bash",
            ".ps1": "powershell",
        }

    def run(
        self,
        file_paths: list[str],
        repository_path: str,
        repository_name: str,
        commit_hash: str,
    ) -> list[CodeChunk]:
        """
        Chunk multiple files into code chunks.

        Parameters
        ----------
        file_paths : List[str]
            List of file paths to chunk.
        repository_path : str
            Path to the repository root.
        repository_name : str
            Name of the repository.
        commit_hash : str
            Current commit hash.

        Returns
        -------
        List[CodeChunk]
            List of all generated code chunks.
        """
        all_chunks = []
        repo_path = Path(repository_path)

        logger.info(f"Chunking {len(file_paths)} files")

        for file_path in file_paths:
            try:
                result = self.chunk_file(
                    file_path, repo_path, repository_name, commit_hash
                )
                if result.success:
                    all_chunks.extend(result.chunks)
                    logger.debug(f"Chunked {file_path}: {len(result.chunks)} chunks")
                else:
                    logger.warning(
                        f"Failed to chunk {file_path}: {result.error_message}"
                    )
            except Exception:
                logger.exception(f"Unexpected error chunking {file_path}")

        logger.info(
            f"Generated {len(all_chunks)} total chunks from {len(file_paths)} files"
        )
        return all_chunks

    def chunk_file(
        self,
        file_path: str,
        repository_path: Path,
        repository_name: str,
        commit_hash: str,
    ) -> ChunkingResult:
        """
        Chunk a single file into semantic code chunks.

        Parameters
        ----------
        file_path : str
            Relative path to the file within the repository.
        repository_path : Path
            Path to the repository root.
        repository_name : str
            Name of the repository.
        commit_hash : str
            Current commit hash.

        Returns
        -------
        ChunkingResult
            Result of the chunking operation.
        """
        start_time = time.time()
        full_path = repository_path / file_path

        try:
            # Read file content
            with open(full_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Detect language
            language = self._detect_language(file_path)
            if not language:
                return ChunkingResult(
                    file_path=file_path,
                    success=False,
                    error_message=(
                        f"Unsupported file extension: {Path(file_path).suffix}"
                    ),
                    processing_time=time.time() - start_time,
                )

            # Check if file is too large or too small
            if not self._validate_file_content(content, file_path):
                return ChunkingResult(
                    file_path=file_path,
                    success=False,
                    error_message="File content validation failed",
                    processing_time=time.time() - start_time,
                    language=language,
                )

            # Chunk the file
            chunks = self._chunk_content(
                content, file_path, repository_name, commit_hash, language
            )

            return ChunkingResult(
                file_path=file_path,
                chunks=chunks,
                success=True,
                processing_time=time.time() - start_time,
                language=language,
            )

        except UnicodeDecodeError as e:
            return ChunkingResult(
                file_path=file_path,
                success=False,
                error_message=f"Unicode decode error: {e}",
                processing_time=time.time() - start_time,
            )
        except FileNotFoundError:
            return ChunkingResult(
                file_path=file_path,
                success=False,
                error_message="File not found",
                processing_time=time.time() - start_time,
            )
        except Exception:
            return ChunkingResult(
                file_path=file_path,
                success=False,
                error_message=f"Unexpected error: {e}",
                processing_time=time.time() - start_time,
            )

    def _detect_language(self, file_path: str) -> str | None:
        """
        Detect programming language from file extension.

        Parameters
        ----------
        file_path : str
            Path to the file.

        Returns
        -------
        Optional[str]
            Detected language name, or None if unsupported.
        """
        suffix = Path(file_path).suffix.lower()
        return self.language_map.get(suffix)

    def _validate_file_content(self, content: str, file_path: str) -> bool:
        """
        Validate file content before chunking.

        Parameters
        ----------
        content : str
            File content.
        file_path : str
            File path for logging.

        Returns
        -------
        bool
            True if content is valid for chunking.
        """
        # Check if content is empty
        if not content.strip():
            logger.debug(f"Skipping empty file: {file_path}")
            return False

        # Check if file is binary (contains null bytes)
        if "\x00" in content:
            logger.debug(f"Skipping binary file: {file_path}")
            return False

        # Check line count
        lines = content.splitlines()
        if len(lines) < self.chunking_config.min_chunk_size:
            logger.debug(f"Skipping small file: {file_path} ({len(lines)} lines)")
            return False

        return True

    def _chunk_content(
        self,
        content: str,
        file_path: str,
        repository_name: str,
        commit_hash: str,
        language: str,
    ) -> list[CodeChunk]:
        """
        Chunk file content using astchunk.

        Parameters
        ----------
        content : str
            File content to chunk.
        file_path : str
            Relative file path.
        repository_name : str
            Repository name.
        commit_hash : str
            Commit hash.
        language : str
            Programming language.

        Returns
        -------
        List[CodeChunk]
            List of generated code chunks.
        """
        chunks = []

        try:
            # Get language-specific configuration
            lang_config = self.chunking_config.languages.get(language, {})
            min_lines = getattr(lang_config, "min_lines", 5)

            # Use astchunk to parse and chunk the content
            try:
                # Create ASTChunkBuilder with configuration
                builder = astchunk.ASTChunkBuilder(
                    max_chunk_size=self.chunking_config.max_chunk_size,
                    language=language,
                    metadata_template="basic",
                )

                # Chunk the content
                code_windows = builder.chunkify(
                    content,
                    repo_level_metadata={
                        "file_path": file_path,
                        "repository": repository_name,
                    },
                )

                # Convert to our format
                for _i, window in enumerate(code_windows):
                    chunk_content = window.get("content", "")
                    metadata = window.get("metadata", {})

                    start_line = metadata.get("start_line", 1)
                    end_line = metadata.get("end_line", start_line)

                    # Skip chunks that are too small
                    if end_line - start_line + 1 < min_lines:
                        continue

                    # Extract symbol information if available
                    symbol_name = metadata.get("symbol_name")
                    symbol_type = metadata.get("symbol_type")

                    # Create CodeChunk
                    chunk = CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        repository=repository_name,
                        start_line=start_line,
                        end_line=end_line,
                        commit_hash=commit_hash,
                        language=language,
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                    )

                    chunks.append(chunk)

            except Exception:
                logger.warning(
                    f"AST chunking failed for {file_path}: {e}. "
                    "Falling back to line-based chunking."
                )
                chunks = self._create_fallback_chunks(
                    content, file_path, repository_name, commit_hash, language
                )

        except Exception:
            logger.exception(f"Error chunking content for {file_path}")
            # Fallback to simple line-based chunking
            chunks = self._create_fallback_chunks(
                content, file_path, repository_name, commit_hash, language
            )

        return chunks

    def _create_fallback_chunks(
        self,
        content: str,
        file_path: str,
        repository_name: str,
        commit_hash: str,
        language: str,
    ) -> list[CodeChunk]:
        """
        Create fallback chunks when all parsing fails.

        Parameters
        ----------
        content : str
            File content.
        file_path : str
            File path.
        repository_name : str
            Repository name.
        commit_hash : str
            Commit hash.
        language : str
            Programming language.

        Returns
        -------
        List[CodeChunk]
            List of fallback chunks.
        """
        # Create a single chunk for the entire file if it's not too large
        lines = content.splitlines()
        if len(lines) <= self.chunking_config.max_chunk_size:
            return [
                CodeChunk(
                    content=content,
                    file_path=file_path,
                    repository=repository_name,
                    start_line=1,
                    end_line=len(lines),
                    commit_hash=commit_hash,
                    language=language,
                    symbol_name=None,
                    symbol_type="file",
                )
            ]

        # Split into multiple chunks
        chunks = []
        chunk_size = self.chunking_config.max_chunk_size
        overlap = self.chunking_config.overlap_lines

        start = 0
        chunk_num = 1
        while start < len(lines):
            end = min(start + chunk_size, len(lines))

            chunk_lines = lines[start:end]
            chunk_content = "\n".join(chunk_lines)

            chunk = CodeChunk(
                content=chunk_content,
                file_path=file_path,
                repository=repository_name,
                start_line=start + 1,
                end_line=end,
                commit_hash=commit_hash,
                language=language,
                symbol_name=f"chunk_{chunk_num}",
                symbol_type="chunk",
            )

            chunks.append(chunk)
            chunk_num += 1

            # Move start position with overlap
            start = end - overlap if end < len(lines) else end

        return chunks
