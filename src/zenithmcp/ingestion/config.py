"""Configuration management for the ingestion pipeline."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""

    host: str = "localhost"
    port: int = 6333
    collection_name: str = "zenithmcp_code_chunks"
    vector_size: int = 768
    distance: str = "Cosine"
    timeout: float = 30.0
    prefer_grpc: bool = False


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = "microsoft/graphcodebert-base"
    batch_size: int = 32
    max_length: int = 512
    device: str = "auto"
    cache_dir: str = ".cache/huggingface"


class LanguageConfig(BaseModel):
    """Language-specific chunking configuration."""

    chunk_types: list[str] = Field(default_factory=list)
    min_lines: int = 5


class ChunkingConfig(BaseModel):
    """Code chunking configuration."""

    file_extensions: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    min_chunk_size: int = 50
    max_chunk_size: int = 500
    overlap_lines: int = 10
    languages: dict[str, LanguageConfig] = Field(default_factory=dict)

    @field_validator("languages", mode="before")
    @classmethod
    def parse_languages(cls, v: Any) -> dict[str, LanguageConfig]:
        """Parse language configurations."""
        if isinstance(v, dict):
            return {
                lang: LanguageConfig(**config) if isinstance(config, dict) else config
                for lang, config in v.items()
            }
        return v


class GitConfig(BaseModel):
    """Git configuration."""

    enabled: bool = True
    max_commits: int = 1000
    branches: list[str] = Field(default_factory=list)


class FilesystemConfig(BaseModel):
    """Filesystem watching configuration."""

    enabled: bool = False
    poll_interval: float = 5.0
    full_scan_on_start: bool = True


class DiscoveryConfig(BaseModel):
    """File discovery configuration."""

    state_file: str = ".zenithmcp_state"
    git: GitConfig = Field(default_factory=GitConfig)
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)


class IndexingConfig(BaseModel):
    """Vector indexing configuration."""

    batch_size: int = 100
    create_collection: bool = True
    recreate_on_schema_change: bool = False
    parallel_workers: int = 4
    max_retries: int = 3
    retry_delay: float = 1.0


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None
    max_file_size: str = "10MB"
    backup_count: int = 5


class PerformanceConfig(BaseModel):
    """Performance configuration."""

    max_memory_mb: int = 4096
    max_workers: int = 0
    use_gpu: bool = True
    batch_timeout: float = 300.0


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    incremental: bool = True
    validate_embeddings: bool = True
    skip_failed_files: bool = True
    max_file_size_mb: int = 10
    process_binary_files: bool = False


class Config(BaseModel):
    """Main configuration class."""

    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def load_config(config_path: str | None = None) -> Config:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : Optional[str]
        Path to configuration file. If None, looks for config.yaml in current directory.

    Returns
    -------
    Config
        Loaded configuration object.

    Raises
    ------
    FileNotFoundError
        If configuration file is not found.
    yaml.YAMLError
        If configuration file is invalid YAML.
    """
    if config_path is None:
        config_path = "config.yaml"

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}") from e

    if config_data is None:
        config_data = {}

    return Config(**config_data)


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up logging based on configuration.

    Parameters
    ----------
    config : LoggingConfig
        Logging configuration.
    """
    # Convert string level to logging constant
    level = getattr(logging, config.level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if config.file:
        from logging.handlers import RotatingFileHandler

        # Parse max file size
        max_bytes = _parse_size(config.max_file_size)

        file_handler = RotatingFileHandler(
            config.file, maxBytes=max_bytes, backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def _parse_size(size_str: str) -> int:
    """
    Parse size string like '10MB' to bytes.

    Parameters
    ----------
    size_str : str
        Size string with unit (B, KB, MB, GB).

    Returns
    -------
    int
        Size in bytes.
    """
    size_str = size_str.upper().strip()

    if size_str.endswith("B"):
        if size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        if size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        return int(size_str[:-1])
    return int(size_str)


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns
    -------
    Config
        Global configuration object.

    Raises
    ------
    RuntimeError
        If configuration has not been loaded.
    """
    global _config
    if _config is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.

    Parameters
    ----------
    config : Config
        Configuration object to set as global.
    """
    global _config
    _config = config


def load_and_set_config(config_path: str | None = None) -> Config:
    """
    Load configuration and set it as global.

    Parameters
    ----------
    config_path : Optional[str]
        Path to configuration file.

    Returns
    -------
    Config
        Loaded configuration object.
    """
    config = load_config(config_path)
    set_config(config)
    setup_logging(config.logging)
    return config
