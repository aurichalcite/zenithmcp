"""Tests for configuration management."""

import logging
import tempfile
from pathlib import Path

import pytest
import yaml

from zenithmcp.ingestion.config import (
    Config,
    get_config,
    load_and_set_config,
    load_config,
    set_config,
    setup_logging,
)


class TestConfig:
    """Tests for Config model."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = Config()

        # Check default values
        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
        assert config.qdrant.collection_name == "zenithmcp_code_chunks"
        assert config.qdrant.vector_size == 768

        assert config.embedding.model_name == "microsoft/graphcodebert-base"
        assert config.embedding.batch_size == 32
        assert config.embedding.device == "auto"

        assert config.chunking.min_chunk_size == 50
        assert config.chunking.max_chunk_size == 500
        assert config.chunking.overlap_lines == 10

        assert config.discovery.state_file == ".zenithmcp_state"
        assert config.discovery.git.enabled is True

        assert config.indexing.batch_size == 100
        assert config.indexing.create_collection is True

        assert config.logging.level == "INFO"
        assert config.performance.use_gpu is True
        assert config.pipeline.incremental is True

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = Config(
            qdrant={"host": "remote-host", "port": 9999},
            embedding={"batch_size": 16, "device": "cpu"},
            chunking={"min_chunk_size": 10, "max_chunk_size": 200},
        )

        assert config.qdrant.host == "remote-host"
        assert config.qdrant.port == 9999
        assert config.embedding.batch_size == 16
        assert config.embedding.device == "cpu"
        assert config.chunking.min_chunk_size == 10
        assert config.chunking.max_chunk_size == 200

        # Defaults should still be present
        assert config.qdrant.collection_name == "zenithmcp_code_chunks"
        assert config.embedding.model_name == "microsoft/graphcodebert-base"


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config_from_file(self, temp_config_file):
        """Test loading configuration from YAML file."""
        config = load_config(str(temp_config_file))

        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
        assert config.qdrant.collection_name == "test_collection"
        assert config.embedding.batch_size == 2
        assert config.embedding.device == "cpu"
        assert config.chunking.min_chunk_size == 5
        assert config.chunking.max_chunk_size == 100

    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_config_invalid_yaml(self):
        """Test loading config with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with pytest.raises(yaml.YAMLError):
                load_config(f.name)

            Path(f.name).unlink()

    def test_load_config_empty_file(self):
        """Test loading config from empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            config = load_config(f.name)
            # Should create config with defaults
            assert config.qdrant.host == "localhost"

            Path(f.name).unlink()

    def test_load_config_partial_config(self):
        """Test loading partial configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
qdrant:
  host: custom-host
  port: 7777

embedding:
  batch_size: 8
""")
            f.flush()

            config = load_config(f.name)

            # Custom values
            assert config.qdrant.host == "custom-host"
            assert config.qdrant.port == 7777
            assert config.embedding.batch_size == 8

            # Defaults should still be present
            assert config.qdrant.collection_name == "zenithmcp_code_chunks"
            assert config.embedding.model_name == "microsoft/graphcodebert-base"

            Path(f.name).unlink()


class TestGlobalConfig:
    """Tests for global configuration management."""

    def test_set_and_get_config(self, sample_config):
        """Test setting and getting global config."""
        set_config(sample_config)
        retrieved_config = get_config()

        assert retrieved_config is sample_config
        assert retrieved_config.qdrant.host == sample_config.qdrant.host

    def test_get_config_not_loaded(self):
        """Test getting config when not loaded."""
        # Clear global config
        from zenithmcp.ingestion import config as config_module

        config_module._config = None

        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            get_config()

    def test_load_and_set_config(self, temp_config_file):
        """Test loading and setting config in one call."""
        config = load_and_set_config(str(temp_config_file))

        assert config.qdrant.collection_name == "test_collection"

        # Should be set as global config
        retrieved_config = get_config()
        assert retrieved_config is config


class TestLoggingSetup:
    """Tests for logging setup."""

    def test_setup_logging_console_only(self, sample_config):
        """Test setting up console-only logging."""
        setup_logging(sample_config.logging)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

        # Should have at least one handler (console)
        assert len(root_logger.handlers) >= 1

    def test_setup_logging_with_file(self, sample_config):
        """Test setting up logging with file output."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_file = f.name

        # Update config to include file logging
        sample_config.logging.file = log_file
        setup_logging(sample_config.logging)

        root_logger = logging.getLogger()

        # Should have both console and file handlers
        handler_types = [type(h).__name__ for h in root_logger.handlers]
        assert "RichHandler" in handler_types or "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types

        # Cleanup
        Path(log_file).unlink(missing_ok=True)

    def test_setup_logging_different_levels(self, sample_config):
        """Test setting up logging with different levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            sample_config.logging.level = level
            setup_logging(sample_config.logging)

            root_logger = logging.getLogger()
            expected_level = getattr(logging, level)
            assert root_logger.level == expected_level


class TestLanguageConfig:
    """Tests for language-specific configuration."""

    def test_language_config_parsing(self):
        """Test parsing language configurations."""
        config_data = {
            "chunking": {
                "languages": {
                    "python": {
                        "chunk_types": ["function", "class"],
                        "min_lines": 5,
                    },
                    "javascript": {
                        "chunk_types": ["function", "class", "arrow_function"],
                        "min_lines": 3,
                    },
                }
            }
        }

        config = Config(**config_data)

        assert "python" in config.chunking.languages
        assert "javascript" in config.chunking.languages

        python_config = config.chunking.languages["python"]
        assert python_config.chunk_types == ["function", "class"]
        assert python_config.min_lines == 5

        js_config = config.chunking.languages["javascript"]
        assert js_config.chunk_types == ["function", "class", "arrow_function"]
        assert js_config.min_lines == 3

    def test_empty_language_config(self):
        """Test with empty language configuration."""
        config = Config(chunking={"languages": {}})

        assert config.chunking.languages == {}


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_config_values(self):
        """Test that valid configuration values are accepted."""
        config = Config(
            qdrant={
                "host": "localhost",
                "port": 6333,
                "vector_size": 768,
                "distance": "Cosine",
                "timeout": 30.0,
            },
            embedding={
                "batch_size": 32,
                "max_length": 512,
                "device": "auto",
            },
            chunking={
                "min_chunk_size": 10,
                "max_chunk_size": 1000,
                "overlap_lines": 5,
            },
        )

        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
        assert config.embedding.batch_size == 32
        assert config.chunking.min_chunk_size == 10

    def test_config_with_lists(self):
        """Test configuration with list values."""
        config = Config(
            chunking={
                "file_extensions": [".py", ".js", ".ts"],
                "exclude_patterns": ["*/node_modules/*", "*/__pycache__/*"],
            },
            discovery={
                "git": {
                    "branches": ["main", "develop"],
                }
            },
        )

        assert config.chunking.file_extensions == [".py", ".js", ".ts"]
        assert config.chunking.exclude_patterns == [
            "*/node_modules/*",
            "*/__pycache__/*",
        ]
        assert config.discovery.git.branches == ["main", "develop"]
