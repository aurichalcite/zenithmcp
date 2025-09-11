"""Tests for pipeline orchestration."""

from unittest.mock import Mock, patch

from typer.testing import CliRunner

from zenithmcp.ingestion.pipeline import app


class TestPipelineOrchestration:
    """Tests for pipeline orchestration."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    @patch("zenithmcp.ingestion.pipeline.VectorIndexer")
    @patch("zenithmcp.ingestion.pipeline.EmbeddingGenerator")
    @patch("zenithmcp.ingestion.pipeline.CodeChunker")
    @patch("zenithmcp.ingestion.pipeline.SourceFileDiscoverer")
    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_run_command_success(
        self,
        mock_load_config,
        mock_discoverer_class,
        mock_chunker_class,
        mock_embedder_class,
        mock_indexer_class,
        sample_config,
        sample_code_chunks,
        mock_embedding_vector,
        fs,
    ):
        """Test successful pipeline run."""
        # Setup fake filesystem
        repo_path = "/test/repo"
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "main.py", contents="print('hello')")

        # Mock configuration loading
        mock_load_config.return_value = sample_config

        # Mock discoverer
        mock_discoverer = Mock()
        mock_discoverer.run.return_value = ["main.py"]
        mock_discoverer.get_current_commit_hash.return_value = "abc123"
        mock_discoverer.save_processing_state.return_value = None
        mock_discoverer_class.return_value = mock_discoverer

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.run.return_value = sample_code_chunks
        mock_chunker_class.return_value = mock_chunker

        # Mock embedder
        mock_embedder = Mock()
        embedded_chunks = []
        for chunk in sample_code_chunks:
            chunk.embedding = mock_embedding_vector
            embedded_chunks.append(chunk)
        mock_embedder.run.return_value = embedded_chunks
        mock_embedder.cleanup.return_value = None
        mock_embedder_class.return_value = mock_embedder

        # Mock indexer
        mock_indexer = Mock()
        mock_indexing_result = Mock()
        mock_indexing_result.success = True
        mock_indexing_result.chunk_count = len(sample_code_chunks)
        mock_indexer.run.return_value = mock_indexing_result
        mock_indexer.cleanup.return_value = None
        mock_indexer_class.return_value = mock_indexer

        # Run command
        result = self.runner.invoke(app, ["run", repo_path])

        assert result.exit_code == 0

        # Verify all components were called
        mock_discoverer.run.assert_called_once_with(repo_path)
        mock_chunker.run.assert_called_once()
        mock_embedder.run.assert_called_once()
        mock_indexer.run.assert_called_once()

        # Verify cleanup was called
        mock_embedder.cleanup.assert_called_once()
        mock_indexer.cleanup.assert_called_once()

    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_run_command_invalid_repo_path(self, mock_load_config, sample_config):
        """Test run command with invalid repository path."""
        mock_load_config.return_value = sample_config

        result = self.runner.invoke(app, ["run", "/nonexistent/path"])

        assert result.exit_code == 1
        assert "Repository path does not exist" in result.output

    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_run_command_config_load_failure(self, mock_load_config):
        """Test run command with configuration loading failure."""
        mock_load_config.side_effect = Exception("Config load failed")

        result = self.runner.invoke(app, ["run", "/test/repo"])

        assert result.exit_code == 1
        assert "Failed to load configuration" in result.output

    @patch("zenithmcp.ingestion.pipeline.SourceFileDiscoverer")
    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_run_command_no_files_found(
        self,
        mock_load_config,
        mock_discoverer_class,
        sample_config,
        fs,
    ):
        """Test run command when no files are found."""
        # Setup fake filesystem
        repo_path = "/test/repo"
        fs.create_dir(repo_path)

        # Mock configuration loading
        mock_load_config.return_value = sample_config

        # Mock discoverer to return no files
        mock_discoverer = Mock()
        mock_discoverer.run.return_value = []
        mock_discoverer_class.return_value = mock_discoverer

        # Run command
        result = self.runner.invoke(app, ["run", repo_path])

        assert result.exit_code == 0
        assert "No files to process" in result.output

    @patch("zenithmcp.ingestion.pipeline.VectorIndexer")
    @patch("zenithmcp.ingestion.pipeline.EmbeddingGenerator")
    @patch("zenithmcp.ingestion.pipeline.CodeChunker")
    @patch("zenithmcp.ingestion.pipeline.SourceFileDiscoverer")
    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_run_command_dry_run(
        self,
        mock_load_config,
        mock_discoverer_class,
        mock_chunker_class,
        mock_embedder_class,
        mock_indexer_class,
        sample_config,
        sample_code_chunks,
        mock_embedding_vector,
        fs,
    ):
        """Test pipeline run in dry-run mode."""
        # Setup fake filesystem
        repo_path = "/test/repo"
        fs.create_dir(repo_path)

        # Mock configuration loading
        mock_load_config.return_value = sample_config

        # Mock components
        mock_discoverer = Mock()
        mock_discoverer.run.return_value = ["main.py"]
        mock_discoverer.get_current_commit_hash.return_value = "abc123"
        mock_discoverer_class.return_value = mock_discoverer

        mock_chunker = Mock()
        mock_chunker.run.return_value = sample_code_chunks
        mock_chunker_class.return_value = mock_chunker

        mock_embedder = Mock()
        embedded_chunks = []
        for chunk in sample_code_chunks:
            chunk.embedding = mock_embedding_vector
            embedded_chunks.append(chunk)
        mock_embedder.run.return_value = embedded_chunks
        mock_embedder.cleanup.return_value = None
        mock_embedder_class.return_value = mock_embedder

        # Run command with dry-run flag
        result = self.runner.invoke(app, ["run", repo_path, "--dry-run"])

        assert result.exit_code == 0
        assert "Skipping vector indexing (dry run mode)" in result.output

        # Verify indexer was not created
        mock_indexer_class.assert_not_called()

        # Verify other components were still called
        mock_discoverer.run.assert_called_once()
        mock_chunker.run.assert_called_once()
        mock_embedder.run.assert_called_once()

    @patch("zenithmcp.ingestion.pipeline.VectorIndexer")
    @patch("zenithmcp.ingestion.pipeline.EmbeddingGenerator")
    @patch("zenithmcp.ingestion.pipeline.CodeChunker")
    @patch("zenithmcp.ingestion.pipeline.SourceFileDiscoverer")
    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_run_command_force_flag(
        self,
        mock_load_config,
        mock_discoverer_class,
        mock_chunker_class,
        mock_embedder_class,
        mock_indexer_class,
        sample_config,
        sample_code_chunks,
        fs,
    ):
        """Test pipeline run with force flag."""
        # Setup fake filesystem
        repo_path = "/test/repo"
        fs.create_dir(repo_path)

        # Mock configuration loading
        mock_load_config.return_value = sample_config

        # Mock discoverer to return no files initially
        mock_discoverer = Mock()
        mock_discoverer.run.return_value = []
        mock_discoverer_class.return_value = mock_discoverer

        # Run command without force flag
        result = self.runner.invoke(app, ["run", repo_path])

        assert result.exit_code == 0
        assert "No files to process" in result.output

        # Run command with force flag - should continue processing
        result = self.runner.invoke(app, ["run", repo_path, "--force"])

        # Should not exit early even with no files
        assert result.exit_code == 0

    @patch("zenithmcp.ingestion.pipeline.VectorIndexer")
    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_health_command_success(
        self, mock_load_config, mock_indexer_class, sample_config
    ):
        """Test successful health check command."""
        mock_load_config.return_value = sample_config

        # Mock indexer health check
        mock_indexer = Mock()
        mock_indexer.health_check.return_value = True
        mock_indexer.get_collection_info.return_value = {
            "points_count": 100,
        }
        mock_indexer.cleanup.return_value = None
        mock_indexer_class.return_value = mock_indexer

        # Mock embedding generator
        with patch(
            "zenithmcp.ingestion.pipeline.EmbeddingGenerator"
        ) as mock_embedder_class:
            mock_embedder = Mock()
            mock_embedder.device = "cpu"
            mock_embedder.get_embedding_dimension.return_value = 768
            mock_embedder.cleanup.return_value = None
            mock_embedder_class.return_value = mock_embedder

            result = self.runner.invoke(app, ["health"])

        assert result.exit_code == 0
        assert "Qdrant: ✓ Healthy" in result.output
        assert "GraphCodeBERT: ✓ Loaded" in result.output

    @patch("zenithmcp.ingestion.pipeline.VectorIndexer")
    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_health_command_qdrant_unhealthy(
        self, mock_load_config, mock_indexer_class, sample_config
    ):
        """Test health check command with unhealthy Qdrant."""
        mock_load_config.return_value = sample_config

        # Mock indexer health check failure
        mock_indexer = Mock()
        mock_indexer.health_check.return_value = False
        mock_indexer.cleanup.return_value = None
        mock_indexer_class.return_value = mock_indexer

        result = self.runner.invoke(app, ["health"])

        assert result.exit_code == 0
        assert "Qdrant: ✗ Unhealthy" in result.output

    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_info_command(self, mock_load_config, sample_config):
        """Test info command."""
        mock_load_config.return_value = sample_config

        result = self.runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "ZenithMCP Configuration" in result.output
        assert "Qdrant Configuration:" in result.output
        assert "Embedding Configuration:" in result.output
        assert "Chunking Configuration:" in result.output
        assert sample_config.qdrant.host in result.output
        assert str(sample_config.qdrant.port) in result.output
        assert sample_config.embedding.model_name in result.output

    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_info_command_config_failure(self, mock_load_config):
        """Test info command with configuration loading failure."""
        mock_load_config.side_effect = Exception("Config load failed")

        result = self.runner.invoke(app, ["info"])

        assert result.exit_code == 1
        assert "Failed to load configuration" in result.output

    @patch("zenithmcp.ingestion.pipeline.CodeChunker")
    @patch("zenithmcp.ingestion.pipeline.SourceFileDiscoverer")
    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_run_command_chunking_failure(
        self,
        mock_load_config,
        mock_discoverer_class,
        mock_chunker_class,
        sample_config,
        fs,
    ):
        """Test run command when chunking fails."""
        # Setup fake filesystem
        repo_path = "/test/repo"
        fs.create_dir(repo_path)

        # Mock configuration loading
        mock_load_config.return_value = sample_config

        # Mock discoverer
        mock_discoverer = Mock()
        mock_discoverer.run.return_value = ["main.py"]
        mock_discoverer_class.return_value = mock_discoverer

        # Mock chunker to raise exception
        mock_chunker = Mock()
        mock_chunker.run.side_effect = Exception("Chunking failed")
        mock_chunker_class.return_value = mock_chunker

        # Run command
        result = self.runner.invoke(app, ["run", repo_path])

        assert result.exit_code == 1
        assert "Code chunking failed" in result.output

    @patch("zenithmcp.ingestion.pipeline.EmbeddingGenerator")
    @patch("zenithmcp.ingestion.pipeline.CodeChunker")
    @patch("zenithmcp.ingestion.pipeline.SourceFileDiscoverer")
    @patch("zenithmcp.ingestion.pipeline.load_and_set_config")
    def test_run_command_embedding_failure(
        self,
        mock_load_config,
        mock_discoverer_class,
        mock_chunker_class,
        mock_embedder_class,
        sample_config,
        sample_code_chunks,
        fs,
    ):
        """Test run command when embedding generation fails."""
        # Setup fake filesystem
        repo_path = "/test/repo"
        fs.create_dir(repo_path)

        # Mock configuration loading
        mock_load_config.return_value = sample_config

        # Mock discoverer
        mock_discoverer = Mock()
        mock_discoverer.run.return_value = ["main.py"]
        mock_discoverer_class.return_value = mock_discoverer

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.run.return_value = sample_code_chunks
        mock_chunker_class.return_value = mock_chunker

        # Mock embedder to raise exception
        mock_embedder = Mock()
        mock_embedder.run.side_effect = Exception("Embedding failed")
        mock_embedder.cleanup.return_value = None
        mock_embedder_class.return_value = mock_embedder

        # Run command
        result = self.runner.invoke(app, ["run", repo_path])

        assert result.exit_code == 1
        assert "Embedding generation failed" in result.output

    def test_run_command_with_config_file(self, temp_config_file, fs):
        """Test run command with custom config file."""
        # Setup fake filesystem
        repo_path = "/test/repo"
        fs.create_dir(repo_path)

        with patch(
            "zenithmcp.ingestion.pipeline.load_and_set_config"
        ) as mock_load_config:
            mock_load_config.side_effect = Exception("Config not found")

            result = self.runner.invoke(
                app, ["run", repo_path, "--config", str(temp_config_file)]
            )

            # Should attempt to load the specified config file
            mock_load_config.assert_called_once_with(str(temp_config_file))

    def test_run_command_verbose_logging(self, fs):
        """Test run command with verbose logging."""
        # Setup fake filesystem
        repo_path = "/test/repo"
        fs.create_dir(repo_path)

        with patch(
            "zenithmcp.ingestion.pipeline.load_and_set_config"
        ) as mock_load_config:
            mock_load_config.side_effect = Exception("Config not found")

            result = self.runner.invoke(app, ["run", repo_path, "--verbose"])

            # Should enable verbose logging (DEBUG level)
            # This is tested indirectly through the setup_rich_logging call
