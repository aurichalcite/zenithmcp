"""Tests for embedding generation."""

from unittest.mock import Mock, patch

import torch

from zenithmcp.ingestion.embedding import EmbeddingGenerator
from zenithmcp.ingestion.models import CodeChunk


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    @patch("zenithmcp.ingestion.embedding.AutoModel")
    @patch("zenithmcp.ingestion.embedding.AutoTokenizer")
    def test_init_with_config(
        self, mock_tokenizer_class, mock_model_class, sample_config
    ):
        """Test initializing embedding generator with config."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        generator = EmbeddingGenerator(sample_config)

        assert generator.config is sample_config
        assert generator.embedding_config is sample_config.embedding
        assert generator.tokenizer is mock_tokenizer
        assert generator.model is mock_model

    @patch("zenithmcp.ingestion.embedding.torch.cuda.is_available", return_value=False)
    @patch(
        "zenithmcp.ingestion.embedding.torch.backends.mps.is_available",
        return_value=False,
    )
    def test_get_device_cpu_only(self, mock_mps, mock_cuda, sample_config):
        """Test device selection when only CPU is available."""
        # Force CPU device in config
        sample_config.embedding.device = "auto"
        sample_config.performance.use_gpu = True

        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

            assert generator.device == torch.device("cpu")

    @patch("zenithmcp.ingestion.embedding.torch.cuda.is_available", return_value=True)
    def test_get_device_cuda(self, mock_cuda, sample_config):
        """Test device selection when CUDA is available."""
        sample_config.embedding.device = "auto"
        sample_config.performance.use_gpu = True

        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

            assert generator.device == torch.device("cuda")

    def test_get_device_explicit_cpu(self, sample_config):
        """Test explicit CPU device selection."""
        sample_config.embedding.device = "cpu"

        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

            assert generator.device == torch.device("cpu")

    @patch("zenithmcp.ingestion.embedding.AutoModel")
    @patch("zenithmcp.ingestion.embedding.AutoTokenizer")
    def test_embed_batch_success(
        self, mock_tokenizer_class, mock_model_class, sample_code_chunks, sample_config
    ):
        """Test successful batch embedding generation."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_outputs = Mock()
        # Create mock embeddings (batch_size=2, seq_len=3, hidden_size=4)
        mock_outputs.last_hidden_state = torch.randn(2, 3, 4)
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model

        generator = EmbeddingGenerator(sample_config)

        # Take first 2 chunks for testing
        chunks = sample_code_chunks[:2]
        embedded_chunks = generator._embed_batch(chunks)

        assert len(embedded_chunks) == 2

        # Check that embeddings were added
        for chunk in embedded_chunks:
            assert chunk.embedding is not None
            assert isinstance(chunk.embedding, list)
            assert len(chunk.embedding) == 4  # hidden_size from mock
            assert all(isinstance(x, float) for x in chunk.embedding)

    @patch("zenithmcp.ingestion.embedding.AutoModel")
    @patch("zenithmcp.ingestion.embedding.AutoTokenizer")
    def test_embed_batch_error(
        self, mock_tokenizer_class, mock_model_class, sample_code_chunks, sample_config
    ):
        """Test batch embedding with error."""
        # Mock tokenizer to raise exception
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = Exception("Tokenization failed")
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model_class.from_pretrained.return_value = mock_model

        generator = EmbeddingGenerator(sample_config)

        chunks = sample_code_chunks[:2]
        embedded_chunks = generator._embed_batch(chunks)

        # Should return original chunks without embeddings
        assert len(embedded_chunks) == 2
        for chunk in embedded_chunks:
            assert chunk.embedding is None

    @patch("zenithmcp.ingestion.embedding.AutoModel")
    @patch("zenithmcp.ingestion.embedding.AutoTokenizer")
    def test_run_multiple_batches(
        self, mock_tokenizer_class, mock_model_class, sample_config
    ):
        """Test running embedding generation with multiple batches."""
        # Create more chunks than batch size
        chunks = []
        for i in range(5):  # batch_size is 2, so this will create 3 batches
            chunk = CodeChunk(
                content=f"def func_{i}(): pass",
                file_path=f"file_{i}.py",
                repository="test_repo",
                start_line=1,
                end_line=1,
                commit_hash="abc123",
                language="python",
            )
            chunks.append(chunk)

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 4)
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model

        generator = EmbeddingGenerator(sample_config)
        embedded_chunks = generator.run(chunks)

        assert len(embedded_chunks) == 5

        # All chunks should have embeddings (assuming no errors)
        for chunk in embedded_chunks:
            assert chunk.embedding is not None

    @patch("zenithmcp.ingestion.embedding.AutoModel")
    @patch("zenithmcp.ingestion.embedding.AutoTokenizer")
    def test_embed_single(
        self, mock_tokenizer_class, mock_model_class, sample_code_chunk, sample_config
    ):
        """Test embedding a single chunk."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 4)
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model

        generator = EmbeddingGenerator(sample_config)
        embedded_chunk = generator.embed_single(sample_code_chunk)

        assert embedded_chunk.embedding is not None
        assert len(embedded_chunk.embedding) == 4

    def test_mean_pooling(self, sample_config):
        """Test mean pooling calculation."""
        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

        # Create test tensors
        token_embeddings = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # First sequence
                [
                    [7.0, 8.0],
                    [9.0, 10.0],
                    [0.0, 0.0],
                ],  # Second sequence (last token masked)
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1],  # All tokens valid
                [1, 1, 0],  # Last token masked
            ]
        )

        pooled = generator._mean_pooling(token_embeddings, attention_mask)

        # Check shape
        assert pooled.shape == (2, 2)

        # Check first sequence (mean of all tokens)
        expected_first = torch.tensor([3.0, 4.0])  # (1+3+5)/3, (2+4+6)/3
        assert torch.allclose(pooled[0], expected_first)

        # Check second sequence (mean of valid tokens only)
        expected_second = torch.tensor([8.0, 9.0])  # (7+9)/2, (8+10)/2
        assert torch.allclose(pooled[1], expected_second)

    @patch("zenithmcp.ingestion.embedding.AutoModel")
    @patch("zenithmcp.ingestion.embedding.AutoTokenizer")
    def test_get_embedding_dimension(
        self, mock_tokenizer_class, mock_model_class, sample_config
    ):
        """Test getting embedding dimension."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        generator = EmbeddingGenerator(sample_config)

        assert generator.get_embedding_dimension() == 768

    def test_validate_embeddings_success(
        self, sample_code_chunks, mock_embedding_vector, sample_config
    ):
        """Test successful embedding validation."""
        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

        # Add embeddings to chunks
        for chunk in sample_code_chunks:
            chunk.embedding = mock_embedding_vector

        result = generator.validate_embeddings(sample_code_chunks)

        assert result.success is True
        assert result.chunk_count == len(sample_code_chunks)
        assert result.error_message is None

    def test_validate_embeddings_missing(self, sample_code_chunks, sample_config):
        """Test validation with missing embeddings."""
        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

        # Leave embeddings as None
        result = generator.validate_embeddings(sample_code_chunks)

        assert result.success is False
        assert result.chunk_count == 0
        assert "0/3 chunks have valid embeddings" in result.error_message

    def test_validate_embeddings_wrong_dimension(
        self, sample_code_chunks, sample_config
    ):
        """Test validation with wrong embedding dimensions."""
        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

        # Add embeddings with wrong dimensions
        for chunk in sample_code_chunks:
            chunk.embedding = [0.1, 0.2]  # Wrong dimension (should be 768)

        result = generator.validate_embeddings(sample_code_chunks)

        assert result.success is False
        assert result.chunk_count == 0

    def test_validate_embeddings_invalid_values(
        self, sample_code_chunks, sample_config
    ):
        """Test validation with invalid embedding values."""
        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

        # Add embeddings with invalid values
        for chunk in sample_code_chunks:
            chunk.embedding = ["not", "numeric"] + [0.1] * 766  # Invalid values

        result = generator.validate_embeddings(sample_code_chunks)

        assert result.success is False
        assert result.chunk_count == 0

    @patch("zenithmcp.ingestion.embedding.torch.cuda.empty_cache")
    @patch("zenithmcp.ingestion.embedding.torch.cuda.is_available", return_value=True)
    def test_cleanup(self, mock_cuda_available, mock_empty_cache, sample_config):
        """Test cleanup of resources."""
        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

        # Verify model and tokenizer are set
        assert generator.model is not None
        assert generator.tokenizer is not None

        generator.cleanup()

        # Verify cleanup
        assert generator.model is None
        assert generator.tokenizer is None
        mock_empty_cache.assert_called_once()

    def test_run_empty_chunks(self, sample_config):
        """Test running with empty chunk list."""
        with (
            patch("zenithmcp.ingestion.embedding.AutoModel"),
            patch("zenithmcp.ingestion.embedding.AutoTokenizer"),
        ):
            generator = EmbeddingGenerator(sample_config)

        result = generator.run([])

        assert result == []
