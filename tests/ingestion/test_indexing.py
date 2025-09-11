"""Tests for vector indexing."""

from unittest.mock import Mock, patch

import pytest
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from zenithmcp.ingestion.indexing import VectorIndexer


class TestVectorIndexer:
    """Tests for VectorIndexer class."""

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_init_with_config(self, mock_client_class, sample_config):
        """Test initializing vector indexer with config."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)

        assert indexer.config is sample_config
        assert indexer.qdrant_config is sample_config.qdrant
        assert indexer.indexing_config is sample_config.indexing
        assert indexer.client is mock_client

        # Verify client initialization
        mock_client_class.assert_called_once_with(
            host=sample_config.qdrant.host,
            port=sample_config.qdrant.port,
            timeout=sample_config.qdrant.timeout,
            prefer_grpc=sample_config.qdrant.prefer_grpc,
        )

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_ensure_collection_exists_new_collection(
        self, mock_client_class, sample_config
    ):
        """Test ensuring collection exists when it doesn't exist."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []  # No existing collections
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        indexer._ensure_collection_exists()

        # Should create the collection
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == sample_config.qdrant.collection_name
        assert call_args[1]["vectors_config"].size == sample_config.qdrant.vector_size

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_ensure_collection_exists_existing_collection(
        self, mock_client_class, sample_config
    ):
        """Test ensuring collection exists when it already exists."""
        mock_client = Mock()

        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = sample_config.qdrant.collection_name
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors.size = (
            sample_config.qdrant.vector_size
        )
        mock_client.get_collection.return_value = mock_collection_info

        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        indexer._ensure_collection_exists()

        # Should not create the collection
        mock_client.create_collection.assert_not_called()

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_ensure_collection_exists_wrong_vector_size(
        self, mock_client_class, sample_config
    ):
        """Test ensuring collection exists with wrong vector size."""
        mock_client = Mock()

        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = sample_config.qdrant.collection_name
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        # Mock collection info with wrong vector size
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors.size = 512  # Wrong size
        mock_client.get_collection.return_value = mock_collection_info

        mock_client_class.return_value = mock_client

        # Enable recreation on schema change
        sample_config.indexing.recreate_on_schema_change = True

        indexer = VectorIndexer(sample_config)
        indexer._ensure_collection_exists()

        # Should delete and recreate the collection
        mock_client.delete_collection.assert_called_once_with(
            sample_config.qdrant.collection_name
        )
        mock_client.create_collection.assert_called_once()

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_run_success(
        self,
        mock_client_class,
        sample_code_chunks,
        mock_embedding_vector,
        sample_config,
    ):
        """Test successful indexing run."""
        # Add embeddings to chunks
        for chunk in sample_code_chunks:
            chunk.embedding = mock_embedding_vector

        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Mock successful upsert
        mock_operation_info = Mock()
        mock_operation_info.status = models.UpdateStatus.COMPLETED
        mock_client.upsert.return_value = mock_operation_info

        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        result = indexer.run(sample_code_chunks)

        assert result.success is True
        assert result.chunk_count == len(sample_code_chunks)
        assert result.error_message is None
        assert len(result.chunk_ids) == len(sample_code_chunks)

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_run_no_embeddings(
        self, mock_client_class, sample_code_chunks, sample_config
    ):
        """Test indexing run with chunks that have no embeddings."""
        # Don't add embeddings to chunks

        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        result = indexer.run(sample_code_chunks)

        assert result.success is False
        assert "No chunks with embeddings to index" in result.error_message
        assert result.chunk_count == 0

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_run_empty_chunks(self, mock_client_class, sample_config):
        """Test indexing run with empty chunk list."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        result = indexer.run([])

        assert result.success is True
        assert result.chunk_count == 0
        assert result.processing_time >= 0

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_index_batch_success(
        self,
        mock_client_class,
        sample_code_chunks,
        mock_embedding_vector,
        sample_config,
    ):
        """Test successful batch indexing."""
        # Add embeddings to chunks
        for chunk in sample_code_chunks:
            chunk.embedding = mock_embedding_vector

        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Mock successful upsert
        mock_operation_info = Mock()
        mock_operation_info.status = models.UpdateStatus.COMPLETED
        mock_client.upsert.return_value = mock_operation_info

        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        chunk_ids = indexer._index_batch(sample_code_chunks)

        assert len(chunk_ids) == len(sample_code_chunks)

        # Verify upsert was called
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == sample_config.qdrant.collection_name
        assert len(call_args[1]["points"]) == len(sample_code_chunks)

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_index_batch_with_retry(
        self,
        mock_client_class,
        sample_code_chunks,
        mock_embedding_vector,
        sample_config,
    ):
        """Test batch indexing with retry on failure."""
        # Add embeddings to chunks
        for chunk in sample_code_chunks:
            chunk.embedding = mock_embedding_vector

        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Mock first call to fail, second to succeed
        mock_operation_info = Mock()
        mock_operation_info.status = models.UpdateStatus.COMPLETED
        mock_client.upsert.side_effect = [
            ResponseHandlingException("Network error"),
            mock_operation_info,
        ]

        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        chunk_ids = indexer._index_batch(sample_code_chunks)

        assert len(chunk_ids) == len(sample_code_chunks)

        # Verify upsert was called twice (retry)
        assert mock_client.upsert.call_count == 2

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_index_batch_max_retries_exceeded(
        self,
        mock_client_class,
        sample_code_chunks,
        mock_embedding_vector,
        sample_config,
    ):
        """Test batch indexing when max retries are exceeded."""
        # Add embeddings to chunks
        for chunk in sample_code_chunks:
            chunk.embedding = mock_embedding_vector

        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Mock all calls to fail
        mock_client.upsert.side_effect = ResponseHandlingException("Persistent error")

        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)

        with pytest.raises(ResponseHandlingException):
            indexer._index_batch(sample_code_chunks)

        # Verify upsert was called max_retries + 1 times
        expected_calls = sample_config.indexing.max_retries + 1
        assert mock_client.upsert.call_count == expected_calls

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_delete_chunks(self, mock_client_class, sample_config):
        """Test deleting chunks."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Mock successful delete
        mock_operation_info = Mock()
        mock_operation_info.status = models.UpdateStatus.COMPLETED
        mock_client.delete.return_value = mock_operation_info

        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        chunk_ids = ["id1", "id2", "id3"]
        success = indexer.delete_chunks(chunk_ids)

        assert success is True

        # Verify delete was called
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == sample_config.qdrant.collection_name
        assert call_args[1]["points_selector"].points == chunk_ids

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_delete_chunks_empty_list(self, mock_client_class, sample_config):
        """Test deleting empty chunk list."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        success = indexer.delete_chunks([])

        assert success is True

        # Should not call delete
        mock_client.delete.assert_not_called()

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_get_collection_info(self, mock_client_class, sample_config):
        """Test getting collection information."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors.size = 768
        mock_collection_info.config.params.vectors.distance = models.Distance.COSINE
        mock_collection_info.points_count = 100
        mock_collection_info.status = "green"
        mock_client.get_collection.return_value = mock_collection_info

        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        info = indexer.get_collection_info()

        assert info["vector_size"] == 768
        assert info["points_count"] == 100
        assert info["status"] == "green"

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_health_check_success(self, mock_client_class, sample_config):
        """Test successful health check."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        healthy = indexer.health_check()

        assert healthy is True

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_health_check_failure(self, mock_client_class, sample_config):
        """Test health check failure."""
        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        healthy = indexer.health_check()

        assert healthy is False

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_cleanup(self, mock_client_class, sample_config):
        """Test cleanup of resources."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)

        # Verify client is set
        assert indexer.client is not None

        indexer.cleanup()

        # Verify cleanup
        assert indexer.client is None
        mock_client.close.assert_called_once()

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_create_collection_distance_mapping(self, mock_client_class, sample_config):
        """Test collection creation with different distance metrics."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        # Test different distance metrics
        distance_tests = [
            ("cosine", models.Distance.COSINE),
            ("euclidean", models.Distance.EUCLID),
            ("dot", models.Distance.DOT),
            ("unknown", models.Distance.COSINE),  # Should default to cosine
        ]

        for distance_str, expected_distance in distance_tests:
            sample_config.qdrant.distance = distance_str
            indexer = VectorIndexer(sample_config)
            indexer._create_collection()

            # Check the distance parameter in the create_collection call
            call_args = mock_client.create_collection.call_args
            vectors_config = call_args[1]["vectors_config"]
            assert vectors_config.distance == expected_distance

            mock_client.reset_mock()

    @patch("zenithmcp.ingestion.indexing.QdrantClient")
    def test_index_chunks_in_batches(
        self, mock_client_class, mock_embedding_vector, sample_config
    ):
        """Test indexing chunks in multiple batches."""
        # Create more chunks than batch size
        chunks = []
        for i in range(25):  # batch_size is 10, so this will create 3 batches
            chunk = Mock()
            chunk.embedding = mock_embedding_vector
            chunk.to_qdrant_point.return_value = {
                "id": f"id_{i}",
                "vector": mock_embedding_vector,
                "payload": {"content": f"content_{i}"},
            }
            chunks.append(chunk)

        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Mock successful upsert
        mock_operation_info = Mock()
        mock_operation_info.status = models.UpdateStatus.COMPLETED
        mock_client.upsert.return_value = mock_operation_info

        mock_client_class.return_value = mock_client

        indexer = VectorIndexer(sample_config)
        indexed_ids = indexer._index_chunks_in_batches(chunks)

        assert len(indexed_ids) == 25

        # Should have called upsert 3 times (3 batches)
        assert mock_client.upsert.call_count == 3
