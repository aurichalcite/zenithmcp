"""Vector indexing using Qdrant."""

import logging
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from zenithmcp.ingestion.config import Config, get_config
from zenithmcp.ingestion.models import CodeChunk, IndexingResult

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Indexes code chunks into Qdrant vector database."""

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize the vector indexer.

        Parameters
        ----------
        config : Optional[Config]
            Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.qdrant_config = self.config.qdrant
        self.indexing_config = self.config.indexing

        self.client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Qdrant client."""
        logger.info(
            f"Initializing Qdrant client: "
            f"{self.qdrant_config.host}:{self.qdrant_config.port}"
        )

        try:
            self.client = QdrantClient(
                host=self.qdrant_config.host,
                port=self.qdrant_config.port,
                timeout=self.qdrant_config.timeout,
                prefer_grpc=self.qdrant_config.prefer_grpc,
            )

            # Test connection
            collections = self.client.get_collections()
            logger.info(
                f"Connected to Qdrant. Found {len(collections.collections)} collections."
            )

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def run(self, chunks: list[CodeChunk]) -> IndexingResult:
        """
        Index a list of code chunks into Qdrant.

        Parameters
        ----------
        chunks : List[CodeChunk]
            List of code chunks to index.

        Returns
        -------
        IndexingResult
            Result of the indexing operation.
        """
        if not chunks:
            return IndexingResult(success=True, processing_time=0.0)

        logger.info(f"Indexing {len(chunks)} chunks to Qdrant")
        start_time = time.time()

        try:
            # Ensure collection exists
            self._ensure_collection_exists()

            # Filter chunks with embeddings
            valid_chunks = [chunk for chunk in chunks if chunk.embedding is not None]
            if len(valid_chunks) != len(chunks):
                logger.warning(
                    f"Only {len(valid_chunks)}/{len(chunks)} chunks have embeddings. "
                    "Skipping chunks without embeddings."
                )

            if not valid_chunks:
                return IndexingResult(
                    success=False,
                    error_message="No chunks with embeddings to index",
                    processing_time=time.time() - start_time,
                )

            # Index chunks in batches
            indexed_ids = self._index_chunks_in_batches(valid_chunks)

            processing_time = time.time() - start_time
            success = len(indexed_ids) == len(valid_chunks)

            result = IndexingResult(
                chunk_ids=indexed_ids,
                success=success,
                error_message=None
                if success
                else f"Only {len(indexed_ids)}/{len(valid_chunks)} chunks indexed",
                processing_time=processing_time,
            )

            logger.info(
                f"Indexing completed: {len(indexed_ids)} chunks in {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            error_msg = f"Indexing failed: {e}"
            logger.error(error_msg)
            return IndexingResult(
                success=False,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _ensure_collection_exists(self) -> None:
        """Ensure the target collection exists with correct configuration."""
        collection_name = self.qdrant_config.collection_name

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_collections = {col.name for col in collections.collections}

            if collection_name in existing_collections:
                # Verify collection configuration
                collection_info = self.client.get_collection(collection_name)

                # Check vector size
                vector_size = collection_info.config.params.vectors.size
                if vector_size != self.qdrant_config.vector_size:
                    if self.indexing_config.recreate_on_schema_change:
                        logger.warning(
                            f"Vector size mismatch: expected {self.qdrant_config.vector_size}, "
                            f"got {vector_size}. Recreating collection."
                        )
                        self.client.delete_collection(collection_name)
                        self._create_collection()
                    else:
                        raise ValueError(
                            f"Vector size mismatch: expected {self.qdrant_config.vector_size}, "
                            f"got {vector_size}. Set recreate_on_schema_change=true to fix."
                        )
                else:
                    logger.debug(
                        f"Collection {collection_name} exists with correct configuration"
                    )
            elif self.indexing_config.create_collection:
                logger.info(f"Creating collection: {collection_name}")
                self._create_collection()
            else:
                raise ValueError(f"Collection {collection_name} does not exist")

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def _create_collection(self) -> None:
        """Create a new collection with the configured parameters."""
        collection_name = self.qdrant_config.collection_name

        # Map distance string to Qdrant distance enum
        distance_map = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }

        distance = distance_map.get(
            self.qdrant_config.distance.lower(), models.Distance.COSINE
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.qdrant_config.vector_size,
                distance=distance,
            ),
        )

        logger.info(f"Created collection {collection_name}")

    def _index_chunks_in_batches(self, chunks: list[CodeChunk]) -> list[str]:
        """
        Index chunks in batches for efficiency.

        Parameters
        ----------
        chunks : List[CodeChunk]
            Chunks to index.

        Returns
        -------
        List[str]
            List of successfully indexed chunk IDs.
        """
        indexed_ids = []
        batch_size = self.indexing_config.batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            try:
                batch_ids = self._index_batch(batch)
                indexed_ids.extend(batch_ids)

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Indexed {i + len(batch)}/{len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed to index batch {i // batch_size + 1}: {e}")
                # Continue with next batch
                continue

        return indexed_ids

    def _index_batch(self, chunks: list[CodeChunk]) -> list[str]:
        """
        Index a batch of chunks.

        Parameters
        ----------
        chunks : List[CodeChunk]
            Batch of chunks to index.

        Returns
        -------
        List[str]
            List of successfully indexed chunk IDs.
        """
        if not chunks:
            return []

        # Convert chunks to Qdrant points
        points = []
        for chunk in chunks:
            try:
                point_data = chunk.to_qdrant_point()
                point = models.PointStruct(
                    id=point_data["id"],
                    vector=point_data["vector"],
                    payload=point_data["payload"],
                )
                points.append(point)
            except Exception as e:
                logger.error(f"Failed to convert chunk {chunk.id} to Qdrant point: {e}")
                continue

        if not points:
            return []

        # Upsert points with retry logic
        max_retries = self.indexing_config.max_retries
        retry_delay = self.indexing_config.retry_delay

        for attempt in range(max_retries + 1):
            try:
                operation_info = self.client.upsert(
                    collection_name=self.qdrant_config.collection_name,
                    points=points,
                )

                # Check if operation was successful
                if operation_info.status == models.UpdateStatus.COMPLETED:
                    return [point.id for point in points]
                logger.warning(f"Upsert operation status: {operation_info.status}")

            except (ResponseHandlingException, UnexpectedResponse) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Upsert attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Upsert failed after {max_retries + 1} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error during upsert: {e}")
                raise

        return []

    def delete_chunks(self, chunk_ids: list[str]) -> bool:
        """
        Delete chunks from the vector database.

        Parameters
        ----------
        chunk_ids : List[str]
            List of chunk IDs to delete.

        Returns
        -------
        bool
            True if deletion was successful.
        """
        if not chunk_ids:
            return True

        try:
            operation_info = self.client.delete(
                collection_name=self.qdrant_config.collection_name,
                points_selector=models.PointIdsList(
                    points=chunk_ids,
                ),
            )

            success = operation_info.status == models.UpdateStatus.COMPLETED
            if success:
                logger.info(f"Deleted {len(chunk_ids)} chunks")
            else:
                logger.warning(f"Delete operation status: {operation_info.status}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns
        -------
        dict
            Collection information.
        """
        try:
            collection_info = self.client.get_collection(
                self.qdrant_config.collection_name
            )
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy and accessible.

        Returns
        -------
        bool
            True if Qdrant is healthy.
        """
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up client resources."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

        logger.debug("Vector indexer cleaned up")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
