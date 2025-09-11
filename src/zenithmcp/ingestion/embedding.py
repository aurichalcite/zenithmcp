"""Embedding generation using GraphCodeBERT."""

import logging
import time

import torch
from transformers import AutoModel, AutoTokenizer

from zenithmcp.ingestion.config import Config, get_config
from zenithmcp.ingestion.models import CodeChunk, EmbeddingResult

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates vector embeddings for code chunks using GraphCodeBERT."""

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize the embedding generator.

        Parameters
        ----------
        config : Optional[Config]
            Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.embedding_config = self.config.embedding

        self.model = None
        self.tokenizer = None
        self.device = None

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the GraphCodeBERT model and tokenizer."""
        logger.info(f"Initializing embedding model: {self.embedding_config.model_name}")

        try:
            # Determine device
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_config.model_name,
                cache_dir=self.embedding_config.cache_dir,
                trust_remote_code=True,
            )

            # Load model
            self.model = AutoModel.from_pretrained(
                self.embedding_config.model_name,
                cache_dir=self.embedding_config.cache_dir,
                trust_remote_code=True,
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _get_device(self) -> torch.device:
        """
        Determine the best device to use for inference.

        Returns
        -------
        torch.device
            Device to use for model inference.
        """
        device_config = self.embedding_config.device.lower()

        if device_config == "auto":
            if torch.cuda.is_available() and self.config.performance.use_gpu:
                return torch.device("cuda")
            if torch.backends.mps.is_available() and self.config.performance.use_gpu:
                return torch.device("mps")
            return torch.device("cpu")
        if device_config == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        if device_config == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cpu")

    def run(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """
        Generate embeddings for a list of code chunks.

        Parameters
        ----------
        chunks : List[CodeChunk]
            List of code chunks to embed.

        Returns
        -------
        List[CodeChunk]
            List of chunks with embeddings populated.
        """
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()

        # Process chunks in batches
        batch_size = self.embedding_config.batch_size
        embedded_chunks = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                embedded_batch = self._embed_batch(batch)
                embedded_chunks.extend(embedded_batch)

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed to embed batch {i // batch_size + 1}: {e}")
                # Add chunks without embeddings to maintain order
                embedded_chunks.extend(batch)

        processing_time = time.time() - start_time
        success_count = sum(
            1 for chunk in embedded_chunks if chunk.embedding is not None
        )

        logger.info(
            f"Embedding generation completed: {success_count}/{len(chunks)} successful "
            f"in {processing_time:.2f}s"
        )

        return embedded_chunks

    def _embed_batch(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """
        Generate embeddings for a batch of chunks.

        Parameters
        ----------
        chunks : List[CodeChunk]
            Batch of chunks to embed.

        Returns
        -------
        List[CodeChunk]
            Chunks with embeddings populated.
        """
        if not chunks:
            return chunks

        try:
            # Prepare input texts
            texts = [chunk.content for chunk in chunks]

            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.embedding_config.max_length,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use mean pooling of last hidden states
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )

                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to CPU and list
                embeddings = embeddings.cpu().numpy().tolist()

            # Update chunks with embeddings
            embedded_chunks = []
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                # Create new chunk with embedding
                chunk_dict = chunk.dict()
                chunk_dict["embedding"] = embedding
                embedded_chunk = CodeChunk(**chunk_dict)
                embedded_chunks.append(embedded_chunk)

            return embedded_chunks

        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            return chunks

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to token embeddings.

        Parameters
        ----------
        token_embeddings : torch.Tensor
            Token embeddings from the model.
        attention_mask : torch.Tensor
            Attention mask for the tokens.

        Returns
        -------
        torch.Tensor
            Mean-pooled embeddings.
        """
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Apply mask and sum
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # Count non-masked tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Calculate mean
        return sum_embeddings / sum_mask

    def embed_single(self, chunk: CodeChunk) -> CodeChunk:
        """
        Generate embedding for a single chunk.

        Parameters
        ----------
        chunk : CodeChunk
            Chunk to embed.

        Returns
        -------
        CodeChunk
            Chunk with embedding populated.
        """
        embedded_chunks = self._embed_batch([chunk])
        return embedded_chunks[0]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.

        Returns
        -------
        int
            Embedding dimension.
        """
        if self.model is None:
            return self.config.qdrant.vector_size

        # Get dimension from model config
        return self.model.config.hidden_size

    def validate_embeddings(self, chunks: list[CodeChunk]) -> EmbeddingResult:
        """
        Validate that chunks have valid embeddings.

        Parameters
        ----------
        chunks : List[CodeChunk]
            Chunks to validate.

        Returns
        -------
        EmbeddingResult
            Validation result.
        """
        start_time = time.time()
        valid_chunks = []
        expected_dim = self.get_embedding_dimension()

        for chunk in chunks:
            if chunk.embedding is None:
                continue

            if len(chunk.embedding) != expected_dim:
                logger.warning(
                    f"Invalid embedding dimension for chunk {chunk.id}: "
                    f"expected {expected_dim}, got {len(chunk.embedding)}"
                )
                continue

            if not all(isinstance(x, (int, float)) for x in chunk.embedding):
                logger.warning(f"Invalid embedding values for chunk {chunk.id}")
                continue

            valid_chunks.append(chunk.id)

        success = len(valid_chunks) == len(chunks)
        error_message = (
            None
            if success
            else f"Only {len(valid_chunks)}/{len(chunks)} chunks have valid embeddings"
        )

        return EmbeddingResult(
            chunk_ids=valid_chunks,
            success=success,
            error_message=error_message,
            processing_time=time.time() - start_time,
        )

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Embedding generator cleaned up")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
