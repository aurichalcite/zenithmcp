"""Pipeline orchestration for the ZenithMCP ingestion system."""

import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from zenithmcp.ingestion.chunking import CodeChunker
from zenithmcp.ingestion.config import load_and_set_config
from zenithmcp.ingestion.discovery import SourceFileDiscoverer
from zenithmcp.ingestion.embedding import EmbeddingGenerator
from zenithmcp.ingestion.indexing import VectorIndexer

app = typer.Typer(
    name="zenithmcp-ingestion",
    help="ZenithMCP Data Ingestion Pipeline",
    add_completion=False,
)

console = Console()


def setup_rich_logging(level: str = "INFO") -> None:
    """Set up rich logging with console output."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def run(
    repository_path: str = typer.Argument(
        ..., help="Path to the repository to process"
    ),
    config_path: str | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run without actually indexing to vector database"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force processing even if no changes detected"
    ),
) -> None:
    """
    Run the complete data ingestion pipeline.

    This command orchestrates the entire pipeline:
    1. Discover changed files using Git
    2. Chunk files into semantic code units
    3. Generate embeddings using GraphCodeBERT
    4. Index chunks into Qdrant vector database
    """
    # Set up logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_rich_logging(log_level)
    logger = logging.getLogger(__name__)

    # Load configuration
    try:
        config = load_and_set_config(config_path)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        raise typer.Exit(1)

    # Validate repository path
    repo_path = Path(repository_path).resolve()
    if not repo_path.exists():
        console.print(f"[red]Repository path does not exist: {repository_path}[/red]")
        raise typer.Exit(1)

    repository_name = repo_path.name

    console.print("[bold blue]ZenithMCP Data Ingestion Pipeline[/bold blue]")
    console.print(f"Repository: {repository_name}")
    console.print(f"Path: {repo_path}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    # Initialize pipeline components
    try:
        discoverer = SourceFileDiscoverer(config)
        chunker = CodeChunker(config)
        embedder = EmbeddingGenerator(config)
        indexer = VectorIndexer(config) if not dry_run else None

        logger.info("Pipeline components initialized")
    except Exception as e:
        console.print(f"[red]Failed to initialize pipeline components: {e}[/red]")
        raise typer.Exit(1)

    pipeline_start_time = time.time()

    try:
        # Stage 1: File Discovery
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            discovery_task = progress.add_task("Discovering files...", total=None)

            try:
                file_paths = discoverer.run(str(repo_path))
                progress.update(
                    discovery_task,
                    description=f"Found {len(file_paths)} files to process",
                )

                if not file_paths and not force:
                    console.print(
                        "[yellow]No files to process. Use --force to process all files.[/yellow]"
                    )
                    return

            except Exception as e:
                console.print(f"[red]File discovery failed: {e}[/red]")
                raise typer.Exit(1)

        if not file_paths:
            console.print("[yellow]No files found to process.[/yellow]")
            return

        # Get current commit hash for state tracking
        commit_hash = discoverer.get_current_commit_hash(str(repo_path)) or "unknown"

        # Stage 2: Code Chunking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            chunking_task = progress.add_task("Chunking code files...", total=None)

            try:
                chunks = chunker.run(
                    file_paths, str(repo_path), repository_name, commit_hash
                )
                progress.update(
                    chunking_task, description=f"Generated {len(chunks)} code chunks"
                )

                if not chunks:
                    console.print("[yellow]No code chunks generated.[/yellow]")
                    return

            except Exception as e:
                console.print(f"[red]Code chunking failed: {e}[/red]")
                raise typer.Exit(1)

        # Stage 3: Embedding Generation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            embedding_task = progress.add_task("Generating embeddings...", total=None)

            try:
                embedded_chunks = embedder.run(chunks)
                embedded_count = sum(
                    1 for chunk in embedded_chunks if chunk.embedding is not None
                )
                progress.update(
                    embedding_task, description=f"Generated {embedded_count} embeddings"
                )

                if embedded_count == 0:
                    console.print("[red]No embeddings generated.[/red]")
                    raise typer.Exit(1)

            except Exception as e:
                console.print(f"[red]Embedding generation failed: {e}[/red]")
                raise typer.Exit(1)

        # Stage 4: Vector Indexing
        if not dry_run:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                indexing_task = progress.add_task(
                    "Indexing to vector database...", total=None
                )

                try:
                    indexing_result = indexer.run(embedded_chunks)
                    if indexing_result.success:
                        progress.update(
                            indexing_task,
                            description=f"Indexed {indexing_result.chunk_count} chunks",
                        )
                    else:
                        console.print(
                            f"[red]Indexing failed: "
                            f"{indexing_result.error_message}[/red]"
                        )
                        raise typer.Exit(1)

                except Exception as e:
                    console.print(f"[red]Vector indexing failed: {e}[/red]")
                    raise typer.Exit(1)
        else:
            console.print("[yellow]Skipping vector indexing (dry run mode)[/yellow]")

        # Update processing state
        if not dry_run:
            try:
                discoverer.save_processing_state(
                    str(repo_path),
                    commit_hash,
                    total_files=len(file_paths),
                    total_chunks=len(embedded_chunks),
                )
                logger.info("Processing state saved")
            except Exception as e:
                logger.warning(f"Failed to save processing state: {e}")

        # Display summary
        pipeline_time = time.time() - pipeline_start_time
        _display_summary(
            file_paths,
            embedded_chunks,
            indexing_result if not dry_run else None,
            pipeline_time,
        )

    finally:
        # Cleanup resources
        try:
            embedder.cleanup()
            if indexer:
                indexer.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def _display_summary(
    file_paths: list,
    chunks: list,
    indexing_result: object | None,
    pipeline_time: float,
) -> None:
    """Display pipeline execution summary."""
    table = Table(title="Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Files Processed", str(len(file_paths)))
    table.add_row("Chunks Generated", str(len(chunks)))

    embedded_count = sum(1 for chunk in chunks if chunk.embedding is not None)
    table.add_row("Embeddings Generated", str(embedded_count))

    if indexing_result:
        table.add_row("Chunks Indexed", str(indexing_result.chunk_count))
        table.add_row("Indexing Success", "✓" if indexing_result.success else "✗")

    table.add_row("Total Time", f"{pipeline_time:.2f}s")

    console.print()
    console.print(table)


@app.command()
def health(
    config_path: str | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Check the health of pipeline components."""
    try:
        config = load_and_set_config(config_path)
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        raise typer.Exit(1)

    console.print("[bold blue]ZenithMCP Pipeline Health Check[/bold blue]")
    console.print()

    # Check Qdrant connection
    try:
        indexer = VectorIndexer(config)
        qdrant_healthy = indexer.health_check()
        console.print(f"Qdrant: {'✓ Healthy' if qdrant_healthy else '✗ Unhealthy'}")

        if qdrant_healthy:
            collection_info = indexer.get_collection_info()
            if collection_info:
                console.print(f"  Collection: {config.qdrant.collection_name}")
                console.print(
                    f"  Points: {collection_info.get('points_count', 'Unknown')}"
                )

        indexer.cleanup()
    except Exception as e:
        console.print(f"Qdrant: ✗ Error - {e}")

    # Check embedding model
    try:
        embedder = EmbeddingGenerator(config)
        console.print("GraphCodeBERT: ✓ Loaded")
        console.print(f"  Model: {config.embedding.model_name}")
        console.print(f"  Device: {embedder.device}")
        console.print(f"  Embedding Dimension: {embedder.get_embedding_dimension()}")
        embedder.cleanup()
    except Exception as e:
        console.print(f"GraphCodeBERT: ✗ Error - {e}")


@app.command()
def info(
    config_path: str | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Display configuration information."""
    try:
        config = load_and_set_config(config_path)
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        raise typer.Exit(1)

    console.print("[bold blue]ZenithMCP Configuration[/bold blue]")
    console.print()

    # Qdrant configuration
    console.print("[bold]Qdrant Configuration:[/bold]")
    console.print(f"  Host: {config.qdrant.host}:{config.qdrant.port}")
    console.print(f"  Collection: {config.qdrant.collection_name}")
    console.print(f"  Vector Size: {config.qdrant.vector_size}")
    console.print(f"  Distance: {config.qdrant.distance}")
    console.print()

    # Embedding configuration
    console.print("[bold]Embedding Configuration:[/bold]")
    console.print(f"  Model: {config.embedding.model_name}")
    console.print(f"  Batch Size: {config.embedding.batch_size}")
    console.print(f"  Max Length: {config.embedding.max_length}")
    console.print(f"  Device: {config.embedding.device}")
    console.print()

    # Chunking configuration
    console.print("[bold]Chunking Configuration:[/bold]")
    console.print(f"  File Extensions: {len(config.chunking.file_extensions)} types")
    console.print(
        f"  Exclude Patterns: {len(config.chunking.exclude_patterns)} patterns"
    )
    console.print(
        f"  Chunk Size: {config.chunking.min_chunk_size}-"
        f"{config.chunking.max_chunk_size} lines"
    )
    console.print(f"  Overlap: {config.chunking.overlap_lines} lines")


if __name__ == "__main__":
    app()
