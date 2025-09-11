"""Source file discovery for the ingestion pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path

import git
from git.exc import GitCommandError, InvalidGitRepositoryError

from zenithmcp.ingestion.config import Config, get_config
from zenithmcp.ingestion.models import ProcessingState

logger = logging.getLogger(__name__)


class SourceFileDiscoverer:
    """Discovers new or modified files since last pipeline run."""

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize the file discoverer.

        Parameters
        ----------
        config : Optional[Config]
            Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.discovery_config = self.config.discovery
        self.chunking_config = self.config.chunking

    def run(self, repository_path: str) -> list[str]:
        """
        Discover files that need to be processed.

        Parameters
        ----------
        repository_path : str
            Path to the repository to scan.

        Returns
        -------
        List[str]
            List of file paths that need to be processed.

        Raises
        ------
        ValueError
            If repository path is invalid.
        """
        repo_path = Path(repository_path).resolve()
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repository_path}")

        logger.info(f"Discovering files in repository: {repo_path}")

        # Try Git-based discovery first
        if self.discovery_config.git.enabled:
            try:
                return self._discover_git_changes(repo_path)
            except (InvalidGitRepositoryError, GitCommandError) as e:
                logger.warning(
                    f"Git discovery failed: {e}. Falling back to filesystem scan."
                )

        # Fall back to filesystem discovery
        return self._discover_filesystem_changes(repo_path)

    def _discover_git_changes(self, repo_path: Path) -> list[str]:
        """
        Discover changed files using Git.

        Parameters
        ----------
        repo_path : Path
            Path to the Git repository.

        Returns
        -------
        List[str]
            List of changed file paths.

        Raises
        ------
        InvalidGitRepositoryError
            If the path is not a Git repository.
        GitCommandError
            If Git commands fail.
        """
        try:
            repo = git.Repo(repo_path)
        except InvalidGitRepositoryError as e:
            raise InvalidGitRepositoryError(f"Not a Git repository: {repo_path}") from e

        # Get current commit hash
        current_commit = repo.head.commit.hexsha
        logger.debug(f"Current commit: {current_commit}")

        # Load processing state
        state_file = repo_path / self.discovery_config.state_file
        last_state = self._load_processing_state(state_file, str(repo_path))

        # If no previous state or same commit, determine what to do
        if last_state is None:
            logger.info("No previous processing state found. Processing all files.")
            changed_files = self._get_all_tracked_files(repo)
        elif last_state.last_commit_hash == current_commit:
            logger.info("No new commits since last processing. No files to process.")
            return []
        else:
            logger.info(
                f"Changes detected since {last_state.last_commit_hash[:8]}. "
                "Finding changed files."
            )
            changed_files = self._get_changed_files_between_commits(
                repo, last_state.last_commit_hash, current_commit
            )

        # Filter files by extension and exclusion patterns
        filtered_files = self._filter_files(changed_files, repo_path)

        logger.info(f"Found {len(filtered_files)} files to process")
        return filtered_files

    def _discover_filesystem_changes(self, repo_path: Path) -> list[str]:
        """
        Discover files using filesystem scanning.

        Parameters
        ----------
        repo_path : Path
            Path to the directory to scan.

        Returns
        -------
        List[str]
            List of file paths to process.
        """
        logger.info("Using filesystem discovery")

        # Get all files recursively
        all_files = []
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(repo_path)
                all_files.append(str(relative_path))

        # Filter files
        filtered_files = self._filter_files(all_files, repo_path)

        logger.info(f"Found {len(filtered_files)} files to process")
        return filtered_files

    def _get_all_tracked_files(self, repo: git.Repo) -> list[str]:
        """
        Get all tracked files in the repository.

        Parameters
        ----------
        repo : git.Repo
            Git repository object.

        Returns
        -------
        List[str]
            List of all tracked file paths.
        """
        try:
            # Get all files tracked by Git
            tracked_files = repo.git.ls_files().splitlines()
            return tracked_files
        except GitCommandError as e:
            logger.error(f"Failed to get tracked files: {e}")
            return []

    def _get_changed_files_between_commits(
        self, repo: git.Repo, old_commit: str, new_commit: str
    ) -> list[str]:
        """
        Get files changed between two commits.

        Parameters
        ----------
        repo : git.Repo
            Git repository object.
        old_commit : str
            Old commit hash.
        new_commit : str
            New commit hash.

        Returns
        -------
        List[str]
            List of changed file paths.
        """
        try:
            # Get diff between commits
            diff = repo.git.diff("--name-only", old_commit, new_commit).splitlines()

            # Filter out deleted files by checking if they exist
            existing_files = []
            repo_path = Path(repo.working_dir)

            for file_path in diff:
                full_path = repo_path / file_path
                if full_path.exists() and full_path.is_file():
                    existing_files.append(file_path)

            return existing_files
        except GitCommandError as e:
            logger.error(f"Failed to get diff between commits: {e}")
            return []

    def _filter_files(self, file_paths: list[str], repo_path: Path) -> list[str]:
        """
        Filter files by extension and exclusion patterns.

        Parameters
        ----------
        file_paths : List[str]
            List of file paths to filter.
        repo_path : Path
            Repository root path.

        Returns
        -------
        List[str]
            Filtered list of file paths.
        """
        filtered_files = []
        valid_extensions = set(self.chunking_config.file_extensions)
        exclude_patterns = self.chunking_config.exclude_patterns

        for file_path in file_paths:
            # Check file extension
            path_obj = Path(file_path)
            if path_obj.suffix not in valid_extensions:
                continue

            # Check exclusion patterns
            if self._should_exclude_file(file_path, exclude_patterns):
                continue

            # Check file size if configured
            if self.config.pipeline.max_file_size_mb > 0:
                full_path = repo_path / file_path
                try:
                    file_size_mb = full_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > self.config.pipeline.max_file_size_mb:
                        logger.debug(
                            f"Skipping large file: {file_path} ({file_size_mb:.1f}MB)"
                        )
                        continue
                except OSError:
                    logger.warning(f"Could not get file size for: {file_path}")
                    continue

            filtered_files.append(file_path)

        return filtered_files

    def _should_exclude_file(self, file_path: str, exclude_patterns: list[str]) -> bool:
        """
        Check if file should be excluded based on patterns.

        Parameters
        ----------
        file_path : str
            File path to check.
        exclude_patterns : List[str]
            List of glob patterns to exclude.

        Returns
        -------
        bool
            True if file should be excluded.
        """
        import fnmatch

        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
            # Also check if any parent directory matches
            path_parts = Path(file_path).parts
            for i in range(len(path_parts)):
                partial_path = "/".join(path_parts[: i + 1])
                if fnmatch.fnmatch(partial_path, pattern):
                    return True

        return False

    def _load_processing_state(
        self, state_file: Path, repository: str
    ) -> ProcessingState | None:
        """
        Load processing state from file.

        Parameters
        ----------
        state_file : Path
            Path to state file.
        repository : str
            Repository identifier.

        Returns
        -------
        Optional[ProcessingState]
            Processing state if found, None otherwise.
        """
        if not state_file.exists():
            return None

        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)

            # Handle both single repository and multi-repository state files
            if isinstance(data, dict) and "repository" in data:
                # Single repository format
                if data["repository"] == repository:
                    return ProcessingState.from_dict(data)
            elif isinstance(data, dict):
                # Multi-repository format
                repo_data = data.get(repository)
                if repo_data:
                    return ProcessingState.from_dict(repo_data)

            return None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Could not load processing state: {e}")
            return None

    def save_processing_state(
        self,
        repository_path: str,
        commit_hash: str,
        total_files: int = 0,
        total_chunks: int = 0,
        failed_files: list[str] | None = None,
    ) -> None:
        """
        Save processing state to file.

        Parameters
        ----------
        repository_path : str
            Path to the repository.
        commit_hash : str
            Current commit hash.
        total_files : int
            Total number of files processed.
        total_chunks : int
            Total number of chunks created.
        failed_files : Optional[List[str]]
            List of files that failed processing.
        """
        repo_path = Path(repository_path).resolve()
        state_file = repo_path / self.discovery_config.state_file

        state = ProcessingState(
            repository=str(repo_path),
            last_commit_hash=commit_hash,
            last_processed_at=datetime.now().isoformat(),
            total_files_processed=total_files,
            total_chunks_created=total_chunks,
            failed_files=failed_files or [],
        )

        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.debug(f"Saved processing state to {state_file}")
        except OSError as e:
            logger.error(f"Failed to save processing state: {e}")

    def get_current_commit_hash(self, repository_path: str) -> str | None:
        """
        Get current commit hash of the repository.

        Parameters
        ----------
        repository_path : str
            Path to the repository.

        Returns
        -------
        Optional[str]
            Current commit hash, or None if not a Git repository.
        """
        try:
            repo = git.Repo(repository_path)
            return repo.head.commit.hexsha
        except (InvalidGitRepositoryError, GitCommandError):
            return None
