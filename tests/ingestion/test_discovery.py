"""Tests for source file discovery."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from git.exc import InvalidGitRepositoryError

from zenithmcp.ingestion.discovery import SourceFileDiscoverer
from zenithmcp.ingestion.models import ProcessingState


class TestSourceFileDiscoverer:
    """Tests for SourceFileDiscoverer class."""

    def test_init_with_config(self, sample_config):
        """Test initializing discoverer with config."""
        discoverer = SourceFileDiscoverer(sample_config)

        assert discoverer.config is sample_config
        assert discoverer.discovery_config is sample_config.discovery
        assert discoverer.chunking_config is sample_config.chunking

    def test_init_without_config(self, sample_config):
        """Test initializing discoverer without config."""
        # Set global config
        from zenithmcp.ingestion.config import set_config

        set_config(sample_config)

        discoverer = SourceFileDiscoverer()

        assert discoverer.config is sample_config

    @patch("zenithmcp.ingestion.discovery.git.Repo")
    def test_discover_git_changes_new_repo(self, mock_repo_class, sample_config, fs):
        """Test discovering changes in a new repository."""
        # Setup fake filesystem
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "main.py", contents="print('hello')")
        fs.create_file(repo_path / "utils.js", contents="function test() {}")

        # Mock Git repository
        mock_repo = Mock()
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.git.ls_files.return_value = "main.py\nutils.js\nREADME.md"
        mock_repo_class.return_value = mock_repo

        discoverer = SourceFileDiscoverer(sample_config)

        # Mock _load_processing_state to return None (new repo)
        with patch.object(discoverer, "_load_processing_state", return_value=None):
            files = discoverer._discover_git_changes(repo_path)

        # Should return filtered files (only .py and .js based on config)
        assert "main.py" in files
        assert "utils.js" in files
        assert "README.md" not in files  # Not in file_extensions

    @patch("zenithmcp.ingestion.discovery.git.Repo")
    def test_discover_git_changes_no_changes(self, mock_repo_class, sample_config, fs):
        """Test discovering changes when no new commits."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)

        # Mock Git repository
        mock_repo = Mock()
        current_commit = "abc123"
        mock_repo.head.commit.hexsha = current_commit
        mock_repo_class.return_value = mock_repo

        # Mock processing state with same commit
        mock_state = ProcessingState(
            repository=str(repo_path),
            last_commit_hash=current_commit,
            last_processed_at="2025-01-01T12:00:00",
            total_files_processed=0,
            total_chunks_created=0,
        )

        discoverer = SourceFileDiscoverer(sample_config)

        with patch.object(
            discoverer, "_load_processing_state", return_value=mock_state
        ):
            files = discoverer._discover_git_changes(repo_path)

        # Should return empty list (no changes)
        assert files == []

    @patch("zenithmcp.ingestion.discovery.git.Repo")
    def test_discover_git_changes_with_diff(self, mock_repo_class, sample_config, fs):
        """Test discovering changes with Git diff."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "modified.py", contents="print('modified')")
        fs.create_file(repo_path / "new.js", contents="function new() {}")

        # Mock Git repository
        mock_repo = Mock()
        mock_repo.head.commit.hexsha = "def456"
        mock_repo.working_dir = str(repo_path)
        mock_repo.git.diff.return_value = "modified.py\nnew.js\ndeleted.py"
        mock_repo_class.return_value = mock_repo

        # Mock processing state with different commit
        mock_state = ProcessingState(
            repository=str(repo_path),
            last_commit_hash="abc123",
            last_processed_at="2025-01-01T12:00:00",
            total_files_processed=0,
            total_chunks_created=0,
        )

        discoverer = SourceFileDiscoverer(sample_config)

        with patch.object(
            discoverer, "_load_processing_state", return_value=mock_state
        ):
            files = discoverer._discover_git_changes(repo_path)

        # Should return only existing files
        assert "modified.py" in files
        assert "new.js" in files
        assert "deleted.py" not in files  # File doesn't exist

    def test_discover_filesystem_changes(self, sample_config, fs):
        """Test filesystem-based discovery."""
        # Setup fake filesystem
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "main.py", contents="print('hello')")
        fs.create_file(repo_path / "utils.js", contents="function test() {}")
        fs.create_file(repo_path / "README.md", contents="# Test")
        fs.create_dir(repo_path / "subdir")
        fs.create_file(
            repo_path / "subdir" / "nested.py", contents="def nested(): pass"
        )

        discoverer = SourceFileDiscoverer(sample_config)
        files = discoverer._discover_filesystem_changes(repo_path)

        # Should find files with supported extensions
        assert "main.py" in files
        assert "utils.js" in files
        assert "subdir/nested.py" in files
        assert "README.md" not in files  # Not in file_extensions

    def test_run_git_fallback_to_filesystem(self, sample_config, fs):
        """Test falling back to filesystem when Git fails."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)
        fs.create_file(repo_path / "main.py", contents="print('hello')")

        discoverer = SourceFileDiscoverer(sample_config)

        # Mock Git to raise exception
        with patch(
            "zenithmcp.ingestion.discovery.git.Repo",
            side_effect=InvalidGitRepositoryError,
        ):
            files = discoverer.run(str(repo_path))

        # Should fall back to filesystem discovery
        assert "main.py" in files

    def test_run_invalid_path(self, sample_config):
        """Test running with invalid repository path."""
        discoverer = SourceFileDiscoverer(sample_config)

        with pytest.raises(ValueError, match="Repository path does not exist"):
            discoverer.run("/nonexistent/path")

    def test_filter_files_by_extension(self, sample_config):
        """Test filtering files by extension."""
        discoverer = SourceFileDiscoverer(sample_config)

        files = [
            "main.py",
            "utils.js",
            "config.ts",
            "README.md",
            "binary.exe",
            "data.json",
        ]

        filtered = discoverer._filter_files(files, Path("/test"))

        # Should only include files with configured extensions
        assert "main.py" in filtered
        assert "utils.js" in filtered
        assert "config.ts" in filtered
        assert "README.md" not in filtered
        assert "binary.exe" not in filtered
        assert "data.json" not in filtered

    def test_filter_files_by_exclude_patterns(self, sample_config):
        """Test filtering files by exclusion patterns."""
        discoverer = SourceFileDiscoverer(sample_config)

        files = [
            "src/main.py",
            "node_modules/package/index.js",
            "src/__pycache__/module.py",
            "tests/test_main.py",
        ]

        filtered = discoverer._filter_files(files, Path("/test"))

        # Should exclude files matching patterns
        assert "src/main.py" in filtered
        assert "tests/test_main.py" in filtered
        assert "node_modules/package/index.js" not in filtered
        assert "src/__pycache__/module.py" not in filtered

    def test_should_exclude_file(self, sample_config):
        """Test exclusion pattern matching."""
        discoverer = SourceFileDiscoverer(sample_config)

        # Test various exclusion patterns
        assert discoverer._should_exclude_file(
            "node_modules/package/file.js", ["*/node_modules/*"]
        )
        assert discoverer._should_exclude_file(
            "src/__pycache__/file.py", ["*/__pycache__/*"]
        )
        assert not discoverer._should_exclude_file("src/main.py", ["*/node_modules/*"])
        assert not discoverer._should_exclude_file("tests/test.py", ["*/__pycache__/*"])

    def test_load_processing_state_not_exists(self, sample_config, fs):
        """Test loading processing state when file doesn't exist."""
        state_file = Path("/test/.test_state")
        discoverer = SourceFileDiscoverer(sample_config)

        state = discoverer._load_processing_state(state_file, "/test/repo")

        assert state is None

    def test_load_processing_state_single_repo(self, sample_config, fs):
        """Test loading processing state for single repository."""
        state_file = Path("/test/.test_state")
        state_data = {
            "repository": "/test/repo",
            "last_commit_hash": "abc123",
            "last_processed_at": "2025-01-01T12:00:00",
            "total_files_processed": 5,
            "total_chunks_created": 15,
            "failed_files": [],
        }

        fs.create_file(state_file, contents=json.dumps(state_data))

        discoverer = SourceFileDiscoverer(sample_config)
        state = discoverer._load_processing_state(state_file, "/test/repo")

        assert state is not None
        assert state.repository == "/test/repo"
        assert state.last_commit_hash == "abc123"
        assert state.total_files_processed == 5

    def test_load_processing_state_multi_repo(self, sample_config, fs):
        """Test loading processing state for multiple repositories."""
        state_file = Path("/test/.test_state")
        state_data = {
            "/test/repo1": {
                "repository": "/test/repo1",
                "last_commit_hash": "abc123",
                "last_processed_at": "2025-01-01T12:00:00",
                "total_files_processed": 5,
                "total_chunks_created": 15,
                "failed_files": [],
            },
            "/test/repo2": {
                "repository": "/test/repo2",
                "last_commit_hash": "def456",
                "last_processed_at": "2025-01-01T13:00:00",
                "total_files_processed": 3,
                "total_chunks_created": 8,
                "failed_files": [],
            },
        }

        fs.create_file(state_file, contents=json.dumps(state_data))

        discoverer = SourceFileDiscoverer(sample_config)
        state = discoverer._load_processing_state(state_file, "/test/repo1")

        assert state is not None
        assert state.repository == "/test/repo1"
        assert state.last_commit_hash == "abc123"

    def test_load_processing_state_invalid_json(self, sample_config, fs):
        """Test loading processing state with invalid JSON."""
        state_file = Path("/test/.test_state")
        fs.create_file(state_file, contents="invalid json content")

        discoverer = SourceFileDiscoverer(sample_config)
        state = discoverer._load_processing_state(state_file, "/test/repo")

        assert state is None

    def test_save_processing_state(self, sample_config, fs):
        """Test saving processing state."""
        repo_path = "/test/repo"
        fs.create_dir(repo_path)

        discoverer = SourceFileDiscoverer(sample_config)
        discoverer.save_processing_state(
            repo_path,
            "abc123",
            total_files=5,
            total_chunks=15,
            failed_files=["broken.py"],
        )

        # Check that state file was created
        state_file = Path(repo_path) / sample_config.discovery.state_file
        assert state_file.exists()

        # Load and verify content
        with open(state_file) as f:
            state_data = json.load(f)

        assert state_data["repository"] == str(Path(repo_path).resolve())
        assert state_data["last_commit_hash"] == "abc123"
        assert state_data["total_files_processed"] == 5
        assert state_data["total_chunks_created"] == 15
        assert state_data["failed_files"] == ["broken.py"]

    @patch("zenithmcp.ingestion.discovery.git.Repo")
    def test_get_current_commit_hash(self, mock_repo_class, sample_config):
        """Test getting current commit hash."""
        mock_repo = Mock()
        mock_repo.head.commit.hexsha = "abc123def456"
        mock_repo_class.return_value = mock_repo

        discoverer = SourceFileDiscoverer(sample_config)
        commit_hash = discoverer.get_current_commit_hash("/test/repo")

        assert commit_hash == "abc123def456"

    @patch("zenithmcp.ingestion.discovery.git.Repo")
    def test_get_current_commit_hash_not_git_repo(self, mock_repo_class, sample_config):
        """Test getting commit hash from non-Git repository."""
        mock_repo_class.side_effect = InvalidGitRepositoryError

        discoverer = SourceFileDiscoverer(sample_config)
        commit_hash = discoverer.get_current_commit_hash("/test/repo")

        assert commit_hash is None

    def test_filter_files_large_file(self, sample_config, fs):
        """Test filtering out large files."""
        repo_path = Path("/test/repo")
        fs.create_dir(repo_path)

        # Create a file that exceeds the size limit
        large_content = "x" * (6 * 1024 * 1024)  # 6MB (exceeds 5MB limit)
        fs.create_file(repo_path / "large.py", contents=large_content)
        fs.create_file(repo_path / "small.py", contents="print('small')")

        discoverer = SourceFileDiscoverer(sample_config)
        files = ["large.py", "small.py"]

        filtered = discoverer._filter_files(files, repo_path)

        # Large file should be filtered out
        assert "small.py" in filtered
        assert "large.py" not in filtered
