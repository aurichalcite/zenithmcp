"""Test basic project setup and version."""

import zenithmcp


def test_version() -> None:
    """Test that the version is correctly set."""
    assert zenithmcp.__version__ == "0.1.0"


def test_imports() -> None:
    """Test that main modules can be imported."""
    from zenithmcp import __version__, retrieval, server

    assert __version__ is not None
    assert retrieval is not None
    assert server is not None
