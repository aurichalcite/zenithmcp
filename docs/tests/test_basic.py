"""Basic tests for ZenithMCP setup."""

import zenithmcp


def test_version() -> None:
    """Test that version is set."""
    assert zenithmcp.__version__ == "0.1.0"


def test_imports() -> None:
    """Test that main modules can be imported."""
    from zenithmcp import retrieval, server

    assert retrieval is not None
    assert server is not None
