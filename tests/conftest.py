"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_code() -> str:
    """Provide sample Python code for testing."""
    return """
def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two integers.'''
    return a + b


class Calculator:
    '''A simple calculator class.'''

    def __init__(self) -> None:
        self.result = 0

    def add(self, value: int) -> None:
        '''Add a value to the result.'''
        self.result += value
"""


@pytest.fixture
def sample_query() -> str:
    """Provide a sample search query."""
    return "calculate sum of two numbers"
