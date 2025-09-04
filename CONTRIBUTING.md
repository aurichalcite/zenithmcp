# Contributing to ZenithMCP

Thank you for your interest in contributing to ZenithMCP! We're excited to have you join our community of developers working to improve AI-assisted software development.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

In summary:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing opinions and experiences

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- GitHub account
- Basic understanding of Python and async programming

### First-Time Contributors

We love helping first-time contributors! Look for issues labeled:
- `good first issue` - Simple tasks perfect for newcomers
- `help wanted` - Tasks where we need community help
- `documentation` - Documentation improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/zenithmcp.git
cd zenithmcp
git remote add upstream https://github.com/zenithmcp/zenithmcp.git
```

### 2. Install uv Package Manager

We use `uv` for fast, reliable dependency management:

```bash
pip install uv
```

### 3. Set Up Development Environment

```bash
# Install all dependencies including dev tools
uv sync --extra dev

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Verify Setup

```bash
# Run tests to ensure everything is working
uv run pytest

# Check linting
uv run ruff check .

# Check formatting
uv run ruff format . --check
```

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check the [issue tracker](https://github.com/zenithmcp/zenithmcp/issues) to avoid duplicates
2. Try to reproduce the issue with the latest version
3. Collect relevant information (Python version, OS, error messages)

When reporting bugs, include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- System information
- Relevant logs or screenshots

### Suggesting Features

We welcome feature suggestions! Please:
1. Check if the feature has already been suggested
2. Clearly describe the use case
3. Explain how it benefits users
4. Consider implementation complexity

### Contributing Code

1. **Find or Create an Issue**: Start by finding an existing issue or creating a new one
2. **Discuss**: Comment on the issue to discuss your approach
3. **Branch**: Create a feature branch from `main`
4. **Code**: Write your code following our standards
5. **Test**: Add tests for your changes
6. **Document**: Update documentation if needed
7. **Submit**: Create a pull request

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

- Write clear, self-documenting code
- Follow the existing code style
- Add or update tests as needed
- Update documentation

### 3. Commit Your Changes

We follow conventional commit messages:

```bash
git commit -m "feat: add new retrieval algorithm"
git commit -m "fix: resolve memory leak in caching service"
git commit -m "docs: update API documentation"
git commit -m "test: add tests for reranking module"
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

### 4. Run Quality Checks

Before submitting, ensure all checks pass:

```bash
# Lint your code
uv run ruff check .

# Format your code
uv run ruff format .

# Type check
uv run mypy src

# Run tests with coverage
uv run pytest --cov

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title describing the change
- Link to related issue(s)
- Description of what changed and why
- Screenshots if applicable
- Checklist of completed tasks

### 6. Review Process

- Maintainers will review your PR within 2-3 business days
- Address feedback constructively
- Make requested changes in new commits
- Once approved, your PR will be merged

## Coding Standards

### Python Style

We use Ruff for linting and formatting, configured in `pyproject.toml`:

- Line length: 88 characters
- Docstring style: NumPy format
- Type hints: Required for all public functions
- Import sorting: Automatic with Ruff

### Code Quality Principles

1. **Readability**: Code should be self-explanatory
2. **Simplicity**: Prefer simple solutions over complex ones
3. **Testability**: Write testable, modular code
4. **Performance**: Consider performance implications
5. **Security**: Never commit secrets or credentials

### Example Code Style

```python
"""Module description."""

from typing import Any, Optional

import numpy as np
from fastapi import FastAPI


class CodeRetriever:
    """Retrieve relevant code snippets.

    Parameters
    ----------
    model_name : str
        Name of the embedding model.
    cache_size : int, optional
        Size of the cache in MB, by default 100.

    Attributes
    ----------
    embeddings : np.ndarray
        Cached embeddings array.
    """

    def __init__(self, model_name: str, cache_size: int = 100) -> None:
        """Initialize the retriever."""
        self.model_name = model_name
        self.cache_size = cache_size
        self.embeddings: Optional[np.ndarray] = None

    async def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search for relevant code snippets.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict[str, Any]]
            List of relevant code snippets.
        """
        # Implementation here
        return []
```

## Testing Guidelines

### Test Structure

Tests should follow the AAA pattern:
- **Arrange**: Set up test data
- **Act**: Execute the code
- **Assert**: Verify results

### Writing Tests

```python
import pytest
from zenithmcp.rag_core import retrieval


class TestCandidateRetriever:
    """Test the CandidateRetriever class."""

    @pytest.fixture
    def retriever(self):
        """Create a retriever instance."""
        return retrieval.CandidateRetriever()

    async def test_search_returns_list(self, retriever):
        """Test that search returns a list."""
        # Arrange
        query = "calculate sum"

        # Act
        results = await retriever.search(query)

        # Assert
        assert isinstance(results, list)
```

### Test Coverage

- Maintain minimum 80% code coverage
- Focus on testing critical paths
- Write both unit and integration tests
- Use mocks for external dependencies

## Documentation

### Docstrings

All public modules, classes, and functions must have docstrings:

```python
def process_code(code: str, language: str = "python") -> dict[str, Any]:
    """Process source code for indexing.

    Takes raw source code and prepares it for vector embedding
    and storage in the database.

    Parameters
    ----------
    code : str
        The source code to process.
    language : str, optional
        Programming language, by default "python".

    Returns
    -------
    dict[str, Any]
        Processed code metadata and chunks.

    Raises
    ------
    ValueError
        If the language is not supported.

    Examples
    --------
    >>> result = process_code("def hello(): pass")
    >>> print(result["chunks"])
    [{"content": "def hello(): pass", "type": "function"}]
    """
```

### Documentation Updates

When adding features:
1. Update relevant docstrings
2. Add to API documentation if public
3. Update README if it's a major feature
4. Add usage examples

## Community

### Getting Help

- **Discord**: Join our [Discord server](https://discord.gg/zenithmcp)
- **Discussions**: Use [GitHub Discussions](https://github.com/zenithmcp/zenithmcp/discussions) for questions
- **Issues**: Report bugs via [GitHub Issues](https://github.com/zenithmcp/zenithmcp/issues)

### Recognition

We value all contributions! Contributors are:
- Listed in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- Mentioned in release notes
- Given credit in documentation

### Becoming a Maintainer

Active contributors may be invited to become maintainers. We look for:
- Consistent, high-quality contributions
- Helpful participation in discussions
- Understanding of the project architecture
- Commitment to the project's vision

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Questions?

Feel free to reach out:
- Open a [Discussion](https://github.com/zenithmcp/zenithmcp/discussions)
- Message us on [Discord](https://discord.gg/zenithmcp)
- Email: contributors@zenithmcp.org

Thank you for helping make ZenithMCP better!
