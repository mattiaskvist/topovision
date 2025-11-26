# Contributing to Topovision

Thank you for your interest in contributing to Topovision!

## Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) installed

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/mattiaskvist/topovision.git
    cd topovision
    ```

2.  **Install dependencies:**

    Use `uv` to sync dependencies and create a virtual environment:

    ```bash
    uv sync
    ```

3.  **Install pre-commit hooks:**

    We use [pre-commit](https://pre-commit.com/) to ensure code quality before each commit.

    ```bash
    uv run pre-commit install
    ```

## Development Workflow

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) to enforce code style and formatting.

-   **Linting:** `ruff check .`
-   **Formatting:** `ruff format .`

We adhere to **Google-style docstrings**. Please ensure all new functions and classes have appropriate docstrings.

### Running Tests

TODO: Add instructions for running tests here once they are established.

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Please ensure your commit messages follow this format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Common types include:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation

## Pull Request Process

1.  Fork the repository and create your branch from `main`.
2.  Make sure your code lints and formats correctly.
3.  Ensure your commit messages are clear and descriptive.
4.  Open a Pull Request against the `main` branch.
