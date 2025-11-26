# TopoVision

KTH AI Society Research project about extracting topographical data using computer vision

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to automatically check and format code before commits.

**Install hooks:**

```bash
uv run pre-commit install
```

**Run hooks manually:**

```bash
uv run pre-commit run --all-files
```

The hooks will automatically run `ruff check` and `ruff format` on all Python files and Jupyter notebooks before each commit.
