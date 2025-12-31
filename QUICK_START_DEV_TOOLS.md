# Quick Start: Development Tools Setup

This is a quick reference guide. For detailed instructions, see [DEVELOPMENT_TOOLS_SETUP.md](DEVELOPMENT_TOOLS_SETUP.md).

## Installation

```bash
# Install all development tools (black, isort, flake8, mypy)
pip install -e ".[dev]"

# OR use the Makefile target
make install-dev
```

## Usage

### Format Code

```bash
# Format code with black and isort
make format

# Check formatting without making changes
make format-check
```

### Lint Code

```bash
# Run flake8 linting
make lint
```

### Type Check

```bash
# Run mypy type checking
make type-check
```

### Run All Checks

```bash
# Run all code quality checks at once
make check-code
```

## Direct Commands (Alternative)

If you prefer not to use Makefile:

```bash
# Format
black trading_system/ && isort trading_system/

# Lint
flake8 trading_system/

# Type check
mypy trading_system/

# Check formatting
black --check trading_system/ && isort --check-only trading_system/
```

## What Was Updated

1. ✅ Added `isort` to dev dependencies in `pyproject.toml`
2. ✅ Added isort configuration to `pyproject.toml` (compatible with black)
3. ✅ Added Makefile targets: `install-dev`, `format`, `format-check`, `lint`, `type-check`, `check-code`
4. ✅ Created `DEVELOPMENT_TOOLS_SETUP.md` with comprehensive guide

## Next Steps

1. Install tools: `make install-dev`
2. Format code: `make format`
3. Run checks: `make check-code`
4. Set up pre-commit hooks (optional): See `DEVELOPMENT_TOOLS_SETUP.md`

