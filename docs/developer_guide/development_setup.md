# Development Environment Setup

This guide will help you set up a development environment for the Trading System.

## Prerequisites

- Python 3.9+ (3.11+ recommended)
- Git
- pip or conda
- (Optional) Docker and Docker Compose

## Setup Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd trade-test
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n trading-system python=3.11
conda activate trading-system
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Set Up Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit. This helps catch issues early and maintain code quality.

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install the git hook scripts
pre-commit install

# (Optional) Run hooks on all files to verify setup
pre-commit run --all-files
```

**What the hooks do**:
- Remove trailing whitespace and fix end-of-file issues
- Check YAML, JSON, and TOML files for syntax errors
- Format code with Black
- Run flake8 linting
- Run mypy type checking
- Check for merge conflicts and debug statements

**Note**: The first run may take a few minutes as it downloads hook dependencies. Subsequent runs are faster.

### 5. Verify Installation

```bash
# Run quick test
./quick_test.sh

# Or manually
pytest tests/ -v
```

## Development Tools

### Type Checking

The project uses type hints. To check types:

```bash
# Using mypy (if installed)
mypy trading_system/

# Using pyright (if installed)
pyright trading_system/
```

### Code Formatting

```bash
# Format code with black (if installed)
black trading_system/ tests/

# Check formatting
black --check trading_system/ tests/
```

### Linting

```bash
# Run flake8 (if installed)
flake8 trading_system/ tests/
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. You can also run them manually:

```bash
# Run all hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run black --all-files
pre-commit run flake8 --all-files
pre-commit run mypy --all-files
```

**Troubleshooting**:
- If hooks fail, they will show you what needs to be fixed
- Some hooks (like Black) will auto-fix issues - just stage the changes and commit again
- To skip hooks for a commit (not recommended): `git commit --no-verify`

## Running Tests

See [Testing Guide](../../TESTING_GUIDE.md) for comprehensive testing instructions.

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_loading.py -v

# Run with coverage
pytest tests/ --cov=trading_system --cov-report=html
```

## Building Documentation

### Sphinx API Documentation

```bash
cd docs/
make html
```

Generated documentation will be in `docs/_build/html/`.

## Docker Development

You can also develop using Docker:

```bash
# Build image
docker-compose build

# Run tests
docker-compose run --rm trading-system pytest tests/ -v

# Interactive shell
docker-compose run --rm trading-system /bin/bash
```

## IDE Setup

### VS Code

Recommended extensions:
- Python
- Pylance
- Python Test Explorer
- Black Formatter

### PyCharm

- Configure Python interpreter to use virtual environment
- Set up pytest as test runner
- Enable type checking

## Next Steps

- Read [Code Style & Standards](code_style.md)
- Review [Architecture Overview](../../agent-files/01_ARCHITECTURE_OVERVIEW.md)
- Check [Testing Guide](../../TESTING_GUIDE.md)

