# Development Environment Setup

This guide will help you set up a development environment for the Trading System.

## Prerequisites

- **Docker and Docker Compose** (Recommended) - See [DOCKER_SETUP.md](../../DOCKER_SETUP.md)
- OR Python 3.9+ (3.11+ recommended) with Git, pip/conda (Alternative)
- Git

## Docker Development (Recommended) ⭐

**Docker is the recommended development setup** as it provides a consistent environment across all systems and avoids environment-specific issues.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd trade-test
```

### 2. Build the Docker Image

```bash
make docker-build
```

### 3. Verify Installation

```bash
# Run unit tests to verify setup
make docker-test-unit
```

### 4. Development Workflow

```bash
# Run tests
make docker-test-unit
make docker-test-integration

# Run specific test file
docker-compose run --rm --entrypoint pytest trading-system tests/test_data_loading.py -v

# Interactive shell for development
docker-compose run --rm trading-system /bin/bash

# Type checking (if tools installed in container)
docker-compose run --rm trading-system mypy trading_system/

# Code formatting (if black installed in container)
docker-compose run --rm trading-system black trading_system/
```

Docker provides a consistent Linux environment that works identically across all machines, eliminating macOS-specific compatibility issues.

For detailed Docker setup, see [DOCKER_SETUP.md](../../DOCKER_SETUP.md).

---

## Native Python Development (Alternative)

If you prefer not to use Docker, you can set up a native Python environment. However, this may lead to environment-specific issues, especially on macOS.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd trade-test
```

### 2. Create Virtual Environment

**Recommended: Use Python 3.11+ for better compatibility (especially on macOS)**

```bash
# Using venv (recommended)
python3.11 -m venv venv  # or python3.10, python3.12
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n trading-system python=3.11
conda activate trading-system
```

**macOS Users**: Python 3.11+ is strongly recommended to avoid NumPy compatibility issues. See [ENVIRONMENT_ISSUE.md](../../ENVIRONMENT_ISSUE.md) for details. **Docker is strongly recommended for macOS.**

### 3. Install Dependencies

**Option A: Automated Setup (Recommended)**

```bash
# Run the automated setup script (detects and fixes common issues)
./scripts/setup_environment.sh

# This will:
# - Check Python version
# - Verify NumPy compatibility (critical on macOS)
# - Install missing dependencies
# - Fix common environment issues
```

**Option B: Manual Installation**

```bash
# Install production dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Or using requirements files
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Note**: The project uses `pyproject.toml` as the source of truth. Using `pip install -e ".[dev]"` is the recommended approach.

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
# Run automated environment check
./scripts/setup_environment.sh --check

# Run quick test
./quick_test.sh

# Or manually verify
python -c "import numpy, pandas, pydantic, yaml; print('Dependencies OK')"
pytest tests/ -v
```

**macOS Users**: If you encounter NumPy segmentation faults, see [ENVIRONMENT_ISSUE.md](../../ENVIRONMENT_ISSUE.md). **Docker is strongly recommended for macOS.**

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

**Using Docker (Recommended):**

```bash
# Run all unit tests
make docker-test-unit

# Run integration tests
make docker-test-integration

# Run specific test file
docker-compose run --rm --entrypoint pytest trading-system tests/test_data_loading.py -v

# Run with coverage
docker-compose run --rm --entrypoint pytest trading-system tests/ \
    --cov=trading_system --cov-report=html
```

**Using Native Installation:**

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

