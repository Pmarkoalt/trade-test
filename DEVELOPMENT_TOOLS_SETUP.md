# Development Tools Setup Guide

This guide explains how to install and use code quality tools for the Trading System project.

## Tools Included

The project uses the following code quality tools:

- **black**: Code formatter (line length: 127)
- **isort**: Import sorter
- **flake8**: Linter (max line length: 127)
- **mypy**: Static type checker

All tools are configured in `pyproject.toml`.

## Quick Start

### Option 1: Install Dev Dependencies (Recommended)

The easiest way to install all development tools is to install the dev dependencies:

```bash
# Install the project with dev dependencies
pip install -e ".[dev]"
```

This installs all development tools including black, isort, flake8, mypy, and testing tools.

### Option 2: Install Tools Individually

If you only want specific tools:

```bash
# Code formatting tools
pip install black isort

# Linting
pip install flake8

# Type checking
pip install mypy types-PyYAML
```

### Option 3: Using Docker

If you're using Docker, tools need to be installed in the Docker image. Edit `Dockerfile` to include:

```dockerfile
RUN pip install -e ".[dev]"
```

Then rebuild:

```bash
make docker-build
```

## Using the Tools

### Code Formatting

#### Format code with Black

```bash
# Format all Python files in trading_system/
black trading_system/

# Format specific files
black trading_system/data/memory_profiler.py

# Check formatting without making changes
black --check trading_system/
```

#### Sort imports with isort

```bash
# Sort imports in all Python files
isort trading_system/

# Sort imports in specific files
isort trading_system/data/memory_profiler.py

# Check import order without making changes
isort --check-only trading_system/
```

#### Format and sort together

```bash
# Format with black, then sort imports
black trading_system/ && isort trading_system/
```

### Linting with Flake8

```bash
# Lint all Python files
flake8 trading_system/

# Lint specific files
flake8 trading_system/data/memory_profiler.py

# Show statistics
flake8 --statistics trading_system/
```

**Configuration**: Flake8 settings are in `pyproject.toml`:
- Max line length: 127
- Ignored errors: E203, E266, E501, W503
- Max complexity: 10

### Type Checking with Mypy

```bash
# Type check all Python files
mypy trading_system/

# Type check specific files
mypy trading_system/data/memory_profiler.py

# Show error codes
mypy --show-error-codes trading_system/
```

**Configuration**: Mypy settings are in `pyproject.toml`. The configuration is lenient (not strict) to allow gradual type hint adoption.

### Using Makefile Targets

Add these targets to your `Makefile` for convenience:

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Run all quality checks
make check-code
```

See the next section for Makefile setup.

## Makefile Setup

Add these targets to your `Makefile`:

```makefile
.PHONY: format format-check lint type-check check-code install-dev

# Install dev dependencies
install-dev:
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	pip install -e ".[dev]"
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

# Format code
format:
	@echo "$(YELLOW)Formatting code with black...$(NC)"
	black trading_system/ tests/
	@echo "$(YELLOW)Sorting imports with isort...$(NC)"
	isort trading_system/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

# Check formatting without making changes
format-check:
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	black --check trading_system/ tests/ || (echo "$(RED)✗ Code formatting issues found$(NC)" && exit 1)
	isort --check-only trading_system/ tests/ || (echo "$(RED)✗ Import sorting issues found$(NC)" && exit 1)
	@echo "$(GREEN)✓ Code formatting OK$(NC)"

# Lint code
lint:
	@echo "$(YELLOW)Linting code with flake8...$(NC)"
	flake8 trading_system/ tests/ || (echo "$(RED)✗ Linting issues found$(NC)" && exit 1)
	@echo "$(GREEN)✓ Linting passed$(NC)"

# Type check
type-check:
	@echo "$(YELLOW)Type checking with mypy...$(NC)"
	mypy trading_system/ || (echo "$(YELLOW)⚠ Type checking issues found (warnings are OK)$(NC)")
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# Run all code quality checks
check-code: format-check lint type-check
	@echo "$(GREEN)✓ All code quality checks passed$(NC)"
```

Then use:

```bash
# Install dev tools
make install-dev

# Format code
make format

# Run all checks
make check-code
```

## Pre-commit Hooks (Recommended)

Pre-commit hooks automatically run code quality checks before each commit. This helps maintain code quality.

### Setup Pre-commit

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install git hooks
pre-commit install

# (Optional) Run hooks on all files to verify setup
pre-commit run --all-files
```

### Pre-commit Configuration

Create `.pre-commit-config.yaml` in the project root:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--config=pyproject.toml']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: ['--config-file=pyproject.toml']
        additional_dependencies: [types-PyYAML]
```

Then install:

```bash
pre-commit install
```

## Docker Usage

If using Docker, run tools inside the container:

```bash
# Format code
docker-compose run --rm trading-system black trading_system/

# Lint code
docker-compose run --rm trading-system flake8 trading_system/

# Type check
docker-compose run --rm trading-system mypy trading_system/

# Format and check everything
docker-compose run --rm trading-system sh -c "black trading_system/ && isort trading_system/ && flake8 trading_system/"
```

**Note**: Make sure the Docker image includes dev dependencies. Update `Dockerfile`:

```dockerfile
RUN pip install -e ".[dev]"
```

## IDE Integration

### VS Code

Install extensions:
- **Python** (Microsoft)
- **Black Formatter** (Microsoft)
- **isort** (Microsoft)
- **Pylance** (Microsoft) - includes mypy support

Configure in `.vscode/settings.json`:

```json
{
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "127"],
  "python.sortImports.args": ["--profile", "black"],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--config=pyproject.toml"],
  "python.analysis.typeCheckingMode": "basic"
}
```

### PyCharm

1. **Black**: Go to Settings → Tools → Black → Enable, set line length to 127
2. **isort**: Go to Settings → Tools → isort → Enable
3. **flake8**: Install via Settings → Tools → External Tools
4. **mypy**: Go to Settings → Languages & Frameworks → Python → Type Checking → Enable mypy

## Troubleshooting

### Tools Not Found

If tools are not found after installation:

```bash
# Check if tools are installed
which black isort flake8 mypy

# If not found, ensure you're in the correct virtual environment
source venv/bin/activate  # or conda activate trading-system

# Reinstall dev dependencies
pip install -e ".[dev]"
```

### Import Errors in mypy

If mypy can't find imports:

1. Ensure the package is installed: `pip install -e .`
2. Check `pyproject.toml` mypy configuration
3. Add type stubs: `pip install types-PyYAML`

### Formatting Conflicts

If black and isort conflict:

1. Use `isort` with `--profile black` flag: `isort --profile black trading_system/`
2. Or configure isort in `pyproject.toml`:

```toml
[tool.isort]
profile = "black"
line_length = 127
```

## Workflow Recommendation

1. **Before committing**:
   ```bash
   make format      # Format code
   make lint        # Check linting
   make type-check  # Type check
   ```

2. **Or use pre-commit hooks** (automatic):
   ```bash
   pre-commit install
   ```

3. **In CI/CD**: Run `make check-code` or use pre-commit in CI

## Additional Resources

- [Black documentation](https://black.readthedocs.io/)
- [isort documentation](https://pycqa.github.io/isort/)
- [Flake8 documentation](https://flake8.pycqa.org/)
- [Mypy documentation](https://mypy.readthedocs.io/)
