# Code Cleanup Completion Guide

This guide provides step-by-step instructions to complete the remaining code cleanup tasks now that the development tools are configured.

## Prerequisites

Before running the cleanup tools, install the development dependencies:

```bash
# Make sure you're in the project directory
cd /Users/pmarko.alt/Desktop/trade-test

# Install dev dependencies (includes black, isort, flake8, mypy)
pip install -e ".[dev]"

# OR if you prefer to install individually:
pip install black isort flake8 mypy types-PyYAML
```

**Note**: If you encounter permission errors, you may need to:
- Use a virtual environment: `python3 -m venv venv && source venv/bin/activate`
- Use `--user` flag: `pip install --user -e ".[dev]"`
- Or use Docker (tools are already configured)

## Step-by-Step Cleanup Process

### Step 1: Format Code with Black and isort

```bash
# Option A: Using Makefile (recommended)
make format

# Option B: Direct commands
black trading_system/ tests/
isort trading_system/ tests/
```

**What this does**:
- Formats all Python files according to Black's style guide (line length: 127)
- Sorts and organizes imports according to isort (compatible with black)

**Expected output**: Files will be reformatted in place. Review the changes with `git diff` before committing.

### Step 2: Check Formatting (Verify Step 1)

```bash
# Option A: Using Makefile
make format-check

# Option B: Direct commands
black --check trading_system/ tests/
isort --check-only trading_system/ tests/
```

**What this does**:
- Checks if code is properly formatted without making changes
- Exits with error code if formatting issues are found

**Expected output**: Should pass if Step 1 completed successfully.

### Step 3: Run Linting with Flake8

```bash
# Option A: Using Makefile
make lint

# Option B: Direct command
flake8 trading_system/ tests/
```

**What this does**:
- Checks code for style violations, potential bugs, and code quality issues
- Uses configuration from `pyproject.toml` (line length: 127, complexity: 10)

**Common issues you might see**:
- Line too long (>127 characters)
- Unused imports
- Undefined variables
- Style violations

**How to fix**:
- Review each error message
- Fix manually or use auto-fix where possible
- For line length: Break long lines or use Black formatting
- Re-run until all issues are resolved

### Step 4: Type Checking with Mypy

```bash
# Option A: Using Makefile
make type-check

# Option B: Direct command
mypy trading_system/
```

**What this does**:
- Performs static type checking
- Identifies missing type hints, type mismatches, and type-related errors

**Expected output**:
- May show warnings (these are OK - the config is lenient for gradual adoption)
- Errors should be addressed

**Common issues**:
- Missing return type annotations
- Missing parameter type hints
- Type mismatches

**How to fix**:
- Add type hints following patterns from the improvements already made
- See `CODE_CLEANUP_SUMMARY.md` for examples of type hint improvements

### Step 5: Run All Checks at Once

```bash
# Run all quality checks
make check-code

# This is equivalent to:
make format-check
make lint
make type-check
```

**Expected result**: All checks should pass (or only show acceptable warnings).

## What Has Already Been Completed

### ✅ Type Hint Improvements (Manual)

The following functions have already been improved with type hints:

1. **`trading_system/data/memory_profiler.py`**:
   - `optimize_dataframe_dtypes()` - Added parameter and return type hints
   - `estimate_dataframe_memory()` - Added parameter type hint

2. **`trading_system/data/calendar.py`**:
   - `get_trading_calendar()` - Added return type annotation

3. **`trading_system/indicators/profiling.py`**:
   - `start_profiling()` - Added return type annotation (`-> None`)
   - `print_stats()` - Added return type annotation (`-> None`)
   - `reset()` - Added return type annotation (`-> None`)

### ✅ Configuration Setup

- ✅ Added `isort` to dev dependencies
- ✅ Added isort configuration to `pyproject.toml`
- ✅ Added Makefile targets for all tools
- ✅ Created comprehensive documentation

### ✅ Dead Code Review

- ✅ TODOs are only in template files (expected, not an issue)
- ✅ No obvious commented-out code blocks found
- ✅ Code structure is clean

## Remaining Tasks

### 1. Code Formatting ⏳

**Status**: Tools configured, needs to be run

**Action**: Run `make format` or `black trading_system/ tests/ && isort trading_system/ tests/`

**Estimated time**: 5-10 minutes (automated)

### 2. Linting Fixes ⏳

**Status**: Tools configured, needs to be run and issues fixed

**Action**:
1. Run `make lint` or `flake8 trading_system/`
2. Review and fix any issues found
3. Re-run until clean

**Estimated time**: 15-30 minutes (depends on number of issues)

### 3. Type Checking Improvements ⏳

**Status**: Tools configured, some improvements made, more needed

**Action**:
1. Run `make type-check` or `mypy trading_system/`
2. Review warnings and errors
3. Add type hints to functions/methods missing them
4. Focus on public APIs first, then internal functions

**Estimated time**: 1-2 hours (for comprehensive type hints)

### 4. Documentation Review ⏳

**Status**: Partially reviewed

**Action**:
1. Review docstrings for consistency
2. Ensure all public functions/classes have docstrings
3. Add type information to docstrings where missing
4. Fix docstring format inconsistencies

**Estimated time**: 1-2 hours

## Quick Reference Commands

```bash
# Install tools
make install-dev

# Format code
make format

# Check formatting
make format-check

# Lint code
make lint

# Type check
make type-check

# Run all checks
make check-code
```

## Using Docker (Alternative)

If you prefer to use Docker:

```bash
# First, ensure Dockerfile installs dev dependencies
# Edit Dockerfile and add: RUN pip install -e ".[dev]"

# Build image
make docker-build

# Format code
docker-compose run --rm trading-system black trading_system/

# Lint code
docker-compose run --rm trading-system flake8 trading_system/

# Type check
docker-compose run --rm trading-system mypy trading_system/
```

## Pre-commit Hooks (Recommended)

Once cleanup is complete, set up pre-commit hooks to maintain code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

See `DEVELOPMENT_TOOLS_SETUP.md` for pre-commit configuration details.

## Verification

After completing all steps:

1. ✅ Code is formatted consistently
2. ✅ No linting errors
3. ✅ Type checking passes (or shows only acceptable warnings)
4. ✅ All tests still pass: `make test-unit`
5. ✅ Code is ready for commit

## Troubleshooting

### Tools not found after installation

```bash
# Check if tools are installed
which black isort flake8 mypy

# If not found, ensure virtual environment is activated
source venv/bin/activate  # or conda activate trading-system

# Reinstall
pip install -e ".[dev]"
```

### Permission errors

```bash
# Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Or use --user flag
pip install --user -e ".[dev]"
```

### Formatting conflicts

If black and isort have conflicts:

1. Run black first: `black trading_system/`
2. Then isort: `isort trading_system/`
3. Or use isort's black profile (already configured in `pyproject.toml`)

### Many linting errors

1. Fix errors in batches (e.g., by file or by error type)
2. Use `--statistics` to see error frequency: `flake8 --statistics trading_system/`
3. Focus on high-frequency errors first
4. Some errors can be auto-fixed, review before applying

## Next Steps After Cleanup

1. Review changes: `git diff`
2. Test the codebase: `make test-unit`
3. Commit changes
4. Set up pre-commit hooks
5. Update CI/CD to run `make check-code`

---

**Estimated Total Time**: 2-4 hours for complete cleanup (when tools can be run)

**Current Status**:
- ✅ Tools configured
- ✅ Some manual improvements completed
- ⏳ Automated formatting and linting pending
- ⏳ Comprehensive type checking pending
