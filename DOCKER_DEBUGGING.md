# Docker Debugging Guide for Integration Tests

This guide explains how to use Docker to debug integration test failures, bypassing local environment issues (like NumPy segfaults on macOS).

## Quick Start

### 1. Build the Docker Image

```bash
make docker-build
# OR
docker-compose build
```

### 2. Run Integration Tests

```bash
# Basic run
make docker-test-integration

# With verbose output
make docker-test-integration-verbose

# Capture output to file
make docker-test-integration-capture
```

## Interactive Debugging

### Open Interactive Shell

Get a shell inside the Docker container to run commands manually:

```bash
make docker-debug-shell
# OR
./scripts/debug_integration_tests.sh
# Then select option 7
```

Once inside, you can:
- Run pytest manually: `pytest tests/integration/ -v -k test_name`
- Inspect files: `ls -la /app/tests/integration/`
- Check Python environment: `python -c "import numpy; print(numpy.__version__)"`
- Run Python interactively: `python`

### Use the Interactive Menu

```bash
./scripts/debug_integration_tests.sh
```

This provides a menu with options to:
1. Build Docker image
2. Run all integration tests
3. Run all integration tests (verbose)
4. Run all integration tests (with output capture)
5. Run specific test (interactive)
6. Run specific test (with pdb debugger)
7. Open interactive shell
8. Re-run only failed tests
9. List available integration tests

## Debugging Specific Tests

### Run a Single Test

```bash
# Using Makefile
make docker-debug-test TEST=test_integration_workflow

# Using script (interactive)
./scripts/debug_integration_tests.sh
# Select option 5, then enter test name

# Direct docker-compose
docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
    -vvv --tb=long --showlocals -k test_integration_workflow
```

### Run with Python Debugger (pdb)

This will pause execution at failures or breakpoints:

```bash
make docker-debug-integration ARGS="-k test_name"
# OR
./scripts/debug_integration_tests.sh
# Select option 6, then enter test name
```

### Run Only Failed Tests

After a test run, re-run only the tests that failed:

```bash
make docker-test-integration-last-failed
```

## Advanced Debugging

### Very Verbose Output

```bash
make docker-test-integration-verbose
```

This includes:
- `-vvv`: Maximum verbosity
- `--tb=long`: Long traceback format
- `--showlocals`: Show local variables in tracebacks
- `--log-cli-level=DEBUG`: Debug-level logging

### Capture Output to File

```bash
make docker-test-integration-capture
```

Output is saved to `results/integration_test_output_YYYYMMDD_HHMMSS.txt`

### Custom pytest Arguments

Pass additional pytest arguments:

```bash
make docker-test-integration ARGS="-k test_name --maxfail=1"
```

## Common Debugging Scenarios

### Scenario 1: Test Fails with Import Error

1. Open shell: `make docker-debug-shell`
2. Check Python path: `echo $PYTHONPATH`
3. Try importing: `python -c "from trading_system.data import load_ohlcv_data"`
4. Check installed packages: `pip list | grep trading`

### Scenario 2: Test Fails with Data Loading Error

1. Open shell: `make docker-debug-shell`
2. Check fixtures: `ls -la /app/tests/fixtures/`
3. Check data files: `ls -la /app/data/`
4. Try loading manually: `python -c "from trading_system.data import load_ohlcv_data; print(load_ohlcv_data('/app/tests/fixtures', ['AAPL']))"`

### Scenario 3: Test Fails with Assertion Error

1. Run with verbose output: `make docker-test-integration-verbose ARGS="-k test_name"`
2. Check the long traceback for variable values
3. Use pdb to inspect: `make docker-debug-test TEST=test_name`

### Scenario 4: Test Hangs or Times Out

1. Run with output capture to see where it hangs
2. Open shell and run test manually with timeout
3. Check for infinite loops or blocking operations

## Available Makefile Commands

| Command | Description |
|---------|-------------|
| `make docker-build` | Build Docker image |
| `make docker-test-integration` | Run integration tests |
| `make docker-test-integration-verbose` | Run with very verbose output |
| `make docker-test-integration-capture` | Run and capture output to file |
| `make docker-test-integration-last-failed` | Re-run only failed tests |
| `make docker-debug-shell` | Open interactive shell |
| `make docker-debug-integration` | Run with pdb debugger |
| `make docker-debug-test TEST=name` | Run specific test with debugging |

## Troubleshooting

### Docker Image Won't Build

```bash
# Clean build (no cache)
docker-compose build --no-cache

# Check Docker is running
docker ps
```

### Container Exits Immediately

```bash
# Run with interactive terminal
docker-compose run --rm trading-system /bin/bash
```

### Tests Can't Find Fixtures

Check that volumes are mounted correctly in `docker-compose.yml`:
- `./tests/fixtures:/app/tests/fixtures:ro`
- `./data:/app/data:ro`

### Permission Issues

If you see permission errors:
```bash
# Check file permissions
ls -la tests/fixtures/

# Fix if needed (on host)
chmod -R 644 tests/fixtures/*
```

## Tips

1. **Always build first**: Run `make docker-build` after code changes
2. **Use verbose mode**: Start with `docker-test-integration-verbose` to see detailed output
3. **Capture output**: Use `docker-test-integration-capture` for long test runs
4. **Interactive shell**: Use `docker-debug-shell` to explore and test manually
5. **Incremental debugging**: Run one test at a time with `-k` flag

## Example Workflow

```bash
# 1. Build image
make docker-build

# 2. List available tests
docker-compose run --rm --entrypoint pytest trading-system tests/integration/ --collect-only -q

# 3. Run all tests to see failures
make docker-test-integration

# 4. Run specific failing test with verbose output
make docker-test-integration-verbose ARGS="-k test_integration_workflow"

# 5. If still unclear, use pdb debugger
make docker-debug-test TEST=test_integration_workflow

# 6. Or open shell to investigate
make docker-debug-shell
```

## Next Steps

After identifying the issue:
1. Fix the code
2. Rebuild Docker image: `make docker-build`
3. Re-run the test: `make docker-debug-test TEST=test_name`
4. Once fixed, run all tests: `make docker-test-integration`
