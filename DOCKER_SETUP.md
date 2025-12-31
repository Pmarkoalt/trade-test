# Docker Setup Guide

This guide explains how to set up and use Docker to run tests and the trading system, which avoids environment issues like NumPy segfaults on macOS.

## Prerequisites

### Install Docker Desktop for macOS

1. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download Docker Desktop for Mac (Intel or Apple Silicon)
   - Follow the installation wizard

2. **Verify Installation:**
   ```bash
   docker --version
   docker-compose --version
   ```

3. **Start Docker Desktop:**
   - Open Docker Desktop from Applications
   - Wait for it to start (whale icon in menu bar should be steady)

## Building the Docker Image

Build the Docker image with all dependencies:

```bash
# Build the image
make docker-build

# Or manually
docker-compose build
```

This will:
- Use Python 3.11 (avoids NumPy segfault issues)
- Install all dependencies including dev dependencies (pytest, etc.)
- Copy all test fixtures and code

## Running Tests in Docker

### Unit Tests

```bash
# Using Makefile (recommended)
make docker-test-unit

# Or manually
docker-compose run --rm trading-system pytest tests/ \
    --ignore=tests/integration \
    --ignore=tests/performance \
    --ignore=tests/property \
    -v
```

### Integration Tests

```bash
# Using Makefile
make docker-test-integration

# Or manually
docker-compose run --rm trading-system pytest tests/integration/ -v
```

### Validation Suite

```bash
# Using Makefile
make docker-test-validation

# Or manually
docker-compose run --rm trading-system validate \
    --config /app/tests/fixtures/configs/run_test_config.yaml
```

### All Tests

```bash
# Run unit + integration tests
make docker-test-all
```

## Running Backtests in Docker

```bash
# Run a backtest
docker-compose run --rm trading-system backtest \
    --config /app/configs/run_config.yaml \
    --period train

# Results will be saved to ./results/ on your host machine
```

## Advantages of Docker

1. **Consistent Environment:** Same Python version and dependencies across all machines
2. **No Environment Issues:** Avoids NumPy segfaults, version conflicts, etc.
3. **Isolated:** Doesn't affect your local Python environment
4. **Reproducible:** Same results on macOS, Linux, Windows

## Troubleshooting

### Docker not starting
- Make sure Docker Desktop is running
- Check system requirements (macOS 10.15+)

### Permission errors
- Docker Desktop should handle permissions automatically
- If issues persist, check Docker Desktop settings

### Out of disk space
- Docker images can be large
- Clean up old images: `docker system prune -a`

### Slow performance
- Allocate more resources to Docker Desktop (Settings → Resources)
- Use Docker's built-in caching (subsequent builds are faster)

## Next Steps

Once Docker is set up:
1. Build the image: `make docker-build`
2. Run unit tests: `make docker-test-unit`
3. Run integration tests: `make docker-test-integration`
4. Run validation suite: `make docker-test-validation`

For more information, see:
- [TESTING_GUIDE.md](TESTING_GUIDE.md)
- [README.md](README.md)
- [ENVIRONMENT_ISSUE.md](ENVIRONMENT_ISSUE.md)

