# Quick Guide: Run Pre-commit in Docker

## ✅ Fastest Way

```bash
make docker-precommit
```

That's it! This will:
1. Install pre-commit in the Docker container
2. Run all pre-commit hooks on all files
3. Show you any issues that need fixing

## Alternative Commands

```bash
# Using the script
./scripts/run_precommit_docker.sh

# Using docker-compose directly
docker-compose run --rm --entrypoint bash trading-system \
  -c "pip install pre-commit && pre-commit run --all-files"
```

## What Happens

Pre-commit will check:
- ✅ Code formatting (Black, isort)
- ✅ Linting (flake8)
- ✅ Type checking (mypy)
- ✅ Security (bandit)
- ✅ File validation (YAML, JSON, TOML)
- ✅ And more...

## If Files Are Modified

Some hooks (like Black) will **auto-format** your code. If files are modified:

1. **Review changes**:
   ```bash
   git diff
   ```

2. **Stage changes**:
   ```bash
   git add .
   ```

3. **Commit**:
   ```bash
   git commit -m "Your message"
   ```

## Full Documentation

See [DOCKER_PRECOMMIT.md](DOCKER_PRECOMMIT.md) for detailed documentation.
