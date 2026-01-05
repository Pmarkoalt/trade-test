# Running Pre-commit in Docker

Pre-commit hooks can be run in Docker to ensure a consistent environment.

## Quick Start

### Option 1: Using Makefile (Recommended)

```bash
# Run pre-commit on all files
make docker-precommit
```

### Option 2: Using Script

```bash
# Run pre-commit on all files
./scripts/run_precommit_docker.sh

# Run on staged files only
./scripts/run_precommit_docker.sh run

# Run specific hook
./scripts/run_precommit_docker.sh run black --all-files
./scripts/run_precommit_docker.sh run flake8 --all-files
```

### Option 3: Using Docker Compose Directly

```bash
# Run pre-commit on all files
docker-compose run --rm --entrypoint bash trading-system \
  -c "pip install pre-commit && pre-commit run --all-files"

# Run on staged files only
docker-compose run --rm --entrypoint bash trading-system \
  -c "pip install pre-commit && pre-commit run"
```

## What Gets Checked

Pre-commit will run:
- ✅ **Black** - Code formatting
- ✅ **isort** - Import sorting
- ✅ **flake8** - Linting
- ✅ **mypy** - Type checking
- ✅ **bandit** - Security checks
- ✅ **File checks** - Trailing whitespace, end-of-file, YAML/JSON/TOML validation
- ✅ **And more** - See `.pre-commit-config.yaml` for full list

## Important Notes

### File Modifications

Some hooks (like Black and isort) will **auto-format** your code. If files are modified:

1. **Review changes** on your host machine:
   ```bash
   git diff
   ```

2. **Stage the changes**:
   ```bash
   git add .
   ```

3. **Commit**:
   ```bash
   git commit -m "Your message"
   ```

### Git Hooks (Not Available in Docker)

Pre-commit git hooks (automatic running on commit) **won't work** in Docker because:
- Docker containers don't have access to your `.git/hooks` directory
- Git hooks require a live git repository

**Solution**: Run pre-commit manually in Docker before committing:
```bash
make docker-precommit
git add .
git commit -m "Your message"
```

Or install hooks locally (outside Docker) for automatic running:
```bash
pip install pre-commit
pre-commit install
```

## Troubleshooting

### "pre-commit: command not found"
- Pre-commit is installed automatically in the Docker command
- If issues persist, rebuild the image: `make docker-build`

### "Hook failed" errors
- Review the error messages
- Some hooks auto-fix issues (like formatting)
- Re-run after fixes: `make docker-precommit`

### "No files to check"
- Make sure you're in the project root
- Check that files are tracked by git or explicitly specified

### Type checking errors
- Some files are excluded from type checking (see `.pre-commit-config.yaml`)
- Review mypy errors and fix type issues
- Or add appropriate type ignores if needed

## Pre-commit Configuration

Configuration is in `.pre-commit-config.yaml`. Key sections:

- **Stage 1**: Basic file checks (fast, no modifications)
- **Stage 2**: Auto-formatters (modify files - black, isort)
- **Stage 3**: Linters (check for issues - flake8)
- **Stage 4**: Type checking (mypy)
- **Stage 5**: Security checks (bandit)

## Workflow Recommendation

1. **Before committing**:
   ```bash
   make docker-precommit
   ```

2. **Review changes**:
   ```bash
   git diff
   git status
   ```

3. **Stage and commit**:
   ```bash
   git add .
   git commit -m "Your message"
   ```

This ensures all code quality checks pass before committing!
