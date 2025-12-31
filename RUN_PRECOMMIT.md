# Running Pre-commit

Pre-commit hooks are configured in `.pre-commit-config.yaml`. Here's how to run them:

## Setup (First Time Only)

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Or if using a virtual environment:
python -m pip install pre-commit

# Install the git hooks
pre-commit install
```

## Running Pre-commit

### Option 1: Run on all files
```bash
pre-commit run --all-files
```

### Option 2: Run automatically on git commit (if hooks installed)
```bash
git commit -m "Your message"
# Pre-commit will run automatically
```

### Option 3: Run on staged files only
```bash
pre-commit run
```

## What Pre-commit Does

The configured hooks will:
- ✅ Format code with **black** and **isort**
- ✅ Check for trailing whitespace
- ✅ Validate YAML, JSON, TOML files
- ✅ Check for large files
- ✅ Detect merge conflicts
- ✅ Check for debug statements
- ✅ Fix line endings
- ✅ And more...

## If Hooks Modify Files

If pre-commit modifies files (e.g., auto-formats code):
1. Review the changes: `git diff`
2. Stage the changes: `git add .`
3. Commit again: `git commit -m "Your message"`

## Troubleshooting

**"pre-commit: command not found"**
- Install pre-commit: `pip install pre-commit`
- Or use: `python -m pre-commit run --all-files`

**"No module named pre_commit"**
- Install: `pip install pre-commit`
- Or if using venv: `source venv/bin/activate && pip install pre-commit`

**Hooks failing**
- Check the error messages
- Some hooks auto-fix issues (like formatting)
- Re-run after fixes: `pre-commit run --all-files`

