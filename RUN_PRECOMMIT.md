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
- ✅ **autoflake**: Remove unused imports (F401) and unused variables (F841)
  - Note: Cannot fix F811 (redefinition) errors - these need manual fixes
  - Runs on `trading_system/` and `tests/` directories
- ✅ **isort**: Sort imports
- ✅ **black**: Format code
- ✅ Check for trailing whitespace
- ✅ Validate YAML, JSON, TOML files
- ✅ Check for large files
- ✅ Detect merge conflicts
- ✅ Check for debug statements
- ✅ Fix line endings
- ✅ **flake8**: Lint code (runs after auto-fixers)

## If Hooks Modify Files

**ⓘ This is normal and expected behavior!**

Auto-fixing hooks (autoflake, isort, black) modify files and exit with code 1 to indicate changes were made. The commit appears to "fail", but this is intentional - it ensures you review the changes before committing.

### Recommended: Use the Commit Wrapper Script (One-Step Process)

We provide a wrapper script that automatically handles re-running hooks after fixes:

```bash
# Option 1: Direct usage
./scripts/git-commit-with-hooks.sh -m "your message"

# Option 2: Use git alias (recommended - set up once)
git config alias.commit-hooks '!./scripts/git-commit-with-hooks.sh'
git commit-hooks -m "your message"
```

**What it does:**
1. Runs pre-commit hooks
2. If hooks auto-fix files, automatically stages them
3. Re-runs hooks to verify everything passes
4. Commits if all hooks pass ✅

See `scripts/README.md` for more details.

### Manual Workflow (Two-Step Process)

If you prefer to review changes manually before committing:

```bash
# Step 1: Try to commit
git commit -m "your message"
# → Hooks run, auto-fixers modify files, exit with code 1
# → Commit is blocked (this is intentional!)

# Step 2: Review, stage, and commit again
git status          # See which files were modified
git diff            # Review what was auto-fixed
git add .           # Stage the auto-fixed changes
git commit -m "your message"
# → Hooks run again, no changes needed, exit with code 0
# → Commit succeeds! ✅
```

### Why Review Changes?

- **Safety**: You review what was changed before committing
- **Transparency**: You see exactly what auto-fixers did
- **Control**: You can discard changes if needed (`git checkout -- <file>`)

### Quick Check (Without Committing)

To see what would be auto-fixed without committing:

```bash
# Run pre-commit to see what would be fixed
pre-commit run --all-files

# Check what changed
git status
git diff

# If changes look good, stage and commit normally
git add .
git commit -m "Your message"
```

## Troubleshooting

**"pre-commit: command not found"**
- Install pre-commit: `pip install pre-commit`
- Or use: `python -m pre-commit run --all-files`

**"No module named pre_commit"**
- Install: `pip install pre-commit`
- Or if using venv: `source venv/bin/activate && pip install pre-commit`

**Hooks failing**
- **If auto-fixing hooks (autoflake, isort, black) "fail"**: This is normal! They modify files and exit with code 1. Just `git add .` and commit again.
- **For other hooks**: Check the error messages and fix the issues manually
- **To verify fixes**: Run `pre-commit run --all-files` again after staging changes
