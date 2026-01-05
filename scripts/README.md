# Scripts

This directory contains utility scripts for the trading system.

## Git Commit Wrapper

### `git-commit-with-hooks.sh`

A wrapper script for `git commit` that automatically re-runs pre-commit hooks after auto-fixes.

#### Features

- Automatically runs pre-commit hooks before committing
- If hooks make changes (autoflake, isort, black), automatically stages them
- Re-runs hooks to verify everything passes
- Only commits if all hooks pass

#### Usage

**Option 1: Direct usage**
```bash
./scripts/git-commit-with-hooks.sh -m "Your commit message"
```

**Option 2: Git alias (recommended)**
```bash
# Create a global alias
git config --global alias.commit-hooks '!./scripts/git-commit-with-hooks.sh'

# Or create a local alias (project-specific)
git config alias.commit-hooks '!./scripts/git-commit-with-hooks.sh'

# Use it like normal git commit
git commit-hooks -m "Your commit message"
```

**Option 3: Override git commit (advanced)**
```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
alias git='function _git() { if [[ "$1" == "commit" ]] && [[ -f "./scripts/git-commit-with-hooks.sh" ]]; then shift; ./scripts/git-commit-with-hooks.sh "$@"; else command git "$@"; fi; }; _git'

# Then use git commit normally
git commit -m "Your commit message"
```

#### How It Works

1. Runs `pre-commit run` (on staged files only, like normal git commit)
2. If hooks fail but made changes:
   - Stages modified files with `git add -u`
   - Re-runs pre-commit hooks on staged files
3. If hooks pass, proceeds with `git commit` using your original arguments
4. If hooks fail with errors (no files changed), exits with error

**Note**: This runs hooks BEFORE the commit, then commits. The standard pre-commit hooks will also run during the actual commit, but they should pass since we've already verified them.

#### Benefits

- **No double-commit workflow**: Commits succeed on first attempt if hooks auto-fix
- **Automatic staging**: Auto-fixed files are automatically staged
- **Clear feedback**: Shows what's happening at each step
- **Safe**: Only commits if all hooks pass

#### Example

```bash
$ git add my_file.py
$ git commit-hooks -m "Add new feature"
🔍 Running pre-commit hooks...
autoflake...................................................................Passed
isort......................................................................Passed
black......................................................................Failed
- hook id: black
- files were modified by this hook

📝 Pre-commit hooks made changes. Staging and re-running...
⏳ Staging auto-fixed files...
✅ Files staged
🔍 Re-running pre-commit hooks...
black......................................................................Passed
flake8....................................................................Passed
✅ Pre-commit hooks passed after fixes

🚀 Proceeding with git commit...
[main abc1234] Add new feature
 2 files changed, 50 insertions(+), 30 deletions(-)
```

**Important**: Make sure to stage files first with `git add` before using this script!

**Important**: Make sure to stage files first with `git add` before using this script!
