#!/bin/bash
# Git commit wrapper that automatically re-runs pre-commit hooks after fixes
# Usage: ./scripts/git-commit-with-hooks.sh [git commit arguments]
# Or create an alias: git config --global alias.commit-hooks '!./scripts/git-commit-with-hooks.sh'

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Store original git command arguments
GIT_ARGS=("$@")

# Function to stage modified files
stage_modified_files() {
    echo "⏳ Staging auto-fixed files..."
    # Stage both modified files and files that were already staged
    git add -u  # Stage modified tracked files
    git add "$(git diff --cached --name-only 2>/dev/null || true)"  # Re-stage previously staged files
    echo "✅ Files staged"
}

# Function to run pre-commit hooks on staged files
run_precommit() {
    if command -v pre-commit >/dev/null 2>&1; then
        pre-commit run "$@"
    else
        python -m pre_commit run "$@"
    fi
}

# Check if there are any staged files
if [ -z "$(git diff --cached --name-only)" ]; then
    echo "⚠️  No files staged for commit"
    echo "   Stage files first: git add <files>"
    exit 1
fi

# Step 1: Run pre-commit hooks on staged files
echo "🔍 Running pre-commit hooks..."
if run_precommit; then
    # Hooks passed on first try
    echo "✅ Pre-commit hooks passed"
else
    EXIT_CODE=$?

    # Check if any files were modified (either staged or unstaged)
    if [ -n "$(git diff --name-only)" ] || [ -n "$(git diff --cached --name-only)" ]; then
        echo ""
        echo "📝 Pre-commit hooks made changes. Staging and re-running..."

        # Stage the modified files
        stage_modified_files

        # Re-run hooks on staged files
        echo "🔍 Re-running pre-commit hooks..."
        if run_precommit; then
            echo "✅ Pre-commit hooks passed after fixes"
        else
            echo "❌ Pre-commit hooks still failing after fixes"
            echo "   Please review the errors and fix manually"
            exit 1
        fi
    else
        # No files changed, but hooks failed - likely actual errors
        echo "❌ Pre-commit hooks failed with errors (no files were auto-fixed)"
        exit $EXIT_CODE
    fi
fi

# Step 2: Proceed with git commit (hooks are already installed, so they'll run again)
# But we've already verified they pass, so this should succeed
echo ""
echo "🚀 Proceeding with git commit..."
exec git commit "${GIT_ARGS[@]}"
