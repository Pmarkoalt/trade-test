#!/bin/bash
# Setup script to install pre-commit git hooks
# This will make pre-commit run automatically on every git commit

set -e

echo "Setting up pre-commit hooks..."
echo ""

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "pre-commit not found in PATH"
    echo ""
    echo "Please install pre-commit first:"
    echo "  pip install pre-commit"
    echo "  OR"
    echo "  pip install -e \".[dev]\"  # This installs all dev dependencies including pre-commit"
    echo ""
    exit 1
fi

# Install the git hooks
echo "Installing pre-commit git hooks..."
pre-commit install

echo ""
echo "✓ Pre-commit hooks installed successfully!"
echo ""
echo "Pre-commit will now run automatically on every 'git commit'"
echo ""
echo "To test it, try:"
echo "  pre-commit run --all-files"
echo ""
