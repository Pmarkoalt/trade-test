#!/bin/bash
# Run pre-commit hooks in Docker
# Usage: ./scripts/run_precommit_docker.sh [pre-commit-args]
# Examples:
#   ./scripts/run_precommit_docker.sh                    # Run on all files
#   ./scripts/run_precommit_docker.sh run               # Run on staged files
#   ./scripts/run_precommit_docker.sh run --all-files   # Run on all files
#   ./scripts/run_precommit_docker.sh run black --all-files  # Run specific hook

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running pre-commit in Docker..."
echo "================================="
echo ""

# Default to running on all files if no args provided
ARGS="${@:-run --all-files}"

# Run pre-commit in Docker
# Note: .pre-commit-config.yaml is mounted in docker-compose.yml
docker-compose run --rm \
  --entrypoint bash \
  trading-system \
  -c "cd /app && pip install pre-commit && pre-commit $ARGS"

echo ""
echo "✓ Pre-commit complete!"

