#!/bin/bash
# Script to run integration tests independently
# Usage: ./scripts/run_integration_tests.sh [pytest-args]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================="
echo "Running Integration Tests"
echo "========================================="
echo ""

# Run integration tests
pytest tests/integration/ \
    -v \
    --tb=short \
    "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Integration tests passed${NC}"
else
    echo ""
    echo -e "${RED}✗ Integration tests failed${NC}"
fi

exit $EXIT_CODE

