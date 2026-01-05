#!/bin/bash
# Script to run unit tests independently
# Usage: ./scripts/run_unit_tests.sh [pytest-args]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================="
echo "Running Unit Tests"
echo "========================================="
echo ""

# Run unit tests (excluding integration, performance, and property tests)
pytest tests/ \
    --ignore=tests/integration \
    --ignore=tests/performance \
    --ignore=tests/property \
    -v \
    --tb=short \
    "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Unit tests passed${NC}"
else
    echo ""
    echo -e "${RED}✗ Unit tests failed${NC}"
fi

exit $EXIT_CODE
