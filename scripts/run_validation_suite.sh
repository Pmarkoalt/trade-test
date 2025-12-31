#!/bin/bash
# Script to run validation suite independently
# Usage: ./scripts/run_validation_suite.sh [config-path]
#   Default config: tests/fixtures/configs/run_test_config.yaml

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default config path
DEFAULT_CONFIG="tests/fixtures/configs/run_test_config.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

echo "========================================="
echo "Running Validation Suite"
echo "========================================="
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}✗ Config file not found: $CONFIG_PATH${NC}"
    echo ""
    echo "Usage:"
    echo "  ./scripts/run_validation_suite.sh [config-path]"
    echo ""
    echo "Example:"
    echo "  ./scripts/run_validation_suite.sh tests/fixtures/configs/run_test_config.yaml"
    exit 1
fi

echo "Using config: $CONFIG_PATH"
echo ""

# Run validation suite
python -m trading_system validate --config "$CONFIG_PATH"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Validation suite passed${NC}"
else
    echo ""
    echo -e "${RED}✗ Validation suite failed${NC}"
fi

exit $EXIT_CODE

