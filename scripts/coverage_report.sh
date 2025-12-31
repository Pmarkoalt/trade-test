#!/bin/bash
# Script to generate test coverage report for trading_system
# Usage: ./scripts/coverage_report.sh

set -e

echo "========================================="
echo "Test Coverage Report Generator"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest-cov is installed
echo "1. Checking dependencies..."
if ! python3 -m pytest --version > /dev/null 2>&1; then
    echo -e "   ${RED}✗${NC} pytest not found"
    echo "   Install with: pip install pytest pytest-cov coverage"
    exit 1
fi

if ! python3 -c "import pytest_cov" 2>/dev/null; then
    echo -e "   ${YELLOW}⚠${NC} pytest-cov not found"
    echo "   Install with: pip install pytest-cov"
    exit 1
fi

echo -e "   ${GREEN}✓${NC} Dependencies OK"
echo ""

# Run coverage report
echo "2. Running tests with coverage..."
COVERAGE_OUTPUT=$(python3 -m pytest \
    --cov=trading_system \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    tests/ \
    -v 2>&1)

# Display the output
echo "$COVERAGE_OUTPUT"
echo ""

# Extract coverage percentage from output
COVERAGE_PCT=$(echo "$COVERAGE_OUTPUT" | grep -oP 'TOTAL\s+\d+\s+\d+\s+\K[\d.]+%' | head -1 || echo "N/A")

echo "3. Coverage Summary"
echo "   HTML report: htmlcov/index.html"
echo "   XML report: coverage.xml"
if [ "$COVERAGE_PCT" != "N/A" ]; then
    echo "   Coverage: ${COVERAGE_PCT}"

    # Check if coverage meets target (90%) - works without bc
    COVERAGE_NUM=$(echo "$COVERAGE_PCT" | sed 's/%//' | cut -d. -f1)
    if [ "$COVERAGE_NUM" -ge 90 ] 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} Coverage meets target (≥90%)"
    else
        echo -e "   ${YELLOW}⚠${NC} Coverage below target (90%)"
        echo "   Current: ${COVERAGE_PCT}"
    fi
fi
echo ""

# Open HTML report if on macOS
if [[ "$OSTYPE" == "darwin"* ]] && [ -f "htmlcov/index.html" ]; then
    echo "4. Opening HTML coverage report..."
    open htmlcov/index.html
fi

echo "========================================="
echo "Coverage Report Complete"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Review HTML report: open htmlcov/index.html"
echo "  2. Check for uncovered code paths"
echo "  3. Add tests for uncovered areas if needed"
echo "  4. Update README with coverage badge"
echo ""

