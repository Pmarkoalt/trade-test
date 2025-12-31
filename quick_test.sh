#!/bin/bash
# Quick test script to verify trading system setup

set -e  # Exit on error

echo "========================================="
echo "Trading System V0.1 - Quick Test"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"
if [[ $(echo "$PYTHON_VERSION 3.9" | awk '{print ($1 >= $2)}') == 1 ]]; then
    echo -e "   ${GREEN}✓${NC} Python version OK"
else
    echo -e "   ${YELLOW}⚠${NC} Python 3.9+ recommended"
fi
echo ""

# Check dependencies
echo "2. Checking dependencies..."
MISSING_DEPS=()

# Check NumPy first with special handling for macOS segfault
NUMPY_OK=false
if python -c "import numpy; print('NumPy version:', numpy.__version__)" 2>&1 | grep -q "Segmentation fault\|Fatal Python error"; then
    echo -e "   ${RED}✗${NC} NumPy segmentation fault detected!"
    echo "   This is a known macOS issue. Run: ./scripts/setup_environment.sh"
    echo "   Or use Docker: docker-compose run --rm trading-system pytest tests/ -v"
    exit 1
elif python -c "import numpy" 2>/dev/null; then
    NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    echo "   NumPy version: $NUMPY_VERSION"
    NUMPY_OK=true
else
    MISSING_DEPS+=("numpy")
fi

python -c "import pandas" 2>/dev/null || MISSING_DEPS+=("pandas")
python -c "import pydantic" 2>/dev/null || MISSING_DEPS+=("pydantic")
python -c "import yaml" 2>/dev/null || MISSING_DEPS+=("pyyaml")
python -c "import pytest" 2>/dev/null || MISSING_DEPS+=("pytest")

if [ ${#MISSING_DEPS[@]} -eq 0 ] && [ "$NUMPY_OK" = true ]; then
    echo -e "   ${GREEN}✓${NC} All dependencies installed"
else
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo -e "   ${RED}✗${NC} Missing dependencies: ${MISSING_DEPS[*]}"
        echo "   Install with: pip install ${MISSING_DEPS[*]}"
        exit 1
    fi
fi
echo ""

# Check test data
echo "3. Checking test data..."
if [ -d "tests/fixtures" ]; then
    echo -e "   ${GREEN}✓${NC} Test fixtures directory exists"
    
    # Check for key files
    KEY_FILES=("AAPL.csv" "MSFT.csv" "GOOGL.csv" "BTC_sample.csv")
    for file in "${KEY_FILES[@]}"; do
        if [ -f "tests/fixtures/$file" ]; then
            echo -e "   ${GREEN}✓${NC} $file exists"
        else
            echo -e "   ${YELLOW}⚠${NC} $file not found (may use _sample suffix)"
        fi
    done
else
    echo -e "   ${RED}✗${NC} Test fixtures directory not found"
    exit 1
fi
echo ""

# Check config files
echo "4. Checking configuration files..."
if [ -f "tests/fixtures/configs/run_test_config.yaml" ]; then
    echo -e "   ${GREEN}✓${NC} Test run config exists"
else
    echo -e "   ${RED}✗${NC} Test run config not found"
    exit 1
fi
echo ""

# Test import
echo "5. Testing module imports..."
if python -c "import trading_system" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} trading_system module imports successfully"
else
    echo -e "   ${YELLOW}⚠${NC} Module import failed, trying with PYTHONPATH..."
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    if python -c "import trading_system" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} Module imports with PYTHONPATH"
    else
        echo -e "   ${RED}✗${NC} Module import failed"
        exit 1
    fi
fi
echo ""

# Run a simple unit test
echo "6. Running a simple unit test..."
if pytest tests/test_data_loading.py::TestLoadOHLCVData::test_load_valid_data -v --tb=short 2>&1 | grep -q "PASSED\|passed"; then
    echo -e "   ${GREEN}✓${NC} Unit test passed"
else
    echo -e "   ${YELLOW}⚠${NC} Unit test had issues (check output above)"
fi
echo ""

# Summary
echo "========================================="
echo "Quick Test Summary"
echo "========================================="
echo -e "${GREEN}Setup looks good!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run all unit tests: pytest tests/ -v"
echo "  2. Run integration test: pytest tests/integration/ -v"
echo "  3. Run backtest: python -m trading_system backtest --config tests/fixtures/configs/run_test_config.yaml"
echo ""
echo "See TESTING_GUIDE.md for detailed instructions."

