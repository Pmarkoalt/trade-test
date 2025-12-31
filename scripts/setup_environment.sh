#!/bin/bash
# Environment Setup Script for Trading System
# Automatically detects and fixes common environment issues, especially NumPy on macOS

set -e  # Exit on error (unless --check flag is used)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default behavior
CHECK_ONLY=false
FIX_ISSUES=true
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            FIX_ISSUES=false
            shift
            ;;
        --no-fix)
            FIX_ISSUES=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check       Only check environment, don't fix issues"
            echo "  --no-fix      Check and report, but don't attempt fixes"
            echo "  --verbose     Show detailed output"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Trading System - Environment Setup"
echo "========================================="
echo ""

# Detect OS
OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
    OS_NAME="macOS"
    IS_MACOS=true
elif [[ "$OS" == "Linux" ]]; then
    OS_NAME="Linux"
    IS_MACOS=false
else
    OS_NAME="$OS"
    IS_MACOS=false
fi

echo "Detected OS: $OS_NAME"
echo ""

# Check Python version
echo "1. Checking Python installation..."
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "   ${RED}✗${NC} Python not found"
    echo "   Please install Python 3.9+ from python.org or via conda"
    exit 1
fi

PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

# Check if Python version is adequate
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
    echo -e "   ${RED}✗${NC} Python 3.9+ required (found $PYTHON_VERSION)"
    if [[ "$FIX_ISSUES" == true ]]; then
        echo "   ${YELLOW}⚠${NC} Please upgrade Python to 3.9+ or use Python 3.11+ (recommended)"
    fi
    exit 1
elif [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 11 ]]; then
    echo -e "   ${GREEN}✓${NC} Python version is excellent (3.11+)"
elif [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 9 ]]; then
    echo -e "   ${YELLOW}⚠${NC} Python 3.11+ recommended for better NumPy compatibility on macOS"
fi
echo ""

# Check NumPy compatibility (critical for macOS)
echo "2. Checking NumPy compatibility..."
NUMPY_ISSUE=false

if $PYTHON_CMD -c "import numpy; print('NumPy version:', numpy.__version__)" 2>&1 | grep -q "Segmentation fault\|Fatal Python error"; then
    NUMPY_ISSUE=true
    echo -e "   ${RED}✗${NC} NumPy segmentation fault detected!"
    echo "   This is a known macOS issue with certain NumPy versions"
elif $PYTHON_CMD -c "import numpy" 2>/dev/null; then
    NUMPY_VERSION=$($PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    echo "   NumPy version: $NUMPY_VERSION"

    # Check if version is >= 1.24.0 (better macOS compatibility)
    NUMPY_MAJOR=$(echo $NUMPY_VERSION | cut -d. -f1)
    NUMPY_MINOR=$(echo $NUMPY_VERSION | cut -d. -f2)

    if [[ $NUMPY_MAJOR -eq 1 && $NUMPY_MINOR -lt 24 ]] || [[ $NUMPY_MAJOR -lt 1 ]]; then
        echo -e "   ${YELLOW}⚠${NC} NumPy < 1.24.0 may have macOS compatibility issues"
        if [[ "$IS_MACOS" == true ]]; then
            NUMPY_ISSUE=true
        fi
    else
        echo -e "   ${GREEN}✓${NC} NumPy version is compatible"
    fi
else
    echo -e "   ${YELLOW}⚠${NC} NumPy not installed"
    NUMPY_ISSUE=true
fi
echo ""

# Check other dependencies
echo "3. Checking other dependencies..."
MISSING_DEPS=()

for dep in pandas pydantic yaml pytest; do
    if $PYTHON_CMD -c "import $dep" 2>/dev/null; then
        if [[ "$VERBOSE" == true ]]; then
            echo "   ${GREEN}✓${NC} $dep installed"
        fi
    else
        MISSING_DEPS+=("$dep")
        echo "   ${YELLOW}⚠${NC} $dep not installed"
    fi
done

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo -e "   ${GREEN}✓${NC} All core dependencies installed"
else
    echo "   Missing: ${MISSING_DEPS[*]}"
fi
echo ""

# Fix issues if requested
if [[ "$FIX_ISSUES" == true ]] && [[ "$NUMPY_ISSUE" == true || ${#MISSING_DEPS[@]} -gt 0 ]]; then
    echo "4. Attempting to fix issues..."

    # Detect package manager
    if command -v conda &> /dev/null && [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        echo "   Using conda environment: $CONDA_DEFAULT_ENV"

        if [[ "$NUMPY_ISSUE" == true ]]; then
            echo "   Fixing NumPy issue..."
            echo "   Installing NumPy >= 1.24.0 from conda-forge..."
            conda install -y -c conda-forge "numpy>=1.24.0" || {
                echo "   ${YELLOW}⚠${NC} Conda install failed, trying pip..."
                pip install --upgrade "numpy>=1.24.0"
            }
        fi

        if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
            echo "   Installing missing dependencies..."
            pip install "${MISSING_DEPS[@]}"
        fi
    else
        echo "   Using pip"

        if [[ "$NUMPY_ISSUE" == true ]]; then
            echo "   Fixing NumPy issue..."
            echo "   Installing/upgrading NumPy >= 1.24.0..."
            pip install --upgrade "numpy>=1.24.0"
        fi

        if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
            echo "   Installing missing dependencies..."
            pip install "${MISSING_DEPS[@]}"
        fi
    fi

    echo ""
    echo "5. Verifying fixes..."

    # Re-check NumPy
    if $PYTHON_CMD -c "import numpy; print('NumPy version:', numpy.__version__)" 2>&1 | grep -q "Segmentation fault\|Fatal Python error"; then
        echo -e "   ${RED}✗${NC} NumPy issue persists"
        echo ""
        echo "   ${YELLOW}Recommended solutions:${NC}"
        echo "   1. Use Docker: docker-compose run --rm trading-system pytest tests/ -v"
        echo "   2. Create fresh Python 3.11+ environment:"
        echo "      python3.11 -m venv venv"
        echo "      source venv/bin/activate"
        echo "      pip install -e '.[dev]'"
        echo "   3. See ENVIRONMENT_ISSUE.md for more options"
        exit 1
    else
        echo -e "   ${GREEN}✓${NC} NumPy working correctly"
    fi
else
    if [[ "$CHECK_ONLY" == true ]]; then
        echo "4. Check-only mode (no fixes applied)"
    fi
fi

# Final verification
echo ""
echo "6. Final environment check..."
ALL_GOOD=true

if ! $PYTHON_CMD -c "import numpy, pandas, pydantic, yaml" 2>/dev/null; then
    echo -e "   ${RED}✗${NC} Some imports still failing"
    ALL_GOOD=false
else
    echo -e "   ${GREEN}✓${NC} All core imports successful"
fi

# Check if trading_system can be imported
if $PYTHON_CMD -c "import trading_system" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} trading_system module imports successfully"
else
    echo -e "   ${YELLOW}⚠${NC} trading_system module not found (may need: pip install -e .)"
    if [[ "$FIX_ISSUES" == true ]]; then
        echo "   Installing trading_system in development mode..."
        pip install -e . || echo "   ${YELLOW}⚠${NC} Installation failed (may need to be in project root)"
    fi
fi

echo ""
echo "========================================="
if [[ "$ALL_GOOD" == true ]]; then
    echo -e "${GREEN}Environment setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: pytest tests/ -v"
    echo "  2. Run quick test: ./quick_test.sh"
    echo "  3. See README.md for usage examples"
else
    echo -e "${YELLOW}Environment has some issues${NC}"
    echo ""
    echo "Recommendations:"
    if [[ "$IS_MACOS" == true ]]; then
        echo "  • Use Docker to avoid macOS-specific issues:"
        echo "    docker-compose run --rm trading-system pytest tests/ -v"
    fi
    echo "  • See ENVIRONMENT_ISSUE.md for troubleshooting"
    echo "  • Run with --verbose for more details"
fi
echo "========================================="

