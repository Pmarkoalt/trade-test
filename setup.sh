#!/bin/bash
# Master Setup Script for Trading System
# This script orchestrates all setup steps for a complete development environment
#
# Usage:
#   ./setup.sh              - Run full setup (recommended)
#   ./setup.sh --skip-precommit  - Skip pre-commit hooks setup
#   ./setup.sh --dev-only   - Only install dev dependencies (skip environment checks)
#   ./setup.sh --help       - Show this help message

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
SKIP_PRECOMMIT=false
DEV_ONLY=false
SKIP_ENV_CHECK=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-precommit)
            SKIP_PRECOMMIT=true
            shift
            ;;
        --dev-only)
            DEV_ONLY=true
            SKIP_ENV_CHECK=true
            shift
            ;;
        --skip-env-check)
            SKIP_ENV_CHECK=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Master setup script for Trading System development environment"
            echo ""
            echo "Options:"
            echo "  --skip-precommit    Skip pre-commit hooks installation"
            echo "  --dev-only          Only install dev dependencies (skip environment checks)"
            echo "  --skip-env-check    Skip environment validation checks"
            echo "  --help              Show this help message"
            echo ""
            echo "This script will:"
            echo "  1. Check and set up Python environment"
            echo "  2. Install/upgrade dependencies"
            echo "  3. Install pre-commit git hooks (unless --skip-precommit)"
            echo "  4. Verify installation"
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
echo "Trading System - Master Setup"
echo "========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Environment Setup
if [[ "$SKIP_ENV_CHECK" == false ]]; then
    echo -e "${BLUE}Step 1: Environment Setup${NC}"
    echo "================================"

    if [[ -f "scripts/setup_environment.sh" ]]; then
        echo "Running environment setup script..."
        bash scripts/setup_environment.sh
        echo ""
    else
        echo -e "${YELLOW}⚠${NC} scripts/setup_environment.sh not found, skipping environment checks"
        echo ""
    fi
else
    echo -e "${BLUE}Step 1: Skipping environment checks${NC}"
    echo ""
fi

# Step 2: Install Dependencies
echo -e "${BLUE}Step 2: Installing Dependencies${NC}"
echo "================================="

# Detect Python command
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${YELLOW}⚠${NC} Not in a virtual environment"
    echo "   Consider creating one:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
    echo ""
    if [[ -t 0 ]]; then
        # Only prompt if running interactively
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Setup cancelled"
            exit 1
        fi
    else
        echo "   Continuing automatically (non-interactive mode)..."
    fi
    echo ""
fi

echo "Installing project in development mode with all dependencies..."
if [[ "$DEV_ONLY" == true ]]; then
    echo "Installing dev dependencies..."
    $PYTHON_CMD -m pip install --upgrade pip
    $PYTHON_CMD -m pip install -e ".[dev]"
else
    echo "Installing all dependencies (production + dev)..."
    $PYTHON_CMD -m pip install --upgrade pip
    $PYTHON_CMD -m pip install -e ".[dev]"
fi

echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# Step 3: Pre-commit Hooks Setup
if [[ "$SKIP_PRECOMMIT" == false ]]; then
    echo -e "${BLUE}Step 3: Setting Up Pre-commit Hooks${NC}"
    echo "======================================"

    # Check if pre-commit is installed
    if ! command -v pre-commit &> /dev/null; then
        echo "pre-commit not found in PATH, but should be installed via dependencies"
        echo "Trying to use Python module..."

        if $PYTHON_CMD -m pre_commit --version &> /dev/null; then
            echo "Installing pre-commit git hooks..."
            $PYTHON_CMD -m pre_commit install
            echo -e "${GREEN}✓${NC} Pre-commit hooks installed"
        else
            echo -e "${YELLOW}⚠${NC} pre-commit not available"
            echo "   It should be installed with dev dependencies"
            echo "   You can install it manually: pip install pre-commit"
            echo "   Then run: pre-commit install"
        fi
    else
        echo "Installing pre-commit git hooks..."
        pre-commit install
        echo -e "${GREEN}✓${NC} Pre-commit hooks installed"
    fi
    echo ""
else
    echo -e "${BLUE}Step 3: Skipping pre-commit hooks setup${NC}"
    echo ""
fi

# Step 4: Verify Installation
echo -e "${BLUE}Step 4: Verification${NC}"
echo "===================="

ALL_GOOD=true

# Check if trading_system can be imported
echo "Checking if trading_system module can be imported..."
if $PYTHON_CMD -c "import trading_system" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} trading_system module imports successfully"
else
    echo -e "   ${RED}✗${NC} trading_system module cannot be imported"
    ALL_GOOD=false
fi

# Check key dependencies
echo "Checking key dependencies..."
for dep in numpy pandas pydantic yaml pytest; do
    if $PYTHON_CMD -c "import $dep" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} $dep installed"
    else
        echo -e "   ${RED}✗${NC} $dep not found"
        ALL_GOOD=false
    fi
done

echo ""

# Final Summary
echo "========================================="
if [[ "$ALL_GOOD" == true ]]; then
    echo -e "${GREEN}Setup Complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: pytest tests/ -v"
    echo "     Or use: make test-unit"
    echo ""
    echo "  2. Run quick test: ./quick_test.sh"
    echo ""
    echo "  3. Verify pre-commit hooks (if installed):"
    echo "     pre-commit run --all-files"
    echo ""
    echo "  4. See README.md for usage examples and documentation"
    echo ""
    if [[ "$SKIP_PRECOMMIT" == false ]]; then
        echo -e "${BLUE}Note:${NC} Pre-commit hooks are now installed and will run automatically"
        echo "      on every 'git commit'. To skip them: git commit --no-verify"
    fi
else
    echo -e "${YELLOW}Setup completed with some issues${NC}"
    echo ""
    echo "Recommendations:"
    echo "  • Check error messages above"
    echo "  • Try: pip install -e \".[dev]\""
    echo "  • See ENVIRONMENT_ISSUE.md for troubleshooting"
    echo "  • Consider using Docker: make docker-build"
fi
echo "========================================="
