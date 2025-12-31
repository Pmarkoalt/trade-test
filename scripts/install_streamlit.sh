#!/bin/bash
# Quick script to install Streamlit and dashboard dependencies

set -e

echo "Installing Streamlit and dashboard dependencies..."
echo "================================================"
echo ""

# Check if we're in a virtual environment (recommended)
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider creating one: python -m venv .venv && source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install with optional dependencies
echo "Installing Streamlit with visualization dependencies..."
pip install -e ".[visualization]"

# Or install directly if the above doesn't work
# pip install streamlit>=1.28.0 plotly>=5.0.0

echo ""
echo "✓ Installation complete!"
echo ""
echo "Verify installation:"
echo "  streamlit --version"
echo ""
echo "Test the dashboard:"
echo "  python -m trading_system dashboard --run-id <your_run_id>"
echo "  # Or"
echo "  python -m trading_system trading-dashboard"
echo ""

