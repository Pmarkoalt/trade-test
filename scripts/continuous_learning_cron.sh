#!/bin/bash
# Continuous Learning Cron Script
# Run weekly to retrain ML models when new data is available
#
# Add to crontab with: crontab -e
# 0 0 * * 0 /path/to/trade-test/scripts/continuous_learning_cron.sh >> /path/to/trade-test/logs/retrain.log 2>&1

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"  # Adjust if using different venv

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Timestamp
echo "=============================================="
echo "Continuous Learning Cycle: $(date)"
echo "=============================================="

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment if exists
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
fi

# Run continuous learning cycle
echo "Running ML continuous learning cycle..."
python scripts/ml_continuous_learning.py --cycle --verbose

echo ""
echo "Cycle complete: $(date)"
echo "=============================================="
