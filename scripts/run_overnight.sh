#!/bin/bash
# Overnight Super-Optimization Script for EC2
#
# Usage:
#   ./scripts/run_overnight.sh              # Run with defaults (1000 trials)
#   ./scripts/run_overnight.sh 2000         # Custom trial count
#   ./scripts/run_overnight.sh 2000 equity  # Equity only
#   ./scripts/run_overnight.sh 2000 crypto  # Crypto only

set -e

cd "$(dirname "$0")/.."

# Configuration
TRIALS=${1:-1000}
MODE=${2:-"both"}  # both, equity, crypto

# Auto-detect CPU count
if [[ "$OSTYPE" == "darwin"* ]]; then
    CPU_COUNT=$(sysctl -n hw.ncpu)
else
    CPU_COUNT=$(nproc)
fi
JOBS=$((CPU_COUNT - 1))
JOBS=$((JOBS > 1 ? JOBS : 1))

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="overnight_results/run_${TIMESTAMP}"

echo "============================================================"
echo "OVERNIGHT SUPER-OPTIMIZATION"
echo "============================================================"
echo "Started: $(date)"
echo "Trials per strategy: $TRIALS"
echo "Parallel jobs: $JOBS"
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Build command
CMD="python scripts/overnight_optimization.py --trials $TRIALS --jobs $JOBS --output-dir $OUTPUT_DIR"

if [ "$MODE" == "equity" ]; then
    CMD="$CMD --equity-only"
elif [ "$MODE" == "crypto" ]; then
    CMD="$CMD --crypto-only"
fi

# Run with logging
echo "Running optimization..."
$CMD 2>&1 | tee "logs/overnight_${TIMESTAMP}.log"

echo ""
echo "============================================================"
echo "COMPLETED: $(date)"
echo "Results: $OUTPUT_DIR"
echo "Log: logs/overnight_${TIMESTAMP}.log"
echo "============================================================"
