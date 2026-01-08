#!/bin/bash
# Run 500-trial equity optimization
# Extended parameter ranges for longer trends (S&P 45% 2024-2026)

cd "$(dirname "$0")/.."

# Auto-detect CPU count and use 75% of cores (leave some for system)
if [[ "$OSTYPE" == "darwin"* ]]; then
    CPU_COUNT=$(sysctl -n hw.ncpu)
else
    CPU_COUNT=$(nproc)
fi
AUTO_JOBS=$(( (CPU_COUNT * 3) / 4 ))
AUTO_JOBS=$((AUTO_JOBS > 1 ? AUTO_JOBS : 1))

# Allow override via argument
NUM_JOBS=${1:-$AUTO_JOBS}

echo "=============================================="
echo "EQUITY STRATEGY OPTIMIZATION (500 trials)"
echo "=============================================="
echo "Started: $(date)"
echo "CPU Cores: $CPU_COUNT | Using: $NUM_JOBS parallel jobs"
echo ""

mkdir -p logs

python scripts/optimize_strategy.py \
    --config configs/test_equity_strategy.yaml \
    --trials 500 \
    --jobs $NUM_JOBS \
    --objective sharpe_ratio \
    --verbose 2>&1 | tee logs/equity_optimization_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Completed: $(date)"
