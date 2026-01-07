#!/bin/bash
# Run 500-trial equity optimization
# Extended parameter ranges for longer trends (S&P 45% 2024-2026)

cd "$(dirname "$0")/.."

echo "=============================================="
echo "EQUITY STRATEGY OPTIMIZATION (500 trials)"
echo "=============================================="
echo "Started: $(date)"
echo ""

python scripts/optimize_strategy.py \
    --config configs/test_equity_strategy.yaml \
    --trials 500 \
    --objective sharpe_ratio \
    --verbose 2>&1 | tee logs/equity_optimization_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Completed: $(date)"
