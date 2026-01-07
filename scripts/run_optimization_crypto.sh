#!/bin/bash
# Run 500-trial crypto optimization
# Extended parameter ranges for volatile crypto trends

cd "$(dirname "$0")/.."

echo "=============================================="
echo "CRYPTO STRATEGY OPTIMIZATION (500 trials)"
echo "=============================================="
echo "Started: $(date)"
echo ""

python scripts/optimize_strategy.py \
    --config configs/test_crypto_strategy.yaml \
    --trials 500 \
    --objective sharpe_ratio \
    --verbose 2>&1 | tee logs/crypto_optimization_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Completed: $(date)"
