#!/bin/bash
# Run the Optimization Manager Dashboard
#
# Usage:
#   ./scripts/run_dashboard.sh           # Run on default port 8501
#   ./scripts/run_dashboard.sh 8080      # Run on custom port

set -e

cd "$(dirname "$0")/.."

PORT=${1:-8501}

# Set PostgreSQL URL if not already set
export OPTUNA_STORAGE_URL="${OPTUNA_STORAGE_URL:-postgresql://optuna:optuna123@localhost/optuna_db}"

echo "============================================================"
echo "Trading System - Optimization Manager Dashboard"
echo "============================================================"
echo "Port: $PORT"
echo "PostgreSQL: $OPTUNA_STORAGE_URL"
echo "============================================================"
echo ""
echo "Dashboard URL: http://localhost:$PORT"
echo "For EC2, use: http://<EC2-PUBLIC-IP>:$PORT"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run streamlit
streamlit run dashboard/optimization_manager.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true
