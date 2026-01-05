#!/bin/bash
# Setup script for MCP Server
# This script helps set up the MCP server for local development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Setting up MCP Server for Trading System..."
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ] && [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Warning: No virtual environment found. Consider creating one:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo ""
fi

# Install MCP server dependencies
echo "Installing MCP server dependencies..."
pip install fastapi>=0.100.0 uvicorn[standard]>=0.23.0 python-multipart>=0.0.6

echo ""
echo "✓ Dependencies installed"
echo ""

# Check if .env file exists
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file template..."
    cat > "$ENV_FILE" << EOF
# MCP Server Configuration
MCP_HOST=0.0.0.0
MCP_PORT=8000
# Uncomment and set a token for authentication:
# MCP_API_TOKEN=your-secret-token-here

# Trading System Configuration
PYTHONPATH=$PROJECT_ROOT
PYTHONUNBUFFERED=1
EOF
    echo "✓ Created .env file at $ENV_FILE"
    echo "  Edit it to configure MCP server settings"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Create test script
TEST_SCRIPT="$PROJECT_ROOT/scripts/test_mcp_server.sh"
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Creating test script..."
    cat > "$TEST_SCRIPT" << 'EOF'
#!/bin/bash
# Test MCP Server endpoints

MCP_URL="${MCP_URL:-http://localhost:8000}"

echo "Testing MCP Server at $MCP_URL..."
echo ""

# Test health endpoint
echo "1. Testing /health endpoint..."
curl -s "$MCP_URL/health" | python -m json.tool
echo ""

# Test root endpoint
echo "2. Testing / endpoint..."
curl -s "$MCP_URL/" | python -m json.tool
echo ""

# Test configs endpoint
echo "3. Testing /configs endpoint..."
curl -s "$MCP_URL/configs" | python -m json.tool
echo ""

echo "✓ All tests completed!"
echo ""
echo "For interactive API documentation, visit:"
echo "  $MCP_URL/docs"
EOF
    chmod +x "$TEST_SCRIPT"
    echo "✓ Created test script at $TEST_SCRIPT"
    echo ""
fi

echo "Setup complete!"
echo ""
echo "To start the MCP server:"
echo "  1. Development mode (with auto-reload):"
echo "     python -m uvicorn mcp_server.server:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "  2. Production mode:"
echo "     python -m uvicorn mcp_server.server:app --host 0.0.0.0 --port 8000"
echo ""
echo "  3. Using Docker:"
echo "     docker-compose -f docker-compose.yml -f docker-compose.prod.yml up mcp-server"
echo ""
echo "  4. Test the server:"
echo "     ./scripts/test_mcp_server.sh"
echo ""
echo "  5. View API documentation:"
echo "     open http://localhost:8000/docs"
echo ""
