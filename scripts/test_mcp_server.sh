#!/bin/bash
# Test MCP Server endpoints

MCP_URL="${MCP_URL:-http://localhost:8000}"

echo "Testing MCP Server at $MCP_URL..."
echo "=================================="
echo ""

# Test health endpoint
echo "1. Testing /health endpoint..."
if curl -s -f "$MCP_URL/health" > /dev/null 2>&1; then
    curl -s "$MCP_URL/health" | python3 -m json.tool 2>/dev/null || curl -s "$MCP_URL/health"
    echo ""
else
    echo "  ❌ Health check failed - is the server running?"
    echo ""
    exit 1
fi

# Test root endpoint
echo "2. Testing / endpoint..."
curl -s "$MCP_URL/" | python3 -m json.tool 2>/dev/null || curl -s "$MCP_URL/"
echo ""

# Test configs endpoint
echo "3. Testing /configs endpoint..."
curl -s "$MCP_URL/configs" | python3 -m json.tool 2>/dev/null || curl -s "$MCP_URL/configs"
echo ""

echo "✓ All tests completed!"
echo ""
echo "For interactive API documentation, visit:"
echo "  $MCP_URL/docs"
echo ""

