# Trading System MCP Server

MCP (Model Context Protocol) server for the Trading System, enabling Claude and other AI assistants to interact with the trading system via HTTP API.

## Overview

The MCP server provides a RESTful API interface to the trading system, allowing you to:
- Run backtests via HTTP API
- Execute validation suites
- List available configurations
- Retrieve backtest results
- Enable Claude Desktop integration

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

2. **Run the server**:
   ```bash
   python -m uvicorn mcp_server.server:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Test the server**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/docs  # Interactive API documentation
   ```

### Docker

1. **Build the MCP server image**:
   ```bash
   docker build -f Dockerfile.mcp -t trading-system-mcp:latest .
   ```

2. **Run with docker-compose**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up mcp-server
   ```

3. **Or run standalone**:
   ```bash
   docker run -p 8000:8000 \
     -v $(pwd)/data:/app/data:ro \
     -v $(pwd)/configs:/app/configs:ro \
     -v $(pwd)/results:/app/results \
     trading-system-mcp:latest
   ```

## API Endpoints

### Health Check
- `GET /health` - Check server health
- `GET /` - Root endpoint with API information

### Backtest
- `POST /backtest` - Run a backtest
  ```json
  {
    "config_path": "configs/run_config.yaml",
    "period": "train"
  }
  ```

### Validation
- `POST /validate` - Run validation suite
  ```json
  {
    "config_path": "configs/run_config.yaml"
  }
  ```

### Configuration
- `GET /configs` - List available configuration files

### Results
- `GET /results/{run_id}` - Get backtest results
- `GET /results/{run_id}?period=train` - Get results for specific period

## Authentication (Optional)

To enable authentication, set the `MCP_API_TOKEN` environment variable:

```bash
export MCP_API_TOKEN=your-secret-token-here
```

When authentication is enabled, include the token in the Authorization header:

```bash
curl -H "Authorization: Bearer your-secret-token-here" \
  http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/run_config.yaml", "period": "train"}'
```

## Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Claude Desktop Integration

To connect Claude Desktop to your MCP server:

1. **Edit Claude Desktop config** (macOS):
   ```
   ~/Library/Application Support/Claude/claude_desktop_config.json
   ```

2. **Add MCP server configuration**:
   ```json
   {
     "mcpServers": {
       "trading-system": {
         "url": "http://localhost:8000",
         "apiKey": "your-api-token-if-set"
       }
     }
   }
   ```

3. **Restart Claude Desktop**

Note: Claude Desktop MCP integration may require additional configuration. For direct HTTP API usage, Claude can interact with the server via the REST API endpoints.

## Environment Variables

- `MCP_HOST` - Server host (default: 0.0.0.0)
- `MCP_PORT` - Server port (default: 8000)
- `MCP_API_TOKEN` - Optional API token for authentication
- `PYTHONPATH` - Python path (default: /app)

## Error Handling

The server returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (invalid parameters)
- `401` - Unauthorized (invalid or missing token)
- `404` - Not found (config file, results, etc.)
- `500` - Internal server error
- `504` - Gateway timeout (operation exceeded timeout)

## Security Considerations

1. **Authentication**: Set `MCP_API_TOKEN` in production
2. **CORS**: Configure CORS origins in production (currently allows all)
3. **HTTPS**: Use a reverse proxy (nginx, traefik) or AWS ALB for HTTPS
4. **Network**: Restrict access using security groups/firewalls
5. **Rate Limiting**: Consider adding rate limiting for production use

## Development

### Running Tests

```bash
# Run MCP server tests (if implemented)
pytest tests/mcp_server/
```

### Adding New Endpoints

1. Add endpoint function to `mcp_server/server.py`
2. Use Pydantic models for request/response validation
3. Add appropriate error handling
4. Update this README with endpoint documentation

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000
# Kill process or use different port
export MCP_PORT=8001
```

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/app:$PYTHONPATH
# Or run from project root
python -m mcp_server.server
```

### Docker Issues
```bash
# Check container logs
docker logs trading-system-mcp
# Rebuild image
docker build -f Dockerfile.mcp -t trading-system-mcp:latest .
```

## Related Documentation

- [Deployment Guide](../Deployment.md) - Full deployment instructions
- [Main README](../README.md) - Trading system documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - FastAPI framework docs

