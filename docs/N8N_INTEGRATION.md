# n8n Integration Guide

This guide explains how to integrate the Trading System with [n8n](https://n8n.io/), a workflow automation platform. This integration enables visual workflow management, scheduled backtests, database storage, and alerting.

## Overview

n8n acts as an orchestration layer on top of the trading system, allowing you to:

- Schedule automated backtests (daily, weekly, etc.)
- Run parameter sweeps across configurations
- Store results in databases (PostgreSQL, MongoDB, etc.)
- Send alerts via Slack, email, or webhooks
- Chain multi-stage pipelines (train → validate → holdout)
- Create visual dashboards from backtest results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         n8n                                  │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌───────────┐  │
│  │  Cron   │ → │ HTTP Req │ → │  Parse  │ → │  Store/   │  │
│  │ Trigger │   │ to API   │   │ Results │   │  Alert    │  │
│  └─────────┘   └──────────┘   └─────────┘   └───────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Trading API (FastAPI)                      │
│  POST /backtest     GET /results     GET /metrics           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Trading System                             │
│  BacktestEngine → Portfolio → Execution → Reporting         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose installed
- Trading system codebase cloned

### 2. Start the Stack

```bash
# Build the Docker images (includes FastAPI dependencies)
docker-compose -f docker-compose.n8n.yml build

# Start all services
docker-compose -f docker-compose.n8n.yml up -d

# Verify services are running
docker-compose -f docker-compose.n8n.yml ps
```

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| n8n | http://localhost:5678 | admin / changeme |
| Trading API | http://localhost:8000 | None (internal) |
| API Docs (Swagger) | http://localhost:8000/docs | None |
| PostgreSQL | localhost:5432 | trading / trading_password |

### 4. Change Default Passwords

Before production use, update credentials in `docker-compose.n8n.yml`:

```yaml
n8n:
  environment:
    - N8N_BASIC_AUTH_USER=your_username
    - N8N_BASIC_AUTH_PASSWORD=your_secure_password

postgres:
  environment:
    - POSTGRES_USER=your_db_user
    - POSTGRES_PASSWORD=your_secure_db_password
```

## API Reference

The Trading API (`trading_system/api/server.py`) exposes the following endpoints:

### Health Check

```
GET /health
```

Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Start Backtest (Async)

```
POST /backtest
```

Starts a backtest in the background. Returns immediately with a job ID.

**Request Body:**
```json
{
  "config_path": "/app/custom_configs/backtest_config_production.yaml",
  "period": "train"
}
```

**Response:**
```json
{
  "status": "queued",
  "run_id": "backtest_20240115_103000",
  "message": "Backtest queued for period 'train'"
}
```

### Run Backtest (Sync)

```
POST /backtest/sync
```

Runs a backtest and waits for completion. Use for shorter runs.

**Request Body:**
```json
{
  "config_path": "/app/custom_configs/backtest_config_production.yaml",
  "period": "train"
}
```

**Response:**
```json
{
  "status": "completed",
  "period": "train",
  "results": { ... }
}
```

### Check Job Status

```
GET /job/{job_id}
```

Check the status of an async backtest job.

**Response:**
```json
{
  "status": "completed",
  "config_path": "/app/custom_configs/backtest_config_production.yaml",
  "period": "train",
  "started_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:35:00",
  "results": { ... }
}
```

### List Results

```
GET /results?limit=10
```

List recent backtest run directories.

**Response:**
```json
{
  "results": [
    {
      "run_id": "run_20240115_103000",
      "path": "/app/results/run_20240115_103000",
      "periods": ["train", "validation", "holdout"]
    }
  ]
}
```

### Get Metrics

```
GET /results/{run_id}/{period}/metrics
```

Get the monthly report metrics from a specific run.

**Response:**
```json
{
  "period": "train",
  "sharpe_ratio": 1.45,
  "calmar_ratio": 2.1,
  "max_drawdown": -0.12,
  "total_return": 0.35,
  ...
}
```

### Get Equity Curve

```
GET /results/{run_id}/{period}/equity_curve?limit=100
```

Get equity curve data (optionally limited to last N rows).

**Response:**
```json
[
  {"date": "2024-01-01", "equity": 100000, "cash": 50000, "exposure": 0.5},
  {"date": "2024-01-02", "equity": 100500, "cash": 48000, "exposure": 0.52},
  ...
]
```

### Get Trades

```
GET /results/{run_id}/{period}/trades?limit=50
```

Get trade log data (optionally limited to last N rows).

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "entry_date": "2024-01-05",
    "exit_date": "2024-01-15",
    "pnl": 1250.50,
    "return_pct": 0.025
  },
  ...
]
```

## n8n Workflow Examples

### Example 1: Daily Scheduled Backtest

This workflow runs a backtest every day at 6 AM and sends results to Slack.

**Nodes:**

1. **Cron Trigger**
   - Mode: Every Day
   - Hour: 6
   - Minute: 0

2. **HTTP Request** (Run Backtest)
   - Method: POST
   - URL: `http://trading-api:8000/backtest/sync`
   - Body (JSON):
     ```json
     {
       "config_path": "/app/custom_configs/backtest_config_production.yaml",
       "period": "train"
     }
     ```

3. **IF** (Check Results)
   - Condition: `{{ $json.results.sharpe_ratio > 1.0 }}`

4. **Slack** (Send Notification)
   - Channel: #trading-alerts
   - Message:
     ```
     Backtest completed!
     Sharpe: {{ $json.results.sharpe_ratio }}
     Return: {{ $json.results.total_return }}
     ```

### Example 2: Multi-Period Pipeline

Run train → validate → holdout sequentially, storing results.

**Nodes:**

1. **Manual Trigger** or **Cron Trigger**

2. **HTTP Request** (Train)
   - POST `http://trading-api:8000/backtest/sync`
   - Body: `{"config_path": "...", "period": "train"}`

3. **IF** (Train Passed?)
   - Condition: `{{ $json.results.sharpe_ratio > 0.5 }}`

4. **HTTP Request** (Validate)
   - POST `http://trading-api:8000/backtest/sync`
   - Body: `{"config_path": "...", "period": "validation"}`

5. **IF** (Validate Passed?)
   - Condition: `{{ $json.results.sharpe_ratio > 0.3 }}`

6. **HTTP Request** (Holdout)
   - POST `http://trading-api:8000/backtest/sync`
   - Body: `{"config_path": "...", "period": "holdout"}`

7. **Postgres** (Store Results)
   - Operation: Insert
   - Table: backtest_results
   - Columns: Map from JSON response

### Example 3: Parameter Sweep

Run multiple backtests with different configurations.

**Nodes:**

1. **Manual Trigger**

2. **Function** (Generate Configs)
   ```javascript
   return [
     { json: { config: "config_aggressive.yaml", name: "aggressive" } },
     { json: { config: "config_conservative.yaml", name: "conservative" } },
     { json: { config: "config_balanced.yaml", name: "balanced" } }
   ];
   ```

3. **Split In Batches**
   - Batch Size: 1

4. **HTTP Request** (Run Each Backtest)
   - POST `http://trading-api:8000/backtest/sync`
   - Body: `{"config_path": "/app/custom_configs/{{ $json.config }}", "period": "train"}`

5. **Merge** (Combine Results)

6. **Function** (Find Best)
   ```javascript
   const best = items.reduce((a, b) =>
     a.json.results.sharpe_ratio > b.json.results.sharpe_ratio ? a : b
   );
   return [best];
   ```

7. **Slack** (Report Best)

## Database Schema

When storing results in PostgreSQL, use this schema:

```sql
CREATE TABLE backtest_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    config_path VARCHAR(500),
    period VARCHAR(20),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE backtest_metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES backtest_runs(run_id),
    sharpe_ratio DECIMAL(10, 4),
    calmar_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    total_return DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),
    win_rate DECIMAL(10, 4),
    total_trades INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES backtest_runs(run_id),
    symbol VARCHAR(20),
    entry_date DATE,
    exit_date DATE,
    entry_price DECIMAL(15, 4),
    exit_price DECIMAL(15, 4),
    shares INTEGER,
    pnl DECIMAL(15, 4),
    return_pct DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Volume Mounts

The `docker-compose.n8n.yml` configures shared volumes:

| Host Path | Container Path (n8n) | Container Path (API) | Purpose |
|-----------|---------------------|---------------------|---------|
| `./results` | `/data/results` | `/app/results` | Backtest outputs |
| `./configs` | `/data/configs` | `/app/custom_configs` | Custom configurations |
| `./EXAMPLE_CONFIGS` | `/data/example_configs` | `/app/configs` | Example configurations |
| `./data` | `/data/market_data` | `/app/data` | Market data files |

## Configuration

### Timeouts

For long-running backtests, configure timeouts in `docker-compose.n8n.yml`:

```yaml
n8n:
  environment:
    - EXECUTIONS_TIMEOUT=600        # 10 minutes default
    - EXECUTIONS_TIMEOUT_MAX=3600   # 1 hour max
```

### n8n HTTP Request Timeout

In n8n HTTP Request nodes, set appropriate timeouts:
- Timeout: 600000 (10 minutes in milliseconds)

### Memory Limits

For large backtests, you may need to increase memory:

```yaml
trading-api:
  deploy:
    resources:
      limits:
        memory: 4G
```

## Troubleshooting

### n8n Can't Connect to trading-api

Ensure both services are on the same Docker network:

```bash
docker network inspect trading-network
```

Use `http://trading-api:8000` (container name) not `http://localhost:8000`.

### Backtest Times Out

1. Increase n8n timeout in environment variables
2. Increase HTTP Request node timeout
3. Use async endpoint (`POST /backtest`) and poll for status

### Results Directory Not Found

Verify volume mounts are correct:

```bash
docker-compose -f docker-compose.n8n.yml exec trading-api ls -la /app/results
```

### API Server Not Starting

Check logs:

```bash
docker-compose -f docker-compose.n8n.yml logs trading-api
```

Common issues:
- Missing dependencies: Rebuild with `docker-compose -f docker-compose.n8n.yml build trading-api`
- Port conflict: Change port mapping in docker-compose.n8n.yml

## Local Development (Without Docker)

To run the API locally for development:

```bash
# Install n8n dependencies
pip install -e ".[n8n]"

# Start the API server
uvicorn trading_system.api.server:app --reload --port 8000

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Swagger UI
```

## Security Considerations

1. **Change default passwords** before production use
2. **Use HTTPS** in production (configure reverse proxy)
3. **Restrict network access** to internal services
4. **Don't expose PostgreSQL** externally in production
5. **Use environment variables** for sensitive configuration

## Files Reference

| File | Purpose |
|------|---------|
| `docker-compose.n8n.yml` | Docker Compose for n8n stack |
| `trading_system/api/server.py` | FastAPI server for n8n |
| `trading_system/api/__init__.py` | API module init |
| `pyproject.toml` | Contains `[n8n]` extras |

## Next Steps

1. Start with the Quick Start section above
2. Create a simple scheduled backtest workflow
3. Add database storage for results
4. Set up alerting (Slack, email)
5. Build parameter sweep workflows
6. Create dashboards from stored data

## Related Documentation

- [Docker Setup Guide](../DOCKER_SETUP.md)
- [Configuration Guide](../agent-files/02_CONFIGS_AND_PARAMETERS.md)
- [CLI Commands](../agent-files/15_CLI_COMMANDS.md)
- [Testing Guide](../TESTING_GUIDE.md)
