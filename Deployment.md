# Deployment Guide

This guide covers deploying the Trading System to AWS and enabling Claude integration via MCP (Model Context Protocol) server.

## Quick Start

**For dashboard deployment to cloud platforms (Streamlit Cloud, Railway, Render, etc.):**
See [DASHBOARD_DEPLOYMENT.md](DASHBOARD_DEPLOYMENT.md) for detailed platform-specific guides.

**For local development and testing:**
```bash
# 1. Set up MCP server locally
./scripts/setup_mcp_server.sh

# 2. Start MCP server
python -m uvicorn mcp_server.server:app --reload --host 0.0.0.0 --port 8000

# 3. Test the server
./scripts/test_mcp_server.sh
# Or visit http://localhost:8000/docs for interactive API docs
```

**For AWS deployment:**
1. Follow [Option 1: EC2 with Docker](#option-1-ec2-with-docker-recommended-for-development) (easiest)
2. Or [Option 2: ECS Fargate](#option-2-ecs-fargate-recommended-for-production) (production-ready)

**For Claude integration:**
- See [Claude Integration (MCP Server)](#claude-integration-mcp-server) section
- The MCP server exposes REST API endpoints that Claude can call
- View API documentation at `http://<your-server>:8000/docs`

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Docker Build](#local-docker-build)
3. [AWS Deployment Options](#aws-deployment-options)
   - [Option 1: EC2 with Docker (Recommended for Development)](#option-1-ec2-with-docker-recommended-for-development)
   - [Option 2: ECS Fargate (Recommended for Production)](#option-2-ecs-fargate-recommended-for-production)
   - [Option 3: ECS EC2 Launch Type](#option-3-ecs-ec2-launch-type)
4. [Claude Integration (MCP Server)](#claude-integration-mcp-server)
5. [Dashboard/GUI Deployment](#dashboardgui-deployment)
6. [Security Considerations](#security-considerations)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Development Machine

- **Docker Desktop** or **Docker Engine** (v20.10+)
- **Docker Compose** (v2.0+)
- **AWS CLI** (v2.0+) - [Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- **AWS Account** with appropriate IAM permissions
- **Git** for cloning the repository

### AWS Account Setup

1. **Create an AWS Account** (if you don't have one)
2. **Configure AWS CLI**:
   ```bash
   aws configure
   # Enter your AWS Access Key ID
   # Enter your AWS Secret Access Key
   # Enter default region (e.g., us-east-1)
   # Enter default output format (json)
   ```

3. **Required IAM Permissions**:
   - EC2: Launch instances, create security groups, manage key pairs
   - ECS: Create clusters, tasks, services (if using ECS)
   - ECR: Push/pull images (if using ECR)
   - VPC: Create/manage VPC resources (if needed)
   - IAM: Create roles and policies (for service roles)

---

## Local Docker Build

### Build Docker Image Locally

```bash
# Navigate to project directory
cd trade-test

# Build the Docker image
make docker-build

# Or using docker-compose directly
docker-compose build

# Verify the image was built
docker images | grep trading-system
```

### Test Docker Image Locally

```bash
# Run unit tests
make docker-test-unit

# Run a sample backtest
docker-compose run --rm trading-system backtest \
    --config /app/tests/fixtures/configs/run_test_config.yaml \
    --period train
```

---

## AWS Deployment Options

### Option 1: EC2 with Docker (Recommended for Development)

This option is the simplest and most straightforward for development and testing.

#### Step 1: Create EC2 Instance

```bash
# Create a key pair (if you don't have one)
aws ec2 create-key-pair --key-name trading-system-key --query 'KeyMaterial' --output text > ~/.ssh/trading-system-key.pem
chmod 400 ~/.ssh/trading-system-key.pem

# Create security group
aws ec2 create-security-group \
    --group-name trading-system-sg \
    --description "Security group for Trading System" \
    --query 'GroupId' --output text

# Get your public IP
MY_IP=$(curl -s https://checkip.amazonaws.com)

# Allow SSH from your IP
aws ec2 authorize-security-group-ingress \
    --group-name trading-system-sg \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32

# Allow HTTP/HTTPS for MCP server (adjust ports as needed)
aws ec2 authorize-security-group-ingress \
    --group-name trading-system-sg \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0  # Restrict this in production!

# Launch EC2 instance (Ubuntu 22.04 LTS, t3.medium recommended)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --key-name trading-system-key \
    --security-groups trading-system-sg \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=trading-system}]' \
    --query 'Instances[0].InstanceId' --output text

# Get instance public IP (wait ~30 seconds for instance to start)
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=trading-system" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text)

INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Instance IP: $INSTANCE_IP"
```

#### Step 2: Configure EC2 Instance

```bash
# SSH into the instance
ssh -i ~/.ssh/trading-system-key.pem ubuntu@$INSTANCE_IP

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git
sudo apt-get install -y git

# Log out and log back in for docker group to take effect
exit
ssh -i ~/.ssh/trading-system-key.pem ubuntu@$INSTANCE_IP
```

#### Step 3: Deploy Application

```bash
# Clone repository (or use your preferred method)
git clone <your-repo-url> trade-test
cd trade-test

# Or transfer files using SCP from local machine:
# scp -i ~/.ssh/trading-system-key.pem -r . ubuntu@$INSTANCE_IP:~/trade-test
# ssh -i ~/.ssh/trading-system-key.pem ubuntu@$INSTANCE_IP
# cd ~/trade-test

# Build Docker image
docker-compose build

# Start services (see docker-compose.prod.yml section below)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify containers are running
docker ps

# View logs
docker-compose logs -f
```

#### Step 4: Create Production Docker Compose File

Create `docker-compose.prod.yml` for production:

```yaml
services:
  trading-system:
    restart: unless-stopped
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    # Remove volumes that point to host files in production
    # Instead, copy data/configs into image or use EFS/S3
    volumes: []

  mcp-server:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    image: trading-system-mcp:latest
    container_name: trading-system-mcp
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - MCP_HOST=0.0.0.0
      - MCP_PORT=8000
    depends_on:
      - trading-system
    restart: unless-stopped
    volumes:
      - ./data:/app/data:ro
      - ./configs:/app/configs:ro
      - ./results:/app/results
```

#### Step 5: Set Up Systemd Service (Optional)

Create `/etc/systemd/system/trading-system.service`:

```ini
[Unit]
Description=Trading System Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/trade-test
ExecStart=/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable trading-system
sudo systemctl start trading-system
sudo systemctl status trading-system
```

---

### Option 2: ECS Fargate (Recommended for Production)

This option provides managed container orchestration without managing EC2 instances.

#### Step 1: Push Docker Image to ECR

```bash
# Get AWS account ID and region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)

# Create ECR repository
aws ecr create-repository --repository-name trading-system --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag and push image
docker tag trading-system:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/trading-system:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/trading-system:latest
```

#### Step 2: Create ECS Cluster

```bash
# Create Fargate cluster
aws ecs create-cluster --cluster-name trading-system-cluster --region $AWS_REGION
```

#### Step 3: Create Task Definition

Create `ecs-task-definition.json`:

```json
{
  "family": "trading-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "trading-system",
      "image": "<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/trading-system:latest",
      "essential": true,
      "environment": [
        {"name": "PYTHONPATH", "value": "/app"},
        {"name": "PYTHONUNBUFFERED", "value": "1"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/trading-system",
          "awslogs-region": "<REGION>",
          "awslogs-stream-prefix": "ecs"
        }
      }
    },
    {
      "name": "mcp-server",
      "image": "<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/trading-system-mcp:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "PYTHONPATH", "value": "/app"},
        {"name": "MCP_HOST", "value": "0.0.0.0"},
        {"name": "MCP_PORT", "value": "8000"}
      ],
      "dependsOn": [
        {
          "containerName": "trading-system",
          "condition": "START"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/trading-system-mcp",
          "awslogs-region": "<REGION>",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register task definition:

```bash
# Replace placeholders in JSON file first
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

#### Step 4: Create CloudWatch Log Groups

```bash
aws logs create-log-group --log-group-name /ecs/trading-system --region $AWS_REGION
aws logs create-log-group --log-group-name /ecs/trading-system-mcp --region $AWS_REGION
```

#### Step 5: Create ECS Service

```bash
# Get default VPC and subnets
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[*].SubnetId' --output text | tr '\t' ',')

# Create security group
SG_ID=$(aws ec2 create-security-group \
    --group-name trading-system-ecs-sg \
    --description "Security group for Trading System ECS" \
    --vpc-id $VPC_ID \
    --query 'GroupId' --output text)

# Allow inbound traffic on port 8000
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Create IAM role for ECS tasks (if not exists)
# See AWS documentation for creating ecsTaskExecutionRole

# Create service
aws ecs create-service \
    --cluster trading-system-cluster \
    --service-name trading-system-service \
    --task-definition trading-system \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}"
```

---

### Option 3: ECS EC2 Launch Type

Similar to Fargate but requires managing EC2 instances. See AWS ECS documentation for details.

---

## Claude Integration (MCP Server)

To enable Claude to connect to your trading system, you need to set up an MCP (Model Context Protocol) server that exposes the trading system's functionality via HTTP.

### Option A: Simple FastAPI MCP Server (Recommended)

#### Step 1: Create MCP Server

Create `mcp_server/server.py`:

```python
"""MCP Server for Trading System."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import subprocess
import json
import os
from pathlib import Path

app = FastAPI(title="Trading System MCP Server", version="1.0.0")

BASE_DIR = Path("/app")


class BacktestRequest(BaseModel):
    config_path: str
    period: str = "train"  # train, validation, holdout


class ValidateRequest(BaseModel):
    config_path: str


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run a backtest."""
    config_full_path = BASE_DIR / request.config_path.lstrip("/")

    if not config_full_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {request.config_path}")

    try:
        result = subprocess.run(
            [
                "python", "-m", "trading_system", "backtest",
                "--config", str(config_full_path),
                "--period", request.period
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(BASE_DIR)
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Backtest failed: {result.stderr}"
            )

        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Backtest timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def run_validation(request: ValidateRequest):
    """Run validation suite."""
    config_full_path = BASE_DIR / request.config_path.lstrip("/")

    if not config_full_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {request.config_path}")

    try:
        result = subprocess.run(
            [
                "python", "-m", "trading_system", "validate",
                "--config", str(config_full_path)
            ],
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(BASE_DIR)
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Validation failed: {result.stderr}"
            )

        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Validation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/configs")
async def list_configs():
    """List available configuration files."""
    config_dirs = [
        BASE_DIR / "configs",
        BASE_DIR / "EXAMPLE_CONFIGS",
        BASE_DIR / "tests" / "fixtures" / "configs"
    ]

    configs = []
    for config_dir in config_dirs:
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                configs.append({
                    "name": config_file.name,
                    "path": str(config_file.relative_to(BASE_DIR))
                })

    return {"configs": configs}


@app.get("/results/{run_id}")
async def get_results(run_id: str, period: Optional[str] = None):
    """Get backtest results."""
    results_dir = BASE_DIR / "results" / run_id

    if not results_dir.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run_id: {run_id}")

    if period:
        results_dir = results_dir / period

    if not results_dir.exists():
        raise HTTPException(status_code=404, detail=f"Period not found: {period}")

    results = {}
    for result_file in results_dir.glob("*.json"):
        with open(result_file, "r") as f:
            results[result_file.stem] = json.load(f)

    return {"run_id": run_id, "period": period, "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Step 2: Create MCP Server Dockerfile

Create `Dockerfile.mcp`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy MCP server code
COPY mcp_server/ ./mcp_server/

# Copy trading system (or mount as volume)
COPY trading_system/ ./trading_system/
COPY pyproject.toml .
COPY pytest.ini .

# Install trading system
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Run MCP server
CMD ["python", "-m", "uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 3: Create MCP Server Package Structure

Create the directory structure:

```bash
mkdir -p mcp_server
touch mcp_server/__init__.py
```

#### Step 4: Update Requirements

Add to `requirements.txt` (if not already present):

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
```

#### Step 5: Build and Run MCP Server

```bash
# Build MCP server image
docker build -f Dockerfile.mcp -t trading-system-mcp:latest .

# Or add to docker-compose.yml
# Run locally
docker run -p 8000:8000 \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/configs:/app/configs:ro \
    -v $(pwd)/results:/app/results \
    trading-system-mcp:latest

# Test the server
curl http://localhost:8000/health
curl http://localhost:8000/configs
```

### Option B: Claude Desktop Integration

To connect Claude Desktop to your MCP server, add this to Claude Desktop's MCP configuration:

#### macOS Configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "trading-system": {
      "command": "curl",
      "args": [
        "-X", "POST",
        "http://<YOUR_SERVER_IP>:8000/backtest",
        "-H", "Content-Type: application/json",
        "-d", "@-"
      ],
      "env": {
        "TRADING_SYSTEM_API_URL": "http://<YOUR_SERVER_IP>:8000"
      }
    }
  }
}
```

#### Direct HTTP API Usage

Claude can also interact with the MCP server via HTTP API calls. The FastAPI server provides OpenAPI documentation at:

- `http://<YOUR_SERVER_IP>:8000/docs` - Interactive API documentation
- `http://<YOUR_SERVER_IP>:8000/openapi.json` - OpenAPI specification

---

## Dashboard/GUI Deployment

**Answer: You do NOT need a separate host for the GUI.** The Streamlit dashboard can run on the same server/host as your trading system.

Your trading system includes two Streamlit dashboards:
1. **Backtest Results Dashboard** - Visualizes backtest results
2. **Trading Assistant Dashboard** - Live signals, portfolio, news, and performance

### Option 1: Same Docker Compose Setup (Recommended)

Add the dashboard as a service in your `docker-compose.prod.yml`:

```yaml
services:
  # ... existing services ...

  dashboard:
    build:
      context: .
      dockerfile: trading_system/dashboard/Dockerfile
    image: trading-system-dashboard:latest
    container_name: trading-system-dashboard
    restart: unless-stopped
    ports:
      - "8501:8501"  # Streamlit default port
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
      - TRACKING_DB_PATH=/app/data/tracking.db
      - FEATURE_DB_PATH=/app/data/features.db
      # Add API keys if needed
      - MASSIVE_API_KEY=${MASSIVE_API_KEY:-}
      - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY:-}
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs:ro
      - ./results:/app/results
      - ./tests/fixtures:/app/tests/fixtures:ro
    depends_on:
      - trading-system
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Benefits:**
- Same host/server
- Shared volumes (data, results, configs)
- Single docker-compose management
- Lower infrastructure costs

### Option 2: Separate Container on Same EC2 Instance

If you prefer to run the dashboard separately:

```bash
# On your EC2 instance
docker run -d \
  --name trading-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/configs:/app/configs:ro \
  trading-system-dashboard:latest
```

### Option 3: Separate Host (Optional, Not Required)

You can run the dashboard on a separate host if you prefer, but you'll need to:
- Mount/share data and results directories (via NFS, S3, or similar)
- Configure network access between hosts
- Manage two separate deployments

**This is typically only needed if:**
- You have very high traffic and need to scale the dashboard separately
- You want strict network isolation
- You have specific compliance/security requirements

### Accessing the Dashboard

Once deployed:

1. **Backtest Results Dashboard**:
   ```bash
   # Via CLI (inside container or locally)
   python -m trading_system dashboard --run-id <run_id>

   # Or directly via Streamlit
   streamlit run trading_system/reporting/dashboard.py -- --base_path results --run_id <run_id>
   ```

2. **Trading Assistant Dashboard**:
   ```bash
   # Via CLI
   python -m trading_system trading-dashboard

   # Or directly via Streamlit
   streamlit run trading_system/dashboard/app.py
   ```

3. **Via Browser**:
   - Local: `http://localhost:8501`
   - AWS: `http://<your-ec2-ip>:8501`
   - With domain: `http://yourdomain.com:8501`

### Security for Dashboard

Update security group to allow dashboard access:

```bash
# Allow HTTP access on port 8501 (restrict IPs in production!)
aws ec2 authorize-security-group-ingress \
    --group-name trading-system-sg \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0  # Restrict this to your IP in production!
```

**For Production:**
- Use HTTPS via reverse proxy (nginx/traefik)
- Enable dashboard password protection (configured in dashboard settings)
- Restrict access via security groups to known IPs
- Consider VPN access instead of public exposure

### Reverse Proxy Setup (Production)

For production, use nginx as a reverse proxy:

```nginx
# /etc/nginx/sites-available/trading-dashboard
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Then enable HTTPS with Let's Encrypt:
```bash
sudo certbot --nginx -d yourdomain.com
```

---

## Security Considerations

### 1. Network Security

- **Restrict Access**: Use security groups to restrict access to known IPs only
- **HTTPS**: Set up SSL/TLS using a reverse proxy (nginx, traefik) or AWS ALB
- **VPN**: Consider using AWS VPN or private networking for sensitive deployments

### 2. Authentication & Authorization

Add authentication to the MCP server:

```python
# Add to mcp_server/server.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Implement token validation
    if token != os.getenv("MCP_API_TOKEN", "your-secret-token"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

@app.post("/backtest")
async def run_backtest(request: BacktestRequest, token: str = Depends(verify_token)):
    # ... existing code
```

### 3. Data Security

- **Secrets Management**: Use AWS Secrets Manager or Parameter Store for API keys
- **Encryption**: Enable encryption at rest for EBS volumes and S3 buckets
- **Backup**: Regularly backup configuration and results

### 4. Resource Limits

- Set appropriate CPU and memory limits in Docker/ECS
- Implement rate limiting on the MCP server
- Set timeouts for long-running operations

### 5. Logging and Monitoring

- Enable CloudWatch Logs for ECS tasks
- Set up CloudWatch Alarms for errors
- Monitor resource usage (CPU, memory, disk)

---

## Troubleshooting

### Docker Issues

```bash
# Check container logs
docker-compose logs trading-system
docker-compose logs mcp-server

# Restart services
docker-compose restart

# Rebuild images
docker-compose build --no-cache
```

### Network Connectivity

```bash
# Test MCP server endpoint
curl http://localhost:8000/health

# Check if port is open (from EC2)
sudo netstat -tlnp | grep 8000

# Check security group rules
aws ec2 describe-security-groups --group-names trading-system-sg
```

### ECS Issues

```bash
# Check task status
aws ecs describe-tasks --cluster trading-system-cluster --tasks <task-id>

# Check service events
aws ecs describe-services --cluster trading-system-cluster --services trading-system-service

# View logs
aws logs tail /ecs/trading-system --follow
```

### MCP Server Issues

```bash
# Test server locally
python -m uvicorn mcp_server.server:app --reload --host 0.0.0.0 --port 8000

# Check API documentation
open http://localhost:8000/docs
```

### Common Errors

1. **"Connection refused"**: Check security group rules and port configuration
2. **"Config file not found"**: Verify volume mounts and file paths
3. **"Permission denied"**: Check file permissions and Docker user/group
4. **"Out of memory"**: Increase container memory limits

---

## Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Claude Desktop MCP Documentation](https://claude.ai/docs/mcp)
- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)

---

## Next Steps

1. **Set up CI/CD**: Automate deployments using GitHub Actions or AWS CodePipeline
2. **Monitoring**: Set up CloudWatch dashboards and alarms
3. **Scaling**: Configure auto-scaling for ECS services
4. **Backup**: Implement automated backups for results and configurations
5. **Documentation**: Document your specific configuration and workflows
