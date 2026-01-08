# AWS EC2 Deployment Guide

Deploy the Trading System on AWS EC2 for production backtesting, optimization, and scheduled trading operations.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                             │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   EC2 Instance  │    │   S3 Bucket     │                 │
│  │  (t3.xlarge)    │◄──►│  (data/results) │                 │
│  │                 │    └─────────────────┘                 │
│  │  - Trading App  │                                        │
│  │  - Scheduler    │    ┌─────────────────┐                 │
│  │  - ML Models    │◄──►│  CloudWatch     │                 │
│  └─────────────────┘    │  (logs/alerts)  │                 │
│           │             └─────────────────┘                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  Secrets Manager│                                        │
│  │  (API keys)     │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

## Recommended Instance Types

| Use Case | Instance | vCPUs | RAM | Cost/month |
|----------|----------|-------|-----|------------|
| **Development** | t3.medium | 2 | 4GB | ~$30 |
| **Optimization** | t3.xlarge | 4 | 16GB | ~$120 |
| **Production** | c6i.2xlarge | 8 | 16GB | ~$250 |

**Recommendation**: Start with `t3.xlarge` for optimization runs, scale down to `t3.medium` for daily operations.

---

## Quick Start Deployment

### Step 1: Launch EC2 Instance

```bash
# Using AWS CLI (or use Console)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxx \
    --subnet-id subnet-xxxxxxxx \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=trading-system}]'
```

### Step 2: Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<instance-ip>

# Update system
sudo yum update -y

# Install dependencies
sudo yum install -y python3.11 python3.11-pip git docker

# Clone repository
git clone https://github.com/your-repo/trade-test.git
cd trade-test

# Setup Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env
nano .env
```

Add your API keys:
```bash
# .env file
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
POLYGON_API_KEY=your_polygon_key
ALPHA_VANTAGE_API_KEY=your_av_key

# AWS Configuration (for S3 backup)
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=your-trading-bucket
```

### Step 4: Run Initial Test

```bash
# Test the installation
python -m trading_system --help

# Run a quick backtest
python -m trading_system backtest \
    --config configs/test_equity_strategy.yaml \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

---

## Production Setup

### Systemd Service (Auto-restart)

Create `/etc/systemd/system/trading-scheduler.service`:

```ini
[Unit]
Description=Trading System Scheduler
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/trade-test
Environment=PATH=/home/ec2-user/trade-test/.venv/bin
ExecStart=/home/ec2-user/trade-test/.venv/bin/python -m trading_system scheduler start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-scheduler
sudo systemctl start trading-scheduler
```

### CloudWatch Logging

Install CloudWatch agent:
```bash
sudo yum install -y amazon-cloudwatch-agent

# Configure
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

CloudWatch config (`/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json`):
```json
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/home/ec2-user/trade-test/logs/*.log",
            "log_group_name": "trading-system",
            "log_stream_name": "{instance_id}/app"
          }
        ]
      }
    }
  }
}
```

### S3 Backup for Results

Add to crontab (`crontab -e`):
```bash
# Backup results to S3 daily at midnight
0 0 * * * aws s3 sync /home/ec2-user/trade-test/results s3://your-bucket/results/
0 0 * * * aws s3 sync /home/ec2-user/trade-test/models s3://your-bucket/models/
```

---

## Security Best Practices

### Security Group Rules

| Type | Port | Source | Purpose |
|------|------|--------|---------|
| SSH | 22 | Your IP only | Admin access |
| HTTPS | 443 | 0.0.0.0/0 | API calls out |

### IAM Role Permissions

Attach to EC2 instance:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-trading-bucket",
        "arn:aws:s3:::your-trading-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:trading-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

### Store API Keys in Secrets Manager

```bash
# Create secret
aws secretsmanager create-secret \
    --name trading-api-keys \
    --secret-string '{"ALPACA_API_KEY":"xxx","ALPACA_SECRET_KEY":"xxx","POLYGON_API_KEY":"xxx"}'
```

Load in application:
```python
import boto3
import json

def get_secrets():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='trading-api-keys')
    return json.loads(response['SecretString'])
```

---

## Running Optimizations on EC2

### Start Optimization in Background

```bash
# Use screen or tmux for long-running jobs
screen -S optimization

# Run optimization
./scripts/run_optimization_equity.sh

# Detach: Ctrl+A, D
# Reattach: screen -r optimization
```

### Monitor Progress

```bash
# View logs
tail -f logs/equity_optimization_*.log

# Check CPU usage
htop

# Check trials completed
sqlite3 optimization_results/*.db "SELECT COUNT(*) FROM trials"
```

---

## Estimated Costs

| Resource | Monthly Cost |
|----------|-------------|
| EC2 t3.xlarge (on-demand) | ~$120 |
| EC2 t3.xlarge (spot, 70% savings) | ~$36 |
| S3 Storage (50GB) | ~$1 |
| CloudWatch Logs | ~$5 |
| Data Transfer | ~$10 |
| **Total** | **~$50-140/month** |

### Cost Optimization Tips

1. **Use Spot Instances** for optimization (70% savings)
2. **Schedule start/stop** - only run during market hours
3. **Right-size instance** - t3.medium for daily operations

---

## Troubleshooting

### Common Issues

**1. Out of memory during optimization**
```bash
# Check memory
free -h

# Reduce parallel jobs
./scripts/run_optimization_equity.sh 2
```

**2. Permission denied on logs**
```bash
sudo chown -R ec2-user:ec2-user /home/ec2-user/trade-test
```

**3. API rate limits**
```bash
# Check rate limit status in logs
grep "rate limit" logs/*.log
```

---

## Next Steps

1. ✅ Launch EC2 instance
2. ✅ Configure security groups
3. ✅ Setup IAM role
4. ✅ Clone and configure application
5. ✅ Run test backtest
6. ✅ Setup systemd service
7. ✅ Configure CloudWatch logging
8. ✅ Setup S3 backup
9. ✅ Run optimization
10. ✅ Deploy ML models
