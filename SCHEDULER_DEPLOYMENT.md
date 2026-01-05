# Scheduler Deployment Guide

This guide explains how to deploy the Trading System Scheduler as a system service.

## Prerequisites

- Python 3.9+ installed
- Trading system installed in `/opt/trading-system` (or adjust paths accordingly)
- Virtual environment created and activated
- Environment variables configured (API keys, email settings, etc.)

## Linux (systemd)

### 1. Install the Service File

Copy the service file to the systemd directory:

```bash
sudo cp trading-system-scheduler.service /etc/systemd/system/
```

### 2. Edit the Service File

Edit `/etc/systemd/system/trading-system-scheduler.service` and update:

- `User` and `Group` - Set to your user/group
- `WorkingDirectory` - Path to your trading system installation
- `ExecStart` - Path to your Python executable
- `Environment` variables - Add your API keys and email configuration
- `ReadWritePaths` - Adjust paths as needed

### 3. Reload systemd and Start Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable trading-system-scheduler.service

# Start the service
sudo systemctl start trading-system-scheduler.service

# Check status
sudo systemctl status trading-system-scheduler.service

# View logs
sudo journalctl -u trading-system-scheduler.service -f
```

### 4. Service Management

```bash
# Stop the service
sudo systemctl stop trading-system-scheduler.service

# Restart the service
sudo systemctl restart trading-system-scheduler.service

# Disable auto-start on boot
sudo systemctl disable trading-system-scheduler.service
```

## macOS (launchd)

### 1. Install the Service File

Copy the plist file to LaunchAgents directory:

```bash
cp com.trading-system.scheduler.plist ~/Library/LaunchAgents/
```

Or for system-wide installation (requires root):

```bash
sudo cp com.trading-system.scheduler.plist /Library/LaunchDaemons/
```

### 2. Edit the Plist File

Edit the plist file and update:

- `ProgramArguments` - Paths to your Python executable
- `WorkingDirectory` - Path to your trading system installation
- `EnvironmentVariables` - Add your API keys and email configuration
- Log file paths

### 3. Load and Start Service

```bash
# Load the service
launchctl load ~/Library/LaunchAgents/com.trading-system.scheduler.plist

# Start the service
launchctl start com.trading-system.scheduler

# Check status
launchctl list | grep trading-system
```

### 4. Service Management

```bash
# Stop the service
launchctl stop com.trading-system.scheduler

# Unload the service
launchctl unload ~/Library/LaunchAgents/com.trading-system.scheduler.plist

# View logs
tail -f /opt/trading-system/logs/scheduler.out.log
tail -f /opt/trading-system/logs/scheduler.err.log
```

## Environment Variables

The scheduler requires the following environment variables:

### Required

- `EMAIL_RECIPIENTS` - Comma-separated list of email addresses
- `SMTP_PASSWORD` or `SENDGRID_API_KEY` - SMTP password/API key

### Optional

- `MASSIVE_API_KEY` - Massive API key for data fetching (formerly Polygon.io)
- `ALPHA_VANTAGE_API_KEY` - Alpha Vantage API key
- `SMTP_HOST` - SMTP server (default: smtp.sendgrid.net)
- `SMTP_PORT` - SMTP port (default: 587)
- `SMTP_USER` - SMTP username (default: apikey)
- `FROM_EMAIL` - Sender email address
- `FROM_NAME` - Sender display name
- `LOG_LEVEL` - Logging level (default: INFO)
- `DATA_CACHE_PATH` - Path to data cache directory
- `CACHE_TTL_HOURS` - Cache TTL in hours (default: 24)

## Testing

Before deploying as a service, test the scheduler manually:

```bash
# Test scheduler startup
python -m trading_system run-scheduler

# Test signal generation
python -m trading_system run-signals-now --asset-class equity
python -m trading_system run-signals-now --asset-class crypto

# Test email configuration
python -m trading_system send-test-email
```

## Troubleshooting

### Service won't start

1. Check service status and logs:
   - Linux: `sudo systemctl status trading-system-scheduler.service`
   - macOS: `launchctl list | grep trading-system`

2. Verify paths in service file are correct

3. Check environment variables are set correctly

4. Verify Python and dependencies are installed

### Scheduler not running jobs

1. Check scheduler logs for errors

2. Verify timezone settings in `CronRunner`

3. Ensure APScheduler is installed: `pip install apscheduler`

4. Test job execution manually with `run-signals-now`

### Email not sending

1. Test email configuration: `python -m trading_system send-test-email`

2. Verify SMTP credentials are correct

3. Check firewall/network settings

4. Review email service logs

## Security Considerations

1. **API Keys**: Store API keys securely, consider using a secrets manager
2. **File Permissions**: Ensure service files have appropriate permissions (600 for service files)
3. **User Permissions**: Run service as non-root user when possible
4. **Log Files**: Secure log file locations and permissions
5. **Network**: Restrict network access if possible

## Monitoring

Monitor the scheduler service:

- **Linux**: Use `journalctl` or systemd status
- **macOS**: Monitor log files and use `launchctl list`
- **Application**: Check application logs in configured log directory

Set up alerts for:
- Service failures
- Job execution errors
- Email delivery failures
