# Newsletter Module - Multi-Bucket Daily Trading Signals

## Overview

The newsletter module provides automated daily email newsletters with trading signals organized by strategy buckets. This is part of Agent 3's implementation for the PROJECT_NEXT_STEPS.md roadmap.

## Features

- **Multi-Bucket Organization**: Signals grouped by strategy buckets (Safe S&P, Crypto Top-Cap, etc.)
- **HTML Email Templates**: Professional, responsive email templates with Jinja2
- **Market Summary**: SPY and BTC price updates with market regime detection
- **News Integration**: Sentiment-based news digest with positive/negative categorization
- **Action Items**: Clear "What to Buy" and "What to Avoid" sections
- **Scheduled Delivery**: Automated daily newsletter via cron scheduler

## Architecture

### Components

1. **NewsletterGenerator** (`newsletter_generator.py`)
   - Generates newsletter context from signals
   - Formats signals for email display
   - Creates plain text summaries

2. **NewsletterService** (`newsletter_service.py`)
   - Orchestrates newsletter generation and delivery
   - Integrates with EmailService for SMTP
   - Handles both test and production newsletters

3. **Newsletter Job** (`scheduler/jobs/newsletter_job.py`)
   - Generates signals for all configured buckets
   - Fetches market data and news
   - Sends combined newsletter

4. **Email Templates** (`templates/newsletter_daily.html`)
   - Multi-bucket signal display
   - Market overview section
   - News digest with sentiment
   - Action items summary

## Usage

### CLI Commands

#### Send Test Newsletter
```bash
# Send test newsletter with mock data
python -m trading_system send-newsletter --test
# or
python -m trading_system newsletter --test
```

#### Run Newsletter Job
```bash
# Generate signals and send newsletter
python -m trading_system run-newsletter-job
# or
python -m trading_system newsletter-job
```

#### Run Scheduler (includes newsletter)
```bash
# Start scheduler daemon (includes daily newsletter at 5 PM ET)
python -m trading_system run-scheduler
```

### Programmatic Usage

```python
import asyncio
from trading_system.output.email.config import EmailConfig
from trading_system.output.email.newsletter_service import NewsletterService
from trading_system.models.signals import Signal, SignalSide, SignalType

# Configure email
email_config = EmailConfig(
    smtp_host="smtp.sendgrid.net",
    smtp_port=587,
    smtp_user="apikey",
    smtp_password="your_sendgrid_api_key",
    from_email="signals@yourdomain.com",
    from_name="Trading Assistant",
    recipients=["user@example.com"],
)

# Create newsletter service
newsletter_service = NewsletterService(email_config)

# Organize signals by bucket
signals_by_bucket = {
    "safe_sp": [
        Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=150.0,
            stop_price=145.0,
            score=0.85,
            adv20=1000000.0,
        ),
    ],
    "crypto_topCap": [
        Signal(
            symbol="BTC",
            asset_class="crypto",
            date=pd.Timestamp.now(),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="crypto_momentum",
            entry_price=45000.0,
            stop_price=43500.0,
            score=0.78,
            adv20=5000000.0,
        ),
    ],
}

# Send newsletter
success = await newsletter_service.send_daily_newsletter(
    signals_by_bucket=signals_by_bucket,
    market_summary={"spy_price": 450.0, "spy_pct": 0.5},
)
```

## Configuration

### Environment Variables

Required for newsletter functionality:

```bash
# Email Configuration
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASSWORD=your_sendgrid_api_key
FROM_EMAIL=signals@yourdomain.com
FROM_NAME="Trading Assistant"
EMAIL_RECIPIENTS=user1@example.com,user2@example.com

# Data Sources
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
MASSIVE_API_KEY=your_massive_key

# Research (optional)
RESEARCH_ENABLED=true
NEWSAPI_KEY=your_newsapi_key
RESEARCH_LOOKBACK_HOURS=48

# Signal Configuration
MAX_RECOMMENDATIONS=10
MIN_CONVICTION=MEDIUM
NEWS_ENABLED=true
```

### Scheduler Configuration

The newsletter job runs daily at 5:00 PM ET (after market close):

```python
from trading_system.scheduler.cron_runner import CronRunner
from trading_system.scheduler.config import SchedulerConfig

config = SchedulerConfig(enabled=True)
runner = CronRunner(config)
runner.register_jobs()
runner.start()
```

Registered jobs:
- **daily_equity_signals**: 4:30 PM ET
- **daily_crypto_signals**: 12:00 AM UTC
- **daily_newsletter**: 5:00 PM ET

## Strategy Buckets

### Bucket A: Safe S&P (safe_sp)
- **Description**: Safe S&P 500 bets - Low drawdown, realistic capacity
- **Universe**: SP500 or SPY + sector ETFs
- **Focus**: Stable, low-risk equity positions

### Bucket B: Crypto Top-Cap (crypto_topCap)
- **Description**: Aggressive top market cap crypto - Higher turnover
- **Universe**: Dynamic top market cap coins
- **Focus**: Higher volatility, adaptive positioning

### Future Buckets (Planned)
- **Bucket C**: Low-float stock "gamble" (low_float)
- **Bucket D**: Unusual options movement (unusual_options)

## Email Template Structure

The newsletter includes:

1. **Header**: Date and total signal count
2. **Market Overview**: SPY/BTC prices and market regime
3. **Bucket Sections** (for each bucket):
   - Bucket description
   - Buy signals table (symbol, entry, stop, risk %, score, rationale)
   - Sell signals table
4. **News Digest**: Positive/negative sentiment articles
5. **Current Positions**: Portfolio summary (if available)
6. **Action Items**:
   - What to Buy (top picks)
   - What to Avoid (blockers)
7. **Footer**: Timestamp and disclaimer

## Testing

### Unit Tests

```bash
# Run newsletter tests
pytest tests/test_newsletter_generator.py
pytest tests/test_newsletter_service.py
```

### Integration Tests

```bash
# Test newsletter job end-to-end
pytest tests/integration/test_newsletter_job.py
```

### Manual Testing

```bash
# Send test newsletter
python -m trading_system send-newsletter --test

# Check email configuration
python -m trading_system send-test-email
```

## Dependencies

Required packages (included in `pyproject.toml` under `[project.optional-dependencies.live]`):

- `jinja2>=3.1.0` - Email templates
- `apscheduler>=3.10.0` - Job scheduling
- `aiohttp>=3.8.0` - Async HTTP for data fetching

Install with:
```bash
pip install -e ".[live]"
```

## Troubleshooting

### Newsletter not sending

1. **Check email configuration**:
   ```bash
   python -m trading_system send-test-email
   ```

2. **Verify environment variables**:
   ```bash
   echo $EMAIL_RECIPIENTS
   echo $SMTP_PASSWORD
   ```

3. **Check logs**:
   ```bash
   tail -f logs/trading_system.log
   ```

### No signals generated

1. **Verify data sources**:
   ```bash
   python -m trading_system fetch-data --symbols AAPL,MSFT --asset-class equity --days 30
   ```

2. **Check strategy configuration**:
   - Ensure `configs/production_run_config.yaml` exists
   - Verify strategy configs are enabled

3. **Run signal generation manually**:
   ```bash
   python -m trading_system run-signals-now --asset-class equity
   ```

### Template rendering errors

1. **Verify Jinja2 is installed**:
   ```bash
   pip install jinja2
   ```

2. **Check template files exist**:
   ```bash
   ls trading_system/output/email/templates/
   ```

## Integration with Other Agents

### Agent 1 (Integrator)
- Uses canonical `Signal`, `Allocation`, `TradePlan` contracts
- Newsletter consumes `Signal` objects directly

### Agent 2 (Strategy & Data)
- Newsletter job calls strategy signal generation
- Supports multiple buckets (equity, crypto)

### Agent 4 (Paper Trading)
- Newsletter can include portfolio summary from paper trading
- Manual trades can be displayed in "Current Positions"

## Future Enhancements

1. **Personalization**: User-specific signal filtering
2. **Unsubscribe Links**: Email preference management
3. **Mobile Optimization**: Responsive design improvements
4. **Performance Charts**: Embedded performance visualizations
5. **Multi-Language**: i18n support for international users
6. **SMS Alerts**: Optional SMS for high-priority signals
7. **Slack Integration**: Post newsletter to Slack channels

## References

- PROJECT_NEXT_STEPS.md - Phase 2: Daily Newsletter
- trading_system/models/signals.py - Signal data model
- trading_system/scheduler/ - Scheduling infrastructure
- trading_system/output/email/ - Email service components
