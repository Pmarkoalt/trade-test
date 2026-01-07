# News API Setup Guide

This guide explains how to set up API keys for live news sentiment analysis.

## Required API Keys

The trading system supports two news sources. **At least one is required**, but both are recommended for better coverage.

### 1. NewsAPI.org (Recommended)

**Pros**: Broad coverage, multiple sources, free tier available
**Cons**: Free tier limited to 100 requests/day, historical data limited

**Setup**:
1. Go to [https://newsapi.org/register](https://newsapi.org/register)
2. Create a free account
3. Copy your API key from the dashboard

**Free Tier Limits**:
- 100 requests per day
- Articles from last month only
- No commercial use

### 2. Alpha Vantage News API (Recommended)

**Pros**: Pre-computed sentiment scores, financial-focused, ticker-specific news
**Cons**: Rate limited (5 requests/minute on free tier)

**Setup**:
1. Go to [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Fill out the form to get a free API key
3. Copy your API key from the confirmation email

**Free Tier Limits**:
- 5 API calls per minute
- 500 API calls per day

## Environment Configuration

### Option 1: Export Variables (Temporary)

```bash
export NEWSAPI_KEY='your_newsapi_key_here'
export ALPHA_VANTAGE_API_KEY='your_alphavantage_key_here'
```

### Option 2: Create .env File (Recommended)

Create a `.env` file in the project root:

```bash
# News API Keys
NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here
```

Then load it in your shell:
```bash
source .env
# Or use python-dotenv in your scripts
```

### Option 3: Add to Shell Profile (Persistent)

Add to `~/.bashrc`, `~/.zshrc`, or equivalent:

```bash
# Trading System API Keys
export NEWSAPI_KEY='your_newsapi_key_here'
export ALPHA_VANTAGE_API_KEY='your_alphavantage_key_here'
```

## Testing Your Setup

### Quick Test

```bash
# Test single equity
python scripts/test_news_signal.py AAPL

# Test with verbose output
python scripts/test_news_signal.py AAPL --verbose

# Compare multiple symbols
python scripts/test_news_signal.py AAPL MSFT GOOGL NVDA --compare
```

### Expected Output

```
News Sources: NewsAPI.org, Alpha Vantage
Lookback: 48 hours
Max Articles: 10

Fetching news for AAPL...

============================================================
NEWS SENTIMENT ANALYSIS: AAPL
============================================================
╭─────────────────────────────────────────╮
│ Sentiment Metrics                       │
├─────────────────────────────────────────┤
│ Sentiment Score    0.234                │
│ Sentiment Label    POSITIVE             │
│ Articles Analyzed  8                    │
│ Positive Articles  5                    │
│ Negative Articles  2                    │
│ Neutral Articles   1                    │
│ Sentiment Trend    improving            │
│ Most Recent        2.3 hours ago        │
╰─────────────────────────────────────────╯

Top Headlines:
  1. Apple reports strong iPhone sales in emerging markets
  2. Analysts raise AAPL price targets ahead of earnings
  3. Apple Vision Pro shipments exceed expectations

Trading Signal Score: 6.8/10 (NEUTRAL)
Reasoning: Positive news sentiment (8 articles) with improving trend
```

## Troubleshooting

### "No API keys found"

Ensure at least one environment variable is set:
```bash
echo $NEWSAPI_KEY
echo $ALPHA_VANTAGE_API_KEY
```

### "Rate limit exceeded"

- **NewsAPI**: Wait until the next day (100 requests/day limit)
- **Alpha Vantage**: Wait 1 minute between requests (5/minute limit)

The system automatically respects rate limits, but if you run many tests quickly, you may hit them.

### "No recent news articles found"

- Try increasing `--lookback` to 72 or 96 hours
- Some symbols may have limited news coverage
- Check if your API keys are valid

### "vaderSentiment not installed"

```bash
pip install vaderSentiment
# Or install with research dependencies
pip install -e ".[research]"
```

### "aiohttp not installed"

```bash
pip install aiohttp
```

## Integration with Trading System

### In Backtest Config

For backtesting, use synthetic sentiment (no API keys needed):

```yaml
# configs/equity_strategy_with_sentiment.yaml
sentiment:
  enabled: true
  mode: "combined"
  weight: 0.15
```

### In Live Signal Generation

For live trading, configure in your signal config:

```yaml
# configs/signal_config.yaml
signals:
  news_enabled: true
  news_lookback_hours: 48
  news_weight: 0.4
  technical_weight: 0.6
```

### Programmatic Usage

```python
import asyncio
from trading_system.research.config import ResearchConfig
from trading_system.research.news_analyzer import NewsAnalyzer

async def get_sentiment(symbol: str):
    config = ResearchConfig(
        newsapi_key="your_key",
        alpha_vantage_key="your_key",
        lookback_hours=48,
    )

    analyzer = NewsAnalyzer(config)
    result = await analyzer.analyze_symbols([symbol])

    score, reasoning = analyzer.get_news_score_for_signal(symbol, result)
    return score, reasoning

# Run it
score, reason = asyncio.run(get_sentiment("AAPL"))
print(f"Score: {score}/10 - {reason}")
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Add `.env` to `.gitignore`**
3. **Use environment variables** in production
4. **Rotate keys periodically** if exposed
5. **Monitor usage** to detect unauthorized access

## API Cost Comparison

| Provider | Free Tier | Paid Plans |
|----------|-----------|------------|
| NewsAPI | 100 req/day | From $449/mo |
| Alpha Vantage | 500 req/day | From $49.99/mo |

For most backtesting and development needs, free tiers are sufficient.
