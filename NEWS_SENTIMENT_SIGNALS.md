# News Sentiment Signals Integration

## Overview

This document outlines the plan to integrate news sentiment signals into the trading system for:
1. **Backtesting**: Simulate sentiment data for the 2-year test period (2024-2025)
2. **Live Trading**: Fetch real-world news data via APIs
3. **Validation**: Test signal generation for a single equity asset

---

## Current Infrastructure (Already Implemented)

The system has extensive news/sentiment infrastructure ready to use:

### Data Fetching
| Component | Location | Description |
|-----------|----------|-------------|
| NewsAggregator | `trading_system/data_pipeline/sources/news/news_aggregator.py` | Coordinates multiple news sources |
| NewsAPIClient | `trading_system/data_pipeline/sources/news/newsapi_client.py` | NewsAPI.org integration |
| AlphaVantageNewsClient | `trading_system/data_pipeline/sources/news/alpha_vantage_news.py` | Alpha Vantage NEWS_SENTIMENT API |

### Sentiment Analysis
| Component | Location | Description |
|-----------|----------|-------------|
| VADERSentimentAnalyzer | `trading_system/research/sentiment/vader_analyzer.py` | VADER with financial lexicon |
| FinancialLexicon | `trading_system/research/sentiment/financial_lexicon.py` | 100+ financial terms with custom scores |
| SentimentAggregator | `trading_system/research/sentiment/sentiment_aggregator.py` | Per-symbol weighted aggregation |

### Signal Integration
| Component | Location | Description |
|-----------|----------|-------------|
| LiveSignalGenerator | `trading_system/signals/live_signal_generator.py` | Combines technical + news (60/40 weight) |
| NewsAnalyzer | `trading_system/research/news_analyzer.py` | Main orchestrator for news analysis |

---

## Phase 1: Synthetic Sentiment for Backtesting

### Goal
Generate realistic sentiment data for 2024-2025 to enable backtesting strategies that incorporate news signals.

### Approach: Event-Based Sentiment Simulation

Create synthetic sentiment that correlates with actual market events:

```python
# Sentiment simulation modes:
1. RANDOM_WALK      - Brownian motion around neutral
2. PRICE_CORRELATED - Sentiment follows price movements (lagged)
3. EVENT_BASED      - Major events inject sentiment shocks
4. REGIME_BASED     - Bull/bear market sentiment regimes
```

### Implementation Plan

#### 1.1 Create SyntheticSentimentGenerator

**File**: `trading_system/research/sentiment/synthetic_generator.py`

```python
class SyntheticSentimentGenerator:
    """Generate synthetic sentiment data for backtesting."""

    def __init__(
        self,
        mode: str = "price_correlated",
        correlation_lag: int = 1,  # Days sentiment lags price
        noise_std: float = 0.2,
        event_probability: float = 0.05,  # 5% chance of sentiment shock
        seed: int = 42
    ):
        ...

    def generate_for_symbol(
        self,
        symbol: str,
        dates: List[pd.Timestamp],
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        - date
        - symbol
        - sentiment_score (-1.0 to 1.0)
        - sentiment_label (VERY_NEGATIVE to VERY_POSITIVE)
        - confidence (0.0 to 1.0)
        - article_count (simulated)
        - event_type (None, 'earnings', 'upgrade', 'downgrade', etc.)
        """
        ...
```

#### 1.2 Sentiment Simulation Modes

**Mode 1: Price-Correlated**
- Sentiment = sign(price_change) * abs(price_change) * scale + noise
- 1-day lag (news reacts to price)
- Captures momentum-chasing behavior

**Mode 2: Event-Based**
- Inject sentiment shocks on known event dates:
  - Earnings dates (quarterly)
  - Fed meeting dates
  - Major market events (flash crashes, rallies)
- Random surprise events (5% daily probability)

**Mode 3: Regime-Based**
- Detect bull/bear regimes from MA crossovers
- Bull regime: sentiment bias +0.2
- Bear regime: sentiment bias -0.2

#### 1.3 Historical Event Calendar

Create event calendar for 2024-2025:

**File**: `data/sentiment/event_calendar.csv`

```csv
date,event_type,symbols,sentiment_impact,description
2024-01-25,earnings,MSFT,0.3,Strong cloud growth
2024-01-26,earnings,AAPL,-0.2,iPhone sales miss
2024-02-22,earnings,NVDA,0.5,AI demand surge
2024-03-18,fed_meeting,ALL,-0.1,Rate hold signal
...
```

#### 1.4 Integration with BacktestEngine

Modify `BacktestEngine` to accept synthetic sentiment:

```python
class BacktestEngine:
    def __init__(
        self,
        ...
        sentiment_data: Optional[pd.DataFrame] = None,  # NEW
        sentiment_mode: str = "price_correlated"  # NEW
    ):
        if sentiment_data is None and sentiment_mode:
            self.sentiment_data = self._generate_synthetic_sentiment()
```

---

## Phase 2: Real-World News Fetching

### Goal
Fetch and analyze live news for production signal generation.

### API Configuration

#### Alpha Vantage (Primary)
- Already integrated in `AlphaVantageNewsClient`
- Provides pre-computed sentiment scores
- Rate limit: 5 requests/min (free tier)

**Environment Setup**:
```bash
# .env file
ALPHA_VANTAGE_API_KEY=your_key_here
```

#### NewsAPI.org (Secondary)
- Already integrated in `NewsAPIClient`
- Broader coverage, requires VADER analysis
- Rate limit: 100 requests/day (free tier)

**Environment Setup**:
```bash
# .env file
NEWSAPI_KEY=your_key_here
```

### Live Fetching Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Alpha Vantage   │────▶│ NewsAggregator   │────▶│ SentimentAggr.  │
│ NewsAPI.org     │     │ (Deduplication)  │     │ (Per-symbol)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────┐
                                               │ NewsAnalyzer    │
                                               │ (Score 0-10)    │
                                               └─────────────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────┐
                                               │ SignalGenerator │
                                               │ (60% tech,      │
                                               │  40% news)      │
                                               └─────────────────┘
```

---

## Phase 3: Single Equity Test

### Goal
Validate the full pipeline with a single equity (e.g., AAPL) before expanding.

### Test Script

**File**: `scripts/test_news_signal.py`

```python
#!/usr/bin/env python3
"""Test news sentiment signal for a single equity."""

import asyncio
from datetime import datetime
from trading_system.research.news_analyzer import NewsAnalyzer
from trading_system.research.config import ResearchConfig

async def test_single_equity(symbol: str = "AAPL"):
    """Fetch and analyze news for a single symbol."""

    config = ResearchConfig(
        newsapi_key=os.getenv("NEWSAPI_KEY"),
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        lookback_hours=48,
        max_articles_per_symbol=10
    )

    analyzer = NewsAnalyzer(config)

    # Fetch and analyze
    result = await analyzer.analyze_symbols(
        symbols=[symbol],
        current_date=datetime.now()
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"NEWS SENTIMENT ANALYSIS: {symbol}")
    print(f"{'='*60}")

    if symbol in result.symbol_summaries:
        summary = result.symbol_summaries[symbol]
        print(f"Sentiment Score: {summary.aggregated_sentiment:.3f}")
        print(f"Confidence: {summary.confidence:.2f}")
        print(f"Articles Analyzed: {summary.article_count}")
        print(f"Positive: {summary.positive_count}, Negative: {summary.negative_count}")

        print(f"\nTop Headlines:")
        for i, headline in enumerate(summary.top_headlines[:5], 1):
            print(f"  {i}. {headline}")

    # Get trading signal score (0-10)
    news_score = analyzer.get_news_score_for_signal(symbol)
    print(f"\nTrading Signal Score: {news_score:.1f}/10")

    return result

if __name__ == "__main__":
    asyncio.run(test_single_equity("AAPL"))
```

### Expected Output

```
============================================================
NEWS SENTIMENT ANALYSIS: AAPL
============================================================
Sentiment Score: 0.234
Confidence: 0.78
Articles Analyzed: 8
Positive: 5, Negative: 2

Top Headlines:
  1. Apple reports record iPhone sales in China
  2. Analysts raise AAPL price targets ahead of earnings
  3. Apple Vision Pro pre-orders exceed expectations
  4. Concerns mount over App Store regulatory changes
  5. Apple to expand AI features in iOS 18

Trading Signal Score: 6.2/10
```

---

## Implementation Roadmap

### Step 1: Synthetic Sentiment Generator (Backtesting)
- [ ] Create `SyntheticSentimentGenerator` class
- [ ] Implement price-correlated mode
- [ ] Implement event-based mode with calendar
- [ ] Add event calendar for 2024-2025
- [ ] Integrate with `BacktestEngine`
- [ ] Create backtest config option: `sentiment_mode`

### Step 2: Backtest Validation
- [ ] Run backtest with synthetic sentiment
- [ ] Compare results: technical-only vs technical+sentiment
- [ ] Validate sentiment contribution to alpha
- [ ] Tune 60/40 weighting if needed

### Step 3: Live News Testing
- [ ] Set up API keys in `.env`
- [ ] Create `test_news_signal.py` script
- [ ] Test single equity (AAPL)
- [ ] Validate sentiment scores align with market moves
- [ ] Test rate limiting and error handling

### Step 4: Production Integration
- [ ] Add news sentiment to daily signal generation
- [ ] Create cron job for periodic news fetching
- [ ] Add sentiment to dashboard display
- [ ] Monitor and tune confidence thresholds

---

## Configuration Options

### Backtest Config
```yaml
# backtest_config.yaml
sentiment:
  enabled: true
  mode: "price_correlated"  # random_walk, price_correlated, event_based, regime_based
  correlation_lag: 1
  noise_std: 0.2
  event_calendar_path: "data/sentiment/event_calendar.csv"
```

### Signal Config
```yaml
# signal_config.yaml
signals:
  technical_weight: 0.6
  news_weight: 0.4
  news_enabled: true
  news_lookback_hours: 48
  min_news_score_for_boost: 7.0
  max_news_score_for_penalty: 3.0
```

---

## Files to Create

| File | Description |
|------|-------------|
| `trading_system/research/sentiment/synthetic_generator.py` | Synthetic sentiment for backtesting |
| `data/sentiment/event_calendar.csv` | Historical events with sentiment impact |
| `scripts/test_news_signal.py` | Single equity news test script |
| `configs/backtest_config_with_sentiment.yaml` | Backtest config with sentiment enabled |

---

## Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Sentiment-Return Correlation | > 0.1 | Positive correlation between sentiment and next-day returns |
| Signal Improvement | > 5% | Improvement in Sharpe when adding sentiment |
| News Fetch Latency | < 5s | Time to fetch and analyze news for 10 symbols |
| Sentiment Accuracy | > 60% | Correct direction prediction on earnings days |

---

## Next Steps

1. **Immediate**: Create `SyntheticSentimentGenerator` for backtesting
2. **Short-term**: Run backtest experiments with sentiment
3. **Medium-term**: Test live news API with single equity
4. **Long-term**: Full production integration with dashboard

---

## References

- Existing News Infrastructure: `trading_system/research/news_analyzer.py`
- Signal Integration: `trading_system/signals/live_signal_generator.py`
- API Clients: `trading_system/data_pipeline/sources/news/`
- Sentiment Analysis: `trading_system/research/sentiment/`
