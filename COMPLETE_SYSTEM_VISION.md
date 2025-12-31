# Complete Trading System Vision & Implementation Roadmap

**Date**: 2024-12-30
**Status**: Planning Document
**Purpose**: Guide for agents to build the complete trading assistant system

---

## Executive Summary

### Current State
The repository contains a **backtesting and validation framework** (~40% of vision complete).

### Target State
A **fully automated trading assistant** that:
1. Runs daily as a cron job
2. Combines technical analysis with news/sentiment research
3. Uses ML to refine strategies over time
4. Sends email reports with actionable buy/sell recommendations
5. Tracks performance and learns from outcomes

---

# PART 1: COMPLETE SYSTEM ARCHITECTURE

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING ASSISTANT SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   CRON      │    │    DATA     │    │   SIGNAL    │    │   OUTPUT    │  │
│  │  SCHEDULER  │───▶│  PIPELINE   │───▶│  GENERATOR  │───▶│   LAYER     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        │                  ▼                  ▼                  ▼          │
│        │           ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│        │           │  OHLCV API  │    │  TECHNICAL  │    │   EMAIL     │  │
│        │           │  (Polygon,  │    │  ANALYSIS   │    │   REPORTS   │  │
│        │           │  Alpha V.)  │    │   ENGINE    │    │             │  │
│        │           └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        │                  ▼                  ▼                  ▼          │
│        │           ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│        │           │    NEWS     │    │     ML      │    │  DASHBOARD  │  │
│        │           │  SENTIMENT  │    │  REFINEMENT │    │    (WEB)    │  │
│        │           │    API      │    │   ENGINE    │    │             │  │
│        │           └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        │                  ▼                  ▼                  ▼          │
│        │           ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│        │           │   SOCIAL    │    │  STRATEGY   │    │   ALERTS    │  │
│        │           │  SENTIMENT  │    │  OPTIMIZER  │    │  (SMS/Push) │  │
│        │           │  (Twitter)  │    │             │    │             │  │
│        │           └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                                                                   │
│        ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         STORAGE LAYER                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │  │
│  │  │  OHLCV   │  │  NEWS    │  │  SIGNALS │  │PERFORMANCE│            │  │
│  │  │   DB     │  │   DB     │  │   LOG    │  │  TRACKER │            │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     EXISTING BACKTESTING ENGINE                      │  │
│  │  (Current Repository - Strategy validation, risk management, etc.)   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Scheduler Layer (`trading_system/scheduler/`)

```
scheduler/
├── __init__.py
├── cron_runner.py          # Main cron job orchestrator
├── job_definitions.py      # Define all scheduled jobs
├── job_registry.py         # Register and manage jobs
└── health_check.py         # Monitor job health
```

**Responsibilities:**
- Run daily at configurable times (e.g., 4:30 PM ET for equities, midnight UTC for crypto)
- Orchestrate data fetching → analysis → signal generation → reporting
- Handle failures gracefully with retries and alerts
- Log all job executions

### 2. Data Pipeline (`trading_system/data_pipeline/`)

```
data_pipeline/
├── __init__.py
├── orchestrator.py         # Coordinate all data fetching
├── sources/
│   ├── ohlcv/
│   │   ├── polygon_source.py       # Polygon.io API
│   │   ├── alpha_vantage_source.py # Alpha Vantage API
│   │   ├── yahoo_source.py         # Yahoo Finance (backup)
│   │   └── crypto_source.py        # Binance/CoinGecko for crypto
│   ├── news/
│   │   ├── newsapi_source.py       # NewsAPI.org
│   │   ├── alpha_vantage_news.py   # Alpha Vantage News
│   │   ├── benzinga_source.py      # Benzinga (premium)
│   │   └── reddit_source.py        # Reddit sentiment
│   └── social/
│       ├── twitter_source.py       # Twitter/X sentiment
│       └── stocktwits_source.py    # StockTwits
├── transformers/
│   ├── ohlcv_transformer.py        # Standardize OHLCV format
│   ├── news_transformer.py         # Parse and normalize news
│   └── sentiment_scorer.py         # Score sentiment from text
└── storage/
    ├── ohlcv_store.py              # Store OHLCV data
    ├── news_store.py               # Store news articles
    └── sentiment_store.py          # Store sentiment scores
```

### 3. News & Sentiment Analysis (`trading_system/research/`)

```
research/
├── __init__.py
├── news_analyzer.py        # Main news analysis orchestrator
├── sentiment/
│   ├── vader_analyzer.py           # VADER sentiment (rule-based)
│   ├── finbert_analyzer.py         # FinBERT (transformer-based)
│   ├── ensemble_sentiment.py       # Combine multiple models
│   └── crypto_sentiment.py         # Crypto-specific sentiment
├── entity_extraction/
│   ├── ticker_extractor.py         # Extract stock tickers from text
│   ├── crypto_extractor.py         # Extract crypto mentions
│   └── event_classifier.py         # Classify event types (earnings, M&A, etc.)
├── relevance/
│   ├── relevance_scorer.py         # Score news relevance to portfolio
│   └── impact_estimator.py         # Estimate potential price impact
└── summarization/
    ├── article_summarizer.py       # Summarize long articles
    └── daily_digest.py             # Generate daily news digest
```

### 4. Signal Generator (`trading_system/signals/`)

```
signals/
├── __init__.py
├── signal_orchestrator.py  # Main signal generation pipeline
├── generators/
│   ├── technical_signals.py        # Existing technical analysis
│   ├── news_signals.py             # News-based signals
│   ├── sentiment_signals.py        # Sentiment-based signals
│   └── combined_signals.py         # Combine all signal sources
├── filters/
│   ├── quality_filter.py           # Filter low-quality signals
│   ├── capacity_filter.py          # Filter by trading capacity
│   └── correlation_filter.py       # Filter correlated signals
├── rankers/
│   ├── signal_scorer.py            # Score and rank signals
│   ├── conviction_calculator.py    # Calculate conviction levels
│   └── portfolio_optimizer.py      # Optimize signal selection
└── output/
    ├── signal_formatter.py         # Format signals for output
    └── recommendation.py           # Final recommendation object
```

### 5. ML Refinement Engine (`trading_system/ml_refinement/`)

```
ml_refinement/
├── __init__.py
├── feature_store.py        # Store and retrieve ML features
├── training/
│   ├── trainer.py                  # Model training orchestrator
│   ├── hyperparameter_tuner.py     # Hyperparameter optimization
│   └── cross_validator.py          # Walk-forward cross-validation
├── models/
│   ├── signal_quality_model.py     # Predict signal quality
│   ├── regime_classifier.py        # Classify market regime
│   ├── return_predictor.py         # Predict expected returns
│   └── risk_predictor.py           # Predict risk/volatility
├── feedback/
│   ├── outcome_tracker.py          # Track signal outcomes
│   ├── performance_analyzer.py     # Analyze what worked/didn't
│   └── model_updater.py            # Update models with new data
└── optimization/
    ├── parameter_optimizer.py      # Optimize strategy parameters
    └── weight_optimizer.py         # Optimize signal weights
```

### 6. Output Layer (`trading_system/output/`)

```
output/
├── __init__.py
├── email/
│   ├── email_service.py            # Email sending service
│   ├── templates/
│   │   ├── daily_signals.html      # Daily signal report template
│   │   ├── weekly_summary.html     # Weekly performance summary
│   │   ├── alert.html              # Urgent alert template
│   │   └── news_digest.html        # News digest template
│   ├── report_generator.py         # Generate email content
│   └── scheduler.py                # Schedule email delivery
├── alerts/
│   ├── alert_service.py            # Alert orchestrator
│   ├── sms_sender.py               # SMS alerts (Twilio)
│   ├── push_sender.py              # Push notifications
│   └── slack_sender.py             # Slack integration
├── dashboard/
│   ├── app.py                      # Web dashboard (Streamlit/Flask)
│   ├── pages/
│   │   ├── signals.py              # Current signals view
│   │   ├── performance.py          # Performance tracking
│   │   ├── news.py                 # News feed view
│   │   └── settings.py             # User settings
│   └── api/
│       └── endpoints.py            # REST API endpoints
└── exports/
    ├── csv_exporter.py             # Export to CSV
    ├── json_exporter.py            # Export to JSON
    └── pdf_generator.py            # Generate PDF reports
```

### 7. Performance Tracking (`trading_system/tracking/`)

```
tracking/
├── __init__.py
├── signal_tracker.py       # Track all generated signals
├── recommendation_tracker.py # Track recommendations sent
├── outcome_recorder.py     # Record actual outcomes
├── performance_calculator.py # Calculate performance metrics
├── analytics/
│   ├── signal_analytics.py         # Analyze signal performance
│   ├── strategy_analytics.py       # Analyze strategy performance
│   └── attribution.py              # Performance attribution
└── reports/
    ├── performance_report.py       # Generate performance reports
    └── leaderboard.py              # Strategy/signal leaderboard
```

---

## Data Flow Diagram

```
                                    DAILY EXECUTION FLOW

┌──────────────────────────────────────────────────────────────────────────────┐
│ 4:00 PM ET (Equities) / 12:00 AM UTC (Crypto)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [1] CRON TRIGGER                                                            │
│       │                                                                      │
│       ▼                                                                      │
│  [2] FETCH DATA ─────────────────────────────────────────────────────────┐  │
│       │                                                                  │  │
│       ├──▶ OHLCV Data (Polygon/Alpha Vantage)                           │  │
│       ├──▶ News Articles (NewsAPI, Alpha Vantage News)                  │  │
│       ├──▶ Social Sentiment (Twitter, Reddit, StockTwits)               │  │
│       └──▶ Economic Calendar (if applicable)                            │  │
│                                                                          │  │
│       ▼                                                                  │  │
│  [3] PROCESS & ANALYZE ───────────────────────────────────────────────┐ │  │
│       │                                                               │ │  │
│       ├──▶ Compute Technical Indicators (MA, ATR, ROC, Breakouts)     │ │  │
│       ├──▶ Analyze News Sentiment (FinBERT, VADER)                    │ │  │
│       ├──▶ Extract Relevant Entities (tickers, events)                │ │  │
│       └──▶ Score News Relevance & Impact                              │ │  │
│                                                                       │ │  │
│       ▼                                                               │ │  │
│  [4] GENERATE SIGNALS ─────────────────────────────────────────────┐  │ │  │
│       │                                                            │  │ │  │
│       ├──▶ Technical Signals (breakouts, trend following)          │  │ │  │
│       ├──▶ News-Based Signals (sentiment shift, events)            │  │ │  │
│       ├──▶ Combined/ML Signals (ensemble predictions)              │  │ │  │
│       └──▶ Apply Filters (capacity, correlation, quality)          │  │ │  │
│                                                                    │  │ │  │
│       ▼                                                            │  │ │  │
│  [5] RANK & SELECT ─────────────────────────────────────────────┐  │  │ │  │
│       │                                                         │  │  │ │  │
│       ├──▶ Score signals (conviction, risk/reward)              │  │  │ │  │
│       ├──▶ Apply portfolio constraints (max positions, exposure)│  │  │ │  │
│       └──▶ Select top N recommendations                         │  │  │ │  │
│                                                                 │  │  │ │  │
│       ▼                                                         │  │  │ │  │
│  [6] GENERATE OUTPUT ────────────────────────────────────────┐  │  │  │ │  │
│       │                                                      │  │  │  │ │  │
│       ├──▶ Format Daily Signal Report                        │  │  │  │ │  │
│       ├──▶ Generate News Digest                              │  │  │  │ │  │
│       ├──▶ Create Performance Summary                        │  │  │  │ │  │
│       └──▶ Prepare Alerts (if urgent signals)                │  │  │  │ │  │
│                                                              │  │  │  │ │  │
│       ▼                                                      │  │  │  │ │  │
│  [7] DELIVER ─────────────────────────────────────────────┐  │  │  │  │ │  │
│       │                                                   │  │  │  │  │ │  │
│       ├──▶ Send Email Report                              │  │  │  │  │ │  │
│       ├──▶ Update Web Dashboard                           │  │  │  │  │ │  │
│       ├──▶ Send SMS/Push Alerts (if configured)           │  │  │  │  │ │  │
│       └──▶ Log to Database                                │  │  │  │  │ │  │
│                                                           │  │  │  │  │ │  │
│       ▼                                                   │  │  │  │  │ │  │
│  [8] TRACK & LEARN ────────────────────────────────────┐  │  │  │  │  │ │  │
│       │                                                │  │  │  │  │  │ │  │
│       ├──▶ Record all signals generated                │  │  │  │  │  │ │  │
│       ├──▶ Track recommendations delivered             │  │  │  │  │  │ │  │
│       ├──▶ Update outcome tracker (next day)           │  │  │  │  │  │ │  │
│       └──▶ Retrain ML models (weekly)                  │  │  │  │  │  │ │  │
│                                                        │  │  │  │  │  │ │  │
└────────────────────────────────────────────────────────┴──┴──┴──┴──┴──┴─┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### Signal Recommendations Table
```sql
CREATE TABLE signal_recommendations (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(20) NOT NULL,  -- 'equity' or 'crypto'
    signal_type VARCHAR(50) NOT NULL,   -- 'breakout_20d', 'news_sentiment', etc.
    direction VARCHAR(10) NOT NULL,     -- 'BUY' or 'SELL'
    conviction VARCHAR(20) NOT NULL,    -- 'HIGH', 'MEDIUM', 'LOW'
    entry_price DECIMAL(20, 8),
    target_price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    position_size_pct DECIMAL(5, 4),

    -- Signal components
    technical_score DECIMAL(5, 4),
    news_score DECIMAL(5, 4),
    sentiment_score DECIMAL(5, 4),
    combined_score DECIMAL(5, 4),

    -- Metadata
    reasoning TEXT,
    news_headlines TEXT[],
    delivered BOOLEAN DEFAULT FALSE,
    delivered_at TIMESTAMP,

    -- Outcome tracking
    outcome_recorded BOOLEAN DEFAULT FALSE,
    actual_entry_price DECIMAL(20, 8),
    actual_exit_price DECIMAL(20, 8),
    actual_return_pct DECIMAL(10, 6),
    outcome_date TIMESTAMP
);
```

### News Articles Table
```sql
CREATE TABLE news_articles (
    id UUID PRIMARY KEY,
    fetched_at TIMESTAMP NOT NULL,
    source VARCHAR(100) NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    content TEXT,
    url VARCHAR(500),
    published_at TIMESTAMP,

    -- Extracted data
    symbols VARCHAR(20)[],              -- Mentioned tickers
    sentiment_vader DECIMAL(5, 4),
    sentiment_finbert DECIMAL(5, 4),
    sentiment_combined DECIMAL(5, 4),
    relevance_score DECIMAL(5, 4),
    event_type VARCHAR(50),             -- 'earnings', 'merger', 'product', etc.

    -- Processing flags
    processed BOOLEAN DEFAULT FALSE,
    used_in_signal BOOLEAN DEFAULT FALSE
);
```

### Performance Tracking Table
```sql
CREATE TABLE signal_performance (
    id UUID PRIMARY KEY,
    signal_id UUID REFERENCES signal_recommendations(id),

    -- Timing
    signal_date DATE NOT NULL,
    entry_date DATE,
    exit_date DATE,
    holding_days INTEGER,

    -- Prices
    signal_price DECIMAL(20, 8),
    entry_price DECIMAL(20, 8),
    exit_price DECIMAL(20, 8),

    -- Returns
    return_pct DECIMAL(10, 6),
    return_r_multiple DECIMAL(10, 4),

    -- Attribution
    was_followed BOOLEAN,               -- Did user follow recommendation?
    exit_reason VARCHAR(50),            -- 'target', 'stop', 'manual', 'time'

    -- Analysis
    market_return_pct DECIMAL(10, 6),   -- Benchmark return same period
    alpha DECIMAL(10, 6)                -- Signal return - market return
);
```

---

## Email Report Format

### Daily Signal Report Template

```
Subject: Trading Signals for [DATE] - [N] New Recommendations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
             DAILY TRADING SIGNALS - [DATE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MARKET OVERVIEW
─────────────────────────────────────────────────────
SPY: $XXX.XX (+X.XX%)  |  BTC: $XX,XXX (+X.XX%)
Market Regime: [BULLISH/BEARISH/NEUTRAL]
Portfolio Volatility: [LOW/NORMAL/ELEVATED]

═══════════════════════════════════════════════════════
                    🎯 BUY SIGNALS
═══════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│ 1. AAPL - Apple Inc.                    [HIGH CONVICTION]
├─────────────────────────────────────────────────────┤
│ Signal Type: 20-Day Breakout + Positive News Sentiment
│
│ Entry:  $XXX.XX (at open)
│ Target: $XXX.XX (+X.X%)
│ Stop:   $XXX.XX (-X.X%)
│ Size:   X.X% of portfolio
│
│ Technical Score: 8.5/10
│ News Score:      7.2/10
│ Combined:        8.0/10
│
│ 📰 Recent News:
│ • "Apple announces record iPhone sales..." (Positive)
│ • "New AI features driving upgrades..." (Positive)
│
│ 💡 Reasoning: Strong breakout above 20-day high with
│    supporting positive sentiment from product news.
│    MA50 slope is +1.2%, indicating solid uptrend.
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 2. ETH - Ethereum                       [MEDIUM CONVICTION]
├─────────────────────────────────────────────────────┤
│ Signal Type: 55-Day Breakout
│
│ Entry:  $X,XXX.XX (at open)
│ Target: $X,XXX.XX (+X.X%)
│ Stop:   $X,XXX.XX (-X.X%)
│ Size:   X.X% of portfolio
│
│ Technical Score: 7.8/10
│ News Score:      6.0/10 (Neutral)
│ Combined:        7.2/10
│
│ 💡 Reasoning: Breaking above 55-day high with price
│    above MA200. Crypto market showing strength.
└─────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════
                 📈 CURRENT POSITIONS
═══════════════════════════════════════════════════════

Symbol  Entry     Current   P&L      Days  Status
──────────────────────────────────────────────────────
NVDA    $XXX.XX   $XXX.XX   +X.XX%   12    ✅ Above MA20
MSFT    $XXX.XX   $XXX.XX   +X.XX%   8     ✅ Above MA20
BTC     $XX,XXX   $XX,XXX   -X.XX%   5     ⚠️ Near MA20

═══════════════════════════════════════════════════════
                 📰 NEWS DIGEST
═══════════════════════════════════════════════════════

🟢 POSITIVE SENTIMENT
• Tech sector rallies on AI optimism
• Fed signals potential rate cuts

🔴 NEGATIVE SENTIMENT
• China tensions weigh on chipmakers

🟡 WATCH LIST
• FOMC meeting Wednesday - volatility expected

═══════════════════════════════════════════════════════
                 📊 PERFORMANCE (MTD)
═══════════════════════════════════════════════════════

Strategy Return:  +X.XX%
Benchmark (SPY):  +X.XX%
Alpha:            +X.XX%

Win Rate:         XX%
Avg Winner:       +X.XX%
Avg Loser:        -X.XX%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by Trading Assistant v1.0
View Dashboard: https://your-dashboard-url.com
Unsubscribe: [link]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# PART 2: PRIORITIZED MISSING COMPONENTS

## Priority Matrix

| Priority | Component | Effort | Impact | Dependencies |
|----------|-----------|--------|--------|--------------|
| **P0** | Live Data Pipeline | Medium | Critical | None |
| **P0** | Signal Generator (Live) | Medium | Critical | Data Pipeline |
| **P1** | Email Notification System | Low | High | Signal Generator |
| **P1** | Cron/Scheduler | Low | High | All Above |
| **P2** | News API Integration | Medium | High | Data Pipeline |
| **P2** | Basic Sentiment Analysis | Medium | High | News API |
| **P3** | Performance Tracking | Medium | Medium | Signal Generator |
| **P3** | ML Refinement (Basic) | High | Medium | Performance Tracking |
| **P4** | Web Dashboard | High | Medium | All Above |
| **P4** | Advanced Sentiment (FinBERT) | High | Medium | Basic Sentiment |
| **P5** | Social Sentiment | Medium | Low | Sentiment Analysis |
| **P5** | SMS/Push Alerts | Low | Low | Email System |

---

## Detailed Component Specifications

### P0-1: Live Data Pipeline

**Purpose**: Fetch real-time/daily OHLCV data from APIs

**Files to Create**:
```
trading_system/data_pipeline/
├── __init__.py
├── live_data_fetcher.py
├── sources/
│   ├── __init__.py
│   ├── polygon_client.py
│   ├── alpha_vantage_client.py
│   └── binance_client.py
└── cache/
    ├── __init__.py
    └── data_cache.py
```

**Key Implementation**:
```python
# live_data_fetcher.py
class LiveDataFetcher:
    """Fetch live OHLCV data from configured sources."""

    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.equity_source = self._init_equity_source()
        self.crypto_source = self._init_crypto_source()
        self.cache = DataCache(config.cache_path)

    async def fetch_daily_data(
        self,
        symbols: List[str],
        asset_class: str,
        lookback_days: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """Fetch daily OHLCV for symbols."""
        pass

    async def fetch_latest_bar(
        self,
        symbol: str,
        asset_class: str
    ) -> Bar:
        """Fetch most recent bar for a symbol."""
        pass
```

**API Keys Required**:
- Polygon.io (equities) - Free tier: 5 API calls/minute
- Alpha Vantage (backup) - Free tier: 5 API calls/minute
- Binance (crypto) - Free, no key needed for public data

---

### P0-2: Live Signal Generator

**Purpose**: Generate actionable signals for tomorrow's open

**Files to Create**:
```
trading_system/signals/
├── __init__.py
├── live_signal_generator.py
├── signal_types.py
├── recommendation.py
└── filters/
    ├── __init__.py
    └── signal_filter.py
```

**Key Implementation**:
```python
# live_signal_generator.py
class LiveSignalGenerator:
    """Generate live trading signals."""

    def __init__(
        self,
        strategies: List[StrategyInterface],
        portfolio_config: PortfolioConfig,
        data_fetcher: LiveDataFetcher
    ):
        self.strategies = strategies
        self.portfolio_config = portfolio_config
        self.data_fetcher = data_fetcher

    async def generate_daily_signals(
        self,
        current_date: date
    ) -> List[Recommendation]:
        """Generate signals for tomorrow's open."""

        # 1. Fetch latest data
        ohlcv_data = await self.data_fetcher.fetch_daily_data(...)

        # 2. Compute indicators
        features = self._compute_features(ohlcv_data)

        # 3. Generate signals from each strategy
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.check_entry_triggers(features, current_date)
            all_signals.extend(signals)

        # 4. Score and rank signals
        scored_signals = self._score_signals(all_signals)

        # 5. Apply filters and select top N
        recommendations = self._select_recommendations(scored_signals)

        return recommendations
```

**Output Format**:
```python
@dataclass
class Recommendation:
    """A trading recommendation to deliver to user."""
    symbol: str
    asset_class: str
    direction: str  # 'BUY' or 'SELL'
    conviction: str  # 'HIGH', 'MEDIUM', 'LOW'

    entry_price: float
    target_price: float
    stop_price: float
    position_size_pct: float

    technical_score: float
    news_score: Optional[float]
    combined_score: float

    reasoning: str
    news_headlines: List[str]

    generated_at: datetime
```

---

### P1-1: Email Notification System

**Purpose**: Send daily signal reports via email

**Files to Create**:
```
trading_system/output/email/
├── __init__.py
├── email_service.py
├── report_generator.py
├── templates/
│   ├── daily_signals.html
│   ├── weekly_summary.html
│   └── base.html
└── config.py
```

**Key Implementation**:
```python
# email_service.py
class EmailService:
    """Send email notifications."""

    def __init__(self, config: EmailConfig):
        self.config = config
        self.smtp_client = self._init_smtp()
        self.templates = self._load_templates()

    async def send_daily_report(
        self,
        recommendations: List[Recommendation],
        portfolio_summary: PortfolioSummary,
        news_digest: NewsDigest,
        recipients: List[str]
    ) -> bool:
        """Send daily signal report email."""

        # Generate HTML content
        html_content = self.templates['daily_signals'].render(
            recommendations=recommendations,
            portfolio=portfolio_summary,
            news=news_digest,
            generated_at=datetime.now()
        )

        # Send email
        return await self._send_email(
            to=recipients,
            subject=f"Trading Signals for {date.today()}",
            html=html_content
        )
```

**Email Provider Options**:
- SendGrid (recommended) - Free tier: 100 emails/day
- AWS SES - Very cheap, requires AWS account
- Gmail SMTP - Free, but limits apply

---

### P1-2: Cron/Scheduler

**Purpose**: Automate daily execution

**Files to Create**:
```
trading_system/scheduler/
├── __init__.py
├── cron_runner.py
├── jobs/
│   ├── __init__.py
│   ├── daily_signals_job.py
│   ├── data_update_job.py
│   └── weekly_report_job.py
└── config.py
```

**Key Implementation**:
```python
# cron_runner.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class CronRunner:
    """Run scheduled jobs."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.scheduler = AsyncIOScheduler()
        self._register_jobs()

    def _register_jobs(self):
        """Register all scheduled jobs."""

        # Daily signals - 4:30 PM ET (after market close)
        self.scheduler.add_job(
            self._run_daily_signals,
            'cron',
            hour=16,
            minute=30,
            timezone='America/New_York'
        )

        # Crypto signals - midnight UTC
        self.scheduler.add_job(
            self._run_crypto_signals,
            'cron',
            hour=0,
            minute=0,
            timezone='UTC'
        )

        # Weekly summary - Sunday 6 PM ET
        self.scheduler.add_job(
            self._run_weekly_summary,
            'cron',
            day_of_week='sun',
            hour=18,
            timezone='America/New_York'
        )

    async def _run_daily_signals(self):
        """Execute daily signal generation and email."""
        try:
            # 1. Fetch data
            # 2. Generate signals
            # 3. Send email
            # 4. Log results
            pass
        except Exception as e:
            await self._send_error_alert(e)

    def start(self):
        """Start the scheduler."""
        self.scheduler.start()
```

**Deployment Options**:
- Local: Run as systemd service or launchd (macOS)
- Cloud: AWS Lambda + CloudWatch Events, or Railway/Render cron

---

### P2-1: News API Integration

**Purpose**: Fetch relevant news articles

**Files to Create**:
```
trading_system/data_pipeline/sources/news/
├── __init__.py
├── base_news_source.py
├── newsapi_source.py
├── alpha_vantage_news.py
└── aggregator.py
```

**Key Implementation**:
```python
# newsapi_source.py
class NewsAPISource:
    """Fetch news from NewsAPI.org."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    async def fetch_articles(
        self,
        symbols: List[str],
        lookback_hours: int = 24
    ) -> List[NewsArticle]:
        """Fetch news articles mentioning symbols."""

        articles = []
        for symbol in symbols:
            # Search by company name and ticker
            company_name = self._get_company_name(symbol)
            query = f'"{company_name}" OR "{symbol}"'

            response = await self._api_call(
                endpoint="/everything",
                params={
                    "q": query,
                    "from": (datetime.now() - timedelta(hours=lookback_hours)).isoformat(),
                    "sortBy": "publishedAt",
                    "language": "en"
                }
            )

            for article in response['articles']:
                articles.append(NewsArticle(
                    source=article['source']['name'],
                    title=article['title'],
                    summary=article['description'],
                    url=article['url'],
                    published_at=parse_datetime(article['publishedAt']),
                    symbols=[symbol]
                ))

        return articles
```

**API Keys Required**:
- NewsAPI.org - Free tier: 100 requests/day, 1 month old articles
- Alpha Vantage News - Free tier: 5 requests/minute

---

### P2-2: Basic Sentiment Analysis

**Purpose**: Score news sentiment

**Files to Create**:
```
trading_system/research/sentiment/
├── __init__.py
├── vader_analyzer.py
├── keyword_analyzer.py
└── sentiment_aggregator.py
```

**Key Implementation**:
```python
# vader_analyzer.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class VADERSentimentAnalyzer:
    """VADER sentiment analysis for financial news."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Add financial terms
        self._add_financial_lexicon()

    def _add_financial_lexicon(self):
        """Add financial-specific sentiment words."""
        financial_lexicon = {
            'bullish': 2.0,
            'bearish': -2.0,
            'upgrade': 1.5,
            'downgrade': -1.5,
            'beat': 1.0,
            'miss': -1.0,
            'growth': 0.8,
            'decline': -0.8,
            'surge': 1.5,
            'plunge': -1.5,
            'rally': 1.2,
            'selloff': -1.2,
            # ... more terms
        }
        self.analyzer.lexicon.update(financial_lexicon)

    def analyze(self, text: str) -> SentimentScore:
        """Analyze sentiment of text."""
        scores = self.analyzer.polarity_scores(text)

        return SentimentScore(
            positive=scores['pos'],
            negative=scores['neg'],
            neutral=scores['neu'],
            compound=scores['compound'],  # -1 to +1
            label=self._get_label(scores['compound'])
        )

    def _get_label(self, compound: float) -> str:
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        return 'neutral'
```

---

### P3-1: Performance Tracking

**Purpose**: Track signal outcomes to measure and improve

**Files to Create**:
```
trading_system/tracking/
├── __init__.py
├── signal_tracker.py
├── outcome_recorder.py
├── performance_calculator.py
└── storage/
    ├── __init__.py
    └── tracking_db.py
```

**Key Implementation**:
```python
# signal_tracker.py
class SignalTracker:
    """Track all signals and their outcomes."""

    def __init__(self, db: TrackingDatabase):
        self.db = db

    async def record_signal(
        self,
        recommendation: Recommendation,
        delivered: bool = False
    ) -> str:
        """Record a generated signal."""
        signal_id = str(uuid4())

        await self.db.insert_signal(
            id=signal_id,
            symbol=recommendation.symbol,
            direction=recommendation.direction,
            entry_price=recommendation.entry_price,
            target_price=recommendation.target_price,
            stop_price=recommendation.stop_price,
            conviction=recommendation.conviction,
            scores={
                'technical': recommendation.technical_score,
                'news': recommendation.news_score,
                'combined': recommendation.combined_score
            },
            delivered=delivered,
            created_at=datetime.now()
        )

        return signal_id

    async def record_outcome(
        self,
        signal_id: str,
        exit_price: float,
        exit_date: date,
        exit_reason: str
    ):
        """Record the outcome of a signal."""
        signal = await self.db.get_signal(signal_id)

        # Calculate returns
        if signal.direction == 'BUY':
            return_pct = (exit_price - signal.entry_price) / signal.entry_price
        else:
            return_pct = (signal.entry_price - exit_price) / signal.entry_price

        # Calculate R-multiple
        risk = abs(signal.entry_price - signal.stop_price)
        reward = exit_price - signal.entry_price if signal.direction == 'BUY' else signal.entry_price - exit_price
        r_multiple = reward / risk if risk > 0 else 0

        await self.db.update_outcome(
            signal_id=signal_id,
            exit_price=exit_price,
            exit_date=exit_date,
            exit_reason=exit_reason,
            return_pct=return_pct,
            r_multiple=r_multiple
        )
```

---

# PART 3: IMPLEMENTATION ROADMAP

## Phase 1: Minimum Viable Product (MVP)
**Goal**: Daily email with technical-only signals
**Duration**: 2-3 weeks

### Sprint 1.1: Live Data Pipeline (Week 1)
```
Tasks:
├── [1.1.1] Create data_pipeline module structure
├── [1.1.2] Implement Polygon.io client for equities
├── [1.1.3] Implement Binance client for crypto
├── [1.1.4] Add data caching layer
├── [1.1.5] Write tests for data fetching
└── [1.1.6] Create configuration for API keys
```

**Deliverables**:
- `trading_system/data_pipeline/` module
- Working API clients for equities + crypto
- Tests passing

### Sprint 1.2: Live Signal Generator (Week 1-2)
```
Tasks:
├── [1.2.1] Create signals module structure
├── [1.2.2] Adapt existing strategies for live use
├── [1.2.3] Implement Recommendation dataclass
├── [1.2.4] Create signal scoring and ranking
├── [1.2.5] Add portfolio-aware filtering
├── [1.2.6] Write tests for signal generation
└── [1.2.7] Create CLI command for manual signal generation
```

**Deliverables**:
- `trading_system/signals/` module
- CLI command: `trading-system signals generate`
- Tests passing

### Sprint 1.3: Email System (Week 2)
```
Tasks:
├── [1.3.1] Create output/email module structure
├── [1.3.2] Implement SendGrid/SMTP email service
├── [1.3.3] Create HTML email templates
├── [1.3.4] Build report generator
├── [1.3.5] Write tests for email sending
└── [1.3.6] Create CLI command for sending test email
```

**Deliverables**:
- `trading_system/output/email/` module
- HTML email templates
- CLI command: `trading-system email send --test`

### Sprint 1.4: Scheduler & Integration (Week 2-3)
```
Tasks:
├── [1.4.1] Create scheduler module structure
├── [1.4.2] Implement cron runner with APScheduler
├── [1.4.3] Create daily signals job
├── [1.4.4] Add error handling and alerting
├── [1.4.5] Write integration tests
├── [1.4.6] Create deployment documentation
└── [1.4.7] Test end-to-end flow
```

**Deliverables**:
- `trading_system/scheduler/` module
- Working daily cron job
- Deployment instructions

---

## Phase 2: News Integration
**Goal**: Add news sentiment to signal generation
**Duration**: 2 weeks

### Sprint 2.1: News API Integration (Week 4)
```
Tasks:
├── [2.1.1] Create data_pipeline/sources/news structure
├── [2.1.2] Implement NewsAPI.org client
├── [2.1.3] Implement Alpha Vantage News client
├── [2.1.4] Create news article aggregator
├── [2.1.5] Add news storage (SQLite/PostgreSQL)
├── [2.1.6] Write tests for news fetching
└── [2.1.7] Create CLI for fetching news
```

**Deliverables**:
- News fetching from 2 sources
- News storage in database
- CLI command: `trading-system news fetch`

### Sprint 2.2: Sentiment Analysis (Week 4-5)
```
Tasks:
├── [2.2.1] Create research/sentiment structure
├── [2.2.2] Implement VADER analyzer with financial lexicon
├── [2.2.3] Create sentiment aggregator
├── [2.2.4] Integrate sentiment into signal scoring
├── [2.2.5] Update email template with news section
├── [2.2.6] Write tests for sentiment analysis
└── [2.2.7] Test end-to-end with news
```

**Deliverables**:
- Sentiment scoring for news
- News integrated into signals
- Updated email with news digest

---

## Phase 3: Performance Tracking
**Goal**: Track outcomes and measure performance
**Duration**: 1-2 weeks

### Sprint 3.1: Tracking System (Week 6)
```
Tasks:
├── [3.1.1] Create tracking module structure
├── [3.1.2] Design and create database schema
├── [3.1.3] Implement signal tracker
├── [3.1.4] Implement outcome recorder
├── [3.1.5] Create performance calculator
├── [3.1.6] Add tracking to signal generation flow
├── [3.1.7] Write tests for tracking
└── [3.1.8] Create CLI for viewing performance
```

**Deliverables**:
- `trading_system/tracking/` module
- All signals tracked in database
- CLI command: `trading-system performance show`

### Sprint 3.2: Performance Reporting (Week 6-7)
```
Tasks:
├── [3.2.1] Create weekly performance summary email
├── [3.2.2] Add performance section to daily email
├── [3.2.3] Implement performance analytics
├── [3.2.4] Create performance leaderboard
└── [3.2.5] Add weekly report job to scheduler
```

**Deliverables**:
- Weekly performance email
- Performance metrics in daily email

---

## Phase 4: ML Refinement
**Goal**: Use ML to improve signal quality
**Duration**: 2-3 weeks

### Sprint 4.1: Feature Store (Week 8)
```
Tasks:
├── [4.1.1] Create ml_refinement module structure
├── [4.1.2] Design feature store schema
├── [4.1.3] Implement feature extraction from signals
├── [4.1.4] Create feature storage and retrieval
├── [4.1.5] Backfill features from historical data
└── [4.1.6] Write tests for feature store
```

### Sprint 4.2: Signal Quality Model (Week 8-9)
```
Tasks:
├── [4.2.1] Implement signal quality predictor
├── [4.2.2] Create training pipeline
├── [4.2.3] Implement walk-forward validation
├── [4.2.4] Integrate predictions into signal scoring
├── [4.2.5] Add model versioning
└── [4.2.6] Create model retraining job (weekly)
```

### Sprint 4.3: Parameter Optimization (Week 9-10)
```
Tasks:
├── [4.3.1] Implement parameter optimizer
├── [4.3.2] Create optimization job (monthly)
├── [4.3.3] Add guardrails to prevent overfitting
├── [4.3.4] Create optimization report
└── [4.3.5] Test optimization pipeline
```

---

## Phase 5: Dashboard & Polish
**Goal**: Web dashboard and refinements
**Duration**: 2 weeks

### Sprint 5.1: Web Dashboard (Week 11-12)
```
Tasks:
├── [5.1.1] Create Streamlit dashboard app
├── [5.1.2] Implement signals view
├── [5.1.3] Implement performance view
├── [5.1.4] Implement news feed view
├── [5.1.5] Add settings page
├── [5.1.6] Deploy dashboard
└── [5.1.7] Add dashboard link to emails
```

### Sprint 5.2: Advanced Features (Week 12+)
```
Tasks:
├── [5.2.1] Add FinBERT sentiment (if GPU available)
├── [5.2.2] Add social sentiment (Twitter, Reddit)
├── [5.2.3] Add SMS alerts for high-conviction signals
├── [5.2.4] Add push notifications
├── [5.2.5] Create mobile-friendly email templates
└── [5.2.6] Comprehensive documentation
```

---

## Configuration Files to Create

### Main Configuration (`config/trading_config.yaml`)
```yaml
# Trading System Configuration

# API Keys (use environment variables in production)
api_keys:
  polygon: ${POLYGON_API_KEY}
  alpha_vantage: ${ALPHA_VANTAGE_API_KEY}
  newsapi: ${NEWSAPI_API_KEY}
  sendgrid: ${SENDGRID_API_KEY}

# Data Sources
data_sources:
  equity:
    primary: polygon
    fallback: alpha_vantage
  crypto:
    primary: binance

# Universe
universe:
  equity: "SP500"  # or "NASDAQ100"
  crypto:
    - BTC
    - ETH
    - BNB
    - XRP
    - ADA
    - SOL
    - DOT
    - MATIC
    - LTC
    - LINK

# Signal Generation
signals:
  technical:
    enabled: true
    strategies:
      - equity_momentum
      - crypto_momentum
  news:
    enabled: true
    lookback_hours: 48
    min_relevance: 0.5
  combined:
    technical_weight: 0.6
    news_weight: 0.4

# Risk Management
risk:
  max_positions: 8
  max_exposure: 0.80
  risk_per_trade: 0.0075

# Email Settings
email:
  enabled: true
  recipients:
    - your-email@example.com
  daily_time: "16:30"  # ET
  timezone: "America/New_York"

# Scheduler
scheduler:
  enabled: true
  jobs:
    daily_signals:
      cron: "30 16 * * 1-5"  # 4:30 PM ET, Mon-Fri
      timezone: "America/New_York"
    crypto_signals:
      cron: "0 0 * * *"  # Midnight UTC daily
      timezone: "UTC"
    weekly_report:
      cron: "0 18 * * 0"  # 6 PM ET Sunday
      timezone: "America/New_York"
```

---

## Summary: What to Build

| Phase | Duration | Components | Result |
|-------|----------|------------|--------|
| **MVP** | 2-3 weeks | Data pipeline, Signal generator, Email, Scheduler | Daily email with technical signals |
| **News** | 2 weeks | News API, Sentiment analysis | Signals enhanced with news |
| **Tracking** | 1-2 weeks | Signal tracker, Performance calculator | Measure what works |
| **ML** | 2-3 weeks | Feature store, Quality model, Optimizer | Self-improving system |
| **Dashboard** | 2 weeks | Web UI, Advanced features | Complete product |

**Total: ~10-12 weeks to full vision**

---

## Agent Task Format

Each task above can be fed to an agent in this format:

```markdown
## Task: [Task ID] [Task Name]

**Context**:
[Brief description of where this fits in the system]

**Objective**:
[Clear statement of what needs to be built]

**Files to Create/Modify**:
- `path/to/file1.py` - [purpose]
- `path/to/file2.py` - [purpose]

**Requirements**:
1. [Specific requirement 1]
2. [Specific requirement 2]
3. [Specific requirement 3]

**Dependencies**:
- [What must exist before this task]

**Acceptance Criteria**:
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] Tests pass

**Example Usage**:
```python
# How the component should be used
```
```

---

**Document Created**: 2024-12-30
**Last Updated**: 2024-12-30
**Status**: Ready for Implementation
