# Agent Tasks: Phase 1 - MVP Implementation

**Phase Goal**: Daily email with technical-only signals
**Duration**: 2-3 weeks
**Prerequisites**: Existing backtesting framework

---

## Task 1.1.1: Create Data Pipeline Module Structure

**Context**:
The trading system needs a live data pipeline to fetch real-time OHLCV data from APIs. This task sets up the module structure.

**Objective**:
Create the directory structure and base files for the data pipeline module.

**Files to Create**:
```
trading_system/data_pipeline/
├── __init__.py
├── config.py                    # Configuration dataclasses
├── live_data_fetcher.py         # Main orchestrator (stub)
├── exceptions.py                # Custom exceptions
├── sources/
│   ├── __init__.py
│   ├── base_source.py           # Abstract base class
│   ├── polygon_client.py        # Stub
│   ├── alpha_vantage_client.py  # Stub
│   └── binance_client.py        # Stub
└── cache/
    ├── __init__.py
    └── data_cache.py            # Stub
```

**Requirements**:
1. Create all directories and `__init__.py` files
2. Create `config.py` with Pydantic models:
   ```python
   class DataPipelineConfig(BaseModel):
       polygon_api_key: Optional[str] = None
       alpha_vantage_api_key: Optional[str] = None
       cache_path: Path = Path("data/cache")
       cache_ttl_hours: int = 24
   ```
3. Create `exceptions.py` with custom exceptions:
   - `DataFetchError`
   - `APIRateLimitError`
   - `DataValidationError`
4. Create `base_source.py` with abstract base class:
   ```python
   class BaseDataSource(ABC):
       @abstractmethod
       async def fetch_daily_bars(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
           pass

       @abstractmethod
       async def fetch_latest_bar(self, symbol: str) -> Optional[Bar]:
           pass
   ```
5. Create stub implementations for all other files

**Acceptance Criteria**:
- [ ] All files created with proper structure
- [ ] Imports work: `from trading_system.data_pipeline import DataPipelineConfig`
- [ ] No circular import errors
- [ ] Stub classes have proper docstrings

---

## Task 1.1.2: Implement Polygon.io Client for Equities

**Context**:
Polygon.io provides real-time and historical stock data. We need to implement a client that fetches OHLCV data.

**Objective**:
Create a fully functional Polygon.io API client for fetching equity OHLCV data.

**Files to Modify**:
- `trading_system/data_pipeline/sources/polygon_client.py`

**Requirements**:
1. Implement `PolygonClient` class:
   ```python
   class PolygonClient(BaseDataSource):
       def __init__(self, api_key: str, rate_limit_per_minute: int = 5):
           pass

       async def fetch_daily_bars(
           self,
           symbol: str,
           start_date: date,
           end_date: date
       ) -> pd.DataFrame:
           """Fetch daily OHLCV bars from Polygon."""
           pass

       async def fetch_latest_bar(self, symbol: str) -> Optional[Bar]:
           """Fetch the most recent bar."""
           pass

       async def fetch_multiple_symbols(
           self,
           symbols: List[str],
           start_date: date,
           end_date: date
       ) -> Dict[str, pd.DataFrame]:
           """Fetch bars for multiple symbols."""
           pass
   ```

2. API Endpoint: `https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}`

3. Handle rate limiting (5 calls/minute on free tier):
   - Implement exponential backoff
   - Track API calls and wait if needed

4. Response parsing:
   ```python
   # Polygon response format
   {
       "results": [
           {"t": 1234567890000, "o": 100.0, "h": 105.0, "l": 99.0, "c": 104.0, "v": 1000000}
       ]
   }
   # Convert to DataFrame with columns: date, open, high, low, close, volume
   ```

5. Error handling:
   - Raise `DataFetchError` on network errors
   - Raise `APIRateLimitError` on 429 responses
   - Log warnings for missing data

**Dependencies**:
- Task 1.1.1 (module structure)
- `aiohttp` for async HTTP requests
- `pandas` for DataFrame handling

**Acceptance Criteria**:
- [ ] Can fetch 1 year of daily data for AAPL
- [ ] Rate limiting prevents 429 errors
- [ ] Returns properly formatted DataFrame
- [ ] Handles API errors gracefully
- [ ] Unit tests pass

**Test File to Create**: `tests/test_polygon_client.py`
```python
class TestPolygonClient:
    def test_fetch_daily_bars_success(self):
        """Test successful data fetch."""
        pass

    def test_fetch_daily_bars_rate_limit(self):
        """Test rate limit handling."""
        pass

    def test_fetch_multiple_symbols(self):
        """Test fetching multiple symbols."""
        pass
```

---

## Task 1.1.3: Implement Binance Client for Crypto

**Context**:
Binance provides free crypto OHLCV data without API key for public endpoints.

**Objective**:
Create a Binance API client for fetching crypto OHLCV data.

**Files to Modify**:
- `trading_system/data_pipeline/sources/binance_client.py`

**Requirements**:
1. Implement `BinanceClient` class:
   ```python
   class BinanceClient(BaseDataSource):
       BASE_URL = "https://api.binance.com/api/v3"

       def __init__(self, rate_limit_per_minute: int = 20):
           pass

       async def fetch_daily_bars(
           self,
           symbol: str,  # e.g., "BTC" (we'll convert to "BTCUSDT")
           start_date: date,
           end_date: date
       ) -> pd.DataFrame:
           pass

       async def fetch_latest_bar(self, symbol: str) -> Optional[Bar]:
           pass
   ```

2. API Endpoint: `GET /api/v3/klines`
   - Parameters: `symbol=BTCUSDT&interval=1d&startTime=xxx&endTime=xxx`

3. Symbol mapping:
   ```python
   SYMBOL_MAP = {
       "BTC": "BTCUSDT",
       "ETH": "ETHUSDT",
       "BNB": "BNBUSDT",
       # ... etc
   }
   ```

4. Response parsing:
   ```python
   # Binance klines format (array of arrays)
   [
       [1234567890000, "100.0", "105.0", "99.0", "104.0", "1000000", ...]
       # [open_time, open, high, low, close, volume, ...]
   ]
   ```

5. Handle timezone (Binance uses UTC)

**Acceptance Criteria**:
- [ ] Can fetch 1 year of daily data for BTC
- [ ] All 10 crypto symbols work (BTC, ETH, BNB, XRP, ADA, SOL, DOT, MATIC, LTC, LINK)
- [ ] Returns properly formatted DataFrame
- [ ] Handles API errors gracefully
- [ ] Unit tests pass

**Test File to Create**: `tests/test_binance_client.py`

---

## Task 1.1.4: Add Data Caching Layer

**Context**:
To avoid redundant API calls and respect rate limits, we need a caching layer.

**Objective**:
Implement a file-based cache for OHLCV data.

**Files to Modify**:
- `trading_system/data_pipeline/cache/data_cache.py`

**Requirements**:
1. Implement `DataCache` class:
   ```python
   class DataCache:
       def __init__(self, cache_dir: Path, ttl_hours: int = 24):
           pass

       def get(self, key: str) -> Optional[pd.DataFrame]:
           """Get cached data if not expired."""
           pass

       def set(self, key: str, data: pd.DataFrame) -> None:
           """Cache data with timestamp."""
           pass

       def is_valid(self, key: str) -> bool:
           """Check if cache entry exists and is not expired."""
           pass

       def clear(self, key: Optional[str] = None) -> None:
           """Clear specific key or all cache."""
           pass

       def get_cache_key(
           self,
           symbol: str,
           asset_class: str,
           start_date: date,
           end_date: date
       ) -> str:
           """Generate cache key."""
           return f"{asset_class}_{symbol}_{start_date}_{end_date}"
   ```

2. Cache format: Parquet files with metadata
   ```
   cache/
   ├── equity_AAPL_2024-01-01_2024-12-30.parquet
   ├── equity_AAPL_2024-01-01_2024-12-30.meta.json  # {"cached_at": "...", "ttl_hours": 24}
   └── crypto_BTC_2024-01-01_2024-12-30.parquet
   ```

3. TTL handling:
   - Default: 24 hours
   - Check metadata timestamp before returning cached data
   - Delete expired entries on access

**Acceptance Criteria**:
- [ ] Cache stores and retrieves DataFrames correctly
- [ ] TTL expiration works
- [ ] Cache key generation is deterministic
- [ ] Unit tests pass

---

## Task 1.1.5: Implement Live Data Fetcher Orchestrator

**Context**:
The orchestrator coordinates data fetching from multiple sources with caching.

**Objective**:
Implement the main `LiveDataFetcher` class that ties everything together.

**Files to Modify**:
- `trading_system/data_pipeline/live_data_fetcher.py`

**Requirements**:
1. Implement `LiveDataFetcher`:
   ```python
   class LiveDataFetcher:
       def __init__(self, config: DataPipelineConfig):
           self.config = config
           self.cache = DataCache(config.cache_path, config.cache_ttl_hours)
           self.polygon = PolygonClient(config.polygon_api_key) if config.polygon_api_key else None
           self.binance = BinanceClient()

       async def fetch_daily_data(
           self,
           symbols: List[str],
           asset_class: str,  # 'equity' or 'crypto'
           lookback_days: int = 252
       ) -> Dict[str, pd.DataFrame]:
           """Fetch daily OHLCV for multiple symbols.

           Uses cache first, then API.
           """
           pass

       async def fetch_latest_bars(
           self,
           symbols: List[str],
           asset_class: str
       ) -> Dict[str, Bar]:
           """Fetch most recent bar for each symbol."""
           pass

       def _get_source(self, asset_class: str) -> BaseDataSource:
           """Get appropriate data source for asset class."""
           if asset_class == 'equity':
               if not self.polygon:
                   raise DataFetchError("Polygon API key required for equities")
               return self.polygon
           elif asset_class == 'crypto':
               return self.binance
           else:
               raise ValueError(f"Unknown asset class: {asset_class}")
   ```

2. Implement cache-first logic:
   - Check cache for each symbol
   - Fetch only missing data from API
   - Update cache with new data

3. Progress logging:
   ```python
   logger.info(f"Fetching {len(symbols)} {asset_class} symbols")
   logger.info(f"Cache hits: {cache_hits}, API fetches: {api_fetches}")
   ```

**Acceptance Criteria**:
- [ ] Fetches equity data via Polygon (with valid API key)
- [ ] Fetches crypto data via Binance
- [ ] Uses cache to avoid redundant API calls
- [ ] Handles partial cache hits
- [ ] Integration tests pass

---

## Task 1.1.6: Write Tests and Create CLI Command

**Context**:
We need comprehensive tests and a CLI command for manual testing.

**Objective**:
Create tests for the data pipeline and a CLI command.

**Files to Create**:
- `tests/test_data_pipeline.py`
- Update `trading_system/cli.py`

**Requirements**:
1. Create integration test:
   ```python
   class TestLiveDataFetcher:
       @pytest.fixture
       def fetcher(self):
           config = DataPipelineConfig(
               polygon_api_key=os.getenv("POLYGON_API_KEY"),
               cache_path=Path("tests/fixtures/cache")
           )
           return LiveDataFetcher(config)

       @pytest.mark.integration
       async def test_fetch_equity_data(self, fetcher):
           """Test fetching equity data."""
           data = await fetcher.fetch_daily_data(
               symbols=["AAPL", "MSFT"],
               asset_class="equity",
               lookback_days=30
           )
           assert "AAPL" in data
           assert len(data["AAPL"]) >= 20

       @pytest.mark.integration
       async def test_fetch_crypto_data(self, fetcher):
           """Test fetching crypto data."""
           data = await fetcher.fetch_daily_data(
               symbols=["BTC", "ETH"],
               asset_class="crypto",
               lookback_days=30
           )
           assert "BTC" in data
   ```

2. Add CLI command:
   ```python
   @cli.command()
   @click.option("--symbols", required=True, help="Comma-separated symbols")
   @click.option("--asset-class", type=click.Choice(["equity", "crypto"]), required=True)
   @click.option("--days", default=30, help="Lookback days")
   def fetch_data(symbols: str, asset_class: str, days: int):
       """Fetch OHLCV data for symbols."""
       pass
   ```

**Acceptance Criteria**:
- [ ] Unit tests pass without API keys (mocked)
- [ ] Integration tests pass with API keys
- [ ] CLI command works: `trading-system fetch-data --symbols AAPL,MSFT --asset-class equity --days 30`

---

## Task 1.2.1: Create Signals Module Structure

**Context**:
The signals module generates live trading recommendations.

**Objective**:
Create the directory structure for the signals module.

**Files to Create**:
```
trading_system/signals/
├── __init__.py
├── config.py                    # Configuration
├── recommendation.py            # Recommendation dataclass
├── live_signal_generator.py     # Main orchestrator (stub)
├── generators/
│   ├── __init__.py
│   ├── technical_signals.py     # Stub
│   └── combined_signals.py      # Stub
├── filters/
│   ├── __init__.py
│   ├── quality_filter.py        # Stub
│   └── portfolio_filter.py      # Stub
└── rankers/
    ├── __init__.py
    └── signal_scorer.py         # Stub
```

**Requirements**:
1. Create `recommendation.py`:
   ```python
   from dataclasses import dataclass
   from datetime import datetime
   from typing import List, Optional

   @dataclass
   class Recommendation:
       """A trading recommendation to deliver to user."""
       id: str
       symbol: str
       asset_class: str  # 'equity' or 'crypto'
       direction: str    # 'BUY' or 'SELL'
       conviction: str   # 'HIGH', 'MEDIUM', 'LOW'

       # Prices
       current_price: float
       entry_price: float  # Expected fill price (next open)
       target_price: float
       stop_price: float

       # Sizing
       position_size_pct: float
       risk_pct: float

       # Scores
       technical_score: float
       news_score: Optional[float] = None
       sentiment_score: Optional[float] = None
       combined_score: float = 0.0

       # Context
       signal_type: str  # 'breakout_20d', 'breakout_55d', etc.
       reasoning: str
       news_headlines: List[str] = None

       # Metadata
       generated_at: datetime = None
       strategy_name: str = None

       def __post_init__(self):
           if self.generated_at is None:
               self.generated_at = datetime.now()
           if self.news_headlines is None:
               self.news_headlines = []
   ```

2. Create `config.py`:
   ```python
   class SignalConfig(BaseModel):
       max_recommendations: int = 5
       min_conviction: str = "MEDIUM"  # Minimum to include
       technical_weight: float = 1.0
       news_weight: float = 0.0  # Phase 1: technical only
   ```

**Acceptance Criteria**:
- [ ] All files created
- [ ] Imports work correctly
- [ ] Recommendation dataclass is complete

---

## Task 1.2.2: Adapt Existing Strategies for Live Use

**Context**:
The existing backtesting strategies need to be adapted for live signal generation.

**Objective**:
Create a wrapper that uses existing strategies to generate live signals.

**Files to Create/Modify**:
- `trading_system/signals/generators/technical_signals.py`

**Requirements**:
1. Create `TechnicalSignalGenerator`:
   ```python
   class TechnicalSignalGenerator:
       """Generate signals using existing strategy logic."""

       def __init__(
           self,
           strategies: List[StrategyInterface],
           feature_computer: FeatureComputer
       ):
           self.strategies = strategies
           self.feature_computer = feature_computer

       def generate_signals(
           self,
           ohlcv_data: Dict[str, pd.DataFrame],
           current_date: date,
           portfolio_state: Optional[Portfolio] = None
       ) -> List[Signal]:
           """Generate signals for current date.

           Args:
               ohlcv_data: OHLCV data keyed by symbol
               current_date: The date to generate signals for
               portfolio_state: Current portfolio (for exit signals)

           Returns:
               List of Signal objects
           """
           signals = []

           for symbol, data in ohlcv_data.items():
               # Compute features
               features = self.feature_computer.compute(data, symbol)

               # Get latest feature row
               if current_date not in features.index:
                   continue
               feature_row = features.loc[current_date]

               # Check each strategy
               for strategy in self.strategies:
                   if strategy.asset_class != self._get_asset_class(symbol):
                       continue

                   # Check eligibility
                   if not strategy.check_eligibility(feature_row):
                       continue

                   # Check entry triggers
                   entry_signals = strategy.check_entry_triggers(
                       feature_row,
                       current_date
                   )
                   signals.extend(entry_signals)

           return signals
   ```

2. Ensure compatibility with existing:
   - `EquityMomentumStrategy`
   - `CryptoMomentumStrategy`

**Acceptance Criteria**:
- [ ] Generates signals using existing strategy logic
- [ ] Works with live OHLCV data
- [ ] Produces same signals as backtester for same data
- [ ] Unit tests pass

---

## Task 1.2.3: Implement Signal Scoring and Recommendation Creation

**Context**:
Raw signals need to be scored, ranked, and converted to recommendations.

**Objective**:
Implement signal scoring and recommendation creation.

**Files to Modify**:
- `trading_system/signals/rankers/signal_scorer.py`
- `trading_system/signals/live_signal_generator.py`

**Requirements**:
1. Implement `SignalScorer`:
   ```python
   class SignalScorer:
       """Score and rank signals."""

       def __init__(self, config: SignalConfig):
           self.config = config

       def score_signals(
           self,
           signals: List[Signal],
           features: Dict[str, FeatureRow]
       ) -> List[Tuple[Signal, float]]:
           """Score each signal and return sorted list."""
           scored = []

           for signal in signals:
               feature = features.get(signal.symbol)
               if not feature:
                   continue

               # Calculate component scores (0-10 scale)
               breakout_score = self._score_breakout(signal, feature)
               momentum_score = self._score_momentum(feature)

               # Combined score (Phase 1: technical only)
               combined = (
                   breakout_score * 0.6 +
                   momentum_score * 0.4
               )

               scored.append((signal, combined))

           # Sort by score descending
           scored.sort(key=lambda x: x[1], reverse=True)
           return scored

       def _score_breakout(self, signal: Signal, feature: FeatureRow) -> float:
           """Score breakout strength (0-10)."""
           # Breakout clearance above high
           if signal.breakout_type == BreakoutType.FAST_20D:
               clearance = (feature.close - feature.highest_close_20d) / feature.atr14
           else:
               clearance = (feature.close - feature.highest_close_55d) / feature.atr14

           # Normalize to 0-10 (1 ATR clearance = 10)
           return min(clearance * 10, 10)

       def _score_momentum(self, feature: FeatureRow) -> float:
           """Score momentum strength (0-10)."""
           # ROC60 normalized (20% = 10)
           return min(feature.roc60 * 50, 10) if feature.roc60 > 0 else 0
   ```

2. Implement recommendation creation in `LiveSignalGenerator`:
   ```python
   def _create_recommendation(
       self,
       signal: Signal,
       score: float,
       feature: FeatureRow,
       config: StrategyConfig
   ) -> Recommendation:
       """Convert signal to recommendation."""

       # Calculate prices
       entry_price = feature.close  # Will execute at next open
       stop_price = entry_price - (config.exit.hard_stop_atr_mult * feature.atr14)
       target_price = entry_price + (2 * (entry_price - stop_price))  # 2:1 R/R

       # Calculate position size
       risk_amount = 0.0075  # 0.75% risk per trade
       stop_distance = entry_price - stop_price
       position_size_pct = risk_amount / (stop_distance / entry_price)
       position_size_pct = min(position_size_pct, 0.15)  # Max 15%

       # Determine conviction
       if score >= 8:
           conviction = "HIGH"
       elif score >= 6:
           conviction = "MEDIUM"
       else:
           conviction = "LOW"

       return Recommendation(
           id=str(uuid4()),
           symbol=signal.symbol,
           asset_class=signal.asset_class,
           direction="BUY",
           conviction=conviction,
           current_price=feature.close,
           entry_price=entry_price,
           target_price=target_price,
           stop_price=stop_price,
           position_size_pct=position_size_pct,
           risk_pct=risk_amount,
           technical_score=score,
           combined_score=score,
           signal_type=signal.breakout_type.value,
           reasoning=self._generate_reasoning(signal, feature, score),
           strategy_name=signal.strategy_name
       )
   ```

**Acceptance Criteria**:
- [ ] Signals are scored consistently
- [ ] Recommendations have all required fields
- [ ] Position sizing respects risk limits
- [ ] Unit tests pass

---

## Task 1.3.1: Create Email Module Structure

**Context**:
We need to send daily email reports with recommendations.

**Objective**:
Create the email module structure.

**Files to Create**:
```
trading_system/output/
├── __init__.py
├── email/
│   ├── __init__.py
│   ├── config.py
│   ├── email_service.py         # Main email sender
│   ├── report_generator.py      # Generate report content
│   └── templates/
│       ├── base.html
│       ├── daily_signals.html
│       └── styles.css
└── formatters/
    ├── __init__.py
    └── recommendation_formatter.py
```

**Requirements**:
1. Create `config.py`:
   ```python
   class EmailConfig(BaseModel):
       smtp_host: str = "smtp.sendgrid.net"
       smtp_port: int = 587
       smtp_user: str = "apikey"
       smtp_password: str  # SendGrid API key
       from_email: str = "signals@yourdomain.com"
       from_name: str = "Trading Assistant"
       recipients: List[str]
   ```

2. Create HTML template `daily_signals.html` (use the format from COMPLETE_SYSTEM_VISION.md)

**Acceptance Criteria**:
- [ ] All files created
- [ ] Templates render correctly
- [ ] Configuration is complete

---

## Task 1.3.2: Implement Email Service

**Context**:
Need to actually send emails via SMTP/SendGrid.

**Objective**:
Implement the email sending functionality.

**Files to Modify**:
- `trading_system/output/email/email_service.py`

**Requirements**:
1. Implement `EmailService`:
   ```python
   import smtplib
   from email.mime.multipart import MIMEMultipart
   from email.mime.text import MIMEText
   from jinja2 import Environment, FileSystemLoader

   class EmailService:
       def __init__(self, config: EmailConfig):
           self.config = config
           self.jinja_env = Environment(
               loader=FileSystemLoader("trading_system/output/email/templates")
           )

       async def send_daily_report(
           self,
           recommendations: List[Recommendation],
           market_summary: Dict[str, Any],
           date: date
       ) -> bool:
           """Send daily signal report."""

           # Render template
           template = self.jinja_env.get_template("daily_signals.html")
           html_content = template.render(
               recommendations=recommendations,
               market=market_summary,
               date=date.strftime("%B %d, %Y"),
               generated_at=datetime.now().strftime("%I:%M %p ET")
           )

           # Create message
           msg = MIMEMultipart("alternative")
           msg["Subject"] = f"Trading Signals for {date.strftime('%b %d')} - {len(recommendations)} Recommendations"
           msg["From"] = f"{self.config.from_name} <{self.config.from_email}>"
           msg["To"] = ", ".join(self.config.recipients)

           msg.attach(MIMEText(html_content, "html"))

           # Send
           return await self._send_smtp(msg)

       async def _send_smtp(self, msg: MIMEMultipart) -> bool:
           """Send via SMTP."""
           try:
               with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                   server.starttls()
                   server.login(self.config.smtp_user, self.config.smtp_password)
                   server.send_message(msg)
               return True
           except Exception as e:
               logger.error(f"Failed to send email: {e}")
               return False
   ```

2. Add test email functionality:
   ```python
   async def send_test_email(self) -> bool:
       """Send a test email to verify configuration."""
       pass
   ```

**Acceptance Criteria**:
- [ ] Can send HTML emails via SendGrid
- [ ] Templates render correctly
- [ ] Test email works
- [ ] Errors are logged appropriately

---

## Task 1.4.1: Create Scheduler Module

**Context**:
Need to automate daily execution.

**Objective**:
Implement the cron scheduler.

**Files to Create**:
```
trading_system/scheduler/
├── __init__.py
├── config.py
├── cron_runner.py
└── jobs/
    ├── __init__.py
    └── daily_signals_job.py
```

**Requirements**:
1. Use APScheduler for scheduling:
   ```python
   from apscheduler.schedulers.asyncio import AsyncIOScheduler
   from apscheduler.triggers.cron import CronTrigger
   ```

2. Implement `CronRunner`:
   ```python
   class CronRunner:
       def __init__(self, config: SchedulerConfig):
           self.config = config
           self.scheduler = AsyncIOScheduler()

       def register_jobs(self):
           """Register all scheduled jobs."""
           # Daily equity signals - 4:30 PM ET
           self.scheduler.add_job(
               daily_signals_job,
               CronTrigger(
                   hour=16,
                   minute=30,
                   timezone="America/New_York"
               ),
               id="daily_equity_signals",
               kwargs={"asset_class": "equity"}
           )

           # Daily crypto signals - midnight UTC
           self.scheduler.add_job(
               daily_signals_job,
               CronTrigger(
                   hour=0,
                   minute=0,
                   timezone="UTC"
               ),
               id="daily_crypto_signals",
               kwargs={"asset_class": "crypto"}
           )

       def start(self):
           self.scheduler.start()

       def stop(self):
           self.scheduler.shutdown()
   ```

3. Implement `daily_signals_job`:
   ```python
   async def daily_signals_job(asset_class: str):
       """Execute daily signal generation and email."""
       logger.info(f"Starting daily signals job for {asset_class}")

       try:
           # 1. Load configuration
           config = load_config()

           # 2. Initialize components
           data_fetcher = LiveDataFetcher(config.data_pipeline)
           signal_generator = LiveSignalGenerator(config.signals)
           email_service = EmailService(config.email)

           # 3. Fetch data
           symbols = config.universe[asset_class]
           ohlcv_data = await data_fetcher.fetch_daily_data(
               symbols=symbols,
               asset_class=asset_class,
               lookback_days=252
           )

           # 4. Generate signals
           recommendations = await signal_generator.generate(
               ohlcv_data=ohlcv_data,
               asset_class=asset_class,
               current_date=date.today()
           )

           # 5. Send email
           await email_service.send_daily_report(
               recommendations=recommendations,
               market_summary=get_market_summary(ohlcv_data),
               date=date.today()
           )

           logger.info(f"Daily signals job completed: {len(recommendations)} recommendations")

       except Exception as e:
           logger.error(f"Daily signals job failed: {e}")
           # Send error alert
           await send_error_alert(e)
   ```

**Acceptance Criteria**:
- [ ] Scheduler starts and stops cleanly
- [ ] Jobs execute at correct times
- [ ] Errors are caught and logged
- [ ] Can run job manually for testing

---

## Task 1.4.2: Create Main Entry Point and CLI

**Context**:
Need a way to run the scheduler and test components.

**Objective**:
Create main entry point and CLI commands.

**Files to Modify**:
- `trading_system/__main__.py`
- `trading_system/cli.py`

**Requirements**:
1. Add CLI commands:
   ```python
   @cli.command()
   def run_scheduler():
       """Run the scheduler daemon."""
       runner = CronRunner(config)
       runner.register_jobs()
       runner.start()

       # Keep running
       try:
           asyncio.get_event_loop().run_forever()
       except KeyboardInterrupt:
           runner.stop()

   @cli.command()
   @click.option("--asset-class", type=click.Choice(["equity", "crypto"]), required=True)
   def run_signals_now(asset_class: str):
       """Run signal generation immediately (for testing)."""
       asyncio.run(daily_signals_job(asset_class))

   @cli.command()
   def send_test_email():
       """Send a test email to verify configuration."""
       pass
   ```

2. Create systemd/launchd service file for deployment

**Acceptance Criteria**:
- [ ] `trading-system run-scheduler` starts daemon
- [ ] `trading-system run-signals-now --asset-class equity` works
- [ ] `trading-system send-test-email` works
- [ ] Service file created for deployment

---

## Dependencies to Install

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
live = [
    "aiohttp>=3.8.0,<4.0.0",      # Async HTTP
    "apscheduler>=3.10.0,<4.0.0", # Scheduling
    "jinja2>=3.1.0,<4.0.0",       # Email templates
    "python-dotenv>=1.0.0,<2.0.0", # Environment variables
]
```

---

## Environment Variables

Create `.env.example`:
```bash
# API Keys
POLYGON_API_KEY=your_polygon_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWSAPI_API_KEY=your_newsapi_key_here

# Email (SendGrid)
SENDGRID_API_KEY=your_sendgrid_key_here
EMAIL_RECIPIENTS=your@email.com

# Optional
LOG_LEVEL=INFO
```

---

**Document Created**: 2024-12-30
**Phase**: 1 - MVP
**Tasks**: 14 total
