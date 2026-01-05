# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- API documentation generation with Sphinx
- User guide with step-by-step examples
- Troubleshooting guide for common issues
- Migration guide for config and version changes
- Interactive Jupyter notebook examples
- Strategy template generator CLI command
- Pre-commit hooks for code quality
- Enhanced performance benchmarks

## [0.0.2] - 2024-12-19

### Added

#### Core Features
- **Backtest Engine**: Event-driven backtesting engine with daily event loop
  - No lookahead bias enforcement
  - Walk-forward splits (train/validation/holdout)
  - Deterministic results with seeded RNG
  - Comprehensive event logging

- **Strategy Implementations**:
  - Equity momentum strategy with breakout-based entries
  - Crypto momentum strategy with crypto-specific parameters
  - Mean reversion strategy
  - Pairs trading strategy
  - Multi-timeframe strategy
  - Factor-based strategy

- **Data Pipeline**:
  - Multiple data source support (CSV, database, API, Parquet, HDF5)
  - OHLCV data validation and quality checks
  - Missing data handling and gap detection
  - Data loading adapters for various formats

- **Technical Indicators Library**:
  - Moving averages (MA20, MA50, MA200)
  - Average True Range (ATR14)
  - Rate of Change (ROC60)
  - Breakout level calculations
  - Average Dollar Volume (ADV20)
  - Momentum indicators
  - Volatility indicators

- **Execution Simulation**:
  - Realistic slippage models based on ADV and volatility
  - Fee calculation (configurable per market)
  - Capacity constraints (order size vs ADV)
  - Stress slippage during market stress scenarios
  - Fill simulation with partial fills

- **Portfolio Management**:
  - Risk-based position sizing (0.75% risk per trade)
  - Correlation guards for portfolio diversification
  - Volatility scaling
  - Capacity constraints
  - Portfolio state machine with proper update sequence
  - Cash and exposure tracking

- **Validation Suite**:
  - Bootstrap analysis for statistical validation
  - Permutation tests
  - Stress tests (slippage, bear market, range market, flash crash)
  - Sensitivity analysis with parameter grid search
  - Correlation analysis for portfolio monitoring

- **Reporting & Metrics**:
  - CSV outputs (equity curve, trade log, weekly summaries)
  - JSON reports with monthly performance metrics
  - Performance metrics (Sharpe ratio, Calmar ratio, max drawdown, R-multiples, profit factor)
  - Benchmark comparison (SPY/BTC)
  - Report generation CLI command
  - Comparison reports (train vs validation vs holdout)

- **CLI Interface**:
  - Backtest command with period selection
  - Validation suite command
  - Holdout evaluation command
  - Report generation command
  - Rich console output with progress bars
  - Configuration wizard

- **Machine Learning Infrastructure**:
  - ML predictor integration with backtest engine
  - Feature engineering pipeline
  - Model training infrastructure
  - Model versioning and storage
  - ML-enhanced signal scoring (score enhancement, filter, replace modes)

- **Real-time Trading Infrastructure**:
  - Paper trading adapters (Alpaca, Interactive Brokers)
  - Live trading infrastructure
  - Real-time data integration hooks

- **Results Storage**:
  - Database storage for backtest results
  - Results retrieval and analysis
  - Run ID tracking

#### Infrastructure
- **Docker Support**:
  - Dockerfile for containerized execution
  - Docker Compose configuration
  - Volume mounts for data and configs

- **CI/CD**:
  - GitHub Actions workflow
  - Automated testing

- **Dependency Management**:
  - pyproject.toml with proper project metadata
  - Optional dependency groups (dev, database, api, storage, ml, visualization, performance)
  - Version constraints for all dependencies

- **Logging**:
  - Enhanced logging with loguru
  - Structured logging
  - Log file output

- **Type Safety**:
  - Type hints throughout codebase
  - Pydantic models for configuration validation
  - Type checking configuration (mypy)

- **Error Handling**:
  - Comprehensive exception handling
  - Custom exception classes
  - Graceful error recovery

#### Testing
- **Comprehensive Test Suite**:
  - Unit tests for all major components
  - Integration tests for end-to-end workflows
  - Property-based tests with Hypothesis
  - Performance benchmarks
  - Edge case testing
  - Missing data handling tests
  - Fuzz testing

- **Test Fixtures**:
  - Sample equity data (AAPL, MSFT, GOOGL)
  - Sample crypto data (BTC, ETH, SOL)
  - Benchmark data (SPY, BTC)
  - Test configurations
  - Expected trade outputs

#### Documentation
- **Architecture Documentation**:
  - System architecture overview
  - Configuration guide
  - Data pipeline documentation
  - Indicators library documentation
  - Strategy details
  - Backtest engine documentation
  - Validation suite documentation
  - Portfolio state machine documentation

- **User Documentation**:
  - README with quick start guide
  - Testing guide
  - Quick start testing reference
  - FAQ section
  - Example configurations

### Changed
- Initial release - no previous versions

### Fixed
- Initial release - no previous bug fixes

### Security
- Initial release - no security fixes

---

## Version History

- **0.0.2** (2024-12-19): Initial release with core backtesting functionality

---

## Notes

- This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format
- Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
- Dates are in YYYY-MM-DD format
