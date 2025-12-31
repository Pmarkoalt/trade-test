# Production Readiness Checklist

**Date**: 2024-12-19
**Version**: 0.0.2
**Status**: Pre-Production Verification Guide

---

## Overview

This document provides a comprehensive checklist for verifying production readiness before deploying the trading system to a production environment. Follow each section systematically to ensure all critical paths are verified and the system is ready for production use.

---

## 🔴 Critical Pre-Deployment Checklist

### 1. Code Quality & Testing

#### Test Coverage Verification
- [ ] **Generate test coverage report**

  **Note**: If you encounter a segmentation fault when running coverage (numpy import issue on macOS), use Docker instead:

  **Option 1: Using Docker (Recommended - avoids environment issues)**
  ```bash
  make docker-test-coverage
  # Or directly:
  docker-compose run --rm --entrypoint pytest trading-system tests/ \
    --cov=trading_system --cov-report=html --cov-report=term-missing
  ```

  **Option 2: Using Makefile**
  ```bash
  make test-coverage
  ```

  **Option 3: Direct pytest (if environment is stable)**
  ```bash
  pytest --cov=trading_system --cov-report=html --cov-report=term
  ```

  **Option 4: Using coverage script**
  ```bash
  ./scripts/coverage_report.sh
  ```

  - Target: >90% coverage
  - Review uncovered code paths in `htmlcov/index.html`
  - Add tests for any critical uncovered paths
  - **Troubleshooting**: If you see "Segmentation fault" during test collection, this is likely a numpy/macOS compatibility issue. Use Docker (Option 1) to avoid this.

- [ ] **Run full test suite**

  **Note**: If you encounter a segmentation fault when running tests (numpy import issue on macOS), use Docker instead:

  **Option 1: Using Docker (Recommended - avoids environment issues)**
  ```bash
  make docker-test-all
  # Or directly:
  docker-compose run --rm --entrypoint pytest trading-system tests/ -v --tb=short
  ```

  **Option 2: Using Makefile**
  ```bash
  make test-all
  ```

  **Option 3: Direct pytest (if environment is stable)**
  ```bash
  pytest tests/ -v --tb=short
  ```

  **Option 4: Run unit and integration tests separately**
  ```bash
  # Unit tests only
  make test-unit
  # Integration tests only
  make test-integration
  ```

  - All tests must pass
  - No skipped critical tests
  - Verify integration tests run successfully
  - **Troubleshooting**: If you see "Segmentation fault" during test collection, this is likely a numpy/macOS compatibility issue. Use Docker (Option 1) to avoid this.

- [ ] **Run end-to-end integration test**

  **Note**: If you encounter a segmentation fault when running tests (numpy import issue on macOS), use Docker instead:

  **Option 1: Using Docker (Recommended - avoids environment issues)**
  ```bash
  make docker-test-integration
  # Or run specific test:
  docker-compose run --rm --entrypoint pytest trading-system \
    tests/integration/test_end_to_end.py::TestFullBacktest -v
  ```

  **Option 2: Using Makefile (runs all integration tests)**
  ```bash
  make test-integration
  ```

  **Option 3: Direct pytest (if environment is stable)**
  ```bash
  pytest tests/integration/test_end_to_end.py::TestFullBacktest -v
  ```

  **What to verify:**
  - Verify `TestFullBacktest` is not skipped
  - Verify expected trades match specifications
  - Verify all output files are generated correctly
  - Verify metrics are reasonable (finite values, valid ranges)
  - Check that results contain: `total_trades`, `total_return`, `sharpe_ratio`, `max_drawdown`, `win_rate`

  **Troubleshooting**: If you see "Segmentation fault" during test collection, this is likely a numpy/macOS compatibility issue. Use Docker (Option 1) to avoid this.

#### Code Quality Checks
- [ ] **Linter checks**
  ```bash
  flake8 trading_system/ tests/ --max-line-length=127 --extend-ignore=E203,E266,E501,W503
  ```
  - Zero blocking linter errors
  - Review and fix warnings
  - Note: Configuration is in `.flake8` file, so flags are optional

- [ ] **Type checking** (if using mypy)
  ```bash
  mypy trading_system/ --config-file pyproject.toml
  ```
  - No critical type errors
  - Note: Configuration is in `pyproject.toml` file (includes `ignore_missing_imports = true`)

- [ ] **Review TODO comments**
  ```bash
  grep -r "TODO" trading_system/ --exclude-dir=__pycache__
  ```
  - Verify no critical TODOs in production code
  - Document deferred items
  - Note: TODOs in `strategy_template_generator.py` are expected (template placeholders)

---

### 2. Configuration Validation

#### Configuration File Verification
- [ ] **Validate all example configs**
  ```bash
  for config in EXAMPLE_CONFIGS/*.yaml; do
    python -m trading_system config validate --path "$config" || echo "FAILED: $config"
  done
  ```
  - All configs must validate successfully
  - Verify no deprecated parameters
  - Note: Requires dependencies installed (`pip install -r requirements.txt`)

- [ ] **Test production config**
  - Create production configuration file
  - Validate production config structure
  - Verify all required parameters are set
  - Test config loading:
    ```bash
    python -m trading_system config validate --path production_config.yaml
    ```

#### Environment Variables
- [ ] **Verify environment variable handling**
  - Check for hardcoded paths or secrets
  - Verify API keys are loaded from environment variables
  - Test with missing environment variables (should fail gracefully)

- [ ] **Document required environment variables**
  - API keys (if using API data sources)
  - Database connection strings (if using database storage)
  - File paths for data and results

---

### 3. Security Review

#### API Keys & Secrets
- [ ] **Verify no hardcoded secrets**
  ```bash
  grep -r "api_key.*=" trading_system/ --exclude-dir=__pycache__ | grep -v "#"
  grep -r "password.*=" trading_system/ --exclude-dir=__pycache__ | grep -v "#"
  grep -r "secret.*=" trading_system/ --exclude-dir=__pycache__ | grep -v "#"
  ```
  - No hardcoded API keys, passwords, or secrets
  - All secrets loaded from environment variables or secure config

- [ ] **Verify secure credential handling**
  - API keys stored in environment variables
  - Config files do not contain secrets (use `.env` files)
  - `.env` files are in `.gitignore`
  - Database credentials are secure

#### Data Handling
- [x] **Verify data validation**
  - ✅ Input data is validated before processing
    - `validate_ohlcv()` function validates all OHLCV data (columns, OHLC relationships, negative values, date order, duplicates)
    - Used in all data sources: CSV, Parquet, HDF5, API, Database
  - ✅ Malformed data is rejected with clear errors
    - `DataValidationError` raised with descriptive messages including symbol name and specific issues
    - Error messages are actionable (e.g., "Invalid OHLC at dates: ['2023-01-15']")
  - ✅ No SQL injection risks (if using database)
    - All database queries use parameterized queries with placeholders (`?` for SQLite, `%s` for PostgreSQL)
    - User input (symbols, dates) passed as parameters, never concatenated into SQL strings
    - Verified in `trading_system/data/sources/database_source.py`
  - ⚠️ File path validation prevents directory traversal
    - Symbols are normalized (upper case, stripped) in universe loading
    - File paths checked for existence before use
    - Symbols validated against available files list
    - **Note**: No explicit sanitization for `../` patterns, but mitigated by existence checks and symbol validation

- [x] **Review file permissions**
  - ⚠️ Results directory has appropriate permissions
    - Directories created with `mkdir(parents=True, exist_ok=True)` - uses default permissions (typically 755)
    - Files created with default permissions (typically 644)
    - **Recommendation**: Explicitly set permissions in production (e.g., `chmod 755` for directories, `chmod 644` for files)
    - Verified in: `trading_system/storage/database.py`, `trading_system/reporting/report_generator.py`
  - ✅ Config files are read-only in production
    - Docker setup enforces read-only mounts (`:ro` flag) for config directories
    - Verified in `docker-compose.yml`: `./EXAMPLE_CONFIGS:/app/configs:ro`, `./configs:/app/custom_configs:ro`
    - **Note**: No explicit read-only enforcement in Python code (relies on Docker/OS level)
  - ⚠️ Log files have appropriate permissions
    - Log files created using `RotatingFileHandler` and `FileHandler` - uses default permissions (typically 644)
    - **Recommendation**: Explicitly set log file permissions in production (e.g., `chmod 644`)
    - Verified in: `trading_system/logging/logger.py`, `trading_system/cli.py`
  - **Summary**: Codebase relies on default filesystem permissions. Docker enforces read-only configs. Production should explicitly set permissions via deployment scripts or system configuration.

---

### 4. Error Handling Verification

#### Critical Path Error Handling
- [ ] **Test data loading failures**
  ```bash
  # Create a test config with nonexistent data paths
  # Copy an existing config and modify data paths
  cp tests/fixtures/configs/run_test_config.yaml /tmp/test_nonexistent_data_config.yaml
  # Edit the file and set dataset paths to nonexistent directories:
  #   equity_path: "/nonexistent/data/path/equity"
  #   crypto_path: "/nonexistent/data/path/crypto"
  #   benchmark_path: "/nonexistent/data/path/benchmarks"

  # Run backtest with the test config
  python -m trading_system backtest --config /tmp/test_nonexistent_data_config.yaml
  ```
  - Verify graceful error handling (should not crash)
  - Error messages are clear and actionable
  - Error includes data path information
  - Troubleshooting tips are provided in the output
  - Expected: System should raise DataError or ValueError with helpful context

- [ ] **Test invalid configuration**
  ```bash
  # Test 1: Invalid YAML syntax
  # Create a test config file with YAML syntax errors (e.g., missing closing quote, wrong indentation)
  # Then run:
  python -m trading_system backtest --config /tmp/invalid_yaml_config.yaml

  # Test 2: Missing required fields
  # Create a config file missing required sections (e.g., dataset section)
  python -m trading_system backtest --config /tmp/invalid_missing_fields.yaml

  # Test 3: Invalid field values
  # Copy an existing config and modify values to be invalid:
  cp EXAMPLE_CONFIGS/run_config.yaml /tmp/invalid_values.yaml
  # Edit /tmp/invalid_values.yaml and change:
  #   - start_date to "01/01/2023" (wrong format, should be "2023-01-01")
  #   - Or change a percentage field to 150 (should be 0-1 range like 0.50)
  python -m trading_system backtest --config /tmp/invalid_values.yaml

  # Test 4: Use config validate command for cleaner error output
  python -m trading_system config validate --path /tmp/invalid_yaml_config.yaml
  ```
  - Config validation catches errors early (before data loading)
  - Error messages point to specific issues (field names and paths)
  - Error messages include troubleshooting hints and suggestions
  - Formatted error output shows field paths (e.g., "dataset -> start_date")
  - Provides helpful next steps (e.g., "Use 'config template' to generate a template")
  - Expected: System should raise ConfigurationError with formatted error details

- [ ] **Test missing dependencies**
  ```bash
  # Test 1: Missing data files
  # Option: Temporarily rename a data file to simulate missing file
  cp tests/fixtures/configs/run_test_config.yaml /tmp/test_missing_files_config.yaml
  # Temporarily hide a file: mv data/equity/AAPL.csv data/equity/AAPL.csv.bak
  python -m trading_system backtest --config /tmp/test_missing_files_config.yaml
  # Restore: mv data/equity/AAPL.csv.bak data/equity/AAPL.csv
  # Expected: System should log warnings for missing files but continue processing other symbols

  # Test 2: Corrupted data files
  # Create a test directory with a corrupted CSV file
  mkdir -p /tmp/test_corrupted_data/equity
  cp tests/fixtures/AAPL.csv /tmp/test_corrupted_data/equity/AAPL.csv
  # Edit /tmp/test_corrupted_data/equity/AAPL.csv to introduce errors:
  #   - Change one row so low > high (invalid OHLC relationship)
  #   - Or change a price to negative value
  #   - Or remove a required column header

  cp tests/fixtures/configs/run_test_config.yaml /tmp/test_corrupted_config.yaml
  # Edit /tmp/test_corrupted_config.yaml: set dataset.equity_path to "/tmp/test_corrupted_data/equity"
  python -m trading_system backtest --config /tmp/test_corrupted_config.yaml
  # Expected: System should log validation errors and skip the corrupted symbol, continue with others

  # Test 3: Insufficient data for indicators
  # Create a config with a very short date range (insufficient for indicator calculations)
  cp tests/fixtures/configs/run_test_config.yaml /tmp/test_insufficient_data_config.yaml
  # Edit /tmp/test_insufficient_data_config.yaml:
  #   - Change start_date and end_date to only span 10-20 days
  #   - This won't provide enough history for indicators that need 200+ days
  python -m trading_system backtest --config /tmp/test_insufficient_data_config.yaml
  # Expected: Strategies should handle insufficient data gracefully (return None or add to failures list)
  ```
  - Missing data files: System logs warnings, skips missing symbols, continues with available data
  - Corrupted data files: Validation errors logged, corrupted symbols skipped, system continues
  - Missing indicators: Strategies handle insufficient data gracefully (return None or mark as failed)
  - No crashes or unhandled exceptions
  - Error messages clearly identify the problematic symbol/file
  - System remains stable and processes remaining valid data

- [ ] **Test edge cases**
  ```bash
  # Run all edge case tests
  pytest tests/test_edge_cases.py -v

  # Run missing data handling tests
  pytest tests/test_missing_data_handling.py -v

  # Run edge case integration tests (weekend gaps, extreme moves in end-to-end scenarios)
  pytest tests/integration/test_end_to_end.py::TestEdgeCaseIntegration -v

  # Run specific edge case test classes if needed
  pytest tests/test_edge_cases.py::TestExtremePriceMoves -v
  pytest tests/test_edge_cases.py::TestInsufficientCash -v
  pytest tests/test_edge_cases.py::TestFlashCrashScenarios -v
  pytest tests/test_edge_cases.py::TestInvalidOHLCData -v
  ```
  - All edge case tests pass
  - Missing data handling works correctly:
    - Single day missing: system skips signal generation, logs warning, continues
    - 2+ consecutive days missing: symbol marked unhealthy, positions exited
    - No crashes or infinite loops
  - Extreme price moves (>50%) are handled:
    - Detected during validation (warnings logged)
    - Treated as missing data in backtest (bars skipped)
    - System continues processing normally
  - Weekend gaps are handled correctly:
    - Crypto weekend gaps detected (if expected in date range)
    - Weekend penalty applies for crypto trades
    - System continues after gap
  - Invalid OHLC data is rejected:
    - Invalid relationships (low > high, etc.) detected
    - Validation fails with clear error messages
    - Corrupted symbols skipped, system continues
  - Insufficient cash scenarios handled gracefully:
    - Position sizing returns zero when insufficient cash
    - System doesn't crash on capital constraints
  - Correlation guard handles edge cases:
    - Works correctly with <4 positions (not applicable)
    - Handles insufficient return history gracefully
  - Volatility scaling handles insufficient history:
    - Returns 1.0 multiplier when <20 days history
    - No crashes on insufficient data

#### Failure Mode Testing
- [ ] **Test all error paths**
  - Data source failures (API timeouts, network errors)
  - Invalid data formats
  - Insufficient data for indicators
  - Portfolio constraint violations
  - Execution failures

- [ ] **Verify error logging**
  - Errors are logged with sufficient context
  - Stack traces are captured
  - Error messages include troubleshooting hints

---

### 5. Performance Benchmarks

#### Run Performance Tests
- [ ] **Run performance benchmarks**
  ```bash
  pytest tests/performance/ -m performance --benchmark-only
  ```
  - Compare against baseline
  - Verify no significant regressions (>20% slower)
  - Document expected performance characteristics

- [ ] **Test with production-sized datasets**
  ```bash
  # Test with larger dataset (e.g., 50 symbols, 5 years)
  python -m trading_system backtest --config production_config.yaml
  ```
  - Verify performance is acceptable
  - Memory usage is reasonable
  - Processing time scales appropriately

#### Performance Expectations
Based on `docs/PERFORMANCE_CHARACTERISTICS.md`:

- [ ] **Verify indicator computation**
  - Full feature computation: < 50ms per symbol (10K bars)
  - Scales linearly with symbol count

- [ ] **Verify backtest performance**
  - Single day processing: < 100ms (20 symbols)
  - Full 6-month backtest: < 30s (20 symbols)
  - Scales linearly with time period and symbol count

- [ ] **Verify portfolio operations**
  - Equity update (50 positions): < 5ms
  - Exposure calculation: < 2ms

---

### 6. End-to-End Integration Verification

#### Full Workflow Testing
- [ ] **Test complete backtest workflow**
  ```bash
  python -m trading_system backtest --config EXAMPLE_CONFIGS/run_config.yaml --period train
  ```
  - Verify backtest completes successfully
  - Verify all output files are generated
  - Verify metrics are reasonable

- [ ] **Test walk-forward workflow**
  ```bash
  python -m trading_system backtest --config EXAMPLE_CONFIGS/run_config.yaml --period all
  ```
  - Train period completes
  - Validation period completes
  - Holdout period completes (if configured)
  - Results are consistent across periods

- [ ] **Test validation suite**
  ```bash
  python -m trading_system validate --config EXAMPLE_CONFIGS/run_config.yaml
  ```
  - Bootstrap test completes
  - Permutation test completes
  - Stress tests complete
  - Results are saved correctly

#### Output Verification
- [ ] **Verify output files**
  - Equity curve CSV is generated
  - Trade log CSV is generated
  - Monthly reports (JSON) are generated
  - Dashboard visualizations are generated (if enabled)
  - All files have correct format and data

- [ ] **Verify metrics calculation**
  - Sharpe ratio is calculated correctly
  - Max drawdown is calculated correctly
  - Win rate is calculated correctly
  - All metrics are finite and reasonable

---

### 7. Real-World Data Scenarios

#### Test with Production-Like Data
- [ ] **Test with real market data**

  **Step 1: Download real market data**
  ```bash
  # Install yfinance if not already installed
  pip install yfinance

  # Download real historical market data
  python scripts/download_real_market_data.py --output data/real_market_data/

  # Or download specific symbols and date range
  python scripts/download_real_market_data.py \
    --equity-symbols AAPL MSFT GOOGL AMZN TSLA \
    --crypto-symbols BTC ETH SOL \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --output data/real_market_data/
  ```

  **Step 2: Run real market data tests**
  ```bash
  # Run all real market data tests
  pytest tests/integration/test_real_market_data.py -v

  # Run specific test
  pytest tests/integration/test_real_market_data.py::TestRealMarketData::test_real_data_validation -v

  # Test different market conditions
  pytest tests/integration/test_real_market_data.py::TestRealMarketData::test_bull_market_condition -v
  pytest tests/integration/test_real_market_data.py::TestRealMarketData::test_bear_market_condition -v
  pytest tests/integration/test_real_market_data.py::TestRealMarketData::test_range_market_condition -v
  ```

  **What to verify:**
  - ✅ Real data loads correctly (not just test fixtures)
  - ✅ System handles real-world data quality issues:
    - Missing trading days (holidays, weekends)
    - Extreme price moves (>50% daily returns)
    - Low volume days
    - Duplicate dates (if any)
  - ✅ System performs correctly in different market conditions:
    - **Bull market**: Positive trending periods (e.g., 2020-04 to 2021-11)
    - **Bear market**: Negative trending periods (e.g., 2022)
    - **Range market**: Sideways/volatile periods (e.g., 2019)
  - ✅ Full backtest completes with real data
  - ✅ Metrics are reasonable and finite
  - ✅ No crashes or errors with real data quality issues

  **Expected results:**
  - All real data tests should pass
  - System should handle missing days gracefully
  - Extreme moves should be detected and handled (treated as missing data)
  - Backtest should complete successfully with real data
  - Results should show realistic performance metrics

- [ ] **Test with various data sources**

  **Step 1: Run data source integration tests**
  ```bash
  # Run all data source tests
  pytest tests/integration/test_data_sources.py -v

  # Test specific data source
  pytest tests/integration/test_data_sources.py::TestCSVDataSource -v
  pytest tests/integration/test_data_sources.py::TestSQLiteDataSource -v
  pytest tests/integration/test_data_sources.py::TestParquetDataSource -v
  pytest tests/integration/test_data_sources.py::TestHDF5DataSource -v
  ```

  **Step 2: Test CSV data source (primary)**
  ```bash
  # CSV is the default and should always work
  pytest tests/integration/test_data_sources.py::TestCSVDataSource -v
  ```
  - ✅ CSV files load correctly
  - ✅ Date filtering works
  - ✅ Symbol listing works
  - ✅ Date range queries work

  **Step 3: Test SQLite database source**
  ```bash
  # SQLite tests create temporary databases
  pytest tests/integration/test_data_sources.py::TestSQLiteDataSource -v
  ```
  - ✅ SQLite database loads data correctly
  - ✅ Date filtering works
  - ✅ Symbol listing works
  - ✅ Works with load_ohlcv_data() function

  **Step 4: Test PostgreSQL database source (if configured)**
  ```bash
  # Requires PostgreSQL database to be set up
  # Set environment variables:
  export POSTGRES_HOST=localhost
  export POSTGRES_PORT=5432
  export POSTGRES_DB=test_db
  export POSTGRES_USER=test_user
  export POSTGRES_PASSWORD=test_password

  # Then test (if database is available)
  pytest tests/integration/test_data_sources.py::TestPostgreSQLDataSource -v
  ```
  - ⚠️ PostgreSQL tests are optional (skip if database not configured)
  - ✅ If configured, verify data loads correctly
  - ✅ Verify date filtering works
  - ✅ Verify symbol listing works

  **Step 5: Test Parquet data source (if pyarrow available)**
  ```bash
  # Requires pyarrow: pip install pyarrow
  pytest tests/integration/test_data_sources.py::TestParquetDataSource -v
  ```
  - ⚠️ Parquet tests are optional (skip if pyarrow not installed)
  - ✅ If available, verify data loads correctly
  - ✅ Verify works with load_ohlcv_data() function

  **Step 6: Test HDF5 data source (if tables available)**
  ```bash
  # Requires tables (PyTables): pip install tables
  pytest tests/integration/test_data_sources.py::TestHDF5DataSource -v
  ```
  - ⚠️ HDF5 tests are optional (skip if tables not installed)
  - ✅ If available, verify data loads correctly
  - ✅ Verify works with load_ohlcv_data() function

  **Step 7: Test API data sources (if API keys configured)**
  ```bash
  # AlphaVantage API (free tier available)
  export ALPHAVANTAGE_API_KEY=your_api_key
  pytest tests/integration/test_data_sources.py::TestAPIDataSources::test_alphavantage_source_loads_data -v

  # Massive API (requires paid account, formerly Polygon.io)
  export MASSIVE_API_KEY=your_api_key
  pytest tests/integration/test_data_sources.py::TestAPIDataSources::test_massive_source_loads_data -v
  ```
  - ⚠️ API tests are optional (skip if API keys not set)
  - ✅ If configured, verify data loads correctly
  - ✅ Verify rate limiting works
  - ✅ Verify error handling for API failures

  **Step 8: Test data source interchangeability**
  ```bash
  # Verify all sources produce same data format
  pytest tests/integration/test_data_sources.py::TestDataSourceInterchangeability -v
  ```
  - ✅ All sources produce data in same format
  - ✅ All sources work with load_ohlcv_data() function
  - ✅ Data structure is consistent across sources

  **What to verify:**
  - ✅ CSV source works (primary, always available)
  - ✅ SQLite source works (if sqlite3 available)
  - ⚠️ PostgreSQL source works (if database configured)
  - ⚠️ Parquet source works (if pyarrow installed)
  - ⚠️ HDF5 source works (if tables installed)
  - ⚠️ API sources work (if API keys configured)
  - ✅ All sources produce consistent data format
  - ✅ All sources work with load_ohlcv_data() function
  - ✅ Date filtering works for all sources
  - ✅ Symbol listing works for all sources

  **Expected results:**
  - CSV tests should always pass (primary source)
  - SQLite tests should pass (built-in Python module)
  - Other sources may be skipped if dependencies/credentials not available
  - All available sources should produce data in same format
  - All sources should work with the same load_ohlcv_data() interface

- [ ] **Test data quality edge cases**

  **Step 1: Run data quality edge case tests**
  ```bash
  # Run all data quality edge case tests
  pytest tests/integration/test_data_quality_edge_cases.py -v

  # Test specific edge case category
  pytest tests/integration/test_data_quality_edge_cases.py::TestMissingDays -v
  pytest tests/integration/test_data_quality_edge_cases.py::TestExtremePriceMoves -v
  pytest tests/integration/test_data_quality_edge_cases.py::TestLowVolumeDays -v
  pytest tests/integration/test_data_quality_edge_cases.py::TestGapsInData -v
  pytest tests/integration/test_data_quality_edge_cases.py::TestDuplicateDates -v
  ```

  **Step 2: Test missing days (holidays, weekends)**
  ```bash
  pytest tests/integration/test_data_quality_edge_cases.py::TestMissingDays -v
  ```
  - ✅ Equity data correctly handles missing weekends (expected)
  - ✅ Equity data correctly handles missing holidays
  - ✅ Crypto data correctly detects missing days (crypto trades 24/7)
  - ✅ Missing days don't cause validation to fail (warnings only)
  - ✅ Backtest handles missing days gracefully

  **Step 3: Test extreme price moves (>50%)**
  ```bash
  pytest tests/integration/test_data_quality_edge_cases.py::TestExtremePriceMoves -v
  ```
  - ✅ Extreme moves are detected correctly
  - ✅ Extreme moves are handled correctly (treated as missing data per EDGE_CASES.md)
  - ✅ Validation passes with warnings (doesn't fail)
  - ✅ Multiple extreme moves are detected
  - ✅ System skips extreme move bars during signal generation

  **Step 4: Test low volume days**
  ```bash
  pytest tests/integration/test_data_quality_edge_cases.py::TestLowVolumeDays -v
  ```
  - ✅ Low volume days are detected
  - ✅ Zero volume is handled correctly (volume >= 0 is valid)
  - ✅ Very low volume days (< 1000 shares) are handled
  - ✅ Validation passes with low volume days

  **Step 5: Test gaps in data (consecutive missing days)**
  ```bash
  pytest tests/integration/test_data_quality_edge_cases.py::TestGapsInData -v
  ```
  - ✅ Single day gaps are detected
  - ✅ Consecutive gaps (2+ days) are detected
  - ✅ Multiple gaps are detected correctly
  - ✅ Large gaps (>5 days) are handled
  - ✅ Gaps don't cause validation to fail (warnings only)

  **Step 6: Test duplicate dates**
  ```bash
  pytest tests/integration/test_data_quality_edge_cases.py::TestDuplicateDates -v
  ```
  - ✅ Duplicate dates are detected
  - ✅ Validation fails with duplicate dates (errors, not warnings)
  - ✅ Multiple duplicate dates are detected
  - ✅ Backtest rejects data with duplicate dates (or cleans it)

  **Step 7: Test combined edge cases**
  ```bash
  pytest tests/integration/test_data_quality_edge_cases.py::TestCombinedEdgeCases -v
  ```
  - ✅ System handles multiple edge cases simultaneously
  - ✅ Missing days + extreme moves
  - ✅ Low volume + gaps

  **What to verify:**
  - ✅ Missing days (holidays, weekends):
    - Equity: Missing weekends are expected (not errors)
    - Equity: Missing holidays are detected but don't fail validation
    - Crypto: Missing days are detected (crypto should trade 24/7)
    - System continues processing despite missing days
  - ✅ Extreme price moves (>50%):
    - Detected during validation (warnings logged)
    - Treated as missing data per EDGE_CASES.md
    - Bars with extreme moves are skipped during signal generation
    - Validation passes (warnings, not errors)
  - ✅ Low volume days:
    - Detected but don't fail validation
    - Zero volume is valid (volume >= 0)
    - Very low volume days are handled gracefully
  - ✅ Gaps in data:
    - Single day gaps detected
    - Consecutive gaps (2+ days) detected
    - Gap lengths calculated correctly
    - Gaps don't cause validation to fail
  - ✅ Duplicate dates:
    - Detected during validation
    - Cause validation to fail (errors, not warnings)
    - Backtest rejects or cleans duplicate dates

  **Expected results:**
  - All edge case tests should pass
  - Missing days, extreme moves, low volume, and gaps: warnings only (validation passes)
  - Duplicate dates: errors (validation fails)
  - System handles all edge cases gracefully without crashing
  - Backtest continues processing despite data quality issues (except duplicates)

---

### 8. Documentation Verification

#### Documentation Completeness
- [ ] **User guide is complete**
  - Getting started guide works
  - Examples are up-to-date
  - Best practices are documented
  - Troubleshooting guide covers common issues

- [x] **API documentation is complete**
  - All public APIs are documented
  - Examples are provided
  - Parameter descriptions are clear

- [x] **Configuration documentation**
  - All config parameters are documented
  - Example configs are provided
  - Migration guide exists (if applicable)

---

## 🟡 Deployment Checklist

### 9. Infrastructure Setup

#### Docker Deployment (Recommended)
- [ ] **Build Docker image**
  ```bash
  docker build -t trading-system:latest .
  ```
  - Image builds successfully
  - Image size is reasonable
  - No security vulnerabilities in base image

- [ ] **Test Docker container**
  ```bash
  docker run --rm trading-system:latest --help
  docker run --rm -v $(pwd)/EXAMPLE_CONFIGS:/app/configs:ro \
    trading-system:latest config validate --path /app/configs/run_config.yaml
  ```
  - Container runs correctly
  - Volumes mount correctly
  - Commands execute successfully

- [ ] **Test docker-compose**
  ```bash
  docker-compose build
  docker-compose run --rm trading-system --help
  ```
  - Compose file is valid
  - Services start correctly

#### Environment Setup
- [ ] **Verify Python version**
  - Python 3.9+ (3.11+ recommended)
  - All dependencies install correctly
  - No version conflicts

- [ ] **Verify system dependencies**
  - Required system libraries are available
  - File system permissions are correct
  - Network access (if using API sources)

---

### 10. Monitoring & Logging

#### Logging Configuration
- [x] **Verify logging setup**
  - ✅ Logs are written to appropriate location: `{output_dir}/{log_file}` (default: `results/{run_id}/backtest.log`)
    - Location configured via `config.output.base_path` and `config.output.log_file`
    - Directory is created automatically if it doesn't exist
  - ✅ Log levels are configured correctly:
    - Configurable via `config.output.log_level` (default: "INFO")
    - Validated against allowed values: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - Console handler uses configured level, file handler always uses DEBUG for full detail
  - ✅ Log rotation is configured:
    - Using `RotatingFileHandler` with 10MB max size and 5 backup files
    - Additional rotation via loguru if available (10MB rotation, 5 file retention, zip compression)
  - ✅ Structured logging is enabled (optional):
    - `StructuredFormatter` class provides JSON output format
    - Enabled via `config.output.log_json_format` flag (default: false)
    - Includes timestamp, level, logger, message, module, function, line, and exception info

- [x] **Test log output**
  ```bash
  # Run a backtest and capture logs
  python -m trading_system backtest --config EXAMPLE_CONFIGS/run_config.yaml 2>&1 | tee test.log

  # Or check the log file directly
  tail -100 results/{run_id}/train/backtest.log
  ```

  **Verify log content:**
  - ✅ **Logs contain sufficient detail:**
    - Logging initialization message: `"Logging initialized. Log file: {path}, JSON format: {bool}, Rich: {bool}"`
    - Trade events logged: `TRADE_ENTRY`, `TRADE_EXIT`, `TRADE_STOP_HIT`, `TRADE_REJECTED` with prices, quantities, P&L
    - Signal generation logged: `SIGNAL_GENERATED` or `SIGNAL_NOT_GENERATED` with reasons
    - Portfolio snapshots: `PORTFOLIO_SNAPSHOT` with equity, cash, positions, P&L, exposure
    - Daily processing: Error messages include date context when failures occur

  - ✅ **Error messages are clear:**
    - Errors include context: date, step (load_data, compute_features, generate_signals, process_signals, update_portfolio)
    - Error messages are descriptive (e.g., "Backtest error at date: 2023-01-15, step: process_signals")
    - Warnings for data issues: `MISSING_DATA_1DAY`, `DATA_UNHEALTHY`, `CONSECUTIVE_GAP`
    - Exception stack traces are logged (when exceptions occur)

  - ✅ **Performance metrics are logged:**
    - Performance metrics: `PERFORMANCE: {operation} | Duration: {seconds}s | Memory: {MB} MB`
    - Metrics include operation name, duration, and memory usage (if psutil available)
    - Performance context manager logs timing for operations

  **Expected log structure:**
  ```
  INFO - Logging initialized. Log file: results/run_20240101_120000/train/backtest.log, JSON format: False, Rich: True
  INFO - TRADE_ENTRY: AAPL | Price: 150.00 | Qty: 10 | Stop: 145.00 | Trigger: BREAKOUT_20D
  INFO - TRADE_EXIT: AAPL | Exit Price: 155.00 | Reason: PROFIT_TARGET | P&L: 50.00 | R-Multiple: 1.00
  WARNING - TRADE_STOP_HIT: AAPL | Stop Price: 145.00 | Exit Price: 144.50 | P&L: -55.00
  INFO - PORTFOLIO_SNAPSHOT: 2023-01-15 | Equity: $100,050.00 | Cash: $99,450.00 | Positions: 1 | ...
  DEBUG - PERFORMANCE: compute_features | Duration: 0.0234s | Memory: 125.50 MB
  ```

#### Monitoring Recommendations
- [x] **Built-in application monitoring** (available)
  - ✅ **Memory monitoring**:
    - `MemoryProfiler` class tracks RSS, VMS, and memory percentage via psutil
    - `PerformanceContext` logs memory usage for operations
    - Memory snapshots and profiling utilities available
  - ✅ **Performance metrics tracking**:
    - `log_performance_metric()` logs operation duration and memory usage
    - Performance context manager tracks timing and memory deltas
    - Metrics logged to file with structured format
  - ✅ **Live trading monitoring** (if using live trading):
    - `LiveMonitor` class monitors positions, risk limits, and generates alerts
    - Tracks exposure limits, position counts, unrealized P&L, cash levels
    - Risk alerts with configurable thresholds
  - ✅ **Error tracking**:
    - Errors logged with context (date, step, symbol)
    - Exception stack traces captured
    - Warning codes for data issues (MISSING_DATA_1DAY, DATA_UNHEALTHY, etc.)
  - ✅ **Dashboard health checks**:
    - Docker healthcheck configured for dashboard service
    - Health endpoint: `http://localhost:8501/_stcore/health`

- [ ] **External system monitoring** (recommended for production)
  - **System resource monitoring** (CPU, memory, disk):
    ```bash
    # Use system monitoring tools:
    # - Prometheus + Grafana (recommended)
    # - Datadog, New Relic, or similar APM tools
    # - System monitoring: htop, iostat, vmstat
    # - Log aggregation: ELK stack, Splunk, or cloud logging services
    ```
    - Monitor CPU usage during backtests
    - Monitor disk I/O for data loading operations
    - Set up alerts for resource thresholds (e.g., >80% CPU, >90% memory, >85% disk)

  - **Application health checks**:
    ```bash
    # Create health check endpoint (if needed):
    # - HTTP endpoint returning system status
    # - Check database connectivity (if using database storage)
    # - Check data source availability
    # - Verify log file write permissions
    ```

  - **Error rate monitoring**:
    ```bash
    # Parse log files for error rates:
    grep -c "ERROR" results/*/train/backtest.log
    grep -c "WARNING" results/*/train/backtest.log

    # Or use log aggregation tools to:
    # - Track error rates over time
    # - Alert on error rate spikes
    # - Monitor specific error codes
    ```

  - **Performance baseline tracking**:
    ```bash
    # Monitor key performance metrics:
    # - Backtest duration (should scale linearly)
    # - Memory usage (should remain stable)
    # - Feature computation time (should be consistent)
    # - Compare against baseline benchmarks
    ```

  **Recommended monitoring stack:**
  - **Logs**: Centralized logging (ELK, Splunk, or cloud logging)
  - **Metrics**: Prometheus + Grafana for system and application metrics
  - **Alerts**: AlertManager or PagerDuty for critical alerts
  - **APM**: Application Performance Monitoring tool for detailed tracing

---

## 🟢 Post-Deployment Verification

### 11. Smoke Tests

#### Quick Verification
- [ ] **Run smoke test**
  ```bash
  ./quick_test.sh
  # Or manually:
  python -m trading_system config validate --path EXAMPLE_CONFIGS/run_config.yaml
  python -m trading_system backtest --config EXAMPLE_CONFIGS/run_config.yaml --period train
  ```
  - All smoke tests pass
  - System is responsive
  - No critical errors

- [ ] **Verify CLI commands**
  ```bash
  python -m trading_system --help
  python -m trading_system backtest --help
  python -m trading_system validate --help
  ```
  - All commands work correctly
  - Help text is accurate

---

### 12. Production Data Verification

#### Test with Production Data
- [ ] **Load production data**
  - Verify data loading works with production data
  - Verify data validation passes
  - Verify no data quality issues

- [ ] **Run small production backtest**
  - Run backtest on small subset of production data
  - Verify results are reasonable
  - Verify no errors or warnings

---

## 📋 Quick Reference Commands

### Pre-Deployment Verification Script
```bash
#!/bin/bash
# production_verification.sh

set -e

echo "=== Production Readiness Verification ==="

echo "1. Running tests..."
pytest tests/ -v --tb=short

echo "2. Checking test coverage..."
pytest --cov=trading_system --cov-report=term --cov-report=html

echo "3. Running linter..."
flake8 trading_system/ tests/ --max-line-length=100 --extend-ignore=E203,W503

echo "4. Validating configs..."
for config in EXAMPLE_CONFIGS/*.yaml; do
    echo "  Validating $config..."
    python -m trading_system config validate --path "$config" || exit 1
done

echo "5. Running end-to-end test..."
pytest tests/integration/test_end_to_end.py::TestFullBacktest -v

echo "6. Running performance benchmarks..."
pytest tests/performance/ -m performance --benchmark-only

echo "7. Checking for hardcoded secrets..."
if grep -r "api_key.*=" trading_system/ --exclude-dir=__pycache__ | grep -v "#" | grep -v "Optional"; then
    echo "  WARNING: Potential hardcoded API keys found"
    exit 1
fi

echo "=== All checks passed! ==="
```

### Run Full Verification
```bash
chmod +x production_verification.sh
./production_verification.sh
```

---

## 🚨 Troubleshooting Production Issues

### Common Issues and Solutions

#### Issue: Tests Fail
**Symptoms**: Tests fail during verification
**Solutions**:
- Check Python version (3.9+ required)
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check for environment-specific issues
- Review test output for specific error messages

#### Issue: Config Validation Fails
**Symptoms**: Config files fail validation
**Solutions**:
- Verify YAML syntax is correct
- Check for deprecated parameters
- Verify all required fields are present
- Review config schema documentation

#### Issue: Performance Degradation
**Symptoms**: Benchmarks are slower than expected
**Solutions**:
- Check system load (CPU, memory)
- Verify data size matches expected
- Profile slow operations
- Review recent code changes

#### Issue: Missing Data Errors
**Symptoms**: System fails with missing data
**Solutions**:
- Verify data files exist and are readable
- Check data file format (CSV structure)
- Verify date ranges are correct
- Review missing data handling logic

#### Issue: Docker Build Fails
**Symptoms**: Docker image build fails
**Solutions**:
- Check Docker version (20.10+)
- Verify Dockerfile syntax
- Check for network issues (dependency downloads)
- Review base image availability

---

## 📊 Production Readiness Scorecard

Use this scorecard to track progress:

| Category | Status | Notes |
|----------|--------|-------|
| Test Coverage (>90%) | ⬜ | Current: ___% |
| All Tests Pass | ⬜ | |
| End-to-End Test Verified | ⬜ | |
| Config Validation | ⬜ | |
| Security Review | ⬜ | |
| Error Handling Verified | ⬜ | |
| Performance Benchmarks | ⬜ | |
| Real-World Data Tested | ⬜ | |
| Documentation Complete | ⬜ | |
| Docker Deployment | ⬜ | |
| Monitoring Setup | ⬜ | |
| Smoke Tests Pass | ⬜ | |

**Overall Status**: ⬜ Ready for Production

---

## 🎯 Next Steps After Verification

Once all items are checked:

1. **Create production deployment plan**
   - Document deployment steps
   - Create rollback plan
   - Set up monitoring

2. **Schedule deployment**
   - Choose low-traffic window
   - Notify stakeholders
   - Prepare support team

3. **Post-deployment monitoring**
   - Monitor logs for errors
   - Verify system performance
   - Check output files are generated correctly

4. **Document production configuration**
   - Document production-specific settings
   - Create runbook for common operations
   - Update troubleshooting guide with production issues

---

## 📝 Notes

- **Last Updated**: 2024-12-19
- **Next Review**: After critical items completed
- **Maintainer**: Development Team

---

## 🔗 Related Documents

- [MVP_READINESS_ASSESSMENT.md](MVP_READINESS_ASSESSMENT.md) - Overall MVP status
- [docs/PERFORMANCE_CHARACTERISTICS.md](docs/PERFORMANCE_CHARACTERISTICS.md) - Performance benchmarks
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Troubleshooting guide
- [README.md](README.md) - Getting started guide
- [docs/user_guide/getting_started.md](docs/user_guide/getting_started.md) - User guide

